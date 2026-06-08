# Pinocchio 动力学补偿方案：训练与部署一致性设计

> 生成日期：2026-06-08
> 前置文档：[project_structure_analysis.md](project_structure_analysis.md)（第一至三部分）
> 目标：用 Pinocchio 实现重力补偿、Coriolis/离心力补偿、计算力矩控制、自抗扰控制，确保仿真训练与真机部署使用**完全相同的动力学计算**

---

## 1. 补偿目标：到底补偿什么

刚体机器人的运动方程：

```
M(q) q̈ + C(q, q̇) q̇ + g(q) = τ + τ_ext
 │      │          │      │     │
 │      │          │      │     └── 外部扰动（接触力、未建模动态）
 │      │          │      └── 控制力矩
 │      │          └── 重力项（姿态相关，静止时也存在）
 │      └── Coriolis + 离心力（速度相关，旋转/高速运动时显著）
 └── 惯性矩阵（构型相关）
```

| 补偿项 | 物理意义 | 何时显著 | 不补偿的后果 |
|--------|---------|---------|------------|
| **g(q)** — 重力补偿 | 抵抗各连杆自重 | 始终，尤其是人形站立时 | 关节漂移、稳态误差、需要策略额外学习抗重 |
| **C(q,q̇)q̇** — Coriolis + 离心力 | 旋转时连杆间的速度耦合力 | 快速运动、转向、翻身 | 轨迹跟踪滞后、振荡 |
| **M(q)** — 惯性解耦 | 使各关节动态解耦为独立线性系统 | 构型变化大时（蹲下vs站立） | PD 增益在不同构型下效果不一致 |
| **τ_ext** — 未建模扰动 | 摩擦、柔性、负载变化、外部力 | sim2real gap 的主要来源 | 仿真中学到的策略在真机上失效 |

---

## 2. 整体架构：训练与部署共享同一套动力学计算

```
┌─────────────────────────────────────────────────────────────────────┐
│                      共享动力学模型层                                │
│                                                                     │
│  Pinocchio Model (从 URDF/MJCF 构建，一次性)                       │
│    ├── 刚体动力学参数：质量、惯性、连杆长度                          │
│    ├── 关节拓扑：parent-child 链、joint 类型                        │
│    └── 计算接口：rnea(), crba(), aba()                              │
│                                                                     │
│  补偿计算函数 (Python / C++，同一份逻辑)                            │
│    ├── gravity_compensation(q)        → τ_g = g(q)                │
│    ├── coriolis_compensation(q, q̇)    → τ_c = C(q,q̇)q̇           │
│    ├── computed_torque_control(q, q̇, q_d, Kp, Kd)                │
│    │     → τ = M(q)[Kp(q_d-q) + Kd(q̇_d-q̇)] + C(q,q̇)q̇ + g(q) │
│    └── adrc_control(q, q̇, q_d, ...) → CTC + ESO 扰动补偿         │
│                                                                     │
├────────────────────────┬────────────────────────────────────────────┤
│    训练侧 (仿真)       │         部署侧 (真机)                      │
│                        │                                            │
│  输入: MuJoCo qpos/qvel│  输入: 关节编码器 + IMU                    │
│  输出: → pre_step_ctrl │  输出: → 电机力矩指令                      │
│  后端: MuJoCo 物理引擎 │  后端: Unitree SDK / 自研驱动              │
│  采样: 4096 envs 并行  │  采样: 单 env, 实时 <1ms                   │
│  DR: 随机化质量/摩擦   │  DR: 无（真实物理即随机化）                 │
└────────────────────────┴────────────────────────────────────────────┘
```

**一致性保证的核心**：训练和部署使用**同一个** Pinocchio 模型文件 + **同一份**补偿计算代码。区别仅在于状态采样的来源和力矩输出的去向。

---

## 3. 训练侧详细设计

### 3.1 控制流改造

当前 Go2 的控制流（路径 A — MuJoCo position actuator 内建 PD）：

```
策略 actions → apply_action() → ctrl(目标位置) → MuJoCo position actuator(PD) → 力矩 → 物理
```

改造后的控制流（`pre_step_control` + Pinocchio 补偿 + motor actuator）：

```
策略 actions → apply_action() → ctrl(目标位置) → pre_step_control() → 力矩 → 物理
                                              │
                                    ┌─────────▼──────────┐
                                    │ PinocchioDynamics   │
                                    │  .compute_bias(q,q̇) │
                                    │  → g(q) + C(q,q̇)q̇  │
                                    │                      │
                                    │ ComputedTorqueCtrl   │
                                    │  τ = Kp(q_d-q)      │
                                    │    + Kd(0-q̇)        │
                                    │    + g(q)            │
                                    │    + C(q,q̇)q̇        │
                                    │  (或 + ADRC 扰动补偿) │
                                    └──────────────────────┘
```

### 3.2 MuJoCo Actuator 模式切换

**问题**：当前所有机器人的 XML 定义的是 `<position>` actuator，内建 PD。如果 `pre_step_control` 已经计算了力矩（含补偿项），MuJoCo 不应该再做一次 PD，否则力矩会叠加。

**方案：运行时将 position actuator 切换为 motor actuator**

在 env init 阶段，当 `controller_type != "pd"` 时：

```python
def _switch_to_motor_actuators(self):
    """在运行时将 position actuator 切换为 motor (纯力矩透传)。
    
    只修改 MuJoCo model 的 gain/bias 参数，不修改 XML。
    """
    model = self._backend._model
    nu = model.nu
    # 将 gain 设为 1.0 (透传 ctrl 值)
    model.actuator_gainprm[:nu, 0] = 1.0
    # 将 bias 设为 0 (无内建 PD)
    model.actuator_biasprm[:nu, 0] = 0.0
    model.actuator_biasprm[:nu, 1] = 0.0
    model.actuator_biasprm[:nu, 2] = 0.0
```

**何时 `controller_type == "pd"`**：不切换，保持原有 position actuator 行为，确保向后兼容。

### 3.3 Pinocchio 模型构建

#### 3.3.1 从 MJCF 构建（首选）

```python
import pinocchio as pin

def build_pinocchio_model_from_mjcf(mjcf_path: str) -> pin.Model:
    """从现有 MJCF 文件构建 Pinocchio 模型。

    优势：直接复用仿真资产，惯性参数天然一致。
    """
    model = pin.buildModelFromMjcf(mjcf_path)
    return model
```

**验证步骤**（在 Phase 0 必须完成）：

1. 从 MJCF 构建的 Pinocchio 模型的 `nq`、`nv` 是否与 MuJoCo model 一致
2. 关节名称和顺序是否对齐
3. 同一个 `qpos`/`qvel` 下，Pinocchio 的 `rnea(q, v, 0)` 是否与 MuJoCo 的 `data.qfrc_bias` 数值一致（误差 < 1e-6）

#### 3.3.2 从 URDF 构建（备选）

如果 Pinocchio 的 MJCF 加载器不支持某些特性：

```python
def build_pinocchio_model_from_urdf(urdf_path: str) -> pin.Model:
    model = pin.buildModelFromURDF(urdf_path)
    return model
```

**URDF 来源**：

| 机器人 | URDF 获取 | 注意事项 |
|--------|----------|---------|
| Go2 | Unitree `unitree_ros/go2_description` | 惯性参数需与 `go2.xml` 对齐 |
| G1 | Unitree `unitree_ros/g1_description` | 同上 |
| Go2W | 从 Go2 URDF 衍生 + 轮关节 | 需手动添加轮 |

**惯性参数对齐**：如果 URDF 和 MJCF 的惯性参数不一致，必须以 MJCF 为准修改 URDF。否则 Pinocchio 的补偿计算与 MuJoCo 物理不匹配，训练出的策略在仿真中就会行为异常。

### 3.4 状态对齐：MuJoCo ↔ Pinocchio

这是最容易出错的部分。MuJoCo 和 Pinocchio 的广义坐标表示不同：

#### 3.4.1 四元数顺序

| | MuJoCo | Pinocchio |
|--|--------|-----------|
| 顺序 | **wxyz** (scalar-first) | **xyzw** (scalar-last) |
| 示例 | `[w, x, y, z]` | `[x, y, z, w]` |

**转换函数**：

```python
def mj_quat_to_pin_quat(mj_quat: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz → Pinocchio xyzw"""
    return mj_quat[..., [1, 2, 3, 0]]  # (num_envs, 4)
```

#### 3.4.2 浮动基座的广义坐标

| | MuJoCo | Pinocchio |
|--|--------|-----------|
| q (位置+姿态) | `[x, y, z, w, x, y, z]` — 7D | `[x, y, z, qx, qy, qz, qw]` — 7D |
| v (速度) | `[vx, vy, vz, wx, wy, wz]` — 6D | `[vx, vy, vz, wx, wy, wz]` — 6D |
| 速度参考系 | 世界系 | 取决于 `referenceFrame` 设置 |

对于 Go2/G1 等浮动基座机器人，q 的布局是 `pos(3) + quat(4)` + `joint_angles(N)`。

**关键**：MuJoCo 的 `data.qpos` 包含完整 q（含浮动基座），而 `backend.get_dof_pos()` 只返回关节部分（不含基座）。Pinocchio 需要**完整 q**。

#### 3.4.3 状态提取与转换封装

```python
class PinocchioDynamicsModel:
    """封装 Pinocchio 动力学计算，处理 MuJoCo ↔ Pinocchio 状态对齐。"""

    def __init__(self, mjcf_path: str, backend: SimBackend):
        # 1. 构建 Pinocchio 模型
        self._model = pin.buildModelFromMjcf(mjcf_path)
        self._data = self._model.createData()

        # 2. 记录维度
        self.nq = self._model.nq    # Go2: 7 + 12 = 19
        self.nv = self._model.nv    # Go2: 6 + 12 = 18
        self.nu = backend.num_actuators  # Go2: 12

        # 3. 构建关节映射
        self._build_joint_mapping(backend)

    def _build_joint_mapping(self, backend: SimBackend):
        """验证 Pinocchio 和 MuJoCo 的关节名称/顺序对齐。"""
        mj_joint_names = self._get_mj_joint_names(backend)
        pin_joint_names = [self._model.names[i] for i in range(1, self._model.njoints)]
        # 确保名称一一对应，记录任何重排索引
        ...

    def compute_bias_force(
        self,
        base_pos: np.ndarray,      # (num_envs, 3)
        base_quat_mj: np.ndarray,  # (num_envs, 4) — MuJoCo wxyz
        dof_pos: np.ndarray,       # (num_envs, nu)
        base_lin_vel: np.ndarray,  # (num_envs, 3)
        base_ang_vel: np.ndarray,  # (num_envs, 3)
        dof_vel: np.ndarray,       # (num_envs, nu)
    ) -> np.ndarray:
        """计算 g(q) + C(q, q̇)q̇ — 重力 + Coriolis/离心力。

        返回 (num_envs, nv) 的 bias force 向量。
        """
        num_envs = base_pos.shape[0]

        # 1. 组装 Pinocchio 格式的完整 q
        base_quat_pin = mj_quat_to_pin_quat(base_quat_mj)
        q_pin = np.concatenate([base_pos, base_quat_pin, dof_pos], axis=1)  # (num_envs, nq)

        # 2. 组装 Pinocchio 格式的完整 v
        v_pin = np.concatenate([base_lin_vel, base_ang_vel, dof_vel], axis=1)  # (num_envs, nv)

        # 3. 批量计算 RNEA(q, v, a=0) = C(q,v)v + g(q)
        bias = np.zeros((num_envs, self.nv), dtype=np.float64)
        a_zero = np.zeros((self.nv,), dtype=np.float64)
        for i in range(num_envs):
            bias[i] = pin.rnea(self._model, self._data, q_pin[i], v_pin[i], a_zero).copy()

        # 4. 只返回关节部分（去掉浮动基座的前 6 个分量）
        return bias[:, 6:]  # (num_envs, nu)

    def compute_gravity_only(
        self,
        base_pos: np.ndarray,
        base_quat_mj: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        """仅计算 g(q) — 重力项。"""
        base_quat_pin = mj_quat_to_pin_quat(base_quat_mj)
        q_pin = np.concatenate([base_pos, base_quat_pin, dof_pos], axis=1)
        v_zero = np.zeros((self.nv,), dtype=np.float64)
        a_zero = np.zeros((self.nv,), dtype=np.float64)
        num_envs = base_pos.shape[0]
        gravity = np.zeros((num_envs, self.nv), dtype=np.float64)
        for i in range(num_envs):
            # rnea(q, v=0, a=0) = g(q)  (C(q,0)*0 = 0)
            gravity[i] = pin.rnea(self._model, self._data, q_pin[i], v_zero, a_zero).copy()
        return gravity[:, 6:]  # (num_envs, nu)
```

### 3.5 补偿控制器实现

#### 3.5.1 重力补偿 (Gravity Compensation)

最基础的补偿，效果立竿见影：

```python
class GravityCompController(MotorController):
    """PD + 重力补偿。

    τ = Kp * (q_d - q) + Kd * (0 - q̇) + g(q)

    效果：策略不再需要学习对抗重力，关节在零输入时自然悬停。
    """

    def __init__(self, pin_model: PinocchioDynamicsModel, kp, kd):
        self._pin = pin_model
        self._kp = kp   # (nu,) or scalar
        self._kd = kd

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        # 1. 从 backend 读取完整状态
        base_pos = backend.get_base_pos()           # (E, 3)
        base_quat = backend.get_base_quat()         # (E, 4) wxyz
        base_lin_vel = backend.get_base_lin_vel()   # (E, 3)
        base_ang_vel = backend.get_base_ang_vel()   # (E, 3)

        # 2. Pinocchio 计算重力
        g = self._pin.compute_gravity_only(base_pos, base_quat, dof_pos)

        # 3. PD + 重力补偿
        tau = self._kp * (target - dof_pos) - self._kd * dof_vel + g
        return tau
```

#### 3.5.2 计算力矩控制 (Computed Torque Control / CTC)

在重力补偿基础上加入 Coriolis/离心力补偿和惯性解耦：

```python
class CTCController(MotorController):
    """计算力矩控制 (Computed Torque Control)。

    τ = M(q) * [Kp * (q_d - q) + Kd * (q̇_d - q̇)] + C(q,q̇)q̇ + g(q)

    效果：各关节动态被解耦为独立二阶系统，PD 增益全局有效。
    """

    def __init__(self, pin_model: PinocchioDynamicsModel, kp, kd):
        self._pin = pin_model
        self._kp = kp
        self._kd = kd

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        base_pos = backend.get_base_pos()
        base_quat = backend.get_base_quat()
        base_lin_vel = backend.get_base_lin_vel()
        base_ang_vel = backend.get_base_ang_vel()

        # 1. Pinocchio 计算 bias force (g + C*q̇)
        bias = self._pin.compute_bias_force(
            base_pos, base_quat, dof_pos,
            base_lin_vel, base_ang_vel, dof_vel
        )

        # 2. Pinocchio 计算质量矩阵的关节部分
        M_jnt = self._pin.compute_mass_matrix_joint(
            base_pos, base_quat, dof_pos
        )   # (num_envs, nu, nu)

        # 3. PD 期望加速度
        q_dot_desired = np.zeros_like(dof_vel)  # 目标速度 = 0（位置跟踪）
        a_desired = self._kp * (target - dof_pos) + self._kd * (q_dot_desired - dof_vel)

        # 4. CTC 公式
        #    τ = M * a_desired + bias
        tau = np.einsum('eij,ej->ei', M_jnt, a_desired) + bias
        return tau
```

#### 3.5.3 自抗扰控制 (ADRC)

在 CTC 基础上，用扩张状态观测器 (ESO) 估计并补偿未建模扰动：

```python
class ADRCController(MotorController):
    """自抗扰控制器。

    τ = M(q) * [Kp*(q_d - q) + Kd*(q̇_d - q̇) - z₃/b₀] + C(q,q̇)q̇ + g(q)
                                                         ↑ ESO 估计的扰动

    z₃ 估计了 M⁻¹ * (τ_unmodeled) 的等效扰动，实现自适应补偿。
    """

    def __init__(self, pin_model, kp, kd, eso_bw, b0, num_envs, nu):
        self._pin = pin_model
        self._kp = kp
        self._kd = kd
        self._eso_bw = eso_bw       # ESO 带宽 ω₀
        self._b0 = b0               # 控制增益估计
        self._eso_z = np.zeros((num_envs, 3, nu))  # [z₁, z₂, z₃]

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        dt = backend._sim_dt
        w = self._eso_bw

        # ── ESO 更新 ──
        e = dof_pos - self._eso_z[:, 0]             # 位置观测误差
        self._eso_z[:, 0] += dt * (self._eso_z[:, 1] + 3*w*e)
        self._eso_z[:, 1] += dt * (self._eso_z[:, 2] + 3*w**2*e)
        self._eso_z[:, 2] += dt * (w**3*e)

        # ── 误差反馈 ──
        e1 = target - self._eso_z[:, 0]             # 位置误差
        e2 = np.zeros_like(dof_vel) - self._eso_z[:, 1]  # 速度误差
        u0 = self._kp * e1 + self._kd * e2

        # ── 动力学补偿 ──
        base_pos = backend.get_base_pos()
        base_quat = backend.get_base_quat()
        base_lin_vel = backend.get_base_lin_vel()
        base_ang_vel = backend.get_base_ang_vel()
        bias = self._pin.compute_bias_force(...)
        M_jnt = self._pin.compute_mass_matrix_joint(...)

        # ── 扰动补偿 ──
        a_desired = u0 - self._eso_z[:, 2] / self._b0

        # ── 最终力矩 ──
        tau = np.einsum('eij,ej->ei', M_jnt, a_desired) + bias
        return tau

    def reset(self, env_ids):
        self._eso_z[env_ids] = 0.0
```

### 3.6 `pre_step_control` 集成

```python
# 在 Go2JoystickFlatTask.__init__() 中：
if cfg.control_config.controller_type != "pd":
    # 1. 构建 Pinocchio 模型
    mjcf_path = ASSETS_ROOT_PATH / "go2" / "go2.xml"
    self._pin_model = PinocchioDynamicsModel(str(mjcf_path), self._backend)

    # 2. 创建控制器
    self._motor_controller = resolve_controller(cfg.control_config, self._pin_model, self._backend)

    # 3. 切换 actuator 模式
    self._switch_to_motor_actuators()

    # 4. 注册 pre_step_control 回调
    self._backend.set_pre_step_control(self._pre_step_motor_control)

def _pre_step_motor_control(self, backend, policy_ctrl):
    """每个物理 substep 前调用，计算实际力矩。"""
    dof_pos = backend.get_dof_pos()
    dof_vel = backend.get_dof_vel()
    return self._motor_controller.compute_torque(backend, policy_ctrl, dof_pos, dof_vel)
```

### 3.7 性能优化：批量计算

直接循环调用 Pinocchio 的 `rnea` 在 4096 个 env 下太慢（~80ms）。需要向量化。

#### 3.7.1 策略：提取 Pinocchio 参数，手动批量 RNEA

对于 Go2 这种 12-DoF 浮动基座机器人，RNEA 递推公式的关节数很少，可以手动展开：

```python
class VectorizedRNEA:
    """对 Go2/G1 的关节链，手动展开 RNEA 递推公式实现批量计算。

    在初始化时从 Pinocchio 模型提取所有固定参数（惯性、连杆偏移等），
    在每步只做 numpy 矩阵运算，避免 Python 循环。
    """

    def __init__(self, pin_model: pin.Model):
        # 提取每个关节的惯性参数
        self._masses = ...       # (njoints,)
        self._inertias = ...     # (njoints, 3, 3)
        self._joint_axes = ...   # (njoints, 3)
        self._joint_offsets = ... # (njoints, 3)
        # 预计算常量矩阵
        ...

    def compute_bias_batch(self, q_full, v_full):
        """批量计算 C(q,v)v + g(q)。

        Args:
            q_full: (num_envs, nq)
            v_full: (num_envs, nv)
        Returns:
            bias: (num_envs, nv)
        """
        # 从基座向末端正向/逆向递推，全部用 numpy 批量操作
        ...
```

**预期性能**：手动展开的批量 RNEA，4096 env × 12 DoF，约 0.5-2ms，远快于循环的 ~80ms。

#### 3.7.2 渐进优化路径

| 阶段 | 方案 | 性能目标 | 说明 |
|------|------|---------|------|
| Phase 0 | 循环调用 Pinocchio | 无要求（验证正确性） | 减少到 ~100 env 验证 |
| Phase 1 | 手动批量 RNEA | <2ms / 4096 envs | 推荐方案 |
| Phase 2 | C++ 扩展 / Numba JIT | <0.5ms / 4096 envs | 极端性能需求 |

### 3.8 Domain Randomization 与补偿的交互

DR 随机化会改变机器人的动力学参数（质量、摩擦、惯性等），补偿计算需要感知这些变化。

#### 3.8.1 两种策略

| 策略 | 仿真中 Pinocchio 使用的参数 | 部署时 Pinocchio 使用的参数 | sim2real gap |
|------|---------------------------|---------------------------|-------------|
| **A. 用标称参数补偿** | 始终用 MJCF 标称参数 | 用标称参数 | 补偿不随 DR 变化，但策略学会了应对参数偏差 |
| **B. 用随机化后参数补偿** | 每个 env 用各自的随机化参数 | 用标称参数 | 仿真中补偿更精确，但与部署不一致 |

**推荐策略 A**：仿真中始终用标称参数做补偿。理由：
1. 部署时只知道标称参数，无法知道真实参数
2. DR 的目的就是让策略学会应对参数不确定性
3. 用标称参数补偿后，PD/CTC 在标称情况下精确，DR 造成的偏差留给策略学习
4. 这与 ADRC 的设计哲学一致：标称补偿 + ESO 估计残余扰动

#### 3.8.2 DR 参数对补偿的影响

| DR 项 | 影响的补偿项 | 标称补偿下的效果 |
|-------|------------|----------------|
| `base_mass_delta` | g(q) 改变 | 重力补偿不完全，ESO 可吸收 |
| `body_inertia` | M(q), C(q,q̇)q̇ 改变 | CTC 解耦不完美，PD 仍有基本稳定性 |
| `dof_armature` | 等效惯量增大 | MuJoCo 的 armature 不在 Pinocchio 模型中，需要额外处理 |
| `geom_friction` | 不影响补偿 | 影响接触力，属于 τ_ext |
| `gravity` | g(q) 改变 | 可选：同步修改 Pinocchio 的重力向量 |

**armature 的特殊处理**：MuJoCo 的 `armature` 在关节上添加额外惯性，等效于 `I_eff = I + armature`。Pinocchio 模型中不包含 armature，因此 CTC 计算的 M(q) 不含 armature 项。解决方案：

```python
# 从 MuJoCo backend 读取 armature，加到 Pinocchio 的质量矩阵对角线上
armature = backend.get_dof_armature()  # (nv,)
M_jnt_corrected = M_jnt + np.diag(armature[6:])  # 只加关节部分
```

---

## 4. 部署侧详细设计

### 4.1 部署架构

```
┌────────────────────────────────────────────────────────────┐
│                    部署控制器 (C++)                        │
│                                                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ 状态采样      │   │ Pinocchio    │   │ 力矩输出      │  │
│  │              │   │ 动力学计算    │   │              │  │
│  │ IMU → 基座姿态│──▶│              │──▶│ 关节电机指令  │  │
│  │ 编码器 → q   │   │ rnea()       │   │              │  │
│  │ 编码器差分→q̇ │   │ crba()       │   │ Unitree SDK  │  │
│  │              │   │              │   │ 或自研驱动    │  │
│  └──────────────┘   └──────────────┘   └──────────────┘  │
│         │                   │                    │         │
│         └───────────────────┼────────────────────┘         │
│                             │                              │
│                    同一份补偿逻辑                           │
│                  (与 Python 版本对齐)                       │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Pinocchio C++ 部署

Pinocchio 是纯头文件 C++ 库，可以零开销集成到实时控制器中：

```cpp
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>

class RobotDynamicsCompensator {
public:
    RobotDynamicsCompensator(const std::string& urdf_path) {
        pinocchio::urdf::buildModel(urdf_path, model_);
        data_ = pinocchio::Data(model_);
    }

    // 重力补偿
    Eigen::VectorXd gravity_compensation(const Eigen::VectorXd& q) {
        Eigen::VectorXd v_zero = Eigen::VectorXd::Zero(model_.nv);
        Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model_.nv);
        return pinocchio::rnea(model_, data_, q, v_zero, a_zero).tail(model_.nv - 6);
    }

    // 完整 CTC
    Eigen::VectorXd computed_torque_control(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& q_desired,
        const Eigen::VectorXd& Kp,
        const Eigen::VectorXd& Kd
    ) {
        // 1. PD 期望加速度
        Eigen::VectorXd v_desired = Eigen::VectorXd::Zero(model_.nv - 6);
        Eigen::VectorXd a_desired_full = Eigen::VectorXd::Zero(model_.nv);
        a_desired_full.tail(model_.nv - 6) =
            Kp.cwiseProduct(q_desired - q.tail(model_.nv - 6))
            - Kd.cwiseProduct(v.tail(model_.nv - 6));

        // 2. 质量矩阵
        pinocchio::crba(model_, data_, q);
        data_.M.triangularView<Eigen::StrictlyLower>() =
            data_.M.transpose().triangularView<Eigen::StrictlyLower>();

        // 3. bias force
        Eigen::VectorXd bias = pinocchio::rnea(model_, data_, q, v,
            Eigen::VectorXd::Zero(model_.nv));

        // 4. CTC 公式
        Eigen::VectorXd tau = data_.M * a_desired_full + bias;
        return tau.tail(model_.nv - 6);
    }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
};
```

### 4.3 实时性约束

| 控制器 | 计算量 | Go2 (12 DoF) 估计耗时 | 是否满足 1ms 控制周期 |
|--------|-------|---------------------|---------------------|
| 重力补偿 | `rnea(q, 0, 0)` | ~5-10 μs | ✓ |
| CTC | `crba` + `rnea` + 矩阵乘 | ~15-30 μs | ✓ |
| ADRC | CTC + ESO 更新 | ~20-40 μs | ✓ |

Pinocchio 的 C++ 实现高度优化（模板元编程 + SIMD），12-DoF 机器人的 RNEA 在嵌入式 ARM 处理器上也能在 <50μs 内完成。

### 4.4 部署侧状态获取

| 状态 | 传感器 | 频率 | 备注 |
|------|--------|------|------|
| 关节位置 q | 关节编码器 | 1 kHz | 直接读取 |
| 关节速度 q̇ | 编码器差分 + 低通滤波 | 1 kHz | 需要滤波降噪 |
| 基座姿态 | IMU (陀螺仪 + 加速度计) | 200-1000 Hz | 需要 EKF/互补滤波融合 |
| 基座线速度 | IMU 积分 + 里程计 | 200-500 Hz | 漂移问题，短期可信 |
| 基座角速度 | 陀螺仪 | 1 kHz | 直接读取 |

**关键差异**：仿真中状态是精确的，真机上有传感器噪声和延迟。这是 ADRC 的 ESO 能帮助缓解的——ESO 本身就是对有噪声观测的平滑估计器。

### 4.5 部署侧的力矩限幅与安全

```cpp
Eigen::VectorXd clamp_torques(const Eigen::VectorXd& tau,
                              const Eigen::VectorXd& tau_min,
                              const Eigen::VectorXd& tau_max) {
    return tau.cwiseMax(tau_min).cwiseMin(tau_max);
}
```

**安全策略**：
1. 力矩限幅：不超过电机额定力矩
2. 位置限幅：关节不超出物理行程
3. 紧急停止：检测到异常（倾倒、力矩过大）时切回 PD 控制器
4. 补偿力矩上限：限制补偿力矩不超过总力矩的某个比例，防止补偿项主导

---

## 5. 一致性保证：训练 ↔ 部署

### 5.1 一致性金字塔

```
                    ┌─────────────┐
                    │ 5. 行为一致  │  ← 最终目标：策略在仿真和真机上行为一致
                    ├─────────────┤
                    │ 4. 补偿一致  │  ← Pinocchio 同一模型 + 同一计算
                    ├─────────────┤
                    │ 3. 代码一致  │  ← Python 补偿逻辑与 C++ 完全对齐
                    ├─────────────┤
                    │ 2. 参数一致  │  ← URDF/MJCF 惯性参数对齐
                    └─────────────┘
                    │ 1. 模型一致  │  ← 同一个 URDF/MJCF 文件
                    └─────────────┘
```

### 5.2 Level 1：模型一致

**规则**：仿真和部署必须使用**同一个 URDF/MJCF 文件**构建 Pinocchio 模型。

**验证方法**：
```python
# 在仿真和部署的 init 阶段，dump Pinocchio 模型参数并对比
def verify_model_consistency(pin_model_sim, pin_model_deploy):
    assert pin_model_sim.nq == pin_model_deploy.nq
    assert pin_model_sim.nv == pin_model_deploy.nv
    np.testing.assert_allclose(
        pin_model_sim.inertias, pin_model_deploy.inertias, atol=1e-10
    )
    np.testing.assert_allclose(
        pin_model_sim.jointPlacements, pin_model_deploy.jointPlacements, atol=1e-10
    )
```

### 5.3 Level 2：参数一致

**惯性参数对齐清单**：

| 参数 | 来源 | 对齐方式 |
|------|------|---------|
| 连杆质量 | URDF/MJCF `<inertial mass>` | 同文件，天然一致 |
| 连杆惯性 | URDF/MJCF `<inertial diaginertia>` | 同文件，天然一致 |
| 连杆质心 | URDF/MJCF `<inertial pos>` | 同文件，天然一致 |
| 关节轴 | URDF/MJCF `<joint axis>` | 同文件，天然一致 |
| 关节限位 | URDF/MJCF `<joint range>` | 同文件，天然一致 |
| 电机力矩限幅 | URDF/MJCF `<position forcerange>` | 部署侧需要硬编码相同限幅 |
| PD 增益 Kp/Kd | YAML 配置 | 部署侧使用训练时的最终值 |
| ESO 带宽 ω₀ | YAML 配置 | 同上 |
| armature | MJCF `<joint armature>` | **需要特殊处理** |

**armature 对齐**：

MuJoCo 的 armature 不出现在 URDF/Pinocchio 中，但会影响有效惯性。两种处理方式：

1. **方案 A**：在 Pinocchio 模型中手动修改关节惯性矩阵，加上 armature 等效项。这样 CTC 计算的 M(q) 自然包含 armature。
2. **方案 B**：在补偿计算后，额外加上 armature 相关项 `τ_armature = armature * a_desired`。

**推荐方案 A**，因为更干净：

```python
def inject_armature_into_pinocchio_model(pin_model, armature_vec):
    """将 MuJoCo armature 等效为关节惯性增量，注入 Pinocchio 模型。"""
    for i, arm_val in enumerate(armature_vec):
        # 找到对应的关节惯性矩阵，加上 armature
        joint_id = i + 1  # 跳过 universe joint
        pin_model.inertias[joint_id] += arm_val * np.outer(
            pin_model.jointAxes[joint_id], pin_model.jointAxes[joint_id]
        )
```

### 5.4 Level 3：代码一致

**规则**：Python 补偿逻辑和 C++ 补偿逻辑必须**数学等价**。

**验证方法**：

```python
def verify_compensation_consistency():
    """用相同的 q, v 输入，对比 Python 和 C++ 的补偿输出。"""
    # 1. 在 Python 中计算
    bias_py = pin_model_py.compute_bias_force(q, v)

    # 2. 通过 C++ binding 或 subprocess 计算
    bias_cpp = cpp_compensator.compute_bias(q, v)

    # 3. 数值对比
    np.testing.assert_allclose(bias_py, bias_cpp, atol=1e-8)
```

**建议**：将核心补偿逻辑提取为无副作用的纯函数，Python 和 C++ 各实现一份，用同一组测试用例验证。

### 5.5 Level 4：补偿一致

**规则**：训练时和部署时的补偿计算必须使用相同的：
- 同一个 Pinocchio 模型（惯性参数）
- 同一套 Kp/Kd 增益
- 同一个 ESO 带宽 ω₀ 和 b₀
- 同一个重力向量

**配置文件共享**：

```yaml
# deploy_config.yaml — 训练和部署共用的补偿配置
compensation:
  model_path: "go2/go2.xml"     # Pinocchio 模型路径
  controller_type: "adrc"
  kp: [35.0, 35.0, 35.0, 35.0, 35.0, 35.0,
        35.0, 35.0, 35.0, 35.0, 35.0, 35.0]
  kd: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  eso_bandwidth: 10.0
  b0: 1.0
  torque_limits: [23.7, 23.7, 23.7, 23.7, 23.7, 23.7,
                  23.7, 23.7, 23.7, 23.7, 23.7, 23.7]
  use_gravity_comp: true
  use_coriolis_comp: true
  use_inertia_decouple: true  # CTC 中是否解耦 M(q)
```

### 5.6 Level 5：行为一致

**最终验证**：

1. **仿真内对比测试**：同一个策略，分别用 PD 和 CTC/ADRC 控制，对比跟踪性能
2. **Sim-to-real 迁移测试**：仿真训练的策略直接部署到真机，观察行为差异
3. **量化指标**：

| 指标 | 定义 | 目标 |
|------|------|------|
| 跟踪误差 | ‖q_actual - q_desired‖ | CTC/ADRC < PD |
| 力矩平滑度 | ‖τ_{t+1} - τ_t‖ | CTC/ADRC < PD |
| sim2real gap | 仿真 reward - 真机等效 reward | CTC/ADRC < PD |
| 抗扰恢复时间 | 受扰后恢复到稳态的时间 | ADRC < CTC < PD |

---

## 6. 补偿选择指南：何时用哪种补偿

| 场景 | 推荐控制器 | 理由 |
|------|-----------|------|
| 四足平地行走 | 重力补偿 | 速度不大，Coriolis 可忽略；重力补偿即可大幅减轻策略负担 |
| 四足快速跑/转向 | CTC | 高速运动时 Coriolis/离心力显著 |
| 人形站立/行走 | CTC | 质量大、连杆长，重力 + Coriolis 都显著 |
| 人形翻身/翻跟头 | CTC + ADRC | 高速旋转 + 大范围构型变化，需要惯性解耦 + 扰动补偿 |
| 灵巧手操控 | 重力补偿 | 手指质量小，Coriolis 可忽略 |
| manip-loco（臂+腿） | ADRC | 臂负载不确定，ESO 自适应补偿 |

**渐进式部署建议**：

```
PD (基线) → + 重力补偿 → + Coriolis 补偿 → CTC → + ESO (ADRC)
     │             │              │             │           │
     │             │              │             │           └── sim2real gap 最小
     │             │              │             └── 惯性解耦，PD 增益全局有效
     │             │              └── 高速运动时更精确
     │             └── 零成本提升，策略无需重训
     └── 当前状态
```

---

## 7. 验证与测试策略

### 7.1 单元测试

| 测试项 | 内容 | 验证目标 |
|--------|------|---------|
| 四元数转换 | `mj_quat_to_pin_quat` 正反转换 | 往返误差 < 1e-15 |
| 关节映射 | Pinocchio 与 MuJoCo 关节名称/顺序一致 | 无遗漏，无错位 |
| 重力补偿 | 静止姿态下 `τ_g = g(q)` | 补偿后关节加速度 ≈ 0 |
| CTC | 施加阶跃目标，观察跟踪 | 无超调，跟踪误差 < 0.01 rad |
| ADRC | 施加外部扰动（推力），观察恢复 | ESO 估计扰动，恢复时间 < 0.5s |
| 与 MuJoCo 对比 | 同一 q/v 下 Pinocchio rnea vs MuJoCo qfrc_bias | 相对误差 < 1e-6 |

### 7.2 集成测试

| 测试项 | 内容 |
|--------|------|
| PD 等价性 | `controller_type: "pd"` 下行为与现有完全一致 |
| Motor actuator 切换 | 运行时切换后仿真不发散 |
| DR 兼容性 | 开启 DR 后 CTC/ADRC 仍然稳定 |
| Reward 一致性 | `_estimate_pd_torques` 在 CTC 模式下仍正确估计力矩用于 reward |

### 7.3 Sim2real 验证

| 步骤 | 内容 |
|------|------|
| 1. 零重力验证 | 关闭重力，PD 控制器下机器人应漂浮 |
| 2. 重力补偿验证 | 开启重力补偿，PD 输出 0 时机器人应悬停 |
| 3. CTC 阶跃响应 | 施加关节阶跃目标，验证跟踪性能 |
| 4. ADRC 抗扰 | 人工施加外力，验证 ESO 估计和恢复 |
| 5. 策略迁移 | 仿真训练的策略直接部署，量化性能 gap |

---

## 8. 文件组织与模块结构

```
src/unilab/control/                      # 新增模块
├── __init__.py
├── base.py                              # MotorController 抽象基类
├── pd_controller.py                     # PDController
├── gravity_comp_controller.py           # GravityCompController (PD + 重力)
├── ctc_controller.py                    # CTCController (计算力矩控制)
├── adrc_controller.py                   # ADRCController (自抗扰)
├── pinocchio_model.py                   # PinocchioDynamicsModel (状态对齐封装)
├── vectorized_rnea.py                   # VectorizedRNEA (批量计算优化)
├── actuator_switch.py                   # 运行时 position→motor 切换
└── resolver.py                          # resolve_controller() 工厂函数

deploy/                                  # 部署侧（C++，不在 Python 包内）
├── include/
│   └── unilab_deploy/
│       ├── dynamics_compensator.hpp     # C++ Pinocchio 补偿接口
│       ├── adrc_controller.hpp         # C++ ADRC 实现
│       └── robot_config.hpp            # 从 YAML 加载补偿配置
├── src/
│   ├── dynamics_compensator.cpp
│   ├── adrc_controller.cpp
│   └── robot_config.cpp
├── tests/
│   ├── test_consistency.cpp            # 与 Python 补偿输出对比
│   └── test_realtime.cpp              # 实时性基准测试
└── configs/
    ├── go2_compensation.yaml           # Go2 补偿配置（训练部署共享）
    └── g1_compensation.yaml            # G1 补偿配置
```

---

## 9. 实施时间线

| Phase | 内容 | 产出 | 预计时间 |
|-------|------|------|---------|
| **0** | Pinocchio MJCF 加载验证 + 数值一致性验证 | 验证报告 | 1 周 |
| **1** | `MotorController` 抽象 + `PDController` + actuator 切换 | 可运行 PD 等价测试 | 1 周 |
| **2** | `PinocchioDynamicsModel` + `GravityCompController` | 重力补偿训练可用 | 1.5 周 |
| **3** | `CTCController` + `VectorizedRNEA` | CTC 训练可用 | 2 周 |
| **4** | `ADRCController` | ADRC 训练可用 | 1.5 周 |
| **5** | C++ 部署实现 + 一致性测试 | 真机可部署 | 2 周 |
| **6** | 真机验证 + sim2real 量化 | 完整 sim2real 验证 | 2 周 |

**总计约 11 周**。Phase 0-4 为训练侧，Phase 5-6 为部署侧。Phase 2 完成后即可开始真机部署的重力补偿验证（与 Phase 3-4 并行）。
