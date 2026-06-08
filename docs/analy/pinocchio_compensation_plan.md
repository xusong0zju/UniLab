# Pinocchio 动力学补偿方案：G1 + FlashSAC 专项设计

> 生成日期：2026-06-08（v2 — 聚焦 G1 + FlashSAC）
> 前置文档：[project_structure_analysis.md](project_structure_analysis.md)（工程结构 + PD 控制器分析 + 控制器架构）
> 目标：针对 **G1 人形机器人 + FlashSAC 算法**，用 Pinocchio 实现重力补偿、Coriolis/离心力补偿、计算力矩控制，确保仿真训练与真机部署使用**完全相同的动力学计算**，最大化 sim2real 一致性

---

## 1. 核心动机：为什么必须用 Pinocchio 而非 MuJoCo 内建 API

刚体机器人的运动方程：

$$
M(q)\,\ddot{q} + C(q,\dot{q})\,\dot{q} + g(q) = \tau + \tau_{\text{ext}}
$$

| 符号 | 物理意义 | 何时显著 | 不补偿的后果 |
|------|---------|---------|------------|
| $g(q)$ | 重力项 | 始终，尤其人形站立时 | 关节漂移、稳态误差、策略需额外学习抗重 |
| $C(q,\dot{q})\dot{q}$ | Coriolis + 离心力 | 快速运动、转向、翻身 | 轨迹跟踪滞后、振荡 |
| $M(q)$ | 惯性矩阵 | 构型变化大时（蹲下 vs 站立） | PD 增益在不同构型下效果不一致 |
| $\tau_{\text{ext}}$ | 未建模扰动 | sim2real gap 的主要来源 | 仿真策略在真机失效 |

**关键决策**：虽然 MuJoCo 提供了 `data.qfrc_bias`（= $C\dot{q}+g$）和 `mj_fullM`（= $M$），但真机部署时无法调用 MuJoCo API。如果训练时用 MuJoCo 内建动力学做补偿、部署时用 Pinocchio，两者的数值差异（惯性参数来源不同、关节顺序可能不同、armature 处理不同）会直接引入 sim2real gap。**唯一的一致性保证是：训练和部署使用同一个 Pinocchio 模型 + 同一份补偿计算代码。**

| 维度 | MuJoCo 内建 API | Pinocchio |
|------|----------------|-----------|
| sim2real 一致性 | ✗ 真机无法调用 | ✓ 真机和仿真共用 |
| 部署可行性 | ✗ 仅限仿真 | ✓ C++ 纯头文件，可部署到嵌入式端 |
| 建模来源 | MuJoCo 内部模型（与 XML 耦合） | 独立 URDF/MJCF，可与真机 SDK 共享 |
| CTC 实现 | 需提取 `mj_fullM` + `qfrc_bias`，侵入 MuJoCo 内部 | `pin.rnea(q, v, a)` 一行搞定 |

---

## 2. 整体架构：训练与部署共享同一套动力学计算

```
┌─────────────────────────────────────────────────────────────────────┐
│                      共享动力学模型层                                │
│                                                                     │
│  Pinocchio Model (从 MJCF/URDF 构建，一次性)                       │
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

**关键设计决策**：`apply_action()` 仍然输出**目标关节位置**（与现有行为一致），控制器在 `pre_step_control` 回调中完成"目标位置 → 力矩"的映射。这样：
- Env Contract 不变
- 策略网络的输出空间不变
- 控制器替换是 backend 层的局部修改

### 3.2 MuJoCo Actuator 模式切换

**问题**：当前所有机器人的 XML 定义的是 `<position>` actuator，内建 PD。如果 `pre_step_control` 已经计算了力矩（含补偿项），MuJoCo 不应该再做一次 PD，否则力矩会叠加。

**现状盘点**：

| 机器人 | XML actuator 类型 | kp/kv 来源 | forcerange |
|--------|------------------|-----------|------------|
| Go2 | `<position>` | 默认 class: kp=35 | hip: ±23.7, knee: ±45.43 |
| G1 | `<position>` | 每个 actuator 独立 kp/kv | 逐关节不同（±5 ~ ±139） |
| Go2W | `<motor>` (已切换) | N/A | N/A |

**方案：运行时将 position actuator 切换为 motor actuator**

在 env init 阶段，当 `controller_type != "pd"` 时，修改 MuJoCo model 的 gain/bias 参数，使其变为纯力矩透传（等同于 motor actuator），无需修改 XML 文件：

```python
def switch_to_motor_actuators(model) -> None:
    """在运行时将 position actuator 切换为 motor (纯力矩透传)。

    只修改 MuJoCo model 的 gain/bias 参数，不修改 XML。
    切换后 ctrl 值直接作为力矩施加，MuJoCo 不再做内建 PD。

    Args:
        model: mujoco.Model 对象（backend._model）
    """
    nu = model.nu
    # 将 gain 设为 1.0 (透传 ctrl 值)
    model.actuator_gainprm[:nu, 0] = 1.0
    # 将 bias 全部清零 (无内建 PD)
    model.actuator_biasprm[:nu, 0] = 0.0
    model.actuator_biasprm[:nu, 1] = 0.0
    model.actuator_biasprm[:nu, 2] = 0.0
```

**向后兼容**：当 `controller_type == "pd"` 时，不切换，保持原有 position actuator 行为。

**Go2W 已验证**：Go2W 轮腿机器人已经在使用 `pre_step_control` + motor actuator 模式（`compute_go2w_motor_ctrl()`），证明此路径在生产中可行。

### 3.3 Pinocchio 模型构建

#### 3.3.1 从 MJCF 构建（首选）

Pinocchio 3.x 支持 `pin.buildModelFromMjcf()`，可以直接加载现有 MJCF：

```python
import pinocchio as pin

def build_pinocchio_model(mjcf_path: str) -> pin.Model:
    """从现有 MJCF 文件构建 Pinocchio 模型。

    优势：直接复用仿真资产，惯性参数天然一致。
    """
    model = pin.buildModelFromMjcf(mjcf_path)
    return model
```

**验证步骤**（Phase 0 必须完成）：

1. 从 MJCF 构建的 Pinocchio 模型的 `nq`、`nv` 是否与 MuJoCo model 一致
2. 关节名称和顺序是否对齐
3. 同一个 `qpos`/`qvel` 下，Pinocchio 的 `rnea(q, v, 0)` 是否与 MuJoCo 的 `data.qfrc_bias` 数值一致（误差 < 1e-6）

> ⚠️ 需验证：Pinocchio 的 MJCF 加载器是否完整支持所有 MJCF 特性（`freejoint`、`armature`、多重几何体、`frictionloss` 等）。armature 和 frictionloss 不属于刚体动力学，Pinocchio 不应包含它们——但需要确认加载器不会因此报错。

#### 3.3.2 从 URDF 构建（备选）

如果 Pinocchio 的 MJCF 加载器不支持某些特性，回退到 URDF：

```python
model = pin.buildModelFromURDF(urdf_path)
```

**URDF 来源**：

| 机器人 | URDF 获取 | 注意事项 |
|--------|----------|---------|
| Go2 | Unitree `unitree_ros/go2_description` | 惯性参数需与 `go2.xml` 对齐 |
| G1 | Unitree `unitree_ros/g1_description` | 同上 |
| Go2W | 从 Go2 URDF 衍生 + 轮关节 | 需手动添加轮 |

**惯性参数对齐**：如果 URDF 和 MJCF 的惯性参数不一致，**必须以 MJCF 为准修改 URDF**。否则 Pinocchio 的补偿计算与 MuJoCo 物理不匹配，训练出的策略在仿真中就会行为异常。

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
    """封装 Pinocchio 动力学计算，处理 MuJoCo ↔ Pinocchio 状态对齐。

    冷路径：__init__ 中构建模型、验证关节映射
    热路径：compute_bias_force / compute_gravity_only / compute_mass_matrix
    """

    def __init__(self, mjcf_path: str, backend: SimBackend):
        # 1. 构建 Pinocchio 模型
        self._model = pin.buildModelFromMjcf(mjcf_path)
        self._data = self._model.createData()

        # 2. 记录维度
        self.nq = self._model.nq    # Go2: 7 + 12 = 19
        self.nv = self._model.nv    # Go2: 6 + 12 = 18
        self.nu = backend.num_actuators  # Go2: 12

        # 3. 构建关节映射 + 验证一致性
        self._build_joint_mapping(backend)

    def _build_joint_mapping(self, backend: SimBackend):
        """验证 Pinocchio 和 MuJoCo 的关节名称/顺序对齐。

        如果顺序不同，建立重排索引映射。
        """
        # Pinocchio 的关节名（跳过 universe joint 0）
        pin_joint_names = [self._model.names[i] for i in range(1, self._model.njoints)]
        # MuJoCo 的关节名（从 backend 获取）
        mj_joint_names = self._get_mj_joint_names(backend)
        # 验证一一对应，记录重排索引
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

        返回 (num_envs, nu) 的 bias force 向量（仅关节部分，不含浮动基座）。
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
        """仅计算 g(q) — 重力项。

        令 v=0, a=0, 则 rnea(q, 0, 0) = g(q)  （因为 C(q,0)·0 = 0）
        """
        base_quat_pin = mj_quat_to_pin_quat(base_quat_mj)
        q_pin = np.concatenate([base_pos, base_quat_pin, dof_pos], axis=1)
        v_zero = np.zeros((self.nv,), dtype=np.float64)
        a_zero = np.zeros((self.nv,), dtype=np.float64)
        num_envs = base_pos.shape[0]
        gravity = np.zeros((num_envs, self.nv), dtype=np.float64)
        for i in range(num_envs):
            gravity[i] = pin.rnea(self._model, self._data, q_pin[i], v_zero, a_zero).copy()
        return gravity[:, 6:]  # (num_envs, nu)

    def compute_mass_matrix_joint(
        self,
        base_pos: np.ndarray,
        base_quat_mj: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        """计算关节子空间的质量矩阵 M_jnt (num_envs, nu, nu)。

        完整 M 是 (nv, nv)，我们只取关节-关节块 (6:, 6:)。
        """
        base_quat_pin = mj_quat_to_pin_quat(base_quat_mj)
        q_pin = np.concatenate([base_pos, base_quat_pin, dof_pos], axis=1)
        num_envs = base_pos.shape[0]
        M_jnt = np.zeros((num_envs, self.nu, self.nu), dtype=np.float64)
        for i in range(num_envs):
            pin.crba(self._model, self._data, q_pin[i])
            # crba 只填上三角，需要对称化
            self._data.M.triangularView[2] = self._data.M.transpose().triangularView[2]
            M_jnt[i] = np.asarray(self._data.M)[6:, 6:]
        return M_jnt
```

### 3.5 补偿控制器实现

#### 3.5.1 控制器抽象基类

```python
class MotorController(abc.ABC):
    """低层级电机控制器抽象。

    在 pre_step_control 回调中被调用，每个物理 substep 前执行一次。
    输入：策略输出的目标（来自 apply_action）+ 当前机器人状态
    输出：实际施加的力矩
    """

    @abc.abstractmethod
    def compute_torque(
        self,
        backend: SimBackend,
        target: np.ndarray,       # apply_action 输出的目标 (num_envs, nu)
        dof_pos: np.ndarray,      # 当前关节位置 (num_envs, nu)
        dof_vel: np.ndarray,      # 当前关节速度 (num_envs, nu)
    ) -> np.ndarray:
        """计算实际输出力矩。返回 (num_envs, nu) 数组。"""

    @abc.abstractmethod
    def reset(self, env_ids: np.ndarray) -> None:
        """重置指定环境的控制器内部状态（如 ESO 状态）。"""
```

#### 3.5.2 PD 控制器（与现有行为等价）

```python
class PDController(MotorController):
    """标准 PD 控制器。

    τ = Kp * (q_d - q) - Kd * q̇

    与 MuJoCo position actuator 的内建 PD 行为等价，
    用于 controller_type="pd" + motor actuator 模式下的验证。
    """

    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self._kp = kp   # (nu,) or scalar
        self._kd = kd

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        return self._kp * (target - dof_pos) - self._kd * dof_vel

    def reset(self, env_ids):
        pass  # PD 无状态
```

#### 3.5.3 重力补偿控制器

最基础的补偿，效果立竿见影：

```python
class GravityCompController(MotorController):
    """PD + 重力补偿。

    τ = Kp * (q_d - q) - Kd * q̇ + g(q)

    效果：策略不再需要学习对抗重力，关节在零输入时自然悬停。
    适用：四足平地行走、灵巧手操控等速度不大的场景。
    """

    def __init__(self, pin_model: PinocchioDynamicsModel, kp, kd):
        self._pin = pin_model
        self._kp = kp
        self._kd = kd

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        # 1. 从 backend 读取完整状态
        base_pos = backend.get_base_pos()           # (E, 3)
        base_quat = backend.get_base_quat()         # (E, 4) wxyz

        # 2. Pinocchio 计算重力
        g = self._pin.compute_gravity_only(base_pos, base_quat, dof_pos)

        # 3. PD + 重力补偿
        tau = self._kp * (target - dof_pos) - self._kd * dof_vel + g
        return tau

    def reset(self, env_ids):
        pass
```

**物理直觉**：加入重力补偿后，当策略输出 `actions = 0`（即 `target = default_angles`），机器人在默认姿态下所受力矩为 `Kp * 0 - Kd * 0 + g(q_default) = g(q_default)`，恰好抵消重力，机器人悬停。策略只需学习偏离默认姿态的运动，大幅降低了学习难度。

#### 3.5.4 计算力矩控制器 (CTC)

在重力补偿基础上加入 Coriolis/离心力补偿和惯性解耦：

```python
class CTCController(MotorController):
    """计算力矩控制 (Computed Torque Control)。

    τ = M(q) * [Kp * (q_d - q) + Kd * (q̇_d - q̇)] + C(q,q̇)q̇ + g(q)

    效果：各关节动态被解耦为独立二阶系统 q̈_i = Kp_i*(q_d_i - q_i) + Kd_i*(q̇_d_i - q̇_i)，
    PD 增益全局有效，不受构型变化影响。
    适用：人形站立/行走、四足快速跑/转向、翻身等高速运动场景。
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

        # 4. CTC 公式: τ = M * a_desired + bias
        tau = np.einsum('eij,ej->ei', M_jnt, a_desired) + bias
        return tau

    def reset(self, env_ids):
        pass
```

**CTC 的反馈线性化效果**：将 CTC 力矩代入运动方程：

$$
M\ddot{q} + C\dot{q} + g = M[Kp(q_d - q) + Kd(\dot{q}_d - \dot{q})] + C\dot{q} + g
$$

两边消去 $C\dot{q} + g$，再左乘 $M^{-1}$：

$$
\ddot{q} = Kp(q_d - q) + Kd(\dot{q}_d - \dot{q})
$$

每个关节独立为一个二阶系统，PD 增益的含义变得全局一致。

#### 3.5.5 自抗扰控制器 (ADRC)

在 CTC 基础上，用扩张状态观测器 (ESO) 估计并补偿未建模扰动：

```python
class ADRCController(MotorController):
    """自抗扰控制器。

    τ = M(q) * [Kp*(q_d - q) + Kd*(q̇_d - q̇) - z₃/b₀] + C(q,q̇)q̇ + g(q)
                                                         ↑ ESO 估计的扰动

    z₃ 估计了 M⁻¹ * (τ_unmodeled) 的等效扰动，实现自适应补偿。
    适用：人形翻身/翻跟头、manip-loco（臂负载不确定）等需要强鲁棒性的场景。
    """

    def __init__(self, pin_model, kp, kd, eso_bw, b0, num_envs, nu):
        self._pin = pin_model
        self._kp = kp
        self._kd = kd
        self._eso_bw = eso_bw       # ESO 带宽 ω₀
        self._b0 = b0               # 控制增益估计
        self._eso_z = np.zeros((num_envs, 3, nu))  # [z₁, z₂, z₃]

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        dt = backend._sim_dt  # 物理 substep 时间步长
        w = self._eso_bw

        # ── ESO 更新（线性三阶 ESO）──
        e = dof_pos - self._eso_z[:, 0]             # 位置观测误差
        self._eso_z[:, 0] += dt * (self._eso_z[:, 1] + 3*w*e)
        self._eso_z[:, 1] += dt * (self._eso_z[:, 2] + 3*w**2*e)
        self._eso_z[:, 2] += dt * (w**3*e)

        # ── 误差反馈 ──
        e1 = target - self._eso_z[:, 0]             # 位置误差
        e2 = np.zeros_like(dof_vel) - self._eso_z[:, 1]  # 速度误差（目标速度=0）
        u0 = self._kp * e1 + self._kd * e2

        # ── 动力学补偿 ──
        base_pos = backend.get_base_pos()
        base_quat = backend.get_base_quat()
        base_lin_vel = backend.get_base_lin_vel()
        base_ang_vel = backend.get_base_ang_vel()
        bias = self._pin.compute_bias_force(
            base_pos, base_quat, dof_pos,
            base_lin_vel, base_ang_vel, dof_vel
        )
        M_jnt = self._pin.compute_mass_matrix_joint(
            base_pos, base_quat, dof_pos
        )

        # ── 扰动补偿 ──
        a_desired = u0 - self._eso_z[:, 2] / self._b0

        # ── 最终力矩 ──
        tau = np.einsum('eij,ej->ei', M_jnt, a_desired) + bias
        return tau

    def reset(self, env_ids):
        self._eso_z[env_ids] = 0.0
```

**ADRC 降低 sim2real gap 的机理**：

| 来源 | PD 无法处理 | ADRC 如何处理 |
|------|------------|--------------|
| 未建模摩擦 | 产生稳态误差 | ESO 估计为扰动 z₃，前馈补偿 |
| 负载变化 | Kp/Kd 不匹配 → 超调/振荡 | ESO 自适应估计等效质量变化 |
| 外部扰动（推力） | 无抵抗力 | z₃ 实时估计并补偿 |
| 关节柔性 | 模型不匹配 | ESO 将柔性效应纳入扰动估计 |
| 电机饱和 | PD 积分饱和 | ADRC 无积分项，天然抗饱和 |

### 3.6 `pre_step_control` 集成

```python
# 在 Go2JoystickFlatTask.__init__() 中：
if cfg.control_config.controller_type != "pd":
    # 1. 构建 Pinocchio 模型（冷路径，一次性）
    mjcf_path = ASSETS_ROOT_PATH / "go2" / "go2.xml"
    self._pin_model = PinocchioDynamicsModel(str(mjcf_path), self._backend)

    # 2. 创建控制器
    self._motor_controller = resolve_controller(
        cfg.control_config, self._pin_model, self._backend
    )

    # 3. 切换 actuator 模式（冷路径，一次性）
    switch_to_motor_actuators(self._backend._model)

    # 4. 注册 pre_step_control 回调
    self._backend.set_pre_step_control(self._pre_step_motor_control)

def _pre_step_motor_control(self, backend, policy_ctrl):
    """每个物理 substep 前调用，计算实际力矩。"""
    dof_pos = backend.get_dof_pos()
    dof_vel = backend.get_dof_vel()
    return self._motor_controller.compute_torque(backend, policy_ctrl, dof_pos, dof_vel)
```

**调用链**（与 Go2W 已验证的模式一致）：

```
NpEnv.step(actions)
  → apply_action(actions) → ctrl (目标位置)
  → backend.step(ctrl, nsteps)
      → _step_with_pre_step_control(ctrl, nsteps)  ← MuJoCo backend.py:819
          → 每个 substep:
              native_ctrl = self._apply_pre_step_control(ctrl)  ← 在这里做补偿计算
              pool.step(physics_state, nstep=1, control=native_ctrl)
```

### 3.7 配置集成

```python
# 在 ControlConfig 中扩展
@dataclass
class AdvancedControlConfig(PdControlConfig):
    controller_type: str = "pd"    # "pd" | "gravity_comp" | "ctc" | "adrc"
    # CTC 参数
    use_gravity_comp: bool = True
    use_coriolis_comp: bool = True
    use_inertia_decouple: bool = True  # CTC 中是否解耦 M(q)
    # ADRC 参数
    adrc_eso_bandwidth: float = 10.0
    adrc_b0: float = 1.0
    # Pinocchio 模型路径（默认从 assets 自动推断）
    pinocchio_model_path: str | None = None
```

```yaml
# YAML 配置示例
env:
  control_config:
    controller_type: "ctc"
    Kp: 35.0
    Kd: 0.5
    use_gravity_comp: true
    use_coriolis_comp: true
    use_inertia_decouple: true
```

```yaml
# ADRC 配置示例
env:
  control_config:
    controller_type: "adrc"
    Kp: 35.0
    Kd: 0.5
    use_gravity_comp: true
    use_coriolis_comp: true
    use_inertia_decouple: true
    adrc_eso_bandwidth: 10.0
    adrc_b0: 1.0
```

### 3.8 性能优化：批量计算

直接循环调用 Pinocchio 的 `rnea` 在 4096 个 env 下太慢：

| 方案 | 4096 env × 12 DoF 估计耗时 | 说明 |
|------|---------------------------|------|
| 循环调用 Pinocchio | ~80ms | Python 循环开销，不可接受 |
| 手动批量 RNEA (numpy) | ~0.5-2ms | 推荐 |
| C++ 扩展 / Numba JIT | ~0.1-0.5ms | 极端性能需求 |

#### 3.8.1 策略：提取 Pinocchio 参数，手动批量 RNEA

对于 Go2 (12 DoF) / G1 (23 DoF) 等关节数较少的机器人，RNEA 递推公式的步数很少，可以手动展开为纯 numpy 批量操作：

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

#### 3.8.2 渐进优化路径

| 阶段 | 方案 | 性能目标 | 说明 |
|------|------|---------|------|
| Phase 0 | 循环调用 Pinocchio | 无要求（验证正确性） | 减少到 ~100 env 验证 |
| Phase 1 | 手动批量 RNEA | <2ms / 4096 envs | 推荐方案 |
| Phase 2 | C++ 扩展 / Numba JIT | <0.5ms / 4096 envs | 极端性能需求 |

### 3.9 Domain Randomization 与补偿的交互

DR 随机化会改变机器人的动力学参数（质量、摩擦、惯性等），补偿计算需要感知这些变化。

#### 3.9.1 两种策略

| 策略 | 仿真中 Pinocchio 使用的参数 | 部署时 Pinocchio 使用的参数 | sim2real gap |
|------|---------------------------|---------------------------|-------------|
| **A. 用标称参数补偿** | 始终用 MJCF 标称参数 | 用标称参数 | 补偿不随 DR 变化，但策略学会了应对参数偏差 |
| **B. 用随机化后参数补偿** | 每个 env 用各自的随机化参数 | 用标称参数 | 仿真中补偿更精确，但与部署不一致 |

**推荐策略 A**：仿真中始终用标称参数做补偿。理由：

1. 部署时只知道标称参数，无法知道真实参数
2. DR 的目的就是让策略学会应对参数不确定性
3. 用标称参数补偿后，PD/CTC 在标称情况下精确，DR 造成的偏差留给策略学习
4. 这与 ADRC 的设计哲学一致：标称补偿 + ESO 估计残余扰动

#### 3.9.2 DR 参数对补偿的影响

| DR 项 | 影响的补偿项 | 标称补偿下的效果 |
|-------|------------|----------------|
| `base_mass_delta` | g(q) 改变 | 重力补偿不完全，ESO 可吸收 |
| `body_inertia` | M(q), C(q,q̇)q̇ 改变 | CTC 解耦不完美，PD 仍有基本稳定性 |
| `dof_armature` | 等效惯量增大 | MuJoCo 的 armature 不在 Pinocchio 模型中，需额外处理 |
| `geom_friction` | 不影响补偿 | 影响接触力，属于 τ_ext |
| `gravity` | g(q) 改变 | 可选：同步修改 Pinocchio 的重力向量 |

#### 3.9.3 armature 的特殊处理

MuJoCo 的 `armature` 在关节上添加额外惯性，等效于 $I_{\text{eff}} = I + \text{armature}$。Pinocchio 模型中不包含 armature，因此 CTC 计算的 M(q) 不含 armature 项。

**方案 A（推荐）**：在 Pinocchio 模型构建时，将 armature 等效为关节惯性增量，注入模型：

```python
def inject_armature_into_pinocchio_model(pin_model, armature_vec):
    """将 MuJoCo armature 等效为关节惯性增量，注入 Pinocchio 模型。

    armature 的物理意义：在关节轴方向上增加的等效转动惯量。
    等效为在关节惯性矩阵上加上 armature * (axis ⊗ axis)。
    """
    for i, arm_val in enumerate(armature_vec):
        joint_id = i + 1  # 跳过 universe joint
        axis = pin_model.jointAxes[joint_id]
        pin_model.inertias[joint_id] += arm_val * np.outer(axis, axis)
```

**方案 B**：在补偿计算后，额外加上 armature 相关项：

```python
armature = backend.get_dof_armature()  # (nv,)
M_jnt_corrected = M_jnt + np.diag(armature[6:])  # 只加关节部分
```

推荐方案 A，因为更干净——armature 被吸收进模型，后续所有计算自然包含。

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

1. **力矩限幅**：不超过电机额定力矩（Go2: hip ±23.7 Nm, knee ±45.43 Nm）
2. **位置限幅**：关节不超出物理行程
3. **紧急停止**：检测到异常（倾倒、力矩过大）时切回 PD 控制器
4. **补偿力矩上限**：限制补偿力矩不超过总力矩的某个比例，防止补偿项主导

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
| 电机力矩限幅 | URDF/MJCF `<position forcerange>` | 部署侧需硬编码相同限幅 |
| PD 增益 Kp/Kd | YAML 配置 | 部署侧使用训练时的最终值 |
| ESO 带宽 ω₀ | YAML 配置 | 同上 |
| armature | MJCF `<joint armature>` | 需特殊处理（见 §3.9.3） |

### 5.4 Level 3：代码一致

**规则**：Python 补偿逻辑和 C++ 补偿逻辑必须**数学等价**。

**验证方法**：

```python
def verify_compensation_consistency():
    """用相同的 q, v 输入，对比 Python 和 C++ 的补偿输出。"""
    bias_py = pin_model_py.compute_bias_force(q, v)
    bias_cpp = cpp_compensator.compute_bias(q, v)  # 通过 C++ binding 或 subprocess
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
| 人形翻身/翻跟头 | CTC + ADRC | 高速旋转 + 大范围构型变化，需惯性解耦 + 扰动补偿 |
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

## 7. G1 专项设计

### 7.1 G1 机器人现状盘点

#### 7.1.1 运动学拓扑

G1 是 29-DoF 全身人形机器人（含浮动基座），关节树如下：

```
pelvis (freejoint, 7D q / 6D v)
├── left_hip_pitch → left_hip_roll → left_hip_yaw → left_knee → left_ankle_pitch → left_ankle_roll
├── right_hip_pitch → right_hip_roll → right_hip_yaw → right_knee → right_ankle_pitch → right_ankle_roll
└── waist_yaw → waist_roll → waist_pitch
    ├── left_shoulder_pitch → left_shoulder_roll → left_shoulder_yaw → left_elbow → left_wrist_roll → left_wrist_pitch → left_wrist_yaw
    └── right_shoulder_pitch → right_shoulder_roll → right_shoulder_yaw → right_elbow → right_wrist_roll → right_wrist_pitch → right_wrist_yaw
```

| 维度 | 值 | 说明 |
|------|-----|------|
| nq | 7 + 29 = **36** | 浮动基座 7D + 关节角 29D |
| nv | 6 + 29 = **35** | 浮动基座 6D + 关节速度 29D |
| nu (actuator) | **29** | 每个关节一个 position actuator |

#### 7.1.2 Actuator 参数（逐关节不同）

G1 的 actuator 参数与 Go2 不同——**逐关节独立**，且 kp/kv 差异显著：

| 关节组 | 关节 | kp | kv | forcerange (Nm) |
|--------|------|-----|-----|-----------------|
| **腿部** | hip_pitch | 40.179 | 2.558 | ±88 |
| | hip_roll | 99.098 | 6.309 | ±139 |
| | hip_yaw | 40.179 | 2.558 | ±88 |
| | knee | 99.098 | 6.309 | ±139 |
| | ankle_pitch | 28.501 | 1.814 | ±50 |
| | ankle_roll | 28.501 | 1.814 | ±50 |
| **腰部** | waist_yaw | 40.179 | 2.558 | ±88 |
| | waist_roll/pitch | 28.501 | 1.814 | ±50 |
| **手臂** | shoulder_pitch/roll/yaw, elbow, wrist_roll | 14.251 | 0.907 | ±25 |
| | wrist_pitch/yaw | 16.778 | 1.068 | ±5 |

**关键观察**：
- 腿部 hip_roll 和 knee 的 kp=99.098，是 ankle 的 3.5 倍、手臂的 7 倍
- 力矩限幅从 ±5（手腕）到 ±139（hip_roll/knee），跨度 28 倍
- 这些差异意味着**统一的 Kp/Kd 标量不够用**，CTC 必须使用逐关节的 Kp/Kd 向量

#### 7.1.3 惯性参数

G1 的惯性参数在 MJCF 中以 `<inertial>` 显式给出，包含非对角惯性（通过 `quat` 旋转 `diaginertia`）：

| 部件 | 质量 (kg) | 典型 diaginertia |
|------|----------|-----------------|
| pelvis | 3.813 | [0.0105, 0.0093, 0.0079] |
| hip_pitch_link | 1.35 | [0.0018, 0.0015, 0.0012] |
| thigh_link | 1.52 | [0.0025, 0.0024, 0.0015] |
| shin_link | 1.702 | [0.0078, 0.0072, 0.0016] |
| torso_link | 7.818 | [0.1218, 0.1098, 0.0274] |
| shoulder_link | 0.718 | [0.0005, 0.0004, 0.0004] |

**关键观察**：torso_link 质量 7.818 kg 占总质量比例很大，且惯性矩远大于腿部连杆。这意味着**重力项 g(q) 中腰部/躯干贡献最大**，补偿优先级最高。

#### 7.1.4 Armature 与阻尼

| 参数 | 值 | 来源 |
|------|-----|------|
| armature | 0.01（所有关节） | `g1.xml` default class |
| frictionloss | 0.3（所有关节） | `g1.xml` default class |
| damping | 0（默认） | 未显式设置 |

armature=0.01 对 29 个关节都要注入 Pinocchio 模型。

#### 7.1.5 Domain Randomization

G1 的 DR 配置（`G1DomainRandConfig`）：

| DR 项 | 默认值 | 对补偿的影响 |
|--------|--------|------------|
| `randomize_kp` | **True** (0.9~1.1) | 补偿用标称 kp，DR 偏差留给策略 |
| `randomize_kd` | **True** (0.9~1.1) | 同上 |
| `randomize_base_mass` | False | — |
| `randomize_gravity` | False | — |
| `push_robots` | False | — |

**关键**：G1 默认开启 kp/kd 随机化（±10%），这会修改 MuJoCo model 的 `actuator_gainprm` 和 `actuator_biasprm`。在 motor actuator 模式下，这些随机化不再生效（因为 gain/bias 已被切换为透传）。**需要在 `pre_step_control` 中手动实现 kp/kd 随机化**。

#### 7.1.6 当前控制配置

```python
# G1WalkFlatCfg 的 ControlConfig
class G1WalkControlConfig:
    action_scale: float = 1.0          # 注意：Go2 用 0.25，G1 用 1.0
    simulate_action_latency: bool = False
```

**关键差异**：G1 没有 `PdControlConfig` 的 `Kp/Kd` 字段——PD 增益完全由 XML actuator 的 `kp/kv` 定义。引入 Pinocchio 补偿后，需要从 XML 读取这些增益作为控制器的 Kp/Kd 初始值。

### 7.2 G1 的补偿需求分析

#### 7.2.1 重力补偿的必要性

G1 站立时，各关节承受的重力力矩：

| 关节 | 估计重力力矩 (Nm) | 对应 forcerange | 占比 |
|------|------------------|----------------|------|
| hip_pitch | ~30-50 | ±88 | 34-57% |
| hip_roll | ~20-40 | ±139 | 14-29% |
| knee | ~40-60 | ±139 | 29-43% |
| ankle_pitch | ~5-15 | ±50 | 10-30% |
| waist_yaw | ~5-10 | ±88 | 6-11% |

**结论**：重力力矩占电机容量的 10-57%，策略需要花费大量容量学习对抗重力。加入重力补偿后，策略的 action 空间被释放，可以专注于动态平衡和步态。

#### 7.2.2 Coriolis/离心力的显著性

G1 行走时的典型关节速度：

| 关节 | 行走峰值速度 (rad/s) | 跑步峰值速度 (rad/s) |
|------|---------------------|---------------------|
| hip_pitch | 2-4 | 5-8 |
| knee | 3-6 | 8-12 |
| ankle | 1-3 | 3-5 |

Coriolis 项 $C(q,\dot{q})\dot{q}$ 的量级与 $\|\dot{q}\|^2$ 成正比。在跑步速度下，Coriolis 力矩可达 5-20 Nm，不可忽略。

**结论**：G1 行走时重力补偿即可；跑步/快速转向时需要 CTC。

#### 7.2.3 惯性解耦的必要性

G1 从站立到蹲下，髋关节和膝关节的构型变化达 60-90°，有效惯量变化 2-3 倍。固定 Kp 的 PD 控制器在不同构型下阻尼比不同——站立时可能欠阻尼，蹲下时过阻尼。

**结论**：CTC 的惯性解耦对 G1 的全范围运动（蹲下、起身、转身）有显著价值。

### 7.3 G1 + FlashSAC 的特殊考量

#### 7.3.1 FlashSAC 的 off-policy 特性对补偿的影响

FlashSAC 是 off-policy 算法，使用 replay buffer 存储历史 transition。补偿力矩的修改会改变环境的动力学，导致 replay buffer 中旧数据（基于旧动力学）与新数据（基于新动力学）不一致。

**但这不是问题**——补偿是在**环境层面**施加的，策略看到的 obs/reward 不变（仍然是关节位置、速度等）。补偿改变的是"策略输出 → 实际力矩"的映射，而策略的输入空间（obs）和输出空间（action ∈ [-1,1]^29）不变。Replay buffer 中存储的是 (obs, action, reward, next_obs)，这些量在补偿前后语义一致。

**唯一需要注意的**：如果补偿参数在训练中途改变（如从 PD 切换到 CTC），replay buffer 中的旧 transition 会基于旧控制器，导致 off-policy bias。**建议**：补偿参数在训练开始时固定，中途不切换。

#### 7.3.2 FlashSAC 的 actor 网络

FlashSAC actor 使用 `NormalTanhPolicy`（高斯 + tanh 压缩），输出 actions ∈ (-1, 1)^29：

```
obs (98D) → FlashSACEmbedder → FlashSACBlock×2 → UnitRMSNorm → NormalTanhPolicy → actions (29D)
```

加入补偿后，actor 的输出仍然是目标关节位置的偏移量（`actions * action_scale + default_angles`），补偿在 `pre_step_control` 中完成。**actor 网络结构不需要任何修改**。

#### 7.3.3 FlashSAC 的 critic 网络与 asymmetric obs

FlashSAC critic 使用额外的 privileged 信息（`critic_obs = 101D`，比 actor 多 3D 的 `linvel`）。补偿力矩不直接出现在 obs 中，但如果需要，可以将补偿力矩作为额外 privileged 信息提供给 critic：

```python
# 可选：将补偿力矩加入 critic obs
critic = np.concatenate([critic_base, linvel, compensation_torque], axis=1)
# critic_obs_dim: 101 → 130
```

**建议初期不加**——让策略自然学习补偿后的动态。如果发现 critic 预测不准，再考虑加入。

#### 7.3.4 FlashSAC 的 reward 设计与补偿的交互

当前 G1WalkFlat 的 reward 配置（FlashSAC 版本）：

```yaml
reward:
  scales:
    tracking_lin_vel: 2.0
    tracking_ang_vel: 1.5
    penalty_ang_vel_xy: -1.0
    penalty_orientation: -10.0
    penalty_action_rate: -5.0    # ← 与补偿强交互
    pose: -0.5
    penalty_feet_ori: -25.0
    feet_phase: 5.0
    alive: 10.0
```

**`penalty_action_rate` 的特殊处理**：

`action_rate` 惩罚的是 $\|a_t - a_{t-1}\|^2$，即策略输出的变化率。加入补偿后，策略输出的是目标位置偏移，补偿力矩的变化不由策略直接控制。如果 `action_rate` 权重过大，策略会倾向于输出平滑的目标轨迹，但补偿力矩可能仍然剧烈变化（如快速运动时 Coriolis 项突变）。

**建议**：
1. 初期保持 `penalty_action_rate` 不变——它惩罚的是策略输出的平滑度，与补偿无关
2. 如果需要惩罚实际力矩的平滑度，新增 `penalty_torque_rate` reward：

```python
def torque_rate(ctx):
    """惩罚实际力矩的变化率（补偿后）。"""
    current_torques = ctx.info.get("applied_torques", np.zeros_like(ctx.dof_vel))
    last_torques = ctx.info.get("last_applied_torques", current_torques)
    return np.sum(np.square(current_torques - last_torques), axis=1)
```

#### 7.3.5 FlashSAC 的探索噪声与补偿

FlashSAC 使用 **Zeta 噪声**（重尾持久噪声）进行探索，而非 SAC 的标准高斯重采样。Zeta 噪声的特点是噪声持续多步才重新采样，产生更平滑的探索轨迹。

这对补偿是**有利的**：平滑的探索轨迹 → 平滑的目标位置 → 补偿力矩也更平滑 → 仿真更稳定。

#### 7.3.6 FlashSAC 的 `num_envs` 与补偿性能

FlashSAC 的 G1WalkFlat 配置使用 `num_envs: 4096`。Pinocchio 补偿在每个 substep 的 `pre_step_control` 中调用，需要批量计算 4096 个 env 的动力学。

| 补偿类型 | 单 env 耗时 | 4096 env 循环耗时 | 批量优化后耗时 |
|---------|-----------|-----------------|-------------|
| 重力补偿 | ~10 μs | ~40ms | ~1ms |
| CTC (crba + rnea) | ~30 μs | ~120ms | ~3ms |
| CTC + armature 修正 | ~35 μs | ~140ms | ~4ms |

**FlashSAC 的 substep 配置**：G1 的 `sim_dt = 0.02/3 ≈ 0.00667s`，`ctrl_dt = 0.02s`，每个 ctrl step 有 3 个 substep。补偿在每个 substep 调用，总耗时 ×3。

**性能要求**：ctrl_dt = 0.02s 内完成 3 次 substep 的补偿。批量优化后 CTC 约 4ms × 3 = 12ms，占 20ms 控制周期的 60%。**可接受但需要优化**。

**优化策略**：
1. **重力补偿模式**：只需 `rnea(q, 0, 0)`，批量优化后 ~1ms × 3 = 3ms，无压力
2. **CTC 模式**：需要 `crba` + `rnea`，必须用批量 RNEA
3. **可选优化**：不是每个 substep 都需要重新计算 M(q)——在 3 个 substep 内 q 变化很小，可以缓存 M(q)，只在第一个 substep 计算

### 7.4 G1 的 Pinocchio 模型构建

#### 7.4.1 从 MJCF 构建

```python
import pinocchio as pin
from unilab.assets import ASSETS_ROOT_PATH

G1_MJCF_PATH = str(ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml")
pin_model = pin.buildModelFromMjcf(G1_MJCF_PATH)
```

**验证清单**：

| 检查项 | 预期值 | 验证方法 |
|--------|--------|---------|
| `pin_model.nq` | 36 | 与 MuJoCo `model.nq` 对比 |
| `pin_model.nv` | 35 | 与 MuJoCo `model.nv` 对比 |
| `pin_model.njoints` | 31 | 1(universe) + 1(freejoint) + 29(joints) |
| 关节名称顺序 | 与 MuJoCo 一致 | 逐个对比 `pin_model.names[i]` vs `mj_id2name` |
| `rnea(q, v, 0)` vs `data.qfrc_bias` | 误差 < 1e-6 | 在随机 q, v 下对比 |

#### 7.4.2 Armature 注入

```python
def inject_g1_armature(pin_model, armature_value=0.01):
    """G1 所有关节的 armature=0.01，注入 Pinocchio 模型。"""
    for joint_id in range(1, pin_model.njoints):  # 跳过 universe
        if pin_model.joints[joint_id].nv == 1:    # 只处理 1-DoF 关节
            axis = np.asarray(pin_model.jointAxes[joint_id]).flatten()
            pin_model.inertias[joint_id] += armature_value * np.outer(axis, axis)
```

#### 7.4.3 从 XML 读取 PD 增益

G1 的 PD 增益定义在 XML actuator 中，而非 `ControlConfig`。需要从 MuJoCo model 读取：

```python
def read_g1_actuator_gains(backend):
    """从 MuJoCo model 读取 G1 的逐关节 kp/kd。"""
    kp, kd = backend.get_actuator_gains()
    # kp: (29,), kd: (29,)
    # 返回值对应 G1 的 29 个 actuator 顺序
    return kp, kd
```

**G1 的 kp/kd 向量**（从 XML 提取）：

```python
G1_KP = np.array([
    # 左腿 (6)
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    # 右腿 (6)
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    # 腰部 (3)
    40.179, 28.501, 28.501,
    # 左臂 (7)
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
    # 右臂 (7)
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
])

G1_KD = np.array([
    # 左腿 (6)
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    # 右腿 (6)
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    # 腰部 (3)
    2.558, 1.814, 1.814,
    # 左臂 (7)
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
    # 右臂 (7)
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
])
```

#### 7.4.4 力矩限幅

```python
G1_TORQUE_LIMITS = np.array([
    # 左腿
    88, 139, 88, 139, 50, 50,
    # 右腿
    88, 139, 88, 139, 50, 50,
    # 腰部
    88, 50, 50,
    # 左臂
    25, 25, 25, 25, 25, 5, 5,
    # 右臂
    25, 25, 25, 25, 25, 5, 5,
])
```

### 7.5 G1 的控制器实现

#### 7.5.1 G1 重力补偿控制器

```python
class G1GravityCompController(MotorController):
    """G1 专用重力补偿控制器。

    τ = Kp * (q_d - q) - Kd * q̇ + g(q)

    特点：
    - 逐关节 Kp/Kd（从 XML 读取）
    - 力矩限幅（逐关节不同）
    - 可选：只补偿腿部+腰部，不补偿手臂（手臂重力小，补偿收益低）
    """

    def __init__(self, pin_model, kp, kd, torque_limits, comp_mask=None):
        self._pin = pin_model
        self._kp = kp                          # (29,)
        self._kd = kd                          # (29,)
        self._torque_limits = torque_limits    # (29, 2)
        # 默认补偿所有关节；可设置 mask 只补偿部分
        self._comp_mask = comp_mask if comp_mask is not None else np.ones(29, dtype=bool)

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        base_pos = backend.get_base_pos()
        base_quat = backend.get_base_quat()

        # Pinocchio 计算重力
        g = self._pin.compute_gravity_only(base_pos, base_quat, dof_pos)  # (E, 29)

        # PD + 选择性重力补偿
        tau = self._kp * (target - dof_pos) - self._kd * dof_vel
        tau += g * self._comp_mask  # 只补偿 mask 中的关节

        # 力矩限幅
        tau = np.clip(tau, self._torque_limits[:, 0], self._torque_limits[:, 1])
        return tau

    def reset(self, env_ids):
        pass
```

**选择性补偿的物理依据**：

| 关节组 | 重力力矩估计 | 补偿收益 | 建议 |
|--------|------------|---------|------|
| 腿部 (12 joints) | 5-60 Nm | **高** | 必须补偿 |
| 腰部 (3 joints) | 5-10 Nm | **中高** | 建议补偿 |
| 手臂 (14 joints) | 0.5-5 Nm | **低** | 可选（手臂重力小，补偿可能引入高频噪声） |

#### 7.5.2 G1 CTC 控制器

```python
class G1CTCController(MotorController):
    """G1 专用计算力矩控制器。

    τ = M(q) * [Kp * (q_d - q) + Kd * (0 - q̇)] + C(q,q̇)q̇ + g(q)

    特点：
    - 逐关节 Kp/Kd
    - 可选：只对腿部+腰部做惯性解耦，手臂保持 PD（减少计算量）
    - 力矩限幅
    - M(q) 缓存：3 个 substep 内只计算一次
    """

    def __init__(self, pin_model, kp, kd, torque_limits,
                 inertia_decouple_mask=None, cache_M=True):
        self._pin = pin_model
        self._kp = kp
        self._kd = kd
        self._torque_limits = torque_limits
        self._decouple_mask = inertia_decouple_mask  # 选择性惯性解耦
        self._cache_M = cache_M
        self._cached_M = None
        self._cache_step = -1

    def compute_torque(self, backend, target, dof_pos, dof_vel, substep_idx=0):
        base_pos = backend.get_base_pos()
        base_quat = backend.get_base_quat()
        base_lin_vel = backend.get_base_lin_vel()
        base_ang_vel = backend.get_base_ang_vel()

        # 1. Bias force (每个 substep 都计算，因为 v 变化)
        bias = self._pin.compute_bias_force(
            base_pos, base_quat, dof_pos,
            base_lin_vel, base_ang_vel, dof_vel
        )

        # 2. 质量矩阵（可选缓存：3 substep 内只算一次）
        if self._cache_M and substep_idx == 0:
            M_jnt = self._pin.compute_mass_matrix_joint(base_pos, base_quat, dof_pos)
            self._cached_M = M_jnt
        elif self._cache_M:
            M_jnt = self._cached_M
        else:
            M_jnt = self._pin.compute_mass_matrix_joint(base_pos, base_quat, dof_pos)

        # 3. PD 期望加速度
        a_desired = self._kp * (target - dof_pos) - self._kd * dof_vel

        # 4. 选择性惯性解耦
        if self._decouple_mask is not None:
            # 对 mask 内的关节做 M·a，其余直接用 a
            M_a = np.einsum('eij,ej->ei', M_jnt, a_desired)
            tau = np.where(self._decouple_mask, M_a, self._kp * (target - dof_pos) - self._kd * dof_vel)
            tau += bias
        else:
            tau = np.einsum('eij,ej->ei', M_jnt, a_desired) + bias

        # 5. 力矩限幅
        tau = np.clip(tau, self._torque_limits[:, 0], self._torque_limits[:, 1])
        return tau

    def reset(self, env_ids):
        self._cached_M = None
```

### 7.6 G1 的 `pre_step_control` 集成

#### 7.6.1 修改 `G1WalkEnv.__init__()`

```python
class G1WalkEnv(G1BaseEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        # ... 现有代码 ...
        super().__init__(cfg, backend, num_envs)

        # ── 动力学补偿初始化 ──
        controller_type = getattr(cfg.control_config, 'controller_type', 'pd')
        if controller_type != 'pd':
            # 1. 从 XML 读取 PD 增益
            base_kp, base_kd = backend.get_actuator_gains()

            # 2. 构建 Pinocchio 模型
            from unilab.control.pinocchio_model import PinocchioDynamicsModel
            mjcf_path = str(ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml")
            self._pin_model = PinocchioDynamicsModel(mjcf_path, backend)

            # 3. 创建控制器
            from unilab.control.resolver import resolve_controller
            self._motor_controller = resolve_controller(
                controller_type, self._pin_model, base_kp, base_kd,
                torque_limits=self._build_torque_limits(backend),
                num_envs=num_envs,
            )

            # 4. 切换 actuator 模式
            from unilab.control.actuator_switch import switch_to_motor_actuators
            switch_to_motor_actuators(backend._model)

            # 5. 注册 pre_step_control
            self._substep_counter = 0
            backend.set_pre_step_control(self._pre_step_motor_control)

        # ... 其余现有代码 ...
```

#### 7.6.2 `pre_step_control` 回调

```python
def _pre_step_motor_control(self, backend, policy_ctrl):
    """每个物理 substep 前调用。"""
    dof_pos = backend.get_dof_pos()
    dof_vel = backend.get_dof_vel()
    tau = self._motor_controller.compute_torque(
        backend, policy_ctrl, dof_pos, dof_vel,
        substep_idx=self._substep_counter % self._cfg.sim_substeps,
    )
    self._substep_counter += 1

    # 记录实际力矩（供 reward / logging 使用）
    self._applied_torques = tau
    return tau
```

#### 7.6.3 kp/kd 随机化的迁移

当前 G1 的 DR 通过修改 MuJoCo model 的 `actuator_gainprm` / `actuator_biasprm` 实现 kp/kd 随机化。切换到 motor actuator 后，这些修改不再有效。

**迁移方案**：在 `pre_step_control` 中使用随机化后的 kp/kd：

```python
class G1WalkDomainRandomizationProvider(LocomotionDRProvider):
    # ... 现有代码 ...

    def build_reset_plan(self, env, num_reset, rng):
        plan = super().build_reset_plan(env, num_reset, rng)
        # 如果使用 motor actuator + 补偿，kp/kd 随机化
        # 需要存储到 env.info 中，供 pre_step_control 使用
        if env.cfg.domain_rand.randomize_kp:
            base_kp = env._base_kp  # 标称 kp
            low, high = env.cfg.domain_rand.kp_multiplier_range
            kp_mult = rng.uniform(low, high, size=(num_reset, len(base_kp)))
            # 存储到 env 的 DR 状态中
            env._kp_multipliers[env._reset_ids] = kp_mult
        # kd 同理
        return plan
```

在 `pre_step_control` 中使用随机化后的增益：

```python
def _pre_step_motor_control(self, backend, policy_ctrl):
    dof_pos = backend.get_dof_pos()
    dof_vel = backend.get_dof_vel()
    # 使用随机化后的 kp/kd
    kp = self._base_kp * self._kp_multipliers
    kd = self._base_kd * self._kd_multipliers
    tau = self._motor_controller.compute_torque_with_gains(
        backend, policy_ctrl, dof_pos, dof_vel, kp=kp, kd=kd
    )
    return tau
```

### 7.7 G1 的 ControlConfig 扩展

```python
@dataclass
class G1CompensationConfig:
    """G1 动力学补偿配置。"""
    controller_type: str = "pd"  # "pd" | "gravity_comp" | "ctc"

    # 重力补偿选项
    gravity_comp_mask: list[bool] | None = None  # None = 补偿所有关节

    # CTC 选项
    inertia_decouple_mask: list[bool] | None = None  # None = 解耦所有关节
    cache_mass_matrix: bool = True  # 3 substep 内缓存 M(q)

    # Pinocchio 模型路径
    pinocchio_model_path: str | None = None  # None = 自动推断
```

```yaml
# FlashSAC + G1 + 重力补偿 配置示例
env:
  control_config:
    controller_type: "gravity_comp"
    action_scale: 1.0
    gravity_comp_mask: null  # 补偿所有关节

# FlashSAC + G1 + CTC 配置示例
env:
  control_config:
    controller_type: "ctc"
    action_scale: 1.0
    inertia_decouple_mask: null  # 解耦所有关节
    cache_mass_matrix: true
```

### 7.8 G1 的观测空间考虑

加入补偿后，观测空间是否需要修改？

| 观测分量 | 当前维度 | 补偿后是否变化 | 说明 |
|---------|---------|--------------|------|
| gyro | 3 | 不变 | IMU 陀螺仪 |
| gravity (upvector) | 3 | 不变 | 重力方向 |
| dof_pos - default | 29 | 不变 | 关节偏差 |
| dof_vel | 29 | 不变 | 关节速度 |
| last_actions | 29 | 不变 | 策略上步输出 |
| commands | 3 | 不变 | 速度指令 |
| gait_phase | 2 | 不变 | 步态相位 |
| **actor obs total** | **98** | **不变** | — |
| linvel (critic only) | 3 | 不变 | 基座线速度 |
| **critic obs total** | **101** | **不变** | — |

**结论**：加入补偿后，**观测空间完全不变**。策略网络结构不需要修改。补偿是"透明"的——策略仍然输出目标位置偏移，补偿在底层将目标转为力矩。

**可选扩展**：如果需要让策略感知补偿状态（如当前重力补偿力矩），可以增加观测维度。但初期不建议——让策略学习补偿后的自然动态更简洁。

---

## 8. FlashSAC 训练流程适配

### 8.1 FlashSAC 训练入口

FlashSAC 通过 `scripts/train_offpolicy.py` 启动，配置路径：

```
conf/offpolicy/config.yaml          # 全局默认
conf/offpolicy/algo/flashsac.yaml   # FlashSAC 算法参数
conf/offpolicy/task/flashsac/g1_walk_flat/mujoco.yaml  # G1 + FlashSAC 任务配置
```

### 8.2 配置修改

在 `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco.yaml` 中添加补偿配置：

```yaml
# @package _global_
training:
  task_name: G1WalkFlat
  sim_backend: mujoco
algo:
  num_envs: 4096
  # ... 现有参数 ...
env:
  control_config:
    controller_type: "gravity_comp"  # 新增：补偿类型
    action_scale: 1.0
    gravity_comp_mask: null
  # ... 现有参数 ...
```

### 8.3 训练流程不变

```
FlashSACRunner.__init__()
  → 创建 env (G1WalkEnv)
    → __init__ 中检测 controller_type != "pd"
      → 构建 PinocchioDynamicsModel
      → 创建 GravityCompController / CTCController
      → switch_to_motor_actuators
      → set_pre_step_control
  → 创建 FlashSACLearner (网络结构不变)
  → 创建 OffPolicyRunner

训练循环不变：
  collect: env.step(actions) → pre_step_control 计算补偿力矩 → MuJoCo step
  update: learner.update_critic / update_actor (与补偿无关)
```

### 8.4 Checkpoint 兼容性

**问题**：旧 checkpoint（PD 控制器训练的）能否在新控制器（重力补偿）下使用？

**答案**：不能直接使用——策略是在 PD 控制器下的动态学到的，切换到重力补偿后动态完全不同，策略输出会导致完全不同的行为。

**迁移方案**：
1. **从头训练**：最简单，推荐
2. **fine-tune**：加载旧 checkpoint，降低学习率，在新控制器下微调。可能收敛更快，但需要验证
3. **渐进切换**：训练初期用 PD，逐步增加补偿比例（`τ = (1-α)·τ_PD + α·τ_comp`，α 从 0 渐增到 1）。复杂，不推荐初期使用

### 8.5 ONNX 导出

FlashSAC 支持通过 `actor.as_export_module()` 导出 ONNX。加入补偿后，actor 网络不变，ONNX 导出不变。**补偿逻辑在部署侧的 C++ 控制器中实现，不在 actor 网络中**。

```
部署架构：
  ONNX actor → actions (目标位置偏移)
    → C++ Pinocchio 补偿控制器 → 力矩
      → Unitree SDK 电机指令
```

---

## 9. G1 部署侧设计

### 9.1 G1 部署配置

```yaml
# g1_compensation.yaml — 训练和部署共享
compensation:
  model_path: "g1/g1.xml"
  controller_type: "gravity_comp"  # 或 "ctc"
  kp: [40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
       40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
       40.179, 28.501, 28.501,
       14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
       14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778]
  kd: [2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
       2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
       2.558, 1.814, 1.814,
       0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
       0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068]
  torque_limits: [88, 139, 88, 139, 50, 50,
                  88, 139, 88, 139, 50, 50,
                  88, 50, 50,
                  25, 25, 25, 25, 25, 5, 5,
                  25, 25, 25, 25, 25, 5, 5]
  gravity_comp_mask: null  # 补偿所有关节
  use_inertia_decouple: false  # 重力补偿模式不需要
```

### 9.2 G1 C++ 部署控制器

```cpp
class G1DynamicsCompensator {
public:
    G1DynamicsCompensator(const std::string& urdf_path,
                          const Eigen::VectorXd& kp,
                          const Eigen::VectorXd& kd,
                          const Eigen::VectorXd& torque_limits)
        : kp_(kp), kd_(kd), torque_limits_(torque_limits) {
        pinocchio::urdf::buildModel(urdf_path, model_);
        data_ = pinocchio::Data(model_);
        nv_joints_ = model_.nv - 6;  // 29
    }

    Eigen::VectorXd compute_torque(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& q_desired
    ) {
        // 重力补偿
        Eigen::VectorXd v_zero = Eigen::VectorXd::Zero(model_.nv);
        Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model_.nv);
        Eigen::VectorXd bias = pinocchio::rnea(model_, data_, q, v_zero, a_zero);
        Eigen::VectorXd g = bias.tail(nv_joints_);

        // PD + 重力补偿
        Eigen::VectorXd tau = kp_.cwiseProduct(q_desired - q.tail(nv_joints_))
                            - kd_.cwiseProduct(v.tail(nv_joints_))
                            + g;

        // 力矩限幅
        tau = tau.cwiseMax(-torque_limits_).cwiseMin(torque_limits_);
        return tau;
    }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    Eigen::VectorXd kp_, kd_, torque_limits_;
    int nv_joints_;
};
```

### 9.3 G1 部署实时性

| 控制器 | Go2 (12 DoF) | G1 (29 DoF) | G1 1ms 周期可行？ |
|--------|-------------|-------------|------------------|
| 重力补偿 | ~5-10 μs | ~15-25 μs | ✓ |
| CTC | ~15-30 μs | ~40-80 μs | ✓ |

G1 的 29 DoF 比 Go2 的 12 DoF 计算量大约 2-3 倍，但在 C++ 中仍然远低于 1ms 控制周期。

---

## 10. G1 + FlashSAC 实施路线图

| Phase | 内容 | 产出 | 预计时间 | 验证标准 |
|-------|------|------|---------|---------|
| **0** | Pinocchio 加载 G1 MJCF 验证 + 数值一致性 | 验证报告 | 3 天 | rnea vs qfrc_bias 误差 < 1e-6，nq=36, nv=35 |
| **1** | `MotorController` 抽象 + `PDController` + actuator 切换 | PD 等价测试 | 3 天 | G1WalkFlat + motor actuator + PD 行为与 position actuator 一致 |
| **2** | `PinocchioDynamicsModel` + `G1GravityCompController` | 重力补偿可用 | 5 天 | G1 静止悬停测试，FlashSAC 训练不发散 |
| **3** | `G1CTCController` + 批量 RNEA 优化 | CTC 可用 | 7 天 | 阶跃跟踪无超调，4096 env × 3 substep < 12ms |
| **4** | kp/kd DR 迁移 + 配置集成 | 完整训练流程 | 3 天 | FlashSAC + gravity_comp + DR 训练稳定 |
| **5** | 训练对比实验 | PD vs gravity_comp vs CTC 性能对比 | 7 天 | 跟踪误差、reward 曲线、sim2real 指标 |
| **6** | C++ 部署 + 真机验证 | 真机可部署 | 10 天 | Python/C++ 一致性 + 真机行为验证 |

**总计约 5.5 周**。Phase 0-4 为训练侧开发，Phase 5 为实验验证，Phase 6 为部署。

**Phase 5 实验设计**：

| 实验 | 控制器 | 算法 | 环境 | 指标 |
|------|--------|------|------|------|
| Baseline | PD (position actuator) | FlashSAC | G1WalkFlat | reward 曲线、跟踪误差 |
| Exp-1 | gravity_comp | FlashSAC | G1WalkFlat | 同上 + 收敛速度对比 |
| Exp-2 | CTC | FlashSAC | G1WalkFlat | 同上 |
| Exp-3 | gravity_comp (legs+waist only) | FlashSAC | G1WalkFlat | 选择性补偿 vs 全补偿 |

---

## 11. G1 + FlashSAC 关键风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Pinocchio MJCF 加载 G1 的 freejoint + 非对角惯性失败 | 阻塞 | 先验证；回退 URDF |
| G1 29 DoF 批量 RNEA 性能不足 | 训练速度下降 >30% | M(q) 缓存 + 手动批量展开 |
| kp/kd DR 迁移后行为不一致 | 训练不稳定 | 单元测试验证随机化前后行为 |
| 重力补偿后策略过度依赖补偿 | 部署时补偿偏差导致崩溃 | DR + 降级测试（关闭补偿后策略仍能基本控制） |
| FlashSAC replay buffer 中新旧数据混合 | off-policy bias | 补偿参数训练前固定，中途不切换 |
| G1 手臂补偿引入高频噪声 | 手臂抖动 | 选择性补偿 mask，手臂不补偿 |
| 力矩限幅截断补偿项 | 补偿不完全 | 监控限幅比例，调整 Kp/Kd 或力矩限幅 |

---

## 12. 验证与测试策略

### 12.1 单元测试

| 测试项 | 内容 | 验证目标 |
|--------|------|---------|
| 四元数转换 | `mj_quat_to_pin_quat` 正反转换 | 往返误差 < 1e-15 |
| 关节映射 | Pinocchio 与 MuJoCo 关节名称/顺序一致 | 无遗漏，无错位 |
| 重力补偿 | 静止姿态下 τ_g = g(q) | 补偿后关节加速度 ≈ 0 |
| CTC | 施加阶跃目标，观察跟踪 | 无超调，跟踪误差 < 0.01 rad |
| ADRC | 施加外部扰动（推力），观察恢复 | ESO 估计扰动，恢复时间 < 0.5s |
| 与 MuJoCo 对比 | 同一 q/v 下 Pinocchio rnea vs MuJoCo qfrc_bias | 相对误差 < 1e-6 |

### 12.2 集成测试

| 测试项 | 内容 |
|--------|------|
| PD 等价性 | `controller_type: "pd"` 下行为与现有完全一致 |
| Motor actuator 切换 | 运行时切换后仿真不发散 |
| DR 兼容性 | 开启 DR 后 CTC/ADRC 仍然稳定 |
| Reward 一致性 | `_estimate_pd_torques` 在 CTC 模式下仍正确估计力矩用于 reward |

### 12.3 Sim2real 验证

| 步骤 | 内容 |
|------|------|
| 1. 零重力验证 | 关闭重力，PD 控制器下机器人应漂浮 |
| 2. 重力补偿验证 | 开启重力补偿，PD 输出 0 时机器人应悬停 |
| 3. CTC 阶跃响应 | 施加关节阶跃目标，验证跟踪性能 |
| 4. ADRC 抗扰 | 人工施加外力，验证 ESO 估计和恢复 |
| 5. 策略迁移 | 仿真训练的策略直接部署，量化性能 gap |

---

## 13. 文件组织与模块结构（G1 专项）

```
src/unilab/control/                      # 新增模块
├── __init__.py
├── base.py                              # MotorController 抽象基类
├── pd_controller.py                     # PDController
├── gravity_comp_controller.py           # GravityCompController (通用)
├── ctc_controller.py                    # CTCController (通用)
├── pinocchio_model.py                   # PinocchioDynamicsModel (状态对齐封装)
├── vectorized_rnea.py                   # VectorizedRNEA (批量计算优化)
├── actuator_switch.py                   # 运行时 position→motor 切换
└── resolver.py                          # resolve_controller() 工厂函数

src/unilab/envs/locomotion/g1/           # 修改
├── joystick.py                          # G1WalkEnv: 添加补偿初始化 + pre_step_control

conf/offpolicy/task/flashsac/
└── g1_walk_flat/mujoco.yaml             # 添加 controller_type 配置

deploy/                                  # 部署侧（C++）
├── include/unilab_deploy/
│   ├── g1_dynamics_compensator.hpp      # G1 专用 C++ 补偿接口
│   └── robot_config.hpp                # 从 YAML 加载补偿配置
├── src/
│   └── g1_dynamics_compensator.cpp
└── configs/
    └── g1_compensation.yaml            # G1 补偿配置（训练部署共享）
```

---

## 附录 A：G1 关键文件速查

| 用途 | 文件路径 |
|------|---------|
| **新增模块** | |
| 控制器抽象基类 | `src/unilab/control/base.py` (待创建) |
| PD 控制器 | `src/unilab/control/pd_controller.py` (待创建) |
| 重力补偿控制器 | `src/unilab/control/gravity_comp_controller.py` (待创建) |
| CTC 控制器 | `src/unilab/control/ctc_controller.py` (待创建) |
| Pinocchio 模型封装 | `src/unilab/control/pinocchio_model.py` (待创建) |
| 批量 RNEA | `src/unilab/control/vectorized_rnea.py` (待创建) |
| Actuator 切换 | `src/unilab/control/actuator_switch.py` (待创建) |
| 控制器工厂 | `src/unilab/control/resolver.py` (待创建) |
| **G1 相关现有模块** | |
| G1 基类 env | `src/unilab/envs/locomotion/g1/base.py` |
| G1 joystick env (G1WalkEnv) | `src/unilab/envs/locomotion/g1/joystick.py` |
| G1 ControlConfig | `src/unilab/envs/locomotion/g1/base.py:26` |
| G1 DomainRandConfig | `src/unilab/envs/locomotion/g1/joystick.py:33` |
| G1 XML (actuator 定义) | `src/unilab/assets/robots/g1/g1.xml:328-361` |
| G1 XML (inertial 定义) | `src/unilab/assets/robots/g1/g1.xml:71-239` |
| G1 场景 XML | `src/unilab/assets/robots/g1/scene_flat.xml` |
| **FlashSAC 相关** | |
| FlashSAC learner | `src/unilab/algos/torch/flash_sac/learner.py` |
| FlashSAC runner | `src/unilab/algos/torch/flash_sac/runner.py` |
| FlashSAC network | `src/unilab/algos/torch/flash_sac/network.py` |
| FlashSAC 算法配置 | `conf/offpolicy/algo/flashsac.yaml` |
| G1+FlashSAC 任务配置 | `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco.yaml` |
| **Backend / Contract** | |
| SimBackend 接口 | `src/unilab/base/backend/base.py` |
| MuJoCo backend (pre_step_control) | `src/unilab/base/backend/mujoco/backend.py:819` |
| MuJoCo backend (get_actuator_gains) | `src/unilab/base/backend/mujoco/backend.py:1322` |
| MuJoCo backend (get_gravity/mass/armature) | `src/unilab/base/backend/mujoco/backend.py:721-731` |
| NpEnv.step() 控制流 | `src/unilab/base/np_env.py:104` |
| Go2W pre_step_control (已验证模式) | `src/unilab/envs/locomotion/go2w/base.py:93` |
| DR kp/kd 随机化 | `src/unilab/dr/dr_utils.py:141-162` |

---

## 附录 B：G1 Actuator 参数完整表

| # | 关节名 | kp | kv | forcerange (Nm) | 关节组 |
|---|--------|-----|-----|-----------------|--------|
| 0 | left_hip_pitch_joint | 40.179 | 2.558 | ±88 | 腿 |
| 1 | left_hip_roll_joint | 99.098 | 6.309 | ±139 | 腿 |
| 2 | left_hip_yaw_joint | 40.179 | 2.558 | ±88 | 腿 |
| 3 | left_knee_joint | 99.098 | 6.309 | ±139 | 腿 |
| 4 | left_ankle_pitch_joint | 28.501 | 1.814 | ±50 | 腿 |
| 5 | left_ankle_roll_joint | 28.501 | 1.814 | ±50 | 腿 |
| 6 | right_hip_pitch_joint | 40.179 | 2.558 | ±88 | 腿 |
| 7 | right_hip_roll_joint | 99.098 | 6.309 | ±139 | 腿 |
| 8 | right_hip_yaw_joint | 40.179 | 2.558 | ±88 | 腿 |
| 9 | right_knee_joint | 99.098 | 6.309 | ±139 | 腿 |
| 10 | right_ankle_pitch_joint | 28.501 | 1.814 | ±50 | 腿 |
| 11 | right_ankle_roll_joint | 28.501 | 1.814 | ±50 | 腿 |
| 12 | waist_yaw_joint | 40.179 | 2.558 | ±88 | 腰 |
| 13 | waist_roll_joint | 28.501 | 1.814 | ±50 | 腰 |
| 14 | waist_pitch_joint | 28.501 | 1.814 | ±50 | 腰 |
| 15 | left_shoulder_pitch_joint | 14.251 | 0.907 | ±25 | 臂 |
| 16 | left_shoulder_roll_joint | 14.251 | 0.907 | ±25 | 臂 |
| 17 | left_shoulder_yaw_joint | 14.251 | 0.907 | ±25 | 臂 |
| 18 | left_elbow_joint | 14.251 | 0.907 | ±25 | 臂 |
| 19 | left_wrist_roll_joint | 14.251 | 0.907 | ±25 | 臂 |
| 20 | left_wrist_pitch_joint | 16.778 | 1.068 | ±5 | 臂 |
| 21 | left_wrist_yaw_joint | 16.778 | 1.068 | ±5 | 臂 |
| 22 | right_shoulder_pitch_joint | 14.251 | 0.907 | ±25 | 臂 |
| 23 | right_shoulder_roll_joint | 14.251 | 0.907 | ±25 | 臂 |
| 24 | right_shoulder_yaw_joint | 14.251 | 0.907 | ±25 | 臂 |
| 25 | right_elbow_joint | 14.251 | 0.907 | ±25 | 臂 |
| 26 | right_wrist_roll_joint | 14.251 | 0.907 | ±25 | 臂 |
| 27 | right_wrist_pitch_joint | 16.778 | 1.068 | ±5 | 臂 |
| 28 | right_wrist_yaw_joint | 16.778 | 1.068 | ±5 | 臂 |

**选择性补偿 mask 示例**（只补偿腿部+腰部）：

```python
# 索引 0-11 = 腿部, 12-14 = 腰部, 15-28 = 手臂
G1_LEG_WAIST_COMP_MASK = np.array(
    [True]*12 + [True]*3 + [False]*14, dtype=bool
)
```
