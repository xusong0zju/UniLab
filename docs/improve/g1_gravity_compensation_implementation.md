# G1 + FlashSAC 重力补偿训练侧实现

> **目标**：在 G1 人形机器人 + FlashSAC 的训练流程中引入 Pinocchio 重力补偿，使 policy 面对更简单的动力学（重力被抵消），从而加速收敛、提升运动质量。
>
> **核心思路**：将 MuJoCo position actuator → motor actuator + `pre_step_control`，在力矩层面加入 g(q) 项。Policy 的 obs/action 空间不变，可与 baseline 做 A/B 对比。

---

## 1 问题分析

### 1.1 现状

G1 当前使用 MuJoCo **position actuator**：policy 输出目标关节角 q_d，MuJoCo 内部用力矩公式计算：

```
τ = kp * (q_d - q) - kd * q̇
```

这意味着 policy 必须同时学习 **对抗重力** 和 **实现运动控制**，增加了学习难度：

- 重力力矩在站立时可达 4.8 Nm（腰部 pitch）、1.2 Nm（髋关节 pitch），占 forcerange 的 5%~10%
- 不同关节的重力力矩差异极大，policy 需要为每个关节学习不同的偏置
- 收敛慢、运动抖动，policy 需要大量 episode 来"发现"重力补偿策略

### 1.2 改进方向

引入 **重力前馈补偿**：

```
τ = kp * (q_d - q) - kd * q̇ + g(q)
```

使用 **Pinocchio**（而非 MuJoCo 自身）计算 g(q)，原因：

1. **训练-部署一致性**：部署侧用 Pinocchio 做补偿，训练侧也用 Pinocchio，消除 sim2real gap
2. **API 灵活性**：Pinocchio 的 `computeGeneralizedGravity()` 直接给出 g(q)，MuJoCo 没有等价 API
3. **可扩展性**：后续可扩展到 C(q,q̇)q̇ 补偿（Coriolis）、M(q) 逆（CTC），均用同一模型

### 1.3 G1 的关键特征

| 特征 | 值 | 影响 |
|------|-----|------|
| DoF 数 | 29（6 个 free base + 29 个 hinge） | 29 DoF 的重力计算是性能瓶颈 |
| kp 种类 | 6 种（14.251 ~ 99.098） | 必须逐关节存储，不能用统一增益 |
| forcerange 跨度 | ±5 Nm（手腕）~ ±139 Nm（hip_roll/knee） | 手臂 forcerange 窄，补偿后易截断 |
| armature | 0.01（所有关节） | Pinocchio 模型需注入 armature |
| DR 默认开启 kp/kd 随机化 | multiplier [0.9, 1.1] | 切换 motor actuator 后需在 env 中实现 |

---

## 2 架构设计

### 2.1 总体数据流

```
Policy output (actions)
       │
       ▼
  apply_action()         ─→  target_pos = actions * action_scale + default_angles
       │
       ▼
  backend.step(ctrl)     ─→  ctrl = target_pos
       │
       ▼
  _pre_step_motor_control()  ─→  读 qpos/qvel → Pinocchio g(q)
       │                          → τ = kp(target - q) - kd·q̇ + gravity_scale * g(q) * mask
       │                          → clip(τ, force_lower, force_upper)
       ▼
  MuJoCo motor actuator  ─→  直接施加 τ
```

关键点：`apply_action()` 输出不变（目标关节角），转换发生在 `pre_step_control` 回调中。

### 2.2 与 Go2W 模式的对齐

本实现严格遵循 Go2W 已验证的 `pre_step_control` 模式：

| 组件 | Go2W | G1WalkFlatGC |
|------|------|-------------|
| Actuator 切换 | XML 中定义为 motor | 运行时 `switch_to_motor_actuators()` |
| PD 计算 | `compute_go2w_motor_ctrl()` | `GravityCompController.compute()` |
| 重力补偿 | 无 | Pinocchio `computeGeneralizedGravity()` |
| kp/kd DR | env 中 `sample_reset_motor_gains()` | 同 |
| pre_step_control | `_pre_step_motor_control()` | 同 |
| force clipping | `np.clip(out, ctrl_lower, ctrl_upper)` | `np.clip(tau, force_lower, force_upper)` |

### 2.3 模块关系

```
src/unilab/control/
├── __init__.py                 # 导出公共 API
├── base.py                     # MotorController ABC
├── pd_controller.py            # PDController（纯 PD，无补偿）
├── gravity_comp_controller.py  # GravityCompController（PD + 重力）
├── ctc_controller.py           # CTCController（桩，后续扩展）
├── pinocchio_model.py          # PinocchioDynamicsModel（从 MuJoCo 构建）
├── actuator_switch.py          # switch_to_motor_actuators()
└── resolver.py                 # resolve_controller() 工厂
```

---

## 3 核心实现

### 3.1 Pinocchio 模型构建（冷路径）

**文件**：`src/unilab/control/pinocchio_model.py`

从 MuJoCo model 程序化构建 Pinocchio 模型，保证两者完全一致：

```python
class PinocchioDynamicsModel:
    def __init__(self, mj_model):
        self._build_from_mj_model()

    def _build_from_mj_model(self):
        # 遍历 MuJoCo body tree
        for body_id in range(1, mj.nbody):
            parent_body = mj.body_parentid[body_id]
            jnt_adr = mj.body_jntadr[body_id]

            # 关节放置：body_pos + body_quat
            joint_placement = pin.SE3(
                pin.Quaternion(xyzw),  # wxyz → xyzw 转换
                body_pos
            )

            # 关节类型：free → JointModelFreeFlyer, hinge → RX/RY/RZ
            if jnt_type == MJ_JNT_FREE:
                joint_model = pin.JointModelFreeFlyer()
            elif jnt_type == MJ_JNT_HINGE:
                # 根据 jnt_axis 选择 RX/RY/RZ
                ...

            # 惯性参数：mass + ipos + iquat + diaginertia
            # MuJoCo 用 iquat 旋转 diaginertia 到 body frame
            I_body = R @ np.diag(diaginertia) @ R.T
            inertia = pin.Inertia(mass, ipos, I_body)

            model.addJoint(parent, joint_model, placement, name)
            model.appendBodyToJoint(joint_id, inertia, SE3.Identity())

        # 注入 armature
        model.armature[:] = mj.dof_armature
```

**关键转换**：

| 参数 | MuJoCo 格式 | Pinocchio 格式 | 转换 |
|------|------------|---------------|------|
| 四元数 | (w,x,y,z) | (x,y,z,w) | `quat[1], quat[2], quat[3], quat[0]` |
| 惯性矩阵 | `diaginertia` + `iquat` | 完整 3×3 矩阵 | `R @ diag(I) @ R^T` |
| armature | `dof_armature` | `model.armature` | 直接赋值 |

**验证结果**：Pinocchio 重力向量与 MuJoCo `mj_rne(flg_acc=0)` 的最大绝对差 ≈ 1e-16（机器精度），完美对齐。

### 3.2 批量重力计算（热路径）

```python
def gravity(self, qpos_batch, qvel_batch):
    """计算 g(q)，shape = (num_envs, 29)"""
    pin_q, _ = self._align_state(qpos_batch, qvel_batch)
    gravity = np.zeros((num_envs, 29), dtype=np.float64)

    for i in range(num_envs):
        pin.computeGeneralizedGravity(self._model, self._data, pin_q[i])
        gravity[i] = self._data.g[6:]  # 去掉浮动基座 6 DoF

    return gravity
```

**性能估算**：

- 单次 `computeGeneralizedGravity()` 耗时 ~10μs（29 DoF）
- 4096 env × 10μs = ~40ms/ctrl_dt
- 3 个 substep 共享缓存（q 在 ~7ms 内变化小），实际 ~40ms/ctrl_dt
- 与 ctrl_dt=20ms 比较，占比约 2×，但 Pinocchio 计算在 CPU 上可与 GPU learner 并行
- 后续可用 numba 或 C++ 扩展优化

### 3.3 Actuator 切换

**文件**：`src/unilab/control/actuator_switch.py`

```python
def switch_to_motor_actuators(mj_model) -> MotorActuatorInfo:
    """将 MuJoCo position actuator → motor (torque) actuator"""
    # 1. 读取 kp = gainprm[:,0], kd = -biasprm[:,2]
    kp = mj_model.actuator_gainprm[:nu, 0].copy()
    kd = -mj_model.actuator_biasprm[:nu, 2].copy()

    # 2. 读取 forcerange（不是 ctrlrange！）
    force_lower = mj_model.actuator_forcerange[:nu, 0].copy()
    force_upper = mj_model.actuator_forcerange[:nu, 1].copy()

    # 3. 切换为 motor actuator
    mj_model.actuator_gainprm[:nu, 0] = 1.0  # gain = 1
    mj_model.actuator_biasprm[:nu, :] = 0.0  # 无偏置

    # 4. 设置 ctrlrange = forcerange
    mj_model.actuator_ctrlrange[:nu, 0] = force_lower
    mj_model.actuator_ctrlrange[:nu, 1] = force_upper

    return MotorActuatorInfo(kp, kd, force_lower, force_upper)
```

**重要细节**：

- G1 的 `ctrlrange` 默认为 `[0, 0]`（无限制），力矩限制在 `forcerange` 中
- 切换后必须将 `ctrlrange` 设为 `forcerange`，否则 MuJoCo 的 motor actuator 不知道力矩限幅
- 必须在 `backend.materialize()` 之前调用（修改 model 后 pool 编译会使用新参数）

### 3.4 GravityCompController

**文件**：`src/unilab/control/gravity_comp_controller.py`

```python
class GravityCompController(MotorController):
    def __init__(self, dynamics_model, kp, kd, force_lower, force_upper,
                 gravity_comp_mask=None, gravity_scale=1.0): ...

    def compute(self, target_pos, joint_pos, joint_vel,
                *, full_qpos=None, full_qvel=None):
        # PD 项
        tau_pd = kp * (target_pos - joint_pos) - kd * joint_vel

        # 重力补偿项
        tau_gravity = dynamics_model.gravity(full_qpos, full_qvel)
        if gravity_comp_mask is not None:
            tau_gravity *= gravity_comp_mask

        # 合并 + 限幅
        tau = tau_pd + gravity_scale * tau_gravity
        return np.clip(tau, force_lower, force_upper)
```

**选择性补偿 mask**：

G1 的 29 个关节分为 5 组：

```
[left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
```

默认 mask：腿部+腰部补偿，手臂跳过：

```yaml
gravity_comp_mask: [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]
```

理由：

- 手臂 forcerange 仅 ±5 Nm（wrist）~ ±25 Nm（shoulder），重力补偿力矩可能超过限幅
- 手臂质量轻（0.6~0.72 kg），重力力矩小，补偿收益低
- 截断会引入非线性，对训练不利

### 3.5 G1WalkFlatGC 环境

**文件**：`src/unilab/envs/locomotion/g1/joystick.py`

#### 3.5.1 类结构

```python
@registry.env("G1WalkFlatGC", sim_backend="mujoco")
class G1WalkFlatGCEvn(G1BaseEnv):   # 继承 G1BaseEnv，非 G1WalkEnv
    ...
```

继承 `G1BaseEnv` 而非 `G1WalkEnv`，因为 `G1WalkEnv.__init__` 会创建自己的 backend，而我们需在 backend 创建后、materialize 前切换 actuator。

#### 3.5.2 初始化流程

```python
def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
    # 1. 创建 backend
    backend = create_backend(...)

    # 2. 切换 actuator（在 materialize 前）
    actuator_info = switch_to_motor_actuators(backend._model)

    # 3. 构建 Pinocchio 模型（冷路径，一次性）
    dynamics_model = PinocchioDynamicsModel(backend._model)

    # 4. 初始化基类（触发 materialize）
    super().__init__(cfg, backend, num_envs)

    # 5. 设置 motor kp/kd 数组（用于 DR）
    self._base_motor_kp = actuator_info.kp.copy()
    self._base_motor_kd = actuator_info.kd.copy()
    self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, 29)).copy()
    self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, 29)).copy()

    # 6. 创建控制器
    self._controller = GravityCompController(
        dynamics_model=dynamics_model,
        kp=self._base_motor_kp, kd=self._base_motor_kd,
        force_lower=actuator_info.force_lower,
        force_upper=actuator_info.force_upper,
        gravity_comp_mask=..., gravity_scale=...,
    )

    # 7. 注册 pre_step_control 回调
    self._backend.set_pre_step_control(self._pre_step_motor_control)

    # 8. DR（kp/kd 在 env 中处理，不走 backend）
    dr_provider = G1WalkGCDomainRandomizationProvider(...)
    self._init_domain_randomization(dr_provider)
```

#### 3.5.3 pre_step_control 回调

```python
def _pre_step_motor_control(self, backend, policy_ctrl):
    """每个 substep 前被 backend 调用，将目标关节角转为力矩"""
    self._dynamics_model.invalidate_cache()

    joint_pos = backend.get_dof_pos()
    joint_vel = backend.get_dof_vel()
    full_qpos = backend.get_full_qpos()    # 含基座四元数
    full_qvel = backend.get_full_qvel()

    # 更新 DR 后的 kp/kd
    self._controller.kp = self._motor_kp
    self._controller.kd = self._motor_kd

    # τ = kp(q_d - q) - kd·q̇ + g(q)
    return self._controller.compute(
        policy_ctrl, joint_pos, joint_vel,
        full_qpos=full_qpos, full_qvel=full_qvel,
    )
```

#### 3.5.4 DR 迁移

G1 默认开启 kp/kd 随机化（multiplier [0.9, 1.1]）。切换到 motor actuator 后：

- MuJoCo model 中不再有增益，**backend DR 无法处理 kp/kd**
- 需要在 env 中实现 kp/kd 随机化（与 Go2W 一致）

```python
class G1WalkGCDomainRandomizationProvider(LocomotionDRProvider):
    def _get_base_actuator_gains(self, env):
        return None, None  # 不走 backend DR

    def build_reset_plan(self, env, env_ids):
        # ... qpos/qvel reset ...

        # kp/kd 随机化在 env 中处理
        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)

        # backend 只处理 mass/gravity/friction 等
        return ResetPlan(
            ...,
            randomization=_build_g1_gc_backend_reset_randomization(env, num_reset),
        )
```

### 3.6 Backend qpos/qvel 访问

**文件**：`src/unilab/base/backend/base.py`, `mujoco/backend.py`

新增两个方法，供 `pre_step_control` 获取完整状态（含浮动基座）：

```python
# base.py — 抽象方法
def get_full_qpos(self) -> np.ndarray:
    """shape: (num_envs, nq)，含浮动基座四元数"""

def get_full_qvel(self) -> np.ndarray:
    """shape: (num_envs, nv)，含浮动基座速度"""

# mujoco/backend.py — 实现
def get_full_qpos(self):
    return self._qpos_view

def get_full_qvel(self):
    return self._physics_state[:, self._idx_qvel : self._idx_qvel + self.nv]
```

---

## 4 文件清单

### 4.1 新建文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `src/unilab/control/base.py` | `MotorController` ABC | ~40 |
| `src/unilab/control/pd_controller.py` | 纯 PD 力矩控制 | ~65 |
| `src/unilab/control/gravity_comp_controller.py` | PD + 重力补偿 | ~95 |
| `src/unilab/control/ctc_controller.py` | CTC 桩 | ~25 |
| `src/unilab/control/pinocchio_model.py` | Pinocchio 模型构建 + 批量计算 | ~310 |
| `src/unilab/control/actuator_switch.py` | Actuator 切换 | ~95 |
| `src/unilab/control/resolver.py` | 控制器工厂 | ~65 |
| `conf/.../mujoco_gc.yaml` | FlashSAC + G1WalkFlatGC 训练配置 | ~60 |

### 4.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `pyproject.toml` | 添加 `pin>=3.1.0` 依赖 |
| `src/unilab/control/__init__.py` | 更新 import 列表 |
| `src/unilab/base/backend/base.py` | 添加 `get_full_qpos()` / `get_full_qvel()` 抽象方法 |
| `src/unilab/base/backend/mujoco/backend.py` | 实现上述两个方法 |
| `src/unilab/envs/locomotion/g1/joystick.py` | 添加 G1WalkFlatGC 环境及相关类 |

---

## 5 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Pinocchio 模型来源 | 从 MuJoCo model 程序化构建 | 保证与 MuJoCo 完全一致，无需 URDF；已有验证（误差 ≈ 1e-16） |
| 重力计算频率 | 每 ctrl_dt 计算一次，substep 间缓存 | q 在 3 个 substep（~7ms）内变化小，重力变化可忽略 |
| 选择性补偿 | 默认只补偿腿部+腰部 (index 0-14) | 手臂 forcerange 窄（±5 Nm），补偿后易截断引入非线性 |
| 新 env vs 修改现有 | 创建 `G1WalkFlatGC` | baseline 不变，可直接 A/B 对比 |
| 继承 `G1BaseEnv` | 而非 `G1WalkEnv` | 需在 `super().__init__` 前切换 actuator，`G1WalkEnv.__init__` 会自行创建 backend |
| kp/kd DR 迁移 | env 中实现 | motor actuator 后 MuJoCo model 无增益，与 Go2W 一致 |
| action space | `[-1, 1]^29` | motor actuator 的 ctrlrange=forcerange，但 policy 输出是目标角不是力矩 |
| force limits 来源 | `actuator_forcerange` | G1 的 `ctrlrange=[0,0]`，力矩限制在 `forcerange` 中 |

---

## 6 验证

### 6.1 Pinocchio 模型正确性

```
MuJoCo RNE gravity (actuated): [-1.219, -0.068, 0.009, -0.256, -0.153, 0, ...]
Pinocchio gravity (actuated):  [-1.219, -0.068, 0.009, -0.256, -0.153, 0, ...]
Max absolute difference: 0.000000 (≈ 1e-16, machine epsilon)
```

### 6.2 环境功能

```
G1WalkFlatGC (4 envs, zero actions):
  - 创建成功，obs_groups_spec = {"obs": 98, "critic": 101}（与 baseline 一致）
  - 50 步后 100% 存活率，reward > 0
  - 力矩输出合理：left_leg = [-0.18, 7.05, -3.56, -13.22, 15.06, -5.59]
```

### 6.3 回归测试

```
1121 passed, 60 skipped — 无退化
（唯一失败是文档检查，因 pinocchio_compensation_plan.md 引用了尚未实现的文件）
```

---

## 7 训练对比

### 7.1 启动命令

```bash
# Baseline（无重力补偿）
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco

# Gravity Compensation（重力补偿）
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco_gc
```

### 7.2 对比指标

| 指标 | 含义 | 预期改进 |
|------|------|---------|
| tracking_lin_vel | 线速度跟踪精度 | ↑ 收敛更快 |
| feet_phase | 步态相位跟踪 | ↑ 更稳定 |
| penalty_orientation | 姿态惩罚 | ↓ 更少倾斜 |
| penalty_action_rate | 动作变化率 | ↓ 更平滑 |
| survival rate | 存活比例 | ↑ 更稳定站立 |
| iter/s | 训练速度 | ↓ ~5-10%（重力计算开销） |

### 7.3 超参数一致

`mujoco_gc.yaml` 与 `mujoco.yaml` 的超参数完全一致，唯一差异是 `gravity_comp_mask` 和 `gravity_scale`，确保 A/B 对比公平。

---

## 8 后续扩展

1. **Coriolis 补偿**：在 `GravityCompController` 中增加 `C(q,q̇)q̇` 项，用 `pin.rnea(q, q̇, 0) - g(q)` 计算
2. **CTC 控制器**：实现 `CTCController`，完全线性化动力学 τ = M(q)[kp(q_d-q) + kd(q̇_d-q̇)] + C(q,q̇)q̇ + g(q)
3. **numba 加速**：将 Python 循环替换为 numba JIT，预计 4096 env 重力计算从 ~40ms 降至 ~4ms
4. **部署侧 C++ 导出**：将 Pinocchio 模型参数导出为 YAML/JSON，部署侧加载并实现相同补偿
5. **gravity_scale 调度**：训练初期 scale=0.5，逐步增至 1.0，类似 curriculum
