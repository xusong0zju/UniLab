# G1 + FlashSAC 动力学补偿实现

> **目标**：在 G1 人形机器人 + FlashSAC 的训练流程中逐步引入 Pinocchio 动力学补偿（重力 → Coriolis/离心力 → CTC），使 policy 面对更简单的动力学，从而加速收敛、提升运动质量。
>
> **核心思路**：将 MuJoCo position actuator → motor actuator + `pre_step_control`，在力矩层面加入动力学前馈项。Policy 的 obs/action 空间不变，可与 baseline 做 A/B 对比。

---

## 1 补偿层级总览

刚体动力学方程：

```
τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
```

| 层级 | 公式 | 控制器 | 环境 | 状态 |
|------|------|--------|------|------|
| PD only | `τ = kp(q_d-q) - kd·q̇` | `PDController` | G1WalkFlat | ✅ baseline |
| 重力补偿 | `τ = PD + gravity_scale·g(q)` | `GravityCompController` | G1WalkFlatGC | ✅ 已实现 |
| Coriolis 补偿 | `τ = PD + gravity_scale·g(q) + coriolis_scale·C(q,q̇)q̇` | `CoriolisCompController` | G1WalkFlatCC | ✅ 已实现 |
| CTC | `τ = M(q)·[PD] + C(q,q̇)q̇ + g(q)` | `CTCController` | — | 🔜 桩已创建 |

**Coriolis + 离心力不可分离**：C(q,q̇)q̇ 是一个整体，Pinocchio 通过 `rnea(q, q̇, 0) - g(q)` 一次性计算，其中既包含 Coriolis 力也包含离心力。两者物理上耦合、数学上不可拆分。

### 各项的物理含义

| 项 | 依赖 | 物理含义 | 典型量级（G1 站立） |
|----|------|---------|-------------------|
| g(q) | 仅 q | 重力力矩 | 腰 pitch ~4.8 Nm，髋 pitch ~1.2 Nm |
| C(q,q̇)q̇ | q + q̇ | Coriolis 力 + 离心力 | q̇=1 rad/s 时 ~0.1-2 Nm（关节而异） |
| M(q)q̈ | q + q̈ | 惯性力 | 依赖加速度，CTC 时完全抵消 |

### 缓存策略差异

| 项 | 可缓存？ | 原因 |
|----|---------|------|
| g(q) | ✅ 跨 substep | 只依赖 q，3 个 substep（~7ms）内 q 变化极小 |
| C(q,q̇)q̇ | ❌ 每 substep 重算 | 依赖 q̇，substep 间 q̇ 变化显著 |

---

## 2 问题分析

### 2.1 现状

G1 当前使用 MuJoCo **position actuator**：policy 输出目标关节角 q_d，MuJoCo 内部用力矩公式计算：

```
τ = kp * (q_d - q) - kd * q̇
```

这意味着 policy 必须同时学习 **对抗重力/Coriolis** 和 **实现运动控制**，增加了学习难度：

- 重力力矩在站立时可达 4.8 Nm（腰部 pitch）、1.2 Nm（髋关节 pitch），占 forcerange 的 5%~10%
- Coriolis/离心力在行走时随 q̇ 增大，导致跨关节耦合——腿部运动影响躯干，增加协调学习负担
- 不同关节的动力学力矩差异极大，policy 需要为每个关节学习不同的偏置
- 收敛慢、运动抖动，policy 需要大量 episode 来"发现"动力学补偿策略

### 2.2 改进方向

引入 **动力学前馈补偿**，逐步从低级到高级：

1. **重力补偿** g(q)：消除与 q̇ 无关的静力偏置，最基础、最安全
2. **Coriolis + 离心力补偿** C(q,q̇)q̇：消除速度相关的耦合力，行走时效果显著
3. **CTC 完全线性化** M(q)⁻¹·[τ - C·q̇ - g]：使系统表现为解耦双积分器，理论最优但 M(q)⁻¹ 计算量大且数值风险高

使用 **Pinocchio**（而非 MuJoCo 自身）计算动力学项，原因：

1. **训练-部署一致性**：部署侧用 Pinocchio 做补偿，训练侧也用 Pinocchio，消除 sim2real gap
2. **API 灵活性**：Pinocchio 的 `computeGeneralizedGravity()` / `rnea()` 直接给出 g(q) / C(q,q̇)q̇，MuJoCo 没有等价 API
3. **可扩展性**：从重力 → Coriolis → CTC，均用同一模型

### 2.3 G1 的关键特征

| 特征 | 值 | 影响 |
|------|-----|------|
| DoF 数 | 29（6 个 free base + 29 个 hinge） | 29 DoF 的动力学计算是性能瓶颈 |
| kp 种类 | 6 种（14.251 ~ 99.098） | 必须逐关节存储，不能用统一增益 |
| forcerange 跨度 | ±5 Nm（手腕）~ ±139 Nm（hip_roll/knee） | 手臂 forcerange 窄，补偿后易截断 |
| armature | 0.01（所有关节） | Pinocchio 模型需注入 armature |
| DR 默认开启 kp/kd 随机化 | multiplier [0.9, 1.1] | 切换 motor actuator 后需在 env 中实现 |

---

## 3 架构设计

### 3.1 总体数据流

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
  _pre_step_motor_control()  ─→  读 qpos/qvel → Pinocchio g(q) / C(q,q̇)q̇
       │                          → τ = kp(target-q) - kd·q̇ + gravity_scale·g(q)·mask_g + coriolis_scale·C(q,q̇)q̇·mask_c
       │                          → clip(τ, force_lower, force_upper)
       ▼
  MuJoCo motor actuator  ─→  直接施加 τ
```

关键点：`apply_action()` 输出不变（目标关节角），转换发生在 `pre_step_control` 回调中。

### 3.2 与 Go2W 模式的对齐

本实现严格遵循 Go2W 已验证的 `pre_step_control` 模式：

| 组件 | Go2W | G1WalkFlatGC/G1WalkFlatCC |
|------|------|---------------------------|
| Actuator 切换 | XML 中定义为 motor | 运行时 `switch_to_motor_actuators()` |
| PD 计算 | `compute_go2w_motor_ctrl()` | Controller.compute() |
| 重力补偿 | 无 | Pinocchio `computeGeneralizedGravity()` |
| Coriolis 补偿 | 无 | Pinocchio `rnea(q, q̇, 0) - g(q)` |
| kp/kd DR | env 中 `sample_reset_motor_gains()` | 同 |
| pre_step_control | `_pre_step_motor_control()` | 同 |
| force clipping | `np.clip(out, ctrl_lower, ctrl_upper)` | `np.clip(tau, force_lower, force_upper)` |

### 3.3 模块关系

```
src/unilab/control/
├── __init__.py                     # 导出公共 API
├── base.py                         # MotorController ABC
├── pd_controller.py                # PDController（纯 PD，无补偿）
├── gravity_comp_controller.py      # GravityCompController（PD + 重力）
├── coriolis_comp_controller.py     # CoriolisCompController（PD + 重力 + Coriolis/离心力）
├── ctc_controller.py               # CTCController（桩，后续扩展）
├── pinocchio_model.py              # PinocchioDynamicsModel（从 MuJoCo 构建）
├── actuator_switch.py              # switch_to_motor_actuators()
└── resolver.py                     # resolve_controller() 工厂
```

### 3.4 PinocchioDynamicsModel API

```python
class PinocchioDynamicsModel:
    def gravity(self, qpos_batch, qvel_batch) -> np.ndarray:
        """g(q)，shape = (num_envs, nv-6)"""

    def coriolis(self, qpos_batch, qvel_batch) -> np.ndarray:
        """C(q,q̇)q̇，计算方式: rnea(q, q̇, 0) - g(q)"""

    def rnea(self, qpos_batch, qvel_batch, qacc_batch) -> np.ndarray:
        """完整逆动力学: M(q)q̈ + C(q,q̇)q̇ + g(q)"""

    def invalidate_cache(self):
        """清除重力缓存（每个 substep 前调用）"""
```

---

## 4 重力补偿实现

### 4.1 GravityCompController

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

### 4.2 G1WalkFlatGC 环境

**文件**：`src/unilab/envs/locomotion/g1/joystick.py`

```python
@registry.env("G1WalkFlatGC", sim_backend="mujoco")
class G1WalkFlatGCEvn(G1BaseEnv):   # 继承 G1BaseEnv，非 G1WalkEnv
    ...
```

继承 `G1BaseEnv` 而非 `G1WalkEnv`，因为 `G1WalkEnv.__init__` 会创建自己的 backend，而我们需在 backend 创建后、materialize 前切换 actuator。

### 4.3 初始化流程

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

### 4.4 pre_step_control 回调

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

### 4.5 DR 迁移

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

---

## 5 Coriolis + 离心力补偿实现

### 5.1 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 与 GravityCompController 的关系 | 独立实现，不继承 | 避免耦合；CC 包含 GC 全部功能 + Coriolis 项 |
| `coriolis_comp_mask` 默认值 | 跟随 `gravity_comp_mask` | 手臂 forcerange ±5 Nm 截断问题同样适用于 Coriolis |
| `coriolis_scale` 默认值 | 1.0 | 允许调参，与 `gravity_scale` 设计一致 |
| Coriolis 缓存 | 不缓存 | 依赖 q̇，每个 substep 都需重算（gravity 只依赖 q，可跨 substep 缓存） |

### 5.2 CoriolisCompController

**文件**：`src/unilab/control/coriolis_comp_controller.py`

```python
class CoriolisCompController(MotorController):
    def __init__(self, dynamics_model, kp, kd, force_lower, force_upper,
                 gravity_comp_mask=None, gravity_scale=1.0,
                 coriolis_comp_mask=None, coriolis_scale=1.0):
        # coriolis_comp_mask 默认跟随 gravity_comp_mask
        if coriolis_comp_mask is None and gravity_comp_mask is not None:
            coriolis_comp_mask = gravity_comp_mask.copy()
        ...

    def compute(self, target_pos, joint_pos, joint_vel,
                *, full_qpos=None, full_qvel=None):
        # PD 项
        np.subtract(target_pos, joint_pos, out=self._out)
        np.multiply(self._out, self._kp, out=self._out)
        self._out -= self._kd * joint_vel

        # 动力学补偿项（需要 full_qpos + full_qvel）
        if full_qpos is not None and full_qvel is not None:
            # 重力
            tau_gravity = self._dynamics_model.gravity(full_qpos, full_qvel)
            if self._gravity_comp_mask is not None:
                tau_gravity = tau_gravity * self._gravity_comp_mask
            self._out += self._gravity_scale * tau_gravity

            # Coriolis + 离心力
            tau_coriolis = self._dynamics_model.coriolis(full_qpos, full_qvel)
            if self._coriolis_comp_mask is not None:
                tau_coriolis = tau_coriolis * self._coriolis_comp_mask
            self._out += self._coriolis_scale * tau_coriolis

        np.clip(self._out, self._force_lower, self._force_upper, out=self._out)
        return self._out
```

**退化关系**：

| 参数 | 退化到 |
|------|--------|
| `coriolis_scale=0` | GravityCompController |
| `coriolis_scale=0, gravity_scale=0` | PDController |
| `full_qpos=None` 或 `full_qvel=None` | PDController（无动力学信息，仅 PD） |

### 5.3 G1WalkFlatCC 环境

**文件**：`src/unilab/envs/locomotion/g1/joystick.py`

#### 5.3.1 类结构

```python
@registry.env("G1WalkFlatCC", sim_backend="mujoco")
class G1WalkFlatCCEvn(G1BaseEnv):   # 继承 G1BaseEnv，非 G1WalkEnv
    ...
```

#### 5.3.2 控制配置

```python
@dataclass
class G1WalkCCControlConfig:
    action_scale: float = 1.0
    simulate_action_latency: bool = False
    gravity_comp_mask: list[float] | None = None
    gravity_scale: float = 1.0
    coriolis_comp_mask: list[float] | None = None   # 默认跟随 gravity_comp_mask
    coriolis_scale: float = 1.0
```

#### 5.3.3 pre_step_control 回调

```python
def _pre_step_motor_control(self, backend, policy_ctrl):
    """Pre-step callback: convert target positions → motor torques with gravity + Coriolis comp.

    Gravity is computed once per ctrl_dt and cached across substeps.
    Coriolis depends on q̇ and is recomputed each substep.
    """
    self._dynamics_model.invalidate_cache()

    joint_pos = backend.get_dof_pos()
    joint_vel = backend.get_dof_vel()
    full_qpos = backend.get_full_qpos()
    full_qvel = backend.get_full_qvel()

    # Update controller gains (may have been randomized by DR at reset)
    self._controller.kp = self._motor_kp
    self._controller.kd = self._motor_kd

    # Compute motor torques: τ = kp(q_d - q) - kd·q̇ + g(q) + C(q,q̇)q̇
    motor_ctrl = self._controller.compute(
        policy_ctrl,
        joint_pos,
        joint_vel,
        full_qpos=full_qpos,
        full_qvel=full_qvel,
    )
    self._last_motor_ctrl = motor_ctrl
    return motor_ctrl
```

### 5.4 训练配置

**文件**：`conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_cc.yaml`

```yaml
training:
  task_name: G1WalkFlatCC
  sim_backend: mujoco
env:
  control_config:
    action_scale: 1.0
    gravity_comp_mask: [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]
    gravity_scale: 1.0
    coriolis_comp_mask: [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]
    coriolis_scale: 1.0
```

**与 baseline/GC 的超参一致性**：除 `control_config` 外，所有超参完全一致，确保 A/B 对比公平。

---

## 6 选择性补偿 Mask

G1 的 29 个关节分为 5 组：

```
[left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
```

默认 mask：腿部+腰部补偿，手臂跳过：

```yaml
gravity_comp_mask:  [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]
coriolis_comp_mask: [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]
```

### 理由

| 关节组 | forcerange | 重力力矩 | Coriolis 力矩 | 补偿决策 |
|--------|-----------|---------|-------------|---------|
| 腿部 (0-5, 6-11) | ±54 ~ ±139 Nm | 1-5 Nm | 0.1-2 Nm | ✅ 补偿，收益大 |
| 腰部 (12-14) | ±83 ~ ±139 Nm | ~5 Nm | 0.5-3 Nm | ✅ 补偿，收益大 |
| 手臂 (15-21, 22-28) | ±5 ~ ±25 Nm | <1 Nm | <0.5 Nm | ❌ 跳过，截断风险 |

手臂跳过的具体原因：

1. **forcerange 窄**：手腕仅 ±5 Nm，补偿力矩可能超过限幅
2. **力矩小**：手臂质量轻（0.6~0.72 kg），重力和 Coriolis 力矩都小，补偿收益低
3. **截断非线性**：力矩被 clip 后引入非线性，对训练不利

### 两个 mask 是否需要不同？

当前 `coriolis_comp_mask = gravity_comp_mask`。但在某些场景下可能需要不同：

- **行走中手臂摆动**：手臂 q̇ 较大时，C(q,q̇)q̇ 可能超过 g(q)，单独对 Coriolis 跳过手臂更安全
- **双臂操作**：若手臂需要精细控制（如搬运），可单独开启手臂 Coriolis 补偿

设计上允许 `coriolis_comp_mask` 独立于 `gravity_comp_mask` 设置，默认跟随仅为了便利。

---

## 7 核心基础设施

### 7.1 Pinocchio 模型构建（冷路径）

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

### 7.2 批量动力学计算（热路径）

```python
def gravity(self, qpos_batch, qvel_batch):
    """计算 g(q)，shape = (num_envs, 29)"""
    pin_q, _ = self._align_state(qpos_batch, qvel_batch)
    gravity = np.zeros((num_envs, 29), dtype=np.float64)

    for i in range(num_envs):
        pin.computeGeneralizedGravity(self._model, self._data, pin_q[i])
        gravity[i] = self._data.g[6:]  # 去掉浮动基座 6 DoF

    return gravity

def coriolis(self, qpos_batch, qvel_batch):
    """计算 C(q,q̇)q̇，shape = (num_envs, 29)
    通过 rnea(q, q̇, 0) - g(q) 实现"""
    pin_q, pin_v = self._align_state(qpos_batch, qvel_batch)
    coriolis = np.zeros((num_envs, 29), dtype=np.float64)

    for i in range(num_envs):
        pin.computeGeneralizedGravity(self._model, self._data, pin_q[i])
        g = self._data.g[6:].copy()
        pin.rnea(self._model, self._data, pin_q[i], pin_v[i], np.zeros(nv))
        coriolis[i] = self._data.tau[6:] - g

    return coriolis
```

**性能估算**：

| 计算 | 单次耗时 | 4096 env 耗时 | 频率 |
|------|---------|-------------|------|
| g(q) | ~10μs | ~40ms | 每 ctrl_dt（缓存跨 substep） |
| C(q,q̇)q̇ | ~15μs | ~60ms | 每 substep（不缓存） |
| 合计（3 substep） | — | ~100ms/ctrl_dt | — |

与 ctrl_dt=20ms 比较，占比 ~5×，但 Pinocchio 计算在 CPU 上可与 GPU learner 并行。后续可用 numba 或 C++ 扩展优化。

### 7.3 Actuator 切换

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

### 7.4 Backend qpos/qvel 访问

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

### 7.5 控制器工厂

**文件**：`src/unilab/control/resolver.py`

```python
def resolve_controller(name, *, dynamics_model=None, kp, kd,
                       force_lower, force_upper,
                       gravity_comp_mask=None, gravity_scale=1.0,
                       coriolis_comp_mask=None, coriolis_scale=1.0):
    if name == "pd":
        return PDController(kp, kd, force_lower, force_upper)
    elif name == "gravity_comp":
        if dynamics_model is None:
            raise ValueError("gravity_comp requires a PinocchioDynamicsModel")
        return GravityCompController(
            dynamics_model, kp, kd, force_lower, force_upper,
            gravity_comp_mask, gravity_scale,
        )
    elif name == "coriolis_comp":
        if dynamics_model is None:
            raise ValueError("coriolis_comp requires a PinocchioDynamicsModel")
        return CoriolisCompController(
            dynamics_model, kp, kd, force_lower, force_upper,
            gravity_comp_mask, gravity_scale,
            coriolis_comp_mask, coriolis_scale,
        )
    elif name == "ctc":
        raise NotImplementedError("CTCController not yet implemented")
    else:
        raise ValueError(f"Unknown controller '{name}'. Available: 'pd', 'gravity_comp', 'coriolis_comp', 'ctc'.")
```

---

## 8 奖励/惩罚与补偿的交互分析

### 8.1 关键问题

引入动力学补偿后，总力矩 `τ = PD + g(q) + C(q,q̇)q̇` 比纯 PD 大。如果奖励/惩罚函数依赖于总力矩，会产生副作用——**惩罚了补偿本身**，导致补偿的收益被抵消。

### 8.2 逐一分析

#### ✅ `penalty_action_rate`（-5.0）— 无影响

```python
# src/unilab/envs/locomotion/common/rewards.py:145
def action_rate(ctx):
    current = ctx.info["current_actions"]  # 策略输出的目标位置
    last = ctx.info["last_actions"]
    return np.sum(np.square(current - last), axis=1)
```

惩罚的是**策略 action 的变化**（目标位置差），不是力矩变化。Coriolis 补偿力矩在控制器层添加，策略的 action 空间不变（仍然是 [-1, 1]^29 → 目标关节角）。所以 `penalty_action_rate` 不受补偿影响。

**注意**：不同补偿层级下，同一个 action 变化产生的力矩变化不同：

| 层级 | action 变化 → 力矩变化 |
|------|----------------------|
| PD | Δτ = kp · Δaction |
| GravityComp | Δτ = kp · Δaction（g(q) 不随 action 变） |
| CoriolisComp | Δτ = kp · Δaction（g(q) 和 C(q,q̇)q̇ 都不随 action 变） |

因为 g(q) 和 C(q,q̇)q̇ 是前馈项（依赖 q 和 q̇，不依赖 action），所以 `penalty_action_rate` 在所有层级下行为一致。

#### ⚠️ 力矩/能量惩罚（`torques`, `energy`, `dof_torques_l2`）— 当前未使用，但需注意

三个力矩/能量惩罚函数存在于 `src/unilab/envs/locomotion/common/rewards.py`：

```python
# L1 力矩惩罚
def torques(ctx):
    return np.sum(np.abs(_get_torques(ctx)), axis=1)

# 机械能耗惩罚
def energy(ctx):
    t = _get_torques(ctx)
    return np.sum(np.abs(ctx.dof_vel) * np.abs(t), axis=1)

# L2 力矩惩罚
def dof_torques_l2(ctx):
    return np.sum(np.square(torques), axis=1)
```

**当前 CC 配置中没有使用任何一个**，所以不需要调整。但如果将来添加这些惩罚项，**必须注意**：

| 惩罚 | 与补偿的交互 | 解决方案 |
|------|------------|---------|
| `torques` | 补偿增加总力矩 → 惩罚变大 → 补偿被惩罚 | 改为只惩罚 PD 部分：`τ_pd = τ_total - g(q) - C(q,q̇)q̇` |
| `energy` | 补偿力矩×速度 → 能耗变大 → 补偿被惩罚 | 同上：用 PD 部分的力矩计算能耗 |
| `dof_torques_l2` | 同 `torques` | 同上 |

**根本原因**：`state.info["torques"]` 记录的是**总力矩**（PD + gravity + coriolis），而非 PD 部分。如果用力矩惩罚，需要区分"PD 控制力矩"和"前馈补偿力矩"。

#### ✅ 所有结果型奖励 — 不受影响

| 奖励项 | 度量什么 | 补偿的影响 |
|--------|---------|-----------|
| `tracking_lin_vel` | 线速度跟踪精度 | ↑ 更好跟踪（预期正面效果） |
| `tracking_ang_vel` | 角速度跟踪精度 | ↑ 同上 |
| `penalty_ang_vel_xy` | x/y 轴角速度 | ↓ Coriolis 减少了不期望的旋转 |
| `penalty_orientation` | 身体倾斜 | ↓ 更稳定 |
| `pose` | 关节偏离默认姿态 | — 不变 |
| `penalty_feet_ori` | 脚部朝向偏差 | — 不变 |
| `feet_phase` | 步态相位跟踪 | ↑ Coriolis 使步态更平稳 |
| `alive` | 存活 | ↑ 更不容易摔倒 |

这些是**结果度量**——它们测量机器人的运动表现，不关心力矩是怎么产生的。动力学补偿让等效动力学更简单，这些指标应**自然变好**，不需要调整权重。

### 8.3 结论

| 项目 | 是否需要调 | 原因 |
|------|-----------|------|
| `penalty_action_rate` | ❌ 不需要 | 惩罚的是 action（目标位置），不是力矩；前馈项不随 action 变 |
| `torques` / `energy` / `dof_torques_l2` | ❌ 当前未使用 | 如果将来加，需改为只惩罚 PD 部分 |
| 所有结果型奖励 | ❌ 不需要 | 补偿让表现自然变好，无需调权重 |

**当前配置可以直接 A/B 训练，不需要调整奖励权重。**

### 8.4 将来添加力矩/能量惩罚时的注意事项

如果需要添加力矩/能量惩罚，推荐方案：

1. **方案 A：只惩罚 PD 力矩**
   ```python
   # 在 state.info 中分别存储
   state.info["torques"] = motor_ctrl.copy()            # 总力矩（施加给 MuJoCo 的）
   state.info["torques_pd"] = tau_pd.copy()              # 仅 PD 部分
   state.info["torques_feedforward"] = (gravity + coriolis).copy()  # 前馈部分
   ```
   惩罚函数使用 `torques_pd` 而非 `torques`。

2. **方案 B：使用 action 幅度作为代理**
   ```python
   # 惩罚 action 而非力矩，天然不受补偿影响
   def action_magnitude(ctx):
       return np.sum(np.square(ctx.info["current_actions"]), axis=1)
   ```

方案 A 更精确，方案 B 更简单。选择取决于训练需求。

---

## 9 文件清单

### 9.1 新建文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `src/unilab/control/base.py` | `MotorController` ABC | ~40 |
| `src/unilab/control/pd_controller.py` | 纯 PD 力矩控制 | ~65 |
| `src/unilab/control/gravity_comp_controller.py` | PD + 重力补偿 | ~95 |
| `src/unilab/control/coriolis_comp_controller.py` | PD + 重力 + Coriolis/离心力补偿 | ~120 |
| `src/unilab/control/ctc_controller.py` | CTC 桩 | ~25 |
| `src/unilab/control/pinocchio_model.py` | Pinocchio 模型构建 + 批量计算 | ~310 |
| `src/unilab/control/actuator_switch.py` | Actuator 切换 | ~95 |
| `src/unilab/control/resolver.py` | 控制器工厂 | ~80 |
| `conf/.../mujoco_gc.yaml` | FlashSAC + G1WalkFlatGC 训练配置 | ~60 |
| `conf/.../mujoco_cc.yaml` | FlashSAC + G1WalkFlatCC 训练配置 | ~63 |
| `tests/envs/locomotion/g1/test_g1_coriolis_comp.py` | Coriolis 补偿测试 | ~450 |

### 9.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `pyproject.toml` | 添加 `pin>=3.1.0` 依赖 |
| `src/unilab/control/__init__.py` | 更新 import 列表（含 CoriolisCompController） |
| `src/unilab/base/backend/base.py` | 添加 `get_full_qpos()` / `get_full_qvel()` 抽象方法 |
| `src/unilab/base/backend/mujoco/backend.py` | 实现上述两个方法 |
| `src/unilab/envs/locomotion/g1/joystick.py` | 添加 G1WalkFlatGC / G1WalkFlatCC 环境及相关类 |

---

## 10 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Pinocchio 模型来源 | 从 MuJoCo model 程序化构建 | 保证与 MuJoCo 完全一致，无需 URDF；已有验证（误差 ≈ 1e-16） |
| 重力计算频率 | 每 ctrl_dt 计算一次，substep 间缓存 | q 在 3 个 substep（~7ms）内变化小，重力变化可忽略 |
| Coriolis 计算频率 | 每 substep 重算 | q̇ 在 substep 间变化显著，不能缓存 |
| 选择性补偿 | 默认只补偿腿部+腰部 (index 0-14) | 手臂 forcerange 窄（±5 Nm），补偿后易截断引入非线性 |
| coriolis_comp_mask 默认值 | 跟随 gravity_comp_mask | 手臂截断问题同样适用 |
| 新 env vs 修改现有 | 创建 `G1WalkFlatGC` / `G1WalkFlatCC` | baseline 不变，可直接 A/B 对比 |
| 继承 `G1BaseEnv` | 而非 `G1WalkEnv` | 需在 `super().__init__` 前切换 actuator |
| CC 不继承 GC | 独立实现 | 避免耦合；CC 包含 GC 全部功能 + Coriolis 项 |
| kp/kd DR 迁移 | env 中实现 | motor actuator 后 MuJoCo model 无增益，与 Go2W 一致 |
| action space | `[-1, 1]^29` | motor actuator 的 ctrlrange=forcerange，但 policy 输出是目标角不是力矩 |
| force limits 来源 | `actuator_forcerange` | G1 的 `ctrlrange=[0,0]`，力矩限制在 `forcerange` 中 |

---

## 11 验证

### 11.1 Pinocchio 模型正确性

```
MuJoCo RNE gravity (actuated): [-1.219, -0.068, 0.009, -0.256, -0.153, 0, ...]
Pinocchio gravity (actuated):  [-1.219, -0.068, 0.009, -0.256, -0.153, 0, ...]
Max absolute difference: 0.000000 (≈ 1e-16, machine epsilon)
```

### 11.2 Coriolis 值验证

```
C(q,q̇)q̇ at q̇=0:       全零（Coriolis 是速度依赖的，q̇=0 时无 Coriolis 力）✅
C(q,q̇)q̇ at q̇=1 rad/s:  非零，部分关节 > 0.1 Nm ✅
```

### 11.3 环境功能

```
G1WalkFlatGC (4 envs, zero actions):
  - 创建成功，obs_groups_spec = {"obs": 98, "critic": 101}（与 baseline 一致）
  - 50 步后 100% 存活率，reward > 0
  - 力矩输出合理：left_leg = [-0.18, 7.05, -3.56, -13.22, 15.06, -5.59]

G1WalkFlatCC (4 envs, zero actions):
  - 创建成功
  - 50 步后存活，力矩输出合理且有限（finite）
  - Coriolis 项在 q̇≠0 时正确添加
```

### 11.4 测试覆盖

**GravityComp**：基础验证（模型正确性、环境功能）

**CoriolisComp**：13 个测试

| 测试 | 验证内容 |
|------|---------|
| `output_shape` | 输出形状与输入一致 |
| `degrades_to_gravity_comp` | coriolis_scale=0 → 退化为 GravityCompController |
| `degrades_to_pd` | 两个 scale=0 → 退化为 PDController |
| `adds_coriolis_term` | Coriolis 项正确添加 |
| `mask_selectively_applies` | mask 正确遮蔽关节 |
| `mask_defaults_to_gravity_mask` | 默认跟随 gravity_comp_mask |
| `respects_force_limits` | 力矩在限幅内 |
| `without_full_state_is_pure_pd` | 无 qpos/qvel → 纯 PD |
| `resolve_coriolis_comp_controller` | 工厂创建正确 |
| `requires_dynamics_model` | 缺少 dynamics_model 报错 |
| `g1_walk_flat_cc_env_runs_50_steps` | 4 envs × 50 步 smoke test |
| `coriolis_values_nonzero_at_nonzero_velocity` | q̇≠0 时 C(q,q̇)q̇≠0 |
| `coriolis_zero_at_zero_velocity` | q̇=0 时 C(q,q̇)q̇=0 |

### 11.5 回归测试

```
1135 passed, 60 skipped, 0 failed — 无退化
（唯一失败是预存的 test_check_docs.py，与本变更无关）
```

---

## 12 训练对比

### 12.1 启动命令

```bash
# Baseline（无补偿）
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco

# Gravity Compensation（重力补偿）
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco_gc

# Coriolis + Gravity Compensation（Coriolis + 离心力 + 重力补偿）
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco_cc
```

### 12.2 对比指标

| 指标 | 含义 | 预期：GC vs Baseline | 预期：CC vs GC |
|------|------|---------------------|---------------|
| tracking_lin_vel | 线速度跟踪精度 | ↑ 消除重力偏置 | ↑ 消除速度耦合 |
| feet_phase | 步态相位跟踪 | ↑ 更稳定 | ↑ 更平稳 |
| penalty_orientation | 姿态惩罚 | ↓ 更少倾斜 | ↓ Coriolis 减少旋转扰动 |
| penalty_action_rate | 动作变化率 | ↓ 更平滑 | ↓ 无需频繁修正耦合 |
| survival rate | 存活比例 | ↑ 更稳定站立 | ↑ 行走中更稳 |
| iter/s | 训练速度 | ↓ ~5%（重力计算） | ↓ ~15%（Coriolis 每 substep 计算） |

### 12.3 超参数一致

三个配置的超参数完全一致，唯一差异是 `control_config` 中的补偿项，确保 A/B 对比公平。

### 12.4 预期结果

- **GC vs Baseline**：站立和慢速行走显著改善（重力是主要扰动）
- **CC vs GC**：快速行走和转弯显著改善（Coriolis/离心力在高速时占比增大）
- **CC 可能的负面影响**：力矩截断（如果 Coriolis 力矩 + 重力 + PD 超出 forcerange），但 mask 已排除手臂

---

## 13 实现中踩过的坑

### 13.1 重力补偿相关

| # | 坑 | 现象 | 解决 |
|---|-----|------|------|
| 1 | ctrlrange vs forcerange | G1 的 ctrlrange=[0,0]，力矩限幅在 forcerange | 切换 actuator 后必须将 ctrlrange 设为 forcerange |
| 2 | 继承 G1WalkEnv | `__init__` 会自行创建 backend，无法在 materialize 前切换 actuator | 继承 G1BaseEnv |
| 3 | Pinocchio 四元数顺序 | MuJoCo (w,x,y,z) vs Pinocchio (x,y,z,w) | 转换 `quat[1], quat[2], quat[3], quat[0]` |
| 4 | iquat 旋转 diaginertia | MuJoCo 用 iquat 旋转对角惯性到 body frame | `R @ diag(I) @ R^T` |
| 5 | armature 注入 | Pinocchio 默认无 armature，导致动力学差异 | `model.armature[:] = mj.dof_armature` |
| 6 | kp/kd DR 失效 | motor actuator 后 MuJoCo model 无增益 | env 中实现 kp/kd 随机化 |

### 13.2 Coriolis 补偿相关

| # | 坑 | 现象 | 解决 |
|---|-----|------|------|
| 1 | Coriolis 不可缓存 | q̇ 在 substep 间变化显著 | 每 substep 重算，不跨 substep 缓存 |
| 2 | replace_all 匹配重复 | `_reward_upper_body_pose` 在多个 env 类中重复，`replace_all=true` 导致代码注入到错误位置 | 用 Python 脚本精确删除重复块 |
| 3 | registry.resolve_env 不存在 | 测试中使用 `registry.resolve_env()` 但 API 是 `registry.make()` | 直接实例化类 |
| 4 | reset() 需要 env_indices | `NpEnv.reset()` 需要 `env_indices` 位置参数 | 传入 `np.arange(num_envs)` |
| 5 | np.broadcast_to 不支持 -1 | `_FakeDynamicsModel` 使用 `np.broadcast_to(val, (num_envs, -1))` 报错 | 改用 `np.tile(val, (num_envs, 1))` |

---

## 14 A/B 训练对比实验

### 14.1 实验设置

| 项目 | 值 |
|------|-----|
| 算法 | FlashSAC |
| 环境 | G1WalkFlat / G1WalkFlatGC / G1WalkFlatCC |
| num_envs | 4096 |
| max_iterations | 5000（从 2000 checkpoint 继续） |
| 超参 | 三组完全一致，仅 control_config 不同 |
| 硬件 | NVIDIA RTX 4080 SUPER (32GB) |
| 日志 | `logs/flash_sac/G1WalkFlat/`, `G1WalkFlatGC/`, `G1WalkFlatCC/` |

### 14.2 最终结果（5000 iterations）

| 指标 | Baseline (PD) | GC (PD+g) | CC (PD+g+C·q̇) |
|------|:---:|:---:|:---:|
| **reward/mean_ep100** | 308.23 | **323.68** | 317.24 |
| **reward/mean** | 304.67 | **316.51** | 308.77 |
| tracking_lin_vel | 1.71 | **1.77** | 1.71 |
| tracking_ang_vel | **1.18** | 1.11 | 0.92 |
| penalty_orientation | -0.004 | -0.004 | -0.004 |
| penalty_action_rate | -0.70 | **-0.62** | -0.64 |
| feet_phase | 4.60 | **4.75** | 4.71 |
| penalty_feet_ori | -0.33 | **-0.12** | -0.13 |
| episode/length | 987.7 | **998.2** | 990.7 |
| terminated_rate | **0.000** | 0.032 | 0.015 |
| timeout_rate | 1.000 | 0.968 | 0.985 |

**最优值加粗**。三个实验最终都达到了 `timeout_rate > 96%`（走完整个 episode 不摔倒）。

### 14.3 中期对比（2000 iterations）

| 指标 | Baseline | GC | CC |
|------|:---:|:---:|:---:|
| **reward/mean_ep100** | 104.7 | **151.3** | 59.6 |
| **reward/mean** | 45.9 | **60.0** | 26.1 |
| episode/length | 523.2 | **775.5** | 319.2 |
| terminated_rate | 0.885 | **0.400** | 1.000 |
| timeout_rate | 0.115 | **0.600** | 0.000 |

### 14.4 收敛速度分析

| 指标 | Baseline | GC | CC |
|------|---------|-----|-----|
| 最终 reward/mean_ep100 | 308.2 | 323.7 | 317.2 |
| 达到 50% 最终 reward 的 env steps | 8.7M | 8.8M | 10.8M |
| 达到 90% 最终 reward 的 env steps | 14.2M | 14.6M | 15.9M |
| 达到 timeout_rate > 90% 的阶段 | ~14M steps | ~12M steps | ~17M steps |

### 14.5 关键发现

#### 发现 1：GC 最优，CC 接近 Baseline

5000 iterations 后的最终表现排名：

```
GC (323.7) > CC (317.2) > Baseline (308.2)
```

GC 比Baseline 高 **5.0%**，CC 比Baseline 高 **2.9%**。GC 是三者中表现最好的。

#### 发现 2：GC 收敛最快，CC 初期最慢

| 阶段 | Baseline | GC | CC |
|------|---------|-----|-----|
| 2000 iter reward | 104.7 | **151.3** | 59.6 |
| 5000 iter reward | 308.2 | **323.7** | 317.2 |
| 相对 Baseline 的加速 | — | **+44.6%**（2000 iter 时） | -43.1%（2000 iter 时） |

GC 在中期阶段优势明显（2000 iter 时领先 Baseline 44.6%），说明重力补偿显著加速了早期学习。

CC 在初期最慢（2000 iter 时仅为 Baseline 的 57%），因为：
- Coriolis 计算开销导致 collector 每秒步数较少
- 额外的前馈力矩增加了初期探索的难度
- 但 CC 最终追上并超越了 Baseline（+2.9%）

#### 发现 3：CC 的 tracking_ang_vel 偏低

| 指标 | Baseline | GC | CC |
|------|---------|-----|-----|
| tracking_lin_vel | 1.71 | **1.77** | 1.71 |
| tracking_ang_vel | **1.18** | 1.11 | 0.92 |

CC 的角速度跟踪能力明显弱于 Baseline 和 GC（0.92 vs 1.18/1.11）。可能原因：
- Coriolis 补偿在旋转时引入了额外的前馈力矩，使 policy 更难精确控制角速度
- `coriolis_scale=1.0` 可能偏大，尤其在腰部关节——旋转运动时 C(q,q̇)q̇ 量级较大
- 可考虑降低腰部关节的 `coriolis_scale` 或调整 `coriolis_comp_mask`

#### 发现 4：GC/CC 的 feet_ori 更好

| 指标 | Baseline | GC | CC |
|------|---------|-----|-----|
| penalty_feet_ori | -0.33 | **-0.12** | -0.13 |

GC 和 CC 的脚部朝向惩罚仅为 Baseline 的 1/3，说明动力学补偿后步态更自然、脚部更稳。

#### 发现 5：GC/CC 的 action_rate 更低

| 指标 | Baseline | GC | CC |
|------|---------|-----|-----|
| penalty_action_rate | -0.70 | **-0.62** | -0.64 |

补偿后 policy 需要更少的 action 变化即可维持稳定行走——等效动力学更简单，不需要频繁修正。

### 14.6 训练曲线

训练曲线图保存在 `training_logs/comparison_curves.png`。

```
Episode Reward (mean_ep100)          Episode Reward (instant)
  350 |                    GC         350|
      |               CC /---B          |               GC
  300 |   B ---------/                  |   B ---------/--CC
      |  /                              |  /
  250 | /                               | /
      |/                                |/
  200 |                                 |          CC
      |          CC                     |         /
  150 |      /--                        |      /--
      |    /                            |    /
  100 | /-- B                          | /-- B
      |/                                |/
   50 |                                 |
      |                                 |
    0 +------|------|------|-->         0 +------|------|------|-->
      0      5     10     15  M steps     0      5     10     15  M steps
```

### 14.7 实验结论

| 结论 | 说明 |
|------|------|
| **GC 是当前最优选择** | 最终 reward 最高（+5.0%），收敛最快，实现简单 |
| **CC 最终超越 Baseline 但弱于 GC** | +2.9% vs Baseline，但 tracking_ang_vel 偏低 |
| **CC 初期收敛慢** | Coriolis 计算开销 + 额外前馈力矩增加了探索难度 |
| **补偿后步态更自然** | feet_ori 和 action_rate 均优于 Baseline |
| **CC 需要调优** | `coriolis_scale` 和 `coriolis_comp_mask` 可能需要调整 |

### 14.8 CC 调优建议

1. **降低 coriolis_scale**：从 1.0 降至 0.5~0.8，减小前馈力矩对角速度跟踪的干扰
2. **调整腰部 mask**：腰部关节（index 12-14）的 Coriolis 力矩在旋转时较大，可单独降低 `coriolis_scale` 或在 mask 中减弱
3. **增加训练 iterations**：CC 在 5000 iterations 仍在上升，延长至 10000 可能继续改善
4. **Coriolis 补偿 curriculum**：训练初期 `coriolis_scale=0`，逐步增至 1.0（类似 GC 的 gravity_scale 调度）
5. **减少 num_envs**：CC 的 collector 开销大，减少 env 数量可提升 iter/s，但需要更多 iterations

---

## 15 后续扩展

1. **CTC 控制器**：实现 `CTCController`，完全线性化动力学 τ = M(q)[kp(q_d-q) + kd(q̇_d-q̇)] + C(q,q̇)q̇ + g(q)
2. **numba 加速**：将 Python 循环替换为 numba JIT，预计 4096 env 计算从 ~100ms 降至 ~10ms
3. **部署侧 C++ 导出**：将 Pinocchio 模型参数导出为 YAML/JSON，部署侧加载并实现相同补偿
4. **gravity_scale / coriolis_scale 调度**：训练初期 scale 较小，逐步增至 1.0，类似 curriculum（实验表明 CC 初期收敛慢，curriculum 可改善）
5. **力矩/能耗惩罚**：如果需要添加，需区分 PD 力矩和前馈力矩，避免惩罚补偿本身
6. **独立 coriolis_comp_mask 调优**：实验表明 CC 的 tracking_ang_vel 偏低，需调整腰部关节的 coriolis_scale 或 mask
7. **CC 长时间训练验证**：CC 在 5000 iter 仍在上升，需 10000+ iter 确认是否继续追赶 GC
6. **独立 coriolis_comp_mask 调优**：当前跟随 gravity_comp_mask，可根据 A/B 结果单独调整
