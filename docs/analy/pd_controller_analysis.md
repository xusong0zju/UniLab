# UniLab PD 控制器架构分析

> 生成日期：2026-06-07
> 目的：理解现有 PD 控制器的分布与实现方式，为引入高级控制器（动力学补偿 / 自抗扰）做架构准备

---

## 1. 核心发现：PD 控制存在于两个层次

UniLab 中的 PD 控制分布在**两个完全不同的层次**，理解这一区分是替换工作的前提：

```
┌─────────────────────────────────────────────────────────────────┐
│ 层次 1: Env 层 apply_action() — 策略输出 → 目标位置             │
│   actions * action_scale + default_angles → ctrl (目标位置)      │
│   这是 "policy → target" 的映射，PD 增益不在此层                │
├─────────────────────────────────────────────────────────────────┤
│ 层次 2: Backend 层 / pre_step_control — 目标位置 → 力矩         │
│   Kp * (target - qpos) - Kd * qvel → torque                    │
│   这是 "target → torque" 的映射，PD 增益在此层生效              │
└─────────────────────────────────────────────────────────────────┘
```

**关键区别**：
- 层次 1 输出的是**目标位置**（position setpoint），交给 MuJoCo/Motrix 的 position actuator
- 层次 2 是**真正的 PD 计算**，把目标位置转为力矩——这正是需要替换的部分

---

## 2. 层次 1：Env 层 `apply_action()` — 目标生成

### 2.1 基类：`LocomotionBaseEnv.apply_action()`

文件：[src/unilab/envs/locomotion/common/base.py:91-102](src/unilab/envs/locomotion/common/base.py#L91-L102)

```python
def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
    state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
    state.info["current_actions"] = actions
    exec_actions = (
        state.info["last_actions"]
        if self._cfg.control_config.simulate_action_latency
        else actions
    )
    ctrl: np.ndarray = (
        exec_actions * self._cfg.control_config.action_scale + self.default_angles
    )
    return ctrl
```

**作用**：将策略输出 `actions`（通常 ∈ [-1, 1]）通过 `action_scale` 缩放后加上 `default_angles`，得到**目标关节位置** `ctrl`。

### 2.2 各子类的 `apply_action()` 变体

| 环境类 | 文件 | 行为特点 |
|--------|------|---------|
| `LocomotionBaseEnv` | `common/base.py:91` | 标准模式：`actions * action_scale + default_angles` |
| `G1JoystickWalkTask` | `g1/joystick.py:629` | 同标准模式 |
| `Go2RoughTask` | `go2/rough.py:357` | 增加 `clip_actions`，用独立 `_action_scale` 数组（hip/非hip 不同缩放） |
| `Go2FootStandTask` | `go2/footstand.py:384` | 累积式：`_motor_targets += exec_actions * action_scale`，目标逐帧累加 |
| `Go2WJoystickFlatTask` | `go2w/joystick.py:363` | 腿部目标位置 + 轮部目标速度，分离处理 |
| `G1MotionTrackingEnv` | `motion_tracking/g1/tracking.py:602` | 同标准模式 + `default_dof_pos_bias` |
| `AllegroInhandEnv` | `manipulation/allegro_inhand/base.py:107` | 累积式：`prev_ctrl + action_scale * clipped_actions` |
| `SharpaInhandEnv` | `manipulation/sharpa_inhand/rotation.py` | 类似 Allegro，声明了 `torque_control` 但当前强制为 `False` |

### 2.3 控制配置继承体系

```
ControlConfigBase (common/base.py:23)
  ├── action_scale: float = 0.25
  ├── simulate_action_latency: bool = False
  │
  └── PdControlConfig (common/base.py:29)  ← Go1/Go2/Go2W 用这个
        ├── Kp: float = 35.0
        └── Kd: float = 0.5

各机器人子类覆写：
  G1: ControlConfig (g1/base.py:26) → 直接继承 ControlConfigBase，无 Kp/Kd
  Go2: ControlConfig (go2/base.py:21) → 继承 PdControlConfig
  Go2W: ControlConfig (go2w/base.py:64) → 继承 PdControlConfig + wheel_Kd
  Go2 Rough: RoughControlConfig (go2/rough.py:69) → + hip/non_hip_action_scale, clip_actions
  Footstand: FootstandControlConfig (go2/footstand.py:56) → + clip_actions
  Sharpa: SharpaControlConfig (sharpa_inhand/base.py:80) → + p_gain, d_gain, torque_control
  Allegro: (内联在 base.py:22) → kp=1.0, kd=0.1
```

---

## 3. 层次 2：Backend 层 — PD 力矩计算（真正需要替换的部分）

PD 力矩计算有**两种实现路径**：

### 3.1 路径 A：MuJoCo Position Actuator（内建 PD）

**这是大多数环境使用的方式。**

MuJoCo XML 中定义 position actuator：
```xml
<actuator>
  <position name="hip" joint="hip_joint" kp="35"/>
</actuator>
```

MuJoCo 内部自动计算：`torque = kp * (target - qpos) - kd * qvel`

**`position_actuator_gains` 覆盖机制**：

文件：[src/unilab/base/backend/mujoco/backend.py:166-178](src/unilab/base/backend/mujoco/backend.py#L166-L178)

```python
def _apply_position_actuator_gains_to_mj_model(
    model, *, kp, kd, actuator_ids=slice(None),
) -> None:
    model.actuator_gainprm[actuator_ids, 0] = kp_arr
    model.actuator_biasprm[actuator_ids, 1] = -kp_arr
    model.actuator_biasprm[actuator_ids, 2] = -kd_arr
```

各环境在创建 backend 时传入 `position_actuator_gains`：

| 环境类 | 文件 | 传入方式 |
|--------|------|---------|
| `Go2JoystickFlatTask` | `go2/joystick.py:115` | `{"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd}` |
| `Go1JoystickFlatTask` | `go1/joystick.py:105` | 同上 |
| `Go2HandStandTask` | `go2/handstand.py:141` | 同上 |
| `Go2ArmManipLocoEnv` | `go2_arm/manip_loco.py:303` | `build_go2_arm_position_gains(cfg)` — 腿/臂不同增益 |
| `AllegroInhandEnv` | `allegro_inhand/rotation.py:272` | `{"kp": cfg.control_config.kp, "kd": cfg.control_config.kd}` |

**G1 环境没有传入**：G1 的 `ControlConfig` 不继承 `PdControlConfig`，没有 `Kp/Kd` 字段。G1 直接使用 XML 中的默认 actuator 增益。

### 3.2 路径 B：`pre_step_control` 回调（显式 PD 计算）

**Go2W 轮腿机器人使用这种方式**，因为它需要对腿部和轮部使用不同的控制逻辑。

文件：[src/unilab/envs/locomotion/go2w/base.py:93-116](src/unilab/envs/locomotion/go2w/base.py#L93-L116)

```python
def compute_go2w_motor_ctrl(
    policy_ctrl, joint_pos, joint_vel,
    leg_kp, leg_kd, wheel_kd,
    ctrl_lower, ctrl_upper, out,
) -> np.ndarray:
    leg_out = out[:, :NUM_LEG_ACTIONS]
    np.subtract(policy_ctrl[:, :NUM_LEG_ACTIONS], joint_pos[:, :NUM_LEG_ACTIONS], out=leg_out)
    np.multiply(leg_out, leg_kp, out=leg_out)
    leg_out -= leg_kd * joint_vel[:, :NUM_LEG_ACTIONS]
    wheel_out = out[:, NUM_LEG_ACTIONS:]
    np.subtract(policy_ctrl[:, NUM_LEG_ACTIONS:], joint_vel[:, NUM_LEG_ACTIONS:], out=wheel_out)
    np.multiply(wheel_out, wheel_kd, out=wheel_out)
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out
```

**注册机制**：

文件：[src/unilab/envs/locomotion/go2w/joystick.py:273](src/unilab/envs/locomotion/go2w/joystick.py#L273)

```python
self._backend.set_pre_step_control(self._pre_step_motor_control)
```

在 `SimBackend.step()` 中，每个物理子步前都会调用此回调，将策略输出的 ctrl 转为实际力矩。

**`pre_step_control` 的调用链**：

```
NpEnv.step(actions)
  → apply_action(actions) → ctrl (目标位置)
  → backend.step(ctrl, nsteps)
      → 如果有 _pre_step_control_fn:
          → _step_with_pre_step_control(ctrl, nsteps)
              → 每个 substep:
                  native_ctrl = self._apply_pre_step_control(ctrl)  ← 在这里做 PD 计算
                  pool.step(physics_state, nstep=1, control=native_ctrl)
```

文件：[src/unilab/base/backend/mujoco/backend.py:819-863](src/unilab/base/backend/mujoco/backend.py#L819-L863)

### 3.3 路径 C：Env 层显式 PD 估算（仅用于 reward，不影响仿真）

一些环境在 `update_state()` 中估算 PD 力矩，**仅用于 reward 计算**，不影响实际仿真：

| 环境 | 文件位置 | 方法 |
|------|---------|------|
| `Go2RoughTask` | `go2/rough.py:523-539` | `_estimate_pd_torques()` |
| `Go2FootStandTask` | `go2/footstand.py:659-666` | `_estimate_pd_torques()` |

这些方法复制了 PD 公式用于计算 reward 中的 torque penalty，**不参与实际控制回路**。

---

## 4. 控制流全景图

以 Go2 Joystick Flat 为例（**路径 A** — 最常见的模式）：

```
策略网络输出 actions ∈ [-1, 1]^{12}
  │
  ▼
LocomotionBaseEnv.apply_action()          ← Env 层
  ctrl = actions * 0.25 + default_angles   ← 目标关节位置
  │
  ▼
NpEnv.step() → backend.step(ctrl, 3)      ← 3 个 sim substeps
  │
  ▼
MuJoCo Position Actuator (内建 PD)        ← Backend 层（MuJoCo 自动）
  torque = Kp * (ctrl - qpos) - Kd * qvel  ← 这是 PD 力矩
  │
  ▼
MuJoCo 物理仿真 mj_step()
```

以 Go2W Joystick Flat 为例（**路径 B** — pre_step_control 模式）：

```
策略网络输出 actions ∈ [-1, 1]^{16}
  │
  ▼
Go2WJoystickFlatTask.apply_action()       ← Env 层
  leg_targets = actions[:,:12] * 0.25 + default_angles
  wheel_vel_targets = actions[:,12:] * 10.0
  ctrl = concat(leg_targets, wheel_vel_targets)
  │
  ▼
NpEnv.step() → backend.step(ctrl, nsteps)
  │
  ▼
_step_with_pre_step_control()             ← Backend 层（每个 substep 调用）
  native_ctrl = _pre_step_motor_control(backend, ctrl)
    │
    ▼
  compute_go2w_motor_ctrl()               ← 显式 PD 计算
    leg_torque = leg_kp * (leg_target - leg_pos) - leg_kd * leg_vel
    wheel_torque = wheel_kd * (wheel_vel_target - wheel_vel)
  │
  ▼
MuJoCo 物理仿真 mj_step()（使用 native_ctrl 作为力矩）
```

---

## 5. 替换高级控制器的切入点分析

### 5.1 推荐切入点：`pre_step_control` 回调

**`pre_step_control` 是替换 PD 控制器的最佳切入点**，理由：

1. **已在 Go2W 中验证**：`compute_go2w_motor_ctrl()` 就是这个模式的生产级实例
2. **每个 substep 调用**：可以在每个物理步读取当前状态（qpos, qvel, sensor data）并计算补偿力矩
3. **不影响 Env Contract**：`apply_action()` 仍然返回目标位置，PD → 高级控制的替换完全在 backend 层完成
4. **Backend 无关**：MuJoCo 和 Motrix 都支持 `pre_step_control`
5. **可读取 backend 状态**：回调签名 `fn(backend, ctrl)` 可以访问 backend 的全部传感器和运动学接口

### 5.2 需要修改的层次

| 修改点 | 说明 | 影响范围 |
|--------|------|---------|
| **新增控制器模块** | `src/unilab/control/` 或 `src/unilab/envs/common/controller.py` | 新增代码，不影响现有 |
| **Env 层 `apply_action()`** | 可能需要输出更丰富的信息（如前馈力矩） | 仅影响使用新控制器的 env |
| **`ControlConfig` 配置** | 新增 `controller_type` / `controller_params` 字段 | 仅影响使用新控制器的 env |
| **`pre_step_control` 回调** | 将 PD 替换为高级控制器 | 仅影响使用新控制器的 env |
| **MuJoCo actuator 模式** | 如果用 `pre_step_control` 做力矩控制，需要将 XML actuator 改为 `motor`（纯力矩） | 仅影响使用新控制器的 env |

### 5.3 具体替换策略

#### 策略一：保留 Position Actuator，添加前馈补偿

**最小改动方案**。保留 MuJoCo position actuator 做 PD，通过 `pre_step_control` 在每个 substep 前注入前馈力矩。

```
ctrl_from_policy = target_position
  │
  ▼
pre_step_control:
  1. 读取 qpos, qvel, sensor data
  2. 计算动力学补偿力矩: τ_ff = M(q)q̈_ref + C(q,q̇)q̇ + g(q)   ← Pinocchio / MuJoCo API
  3. 计算自抗扰补偿: τ_adrc = ADRC(q, q̇, target, disturbance_estimate)
  4. 将 τ_ff + τ_adrc 写入 backend 的外力接口 (xfrc_applied)
  5. 返回原始 ctrl（PD 部分仍由 position actuator 处理）
```

**优点**：改动最小，PD 作为安全底层始终存在
**缺点**：前馈力和 PD 的交互需要仔细调参

#### 策略二：切换为 Motor Actuator + 完全力矩控制

**彻底替换方案**。将 XML 中的 position actuator 替换为 motor actuator，所有力矩计算由 Python 层完成。

```
ctrl_from_policy = target_position
  │
  ▼
pre_step_control:
  1. 读取 qpos, qvel, M(q), C(q,q̇), g(q)
  2. 计算完整控制律:
     τ = M(q) * (Kp*(target-qpos) + Kd*(0-qvel) + q̈_ref) + C(q,q̇)q̇ + g(q)
         ↑ 计算力矩控制 (CTC)           ↑ 动力学补偿
     或:
     τ = PD_torque + τ_gravity_comp + τ_coriolis_comp + τ_adrc
  3. 返回 native_ctrl = τ
```

**优点**：完全控制力矩，可精确实现 CTC / 阻抗 / 自抗扰
**缺点**：需要修改 XML actuator 定义；仿真稳定性需要额外保障

#### 策略三：基于 `pre_step_control` 的统一控制器抽象

**架构最优方案**。定义控制器抽象接口，支持多种控制器实现。

```python
# src/unilab/control/base.py (提议)
class MotorController(abc.ABC):
    """低层级电机控制器抽象"""

    @abc.abstractmethod
    def compute_torque(
        self,
        backend: SimBackend,
        target: np.ndarray,    # 来自 apply_action 的目标
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        """计算实际输出力矩"""

class PDController(MotorController):    # 现有 PD
class CTCController(MotorController):   # 计算力矩控制
class ADRCController(MotorController):  # 自抗扰控制
class ImpedanceController(MotorController):  # 阻抗控制
```

然后在 `ControlConfig` 中通过 `controller_type` 字段选择：

```yaml
# 配置示例
env:
  control_config:
    controller_type: "ctc"      # "pd" | "ctc" | "adrc" | "impedance"
    controller_params:
      kp: 35.0
      kd: 0.5
      use_gravity_comp: true
      use_coriolis_comp: true
      adrc_bandwidth: 10.0
```

---

## 6. 动力学补偿的数据来源

### 6.1 MuJoCo API（推荐，无需引入 Pinocchio）

MuJoCo 本身已经提供完整的动力学量计算：

| 动力学量 | MuJoCo API | 说明 |
|---------|-----------|------|
| 质量矩阵 M(q) | `mj_fullM(model, data)` | 返回 full mass matrix |
| Coriolis + 重力 | `data.qfrc_bias` | 包含 C(q,q̇)q̇ + g(q) |
| 重力向量 | `data.qfrc_gravity` | 仅重力 |
| 关节雅可比 | `mj_jac(model, data, body_id)` | 已有 `get_site_jacobian_w` |
| 惯性矩阵 | `model.body_inertia` | body ipos + inertia |

**优势**：无需额外依赖，与 MuJoCo 物理完全一致，无需数值对齐问题。

### 6.2 Pinocchio（可选，用于 Motrix 后端或独立计算）

如果 Motrix 后端不提供等价的动力学 API，可以用 Pinocchio 独立计算。但需要注意：
- 需要从 backend 导出 URDF/MJCF 给 Pinocchio 建模
- 动力学参数需要与仿真引擎对齐
- 增加计算开销

### 6.3 自抗扰 (ADRC) 的信息需求

ADRC 核心是扩张状态观测器 (ESO)，需要：

| 信息 | 来源 | 说明 |
|------|------|------|
| 关节位置 q | `backend.get_dof_pos()` | 已有 |
| 关节速度 q̇ | `backend.get_dof_vel()` | 已有 |
| 目标位置 | `apply_action()` 的输出 | 已有 |
| 关节加速度 q̈ | 数值差分 `dof_vel / ctrl_dt` | 已有示例（`_estimate_dof_acc`） |
| 扰动估计 | ESO 内部状态 | 需要新增 |
| 力矩反馈 | MuJoCo `data.actuator_force` | 可通过 backend 暴露 |

---

## 7. 关键文件索引

| 用途 | 文件路径 |
|------|---------|
| 基类 apply_action | `src/unilab/envs/locomotion/common/base.py` |
| ControlConfigBase / PdControlConfig | `src/unilab/envs/locomotion/common/base.py:23-33` |
| SimBackend pre_step_control 接口 | `src/unilab/base/backend/base.py:247-265` |
| MuJoCo _step_with_pre_step_control | `src/unilab/base/backend/mujoco/backend.py:819-863` |
| MuJoCo position_actuator_gains 注入 | `src/unilab/base/backend/mujoco/backend.py:166-178` |
| Go2W 显式 PD (pre_step_control 示例) | `src/unilab/envs/locomotion/go2w/base.py:93-116` |
| Go2W 注册 pre_step_control | `src/unilab/envs/locomotion/go2w/joystick.py:273` |
| Go2W _pre_step_motor_control | `src/unilab/envs/locomotion/go2w/joystick.py:391-405` |
| NpEnv.step() 控制流 | `src/unilab/base/np_env.py:104-166` |
| DR 中 kp/kd 随机化 | `src/unilab/dr/dr_utils.py:151-160` |
| Motrix position_actuator kp/kd 覆盖 | `src/unilab/base/backend/motrix/backend.py:1385-1399` |
| Go2 Rough _estimate_pd_torques (reward) | `src/unilab/envs/locomotion/go2/rough.py:523-539` |
| Footstand _estimate_pd_torques (reward) | `src/unilab/envs/locomotion/go2/footstand.py:659-666` |
| Sharpa torque_control 声明 | `src/unilab/envs/manipulation/sharpa_inhand/base.py:88` |
| create_backend position_actuator_gains | `src/unilab/base/backend/__init__.py:38-78` |
| Go2 Arm 分段增益 | `src/unilab/envs/locomotion/go2_arm/base.py:141` |

---

## 8. 总结

1. **PD 控制的实际计算**在 MuJoCo 的 position actuator 内部（路径 A），或通过 `pre_step_control` 回调显式计算（路径 B）
2. **`pre_step_control` 是替换 PD 的最佳切入点**——它已有生产级使用（Go2W），在每个 substep 前可读取全部状态
3. **Env 层 `apply_action()` 不需要大改**——它只负责生成目标，不涉及 PD 增益
4. **MuJoCo 已提供 M(q), C(q,q̇)+g(q) 等动力学量**，无需引入 Pinocchio 即可实现计算力矩控制
5. **推荐采用策略三**：定义 `MotorController` 抽象接口，通过 `ControlConfig.controller_type` 选择实现，保持与现有架构的 contract-driven 设计一致
