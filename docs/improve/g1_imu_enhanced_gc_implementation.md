# G1 IMU-Enhanced 重力补偿实现

> **目标**：基于六方案 A/B 对比的结论，设计并实现 IMU-Enhanced GC 控制器——在保持 GC 有益过补偿的前提下，利用 IMU 残余加速度检测 swing 腿并增强补偿，同时提供扰动修正能力。
>
> **核心创新**：不再从 GC 中减去接触力（CTC(IMU) 的做法），而是利用 IMU 信号**增强** swing 腿的补偿，保持 stance 侧的有益过补偿不变。

---

## 1 背景与动机

### 1.1 六方案 A/B 对比结果

| 方案 | Reward | 补偿策略 | 信号来源 |
|------|--------|---------|---------|
| GC | **323.68** | 重力补偿 g(q) | 模型 |
| CTC(IMU) | 319.84 | GC + IMU 残余力修正 | 模型 + IMU |
| CC | 317.24 | GC + Coriolis C(q,q̇)q̇ | 模型 |
| CTCP | 313.38 | GC + CC + 特权约束力 | 模型 + 仿真器 |
| Baseline | 308.23 | 纯 PD | — |
| CTC(force) | 302.10 | GC + 力传感器 | 力传感器 |

### 1.2 关键发现

1. **GC 的"过补偿"有益**：GC 对所有关节补偿 g(q)，stance 时 GRF 也在抵消重力，PD 看到净向上力 → 相当于"减轻体重"，有利于学习抬腿、维持高度
2. **CTC(IMU) 的 contact_scale 是减法**：从 GC 中减去接触力 → 降低了有益的过补偿 → 不如纯 GC
3. **qacc 路线不通**：CTCP 用差分 qacc 反推约束力，误差放大抵消了特权信息价值
4. **IMU 残余加速度有独有价值**：关节编码器无法感知外力，IMU 可以检测残余加速度 → 可用于接触判断

### 1.3 设计思路转变

| 旧思路（CTC-IMU） | 新思路（IMU-GC） |
|-------------------|-----------------|
| 从 GC 中减去接触力 | 保持 GC 不变，增强 swing 腿补偿 |
| contact_scale 是减法 → 降低过补偿 | swing_boost 是加法 → 保持过补偿 |
| 所有关节统一减 | swing/stance 区分处理 |

---

## 2 方案设计

### 2.1 核心公式

```
τ = PD + gravity_scale · g(q) · (mask + swing_boost · swing_mask)
    + disturbance_scale · τ_disturbance
```

其中：
- `mask`：基础 GC mask（腿+腰=1，手臂=0）
- `swing_mask`：每步动态计算，swing 腿关节=1，其余=0
- `τ_disturbance`：IMU 残余力通过 pelvis Jacobian 映射的扰动修正，仅超过阈值时激活

### 2.2 物理直觉

**Stance 侧**（脚在地面上）：
- GRF 支撑身体 → GC 的过补偿有益 → 保持 `gravity_scale = 1.0`
- 不做任何减法 → 不削弱过补偿的好处

**Swing 侧**（脚在空中）：
- 无 GRF 支撑 → GC 恰好是"正确补偿"而非"过补偿"
- 额外增强 `swing_boost = 0.3` → swing 腿更轻快，有利于足底轨迹跟踪
- 效果：swing 腿的有效 gravity_scale = 1.0 + 0.3 = 1.3

**扰动修正**：
- IMU 残余加速度 = R · a_sensor - [0,0,9.81]
- 当 |f_residual| > threshold → 有意外扰动 → 通过 pelvis Jacobian 加入修正
- 阈值门控避免在正常 stance 时误触发（stance 时残余力大但属于正常 GRF）

### 2.3 Swing 腿检测

采用 **gait_phase + IMU 交叉验证** 的双重判断：

1. **Gait phase 先验**：phase ∈ (π, 2π) → swing，phase ∈ [0, π] → stance
2. **IMU 交叉验证**：如果 IMU 残余力竖直分量 > mg/2 → 有接触 → stance
3. **最终判断**：`swing = gait_says_swing AND NOT imu_says_stance`

交叉验证的作用：
- gait phase 是开环预测，可能因扰动/滑步而失准
- IMU 提供闭环验证：如果 IMU 检测到大的向下残余力，说明脚确实在承重
- 两者取交集，减少误判

### 2.4 G1 关节映射

G1 的 29 个关节顺序：

```
[left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
```

Swing mask 构建：
- 左腿 swing → mask[0:6] = 1.0
- 右腿 swing → mask[6:12] = 1.0
- 腰部不区分 → mask[12:15] = 0.0
- 手臂不参与 GC → mask[15:29] = 0.0

---

## 3 实现细节

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `src/unilab/control/imu_gc_controller.py` | IMUGravityCompController 控制器 |
| `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_imu_gc.yaml` | 训练配置 |
| `tests/envs/locomotion/g1/test_g1_imu_gc.py` | 单元测试（6 项） |

### 3.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/g1/joystick.py` | 新增 G1WalkFlatIMUGCEvn 环境及 Config |
| `src/unilab/control/resolver.py` | 注册 `"imu_gc"` 控制器 |
| `src/unilab/control/__init__.py` | 导出 IMUGravityCompController |

### 3.3 控制器实现

`IMUGravityCompController` 继承 `MotorController`，核心逻辑：

```python
def compute(self, target_pos, joint_pos, joint_vel, *,
            full_qpos=None, full_qvel=None,
            swing_mask_per_env=None, tau_disturbance=None, **kwargs):
    # 1. PD term
    out = kp * (target_pos - joint_pos) - kd * joint_vel

    # 2. Gravity with swing boost
    if full_qpos is not None and full_qvel is not None:
        tau_gravity = dynamics_model.gravity(full_qpos, full_qvel)
        effective_mask = gravity_comp_mask + swing_boost * swing_mask_per_env
        out += gravity_scale * tau_gravity * effective_mask

    # 3. Disturbance correction
    if disturbance_scale != 0 and tau_disturbance is not None:
        out += disturbance_scale * tau_disturbance

    return clip(out, force_lower, force_upper)
```

**设计要点**：
- `swing_mask_per_env` 和 `tau_disturbance` 由 env 在 `_pre_step_motor_control` 中计算后传入
- controller 本身不访问 backend/IMU，保持纯计算职责（符合 Backend isolation 原则）
- `effective_mask` 支持逐环境动态变化（每个 env 的 swing/stance 状态不同）

### 3.4 环境实现

`G1WalkFlatIMUGCEvn` 继承 `G1BaseEnv`，核心方法 `_compute_imu_signals()`：

```python
def _compute_imu_signals(self, backend, gait_phase):
    # 1. IMU 残余加速度
    accel_local = backend.get_sensor_data("pelvis_acceleration")
    R = quaternion_to_rotation_matrix(backend.get_full_qpos())
    accel_world = R @ accel_local
    accel_world[:, 2] -= 9.81
    f_residual = robot_mass * accel_world

    # 2. Swing 检测：gait_phase + IMU 交叉验证
    left_swing_by_phase = gait_phase[:, 0] > π
    right_swing_by_phase = gait_phase[:, 1] > π
    imu_says_stance = f_residual[:, 2] < -0.5 * mg
    left_swing = left_swing_by_phase & ~imu_says_stance
    right_swing = right_swing_by_phase & ~imu_says_stance

    # 3. 构建 swing_mask
    swing_mask[left_swing, 0:6] = 1.0   # 左腿
    swing_mask[right_swing, 6:12] = 1.0  # 右腿

    # 4. 扰动修正（门控）
    Jp_pelvis = backend.get_site_jacobian_w(pelvis_imu_site_id, ...)
    tau_raw = J_pelvis^T @ f_residual
    active = ||f_residual|| > threshold
    tau_disturbance[active] = tau_raw[active]

    return swing_mask, tau_disturbance
```

### 3.5 训练配置

```yaml
env:
  control_config:
    gravity_comp_mask: [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]
    gravity_scale: 1.0
    swing_boost: 0.3          # swing 腿 GC 增强 30%
    disturbance_scale: 0.2    # 扰动修正保守系数
    disturbance_threshold: 5.0  # 5N 激活阈值
```

参数选择依据：
- `swing_boost = 0.3`：swing 腿有效 GC = 1.3，保守增强，避免过度补偿导致 swing 腿飘
- `disturbance_scale = 0.2`：与 CTC(IMU) 的 `contact_scale = 0.5` 对比更保守，因为这里只用于扰动修正而非全面接触补偿
- `disturbance_threshold = 5.0`：G1 总重 ~18kg，mg ≈ 177N，5N ≈ 3% mg，足以过滤正常 stance 时的 GRF 波动

---

## 4 验证

### 4.1 单元测试

6 项测试全部通过：

| 测试 | 验证内容 |
|------|---------|
| `test_output_shape` | 输出形状正确 |
| `test_no_boost_no_disturbance_matches_gc` | swing_boost=0, disturbance_scale=0 时退化为纯 GC |
| `test_swing_boost_applied` | swing_boost 正确增加 swing 关节的补偿 |
| `test_disturbance_correction_applied` | disturbance_scale 正确叠加扰动修正 |
| `test_force_clipping` | 输出被 forcerange 截断 |
| `test_gait_phase_swing_detection` | gait phase → swing/stance 判断逻辑正确 |

### 4.2 Smoke Test

- 4 envs × 50 步：100% 存活，力矩范围 [-87.2, 34.4] Nm
- 64 envs × 50 iters 快速训练：正常完成

### 4.3 训练结果

**配置**：4096 envs × 5000 iters, FlashSAC, swing_boost=0.3, disturbance_scale=0.2

**训练完成** ✅ — wall time: 3330s (~55 min), 最终 reward: **317.53**

**收敛轨迹**：

```
iter  100:   5.86
iter  200:  59.36
iter  500: 317.40
iter 1000:   6.05   ← 注意：iter 编号是 log 点，每 10 iters log 一次
iter 2000:  60.89
iter 3000: 219.67
iter 4000: 287.53
iter 5000: 317.53
```

**七方案对比**：

| 方案 | Reward | 排名 | 相对 Baseline |
|------|--------|------|--------------|
| GC | **323.68** | 1 | +15.5 |
| CTC(IMU) | 319.84 | 2 | +11.6 |
| CC | 317.24 | 3 | +9.0 |
| **IMU-GC** | **317.53** | **3** | **+9.3** |
| CTCP | 313.38 | 5 | +5.2 |
| Baseline | 308.23 | 6 | — |
| CTC(force) | 302.10 | 7 | -6.1 |

**分析**：

1. **IMU-GC (317.53) 与 CC (317.24) 基本持平**，略优于 CTCP (313.38)，但未超越 CTC(IMU) (319.84) 和 GC (323.68)
2. **Swing boost 未带来预期增益**：swing_boost=0.3 增强了 swing 腿的 GC，但整体 reward 反而略低于纯 GC
3. **可能原因**：
   - Swing 阶段增强 GC 使 swing 腿"更轻"，但策略可能已经学会了在纯 GC 下正确控制 swing 腿的轨迹，额外增强反而改变了策略已适应的动力学
   - disturbance_scale=0.2 的扰动修正可能在正常行走时引入了不必要的修正（IMU 残余力在正常行走时并非纯噪声，包含了步态相关的周期性分量）
   - swing/stance 判断基于 gait_phase 开环预测，可能与实际接触状态有偏差

---

## 5 与已有方案的对比

| 特性 | GC | CTC(IMU) | IMU-GC |
|------|-----|----------|--------|
| 重力补偿 | ✅ 全量 | ✅ 全量 | ✅ 全量 + swing 增强 |
| 接触力处理 | ❌ | ⚠️ 减法（削弱过补偿） | ✅ 加法（增强 swing） |
| IMU 使用 | ❌ | ✅ 残余力 | ✅ 残余力 + 接触判断 |
| Swing/Stance 区分 | ❌ | ❌ | ✅ gait+IMU 双重判断 |
| 扰动修正 | ❌ | ❌ | ✅ 门控激活 |
| Sim-to-real 可行性 | ✅ | ✅ | ✅ |
| 参数量（Actor） | 285K | 285K | 285K |

**关键区别**：IMU-GC 是唯一一个利用 IMU 信号做**加法增强**而非**减法修正**的方案，这保留了 GC 的有益过补偿同时提供了更精确的 swing 侧补偿。

---

## 6 策略网络参数量评估

前馈补偿后，策略的"残差任务"变简单：

| 方面 | 无前馈 (Baseline) | 有前馈 (GC/IMU-GC) |
|------|-------------------|---------------------|
| 策略需要隐式学习的 | 重力对抗 + 接触动力学 + 步态 | 只需输出轨迹修正量 |
| 动作幅度 | 大（含重力对抗） | 小（残差修正） |
| 动力学线性度 | 高度非线性 | 更接近线性 |

当前 Actor **285K** 参数（hidden=128, 2 blocks）。理论上可缩小：

| 配置 | 参数量 | 可行性判断 |
|------|--------|-----------|
| hidden=128, 2 blocks（当前） | 285K | ✅ 一定可行 |
| hidden=96, 2 blocks | ~170K | ✅ 大概率可行 |
| hidden=64, 2 blocks | ~75K | ⚠️ 需验证 |
| hidden=128, 1 block | ~140K | ✅ 可行 |
| hidden=64, 1 block | ~35K | ❌ 对 29-dim action 可能不够 |

**结论**：有好的前馈补偿后，**hidden=96 或 hidden=64 + 2 blocks**（75K~170K）是合理的缩小范围，比当前 285K 减少 40%~75%。但这是实证问题，需 A/B 对比验证。
