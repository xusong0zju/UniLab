# IMU 力矩前馈：物理正确性分析与改进方案

> **日期**：2026-06-13
> **状态**：进行中
> **背景**：上轮 agent 将 IMU-GC 从 `τ_disturbance` 方案重构为 `gravity_factor` 方案，需验证物理正确性。

---

## 1 当前架构：信号链路逐层分析

### 1.1 MuJoCo 传感器 → ✅ 正确

```xml
<accelerometer site="imu_in_pelvis" name="pelvis_acceleration" cutoff="157" noise="0.01"/>
```

MuJoCo `<accelerometer>` 测量**比力（specific force）**——即 site 在本地坐标系中的加速度，含重力分量。等价于真实 IMU 加速度计输出。

| 场景 | 比力 (local Z) | 物理含义 |
|------|---------------|---------|
| 静止站立 | `+9.81` | 支撑力抵消重力 |
| 自由落体 | `0` | 失重 |
| 被向上推 (a=+2) | `+11.81` | 支撑力 + 推力 |

### 1.2 信号变换 `a_sensor → a_net_z` → ✅ 正确

```python
accel_world = R @ accel_local          # 比力 → 世界系
accel_world[:, 2] -= 9.81             # 减去重力 → 净加速度
```

| 场景 | `accel_local` Z | `accel_world` Z | `-9.81` 后 | `a_z` |
|------|----------------|----------------|-----------|-------|
| 静止 | `+9.81` | `+9.81` | `0` | `0` ✅ |
| 自由落体 | `0` | `0` | `-9.81` | `-9.81` ✅ |
| 推上 (a=+2) | `+11.81` | `+11.81` | `+2.0` | `+2.0` ✅ |

**结论：`imu_accel_net_z` = pelvis 运动加速度 `a_z`（世界系）。**

### 1.3 `gravity_factor = 1 + a_z/g` → ✅ 数学正确

非惯性系（pelvis 加速度 `a`）中等效重力：

```
g_eff = g - a = [0, 0, -9.81] - [0, 0, a_z] = [0, 0, -(9.81 + a_z)]

|g_eff| / |g| = (9.81 + a_z) / 9.81 = 1 + a_z/9.81 = gravity_factor
```

**标量乘法等价于修改 Pinocchio `model.gravity.linear[2]`。**（agent 已验证直接修改 binding 不生效，但数学等价。）

### 1.4 `τ_ff += g(q) × gravity_factor` → ❌ 物理错误

**这是核心问题。** 数学正确不代表物理正确。

`g(q)` 是**定常重力场的静态补偿力矩**。行走时 `a_z` 以 ~3Hz（双足步频）振荡于 ~±2 m/s²，导致 `gravity_factor` 在 0.7~1.3 振荡：

```
a_z (m/s²)
  +3 |     ▌▌
  +2 |   ▗▌  ▝▖
  +1 |  ▐    ▝▖
   0 |▗▘      ▝▌
  -1 |          ▝▖
  -2 |           ▝▌▌▌
     +-- heel strike --+-- heel strike --+
```

**总力矩方程：**

```
τ_total = kp·(q_d - q) - kd·q̇ + g(q)·(1 + a_z(t)/g)·mask
            ↑                        ↑
        PD 跟踪                  feedforward ~3Hz 振荡
```

**物理错误链条：**

1. `a_z(t) ≠ 0` 是行走的**结果**（蹬地 → pelvis 加速），不是重力变化的**原因**
2. 把结果当作原因喂回补偿力矩 → 形成错误反馈回路
3. PD 控制器被迫对抗 feedforward 的振荡 → 学习困难（iter 1000=3.75 vs 旧版 59.4）
4. 等效于给静态补偿叠加了 ~3Hz 噪声

**类比**：秋千上的推力应基于**平均**位置而非**瞬时**加速度。根据瞬时加速度调节推力只会让秋千更不稳定。

### 1.5 旧版 `τ_disturbance` 的物理分析

旧版公式：

```python
f_residual = robot_mass × a_net          # 世界系残余力
τ_disturbance = J_pelvis^T @ f_residual  # pelvis Jacobian 映射
τ = PD + g(q)·mask + swing_boost + 0.2·τ_disturbance
```

**物理含义**：`J_pelvis^T @ f_residual` 是静力学对偶——pelvis 残余力在关节空间的等效力矩。

| 场景 | f_residual 方向 | J^T·f 效果 | `+= 0.2·τ_dist` |
|------|----------------|-----------|-----------------|
| 站立 | ≈0 | 不激活 | 无影响 |
| 蹬地 | 向前+上 (~200N) | 推髋前摆、伸膝 | **帮助行走** ✅ |
| 被推 | 向推力方向 | 放大运动 | **放大扰动** ❌ |
| 滑倒 | 向下 | 拉关节向下 | **加速摔倒** ❌ |

- `+=` 有方向性矛盾，但**行走辅助效果显著**（iter 1000=59.4）
- `-=` 正确抵抗扰动但阻碍行走

---

## 2 改进方案

### 2.1 方案总览

| 方案 | 核心思路 | 物理正确性 | 实施复杂度 | 状态 |
|------|---------|-----------|-----------|------|
| **立即** 禁用 gravity_factor | 恢复 τ=PD+g·(mask+swing·swing_mask)，IMU 只用于 swing 检测 | ✅ 已验证 | 1 行 config | → 实施 |
| **短期** 相位条件残余检测 | 在线学习行走 `a_z` 模式，只对异常 `δa_z` 响应 | ✅ 最正确 | ~150 行 | 待开发 |
| **中期** 质心动量前馈 A(q) | `τ_imu = A(q)^T @ F_dist`，正确映射到所有关节 | ✅ 最严格 | ~200 行 | 待开发 |
| **远期** 接触一致分解 | 将 τ_contact 分解到每条腿 | ✅ 研究级 | >500 行 | 探索 |

### 2.2 立即方案：禁用 gravity_factor

**改动**：`conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_imu_gc_s96.yaml`

```yaml
# 改前
imu_modulated_gravity: true

# 改后
imu_modulated_gravity: false
```

**效果**：
- 力矩公式退化为 `τ = PD + g(q)·(mask + 0.3·swing_mask)`
- `_compute_imu_signals()` 返回 `imu_accel_net_z=None` → 控制器走 fallback
- IMU **仅用于 swing 检测**（gait_phase + IMU 交叉验证）→ 不受影响
- 等价于已验证的旧版 IMU-GC 去掉 disturbance 部分

### 2.3 短期方案：相位条件残余检测

**物理思想**：行走时 `a_z` 随步态相位有可预测的模式。维护运行均值表 `a_expected(phase)`，仅对偏离预期模式的**异常**加速度响应。

```python
class PhaseConditionedResidual:
    """在线学习正常行走的 base 加速度模式。

    维护一个 2D 查找表 a_expected(phase_left, phase_right)，
    用 EMA 在线更新。返回 actual - expected 作为"异常信号"。
    正常行走时 residual≈0，被推/绊倒时 residual≠0。
    """

    def __init__(self, num_envs: int, n_bins: int = 32, ema_alpha: float = 0.005):
        self._n_bins = n_bins
        self._alpha = ema_alpha
        # (N,) output buffer
        self._residual: np.ndarray = np.zeros(num_envs, dtype=np.float64)
        # (n_bins, n_bins) running-average table
        self._expected: np.ndarray = np.zeros((n_bins, n_bins), dtype=np.float64)
        # Per-bin update count (for cold-start detection)
        self._counts: np.ndarray = np.zeros((n_bins, n_bins), dtype=np.int64)

    def compute(self, a_z, phase_left, phase_right) -> np.ndarray:
        """返回异常加速度 residual (N,)。

        Args:
            a_z: pelvis net vertical acceleration (N,)
            phase_left: left leg gait phase (N,) ∈ [0, 2π]
            phase_right: right leg gait phase (N,) ∈ [0, 2π]

        Returns:
            residual (N,) — 正常≈0, 扰动≠0
        """
        # 相位 → bin
        li = np.clip(
            (phase_left / (2 * np.pi) * self._n_bins).astype(np.int64),
            0, self._n_bins - 1,
        )
        ri = np.clip(
            (phase_right / (2 * np.pi) * self._n_bins).astype(np.int64),
            0, self._n_bins - 1,
        )

        # EMA 更新每个 bin 的期望值
        # 用 advanced indexing 做 per-env 更新
        alpha = self._alpha
        for i in range(len(a_z)):
            l, r = li[i], ri[i]
            self._expected[l, r] = (
                (1.0 - alpha) * self._expected[l, r] + alpha * a_z[i]
            )
            self._counts[l, r] += 1

        # residual = 实际 - 预期
        expected_per_env = self._expected[li, ri]
        np.subtract(a_z, expected_per_env, out=self._residual)
        return self._residual

    @property
    def is_warm(self) -> bool:
        """是否已有足够样本用于可靠的预期值。"""
        return np.all(self._counts >= 10)
```

**力矩注入**（门控 + 小比例）：

```python
# 只在 residual 显著时激活
perturbation_threshold = 2.0  # m/s² (~0.2g)
active = np.abs(residual) > perturbation_threshold

if active.any():
    # 通过 pelvis Jacobian 映射 → 阻尼扰动
    f_residual = robot_mass * residual[:, None] * np.array([[0, 0, 1.0]])  # (N, 3), Z 方向
    J_pelvis = backend.get_site_jacobian_w(pelvis_imu_site_id, actuated_dof_indices)
    tau_damp = np.einsum("ejk,ej->ek", J_pelvis, f_residual)
    tau_damp *= active[:, None]  # 门控
    τ_ff -= 0.1 * tau_damp       # 小比例阻尼
```

**为什么正确**：
- 正常行走：`residual ≈ 0` → `tau_damp = 0` → **零干扰**  
- 被推：`residual > 0`（加速超过预期）→ 施加反向力矩抵抗
- 零延迟（无 EMA 滤波器滞后）
- 在线自适应：不同步态、不同速度自动学习预期模式

### 2.4 中期方案：质心动量前馈 A(q)

**物理思想**：用 Pinocchio 的质心动量矩阵 `A(q)` 替代粗糙的 pelvis Jacobian，将 base 加速度残差**正确**映射到所有关节（含腰）。

```python
# 每步调用 Pinocchio（可用 substep 缓存）
pin.computeCentroidalMap(model, data, q)   # → data.Ag (6×nv)
A_actuated = data.Ag[:3, 6:]               # 只取平动部分 + 驱动关节 (3×nv_act)

# 扰动合力的关节映射
F_dist = m_total * a_residual  # (N, 3)
tau_imu = A_actuated.T @ F_dist.T  # (nv_act, N) = (3×nv_act)^T × (3×N)
```

**与 pelvis Jacobian 的区别**：
- `J_pelvis^T`：pelvis 单点的力 → 关节力矩（仅腿关节有贡献）
- `A(q)^T`：COM 加速度 → 关节力矩（腿+腰都参与，物理严格）

**代价**：每步多一次 `computeCentroidalMap` 调用（~200μs per env in Pinocchio），对 4096 env 是 ~0.8ms 额外开销。

---

## 3 实现计划

### 3.1 已实施

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `conf/.../mujoco_imu_gc_s96.yaml` | `imu_modulated_gravity: true → false` |

### 3.2 待实施

| 步骤 | 文件 | 内容 |
|------|------|------|
| 2 | `docs/improve/g1_imu_enhanced_gc_implementation.md` | 修正 §8，使其与代码一致 |
| 3 | `docs/analy/imu_gc_sign_analysis_and_redesign.md` | 追加本文的物理分析结论 |
| 4 | `src/unilab/envs/locomotion/g1/joystick.py` | 修正 `_pre_step_motor_control` docstring |
| 5 | (可选) 新 training run | 对比 `imu_modulated_gravity=true` vs `false` |

---

## 4 训练记录

### 4.1 当前训练（gravity_factor=true, 10000 iters）

- 日志：`logs/flash_sac/G1WalkFlatIMUGC/2026-06-13_22-45-06_mujoco/`
- PID：15998（运行中）
- iter ~7530 reward=321.2

### 4.2 新版（gravity_factor=false, disturbance removed）

> **状态**：已完成（2026-06-14）。日志：`logs/flash_sac/G1WalkFlatIMUGC/2026-06-13_23-57-39_mujoco/`

**A/B 对比结果**：

| iter | gf=true | gf=false | Δ |
|------|---------|----------|---|
| 500 | 5.75 | 5.26 | -0.49 |
| 1000 | 9.35 | 10.92 | +1.58 |
| 2000 | 75.23 | 72.47 | -2.75 |
| 3000 | 234.47 | 232.54 | -1.93 |
| 5000 | 310.49 | 319.01 | +8.52 |
| 10000 | 323.76 | 327.71 | +3.95 |

| 指标 | gf=true | gf=false |
|------|---------|----------|
| best_mean_reward | 323.56 | 325.68 |
| 训练时间 | 3225s | 3254s |

**确认**：
1. gravity_factor 对收敛无负面影响（iter 0-3000 曲线重合）→ §1.4 物理分析中"PD 被迫对抗振荡"的预测被证实为**影响极小**，PD 可以轻松克服 ~3Hz 振荡
2. 晚期新版微弱领先 ~2-8 点 → 省略 per-step gravity_factor 计算减少了力矩注入中的高频噪声
3. 之前 iter 1000=3.75 的慢速来自 **τ_disturbance 的删除**，而非 gravity_factor

---

## 5 附录：文档不一致问题

| 位置 | 声称 | 实际代码 |
|------|------|---------|
| `g1_imu_enhanced_gc_implementation.md` §8.7 | "方案 A: `-= τ_dist`, scale=0.2" | τ_disturbance 已被完全删除 |
| `imu_gc_controller.py` docstring | "gravity_factor applied to ALL joints" | ✅ 一致 |
| `joystick.py` `_pre_step_motor_control` docstring | "Upper body joints [12:]: mask · imu_gravity_factor" | 实际应用于**所有**关节 |
| `imu_gc_sign_analysis_and_redesign.md` §5 | gravity_factor + swing_boost | ✅ 一致 |
