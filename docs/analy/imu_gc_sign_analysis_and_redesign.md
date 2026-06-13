# IMU-GC 符号分析与 IMU 正确用法探索

> **日期**：2026-06-13
> **状态**：进行中 — 训练正在后台运行（iter ~3200, reward ~209, 10000 iters 目标）

---

## 1 背景

之前 IMU-GC 控制器的公式为：

```
τ = PD + gravity_scale·g(q)·(mask + swing_boost·swing_mask)
    + disturbance_scale·τ_disturbance
```

sim2sim 评估中发现 IMU-GC 的关节跟踪误差（0.2553）比 Baseline（0.2270）大 12.5%，违背"GC 应减少跟踪误差"的直觉，引发了对补偿符号的质疑。

---

## 2 符号分析（已推翻的结论 → 正确结论）

### 2.1 初版分析（错误）

认为所有 `+=` 补偿项符号反了，理由是：
- 站立机器人单步测试：`PD - g(q)` 跟踪误差比 `PD + g(q)` 更小
- 推导：`Mq̈ = τ - qfrc_bias + qfrc_constraint`，站立时 `qfrc_constraint ≈ -qfrc_bias`
- 所以 `τ = +g(q)` 使 `Mq̈ = -qfrc_bias`（重力加倍）

**修正后执行**：将所有 `+=` 改为 `-=`，启动训练，iter 850 reward 仅 -2.7（远低于旧版 iter 500 的 5.9）。

### 2.2 修正结论

初版分析**只看了站立场景**，忽略了行走时 Swing 腿无接触力：

```
Swing 腿（无接触）:
  Mq̈ = τ_ctrl - qfrc_bias + 0
  τ = PD + g(q) → Mq̈ = PD + g(q) - g(q) = PD     ✓ 完美补偿！
  τ = PD - g(q) → Mq̈ = PD - 2·g(q)                ✗ 重力加倍！

Stance 腿（有 GRF）:
  Mq̈ = τ_ctrl - qfrc_bias + qfrc_constraint
  τ = PD + g(q) → Mq̈ = PD + qfrc_constraint       过补偿（有益）
  τ = PD - g(q) → Mq̈ = PD - 2·g(q) + qfrc_constraint  取决于 GRF
```

**正确符号结论**：

| 控制器 | 补偿项 | 符号 | 正确性 | 物理含义 |
|--------|--------|------|--------|---------|
| GravityCompController | gravity | `+= g(q)` | ✅ | 添加支持力矩抵消重力 |
| CoriolisCompController | gravity | `+= g(q)` | ✅ | 同上 |
| CoriolisCompController | coriolis | `+= C(q,q̇)q̇` | ✅ | 抵消科氏力 |
| ContactCompController | gravity | `+= g(q)` | ✅ | 同上 |
| ContactCompController | coriolis | `+= C(q,q̇)q̇` | ✅ | 同上 |
| ContactCompController | contact | `-= τ_contact` | ✅ | 减去 GRF，降低 stance 过补偿 |

**三个控制器（GC/CC/ContactComp）不需要修改。**

---

## 3 IMU Disturbance 符号问题

### 3.1 Disturbance 的方向矛盾

IMU 残余力 `f_residual = mass × (R·accel_local - [0,0,9.81])`：

| 场景 | f_residual 方向 | `+=` 效果 | `-=` 效果 |
|------|----------------|----------|----------|
| 正常行走推蹬 | 向前/上 | 帮助行走 ✅ | 阻碍行走 ❌ |
| 被推（扰动） | 向前 | 放大扰动 ❌ | 抵消扰动 ✅ |
| Swing 下落 | 向下 | 加速下落 ❌ | 减缓下落 ✅ |

**矛盾**：`+=` 对正常行走有帮助但放大扰动；`-=` 抵消扰动但也阻碍正常行走。

### 3.2 根因

disturbance correction 无法区分"正常步行动力学"和"意外扰动"——阈值 5.0 N 太低，正常行走的 GRF 振荡（~245N）远超 5N。

### 3.3 实测验证

`+= τ_dist` (disturbance_scale=0.2, threshold=5.0) 训练：iter 2630 reward=117.12，远慢于旧版 iter 1000=59.4。

---

## 4 IMU 的正确用法（用户指导）

### 4.1 用户关键洞察

> "IMU处测量的加速度（依据姿态去除重力加速度后的净加速度）。它的主要作用在两方面：
> 1、对于IMU以上部分，提供解算力矩补偿的基准，举个极端的例子，若此时IMU处的净加速度是z轴向下的g（重力加速度），则上身无需重力补偿了。
> 2、对于IMU以下部分则有助于接触力计算及重力补偿。"

> "在pinocchio这种框架下，对于腰部的这种IMU量测解算值引入力矩补偿，应该有相关接入方式吧，可不仅仅是重力补偿。IMU的加速度与角速度其实都是有影响的。"

> "下身的足部接触地面时，是有支撑力的，这个对于所谓的肢体重力及其传导推导的力矩补偿是有很大影响的。"

### 4.2 正确理解

1. **上身（IMU 以上）**：IMU 净加速度决定了**有效重力** `g_eff = g - a_base`。在非惯性系中：
   - 站立（a_net=0）：有效重力=1g → 完整补偿
   - 自由落体（a_net_z=-9.81）：有效重力=0 → 无需补偿
   - 被推向上（a_net_z>0）：有效重力>1g → 需更多补偿

2. **下身（IMU 以下）**：IMU 加速度帮助推算 GRF 分布，但当前无法分解到每条腿。stance 腿的 GRF 部分抵消重力 → 补偿全部 g(q) 是"过补偿"（对学习有益）。

3. **Pinocchio/RNEA 框架**：理论上应修改 `model.gravity.linear` 为 `g_eff = [0,0,-9.81] - a_net`，然后重新计算 g(q)。这样 RNEA 自动处理运动学链。但实测 Python binding 修改 `model.gravity.linear` 不生效（可能是 caching 或 binding 问题）。

---

## 5 最终实现方案

### 5.1 当前实现

用 **gravity_factor** 替代 disturbance correction：

```python
# IMU gravity modulation
gravity_factor = clip(1 + a_net_z / 9.81, 0, 2)  # per-env

τ = PD + gravity_scale · g(q) · gravity_factor[:, None] · modulated_mask
```

其中 `modulated_mask = gravity_comp_mask + swing_boost * swing_mask`

**验证结果**（独立测试）：
- Free fall (a_net_z=-9.81): factor=0 → gravity 补偿为 0 ✅
- Pushed up (a_net_z=5.0): factor=1.51 → gravity 增加 51% ✅
- Standing (a_net_z=0): factor=1.0 → 标准补偿 ✅

### 5.2 收敛速度问题

**当前训练**（gravity_factor 方案，10000 iters）收敛速度明显慢于旧版：

| 版本 | iter 500 | iter 1000 | iter 2000 | iter 3000 |
|------|----------|-----------|-----------|-----------|
| 旧版 IMU-GC (`+=g + +=τ_dist`) | 5.9 | 59.4 | 218.8 | 317.4 |
| **新版** (gravity_factor + swing_boost) | 0.26 | 3.75 | ~33 | ~207 |

**根因**：gravity_factor 用每步的 `a_net_z` 实时调制，但行走时 `a_net_z` 随步态周期剧烈波动，导致补偿力矩高频振荡，策略学习困难。

### 5.3 待选改进方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. EMA 滤波 | 低通滤波 `a_net_z`，只保留低频分量 | 过滤步态波动，保留大扰动响应 | EMA 参数需调，引入延迟 |
| B. 只用直流分量 | 行走稳态时 `a_net_z` 均值≈0，factor≈1.0 → 等于普通 GC | 最稳定 | 完全丧失扰动响应 |
| C. 禁用 IMU 调制 | `imu_modulated_gravity=false`，只保留 swing_boost | 已验证有效，训练快 | IMU 只用于 swing 检测 |
| D. Pinocchio g_eff | 修改 model.gravity 为 g-a_base | 物理最正确 | Python binding 不生效，需 per-env RNEA |

---

## 6 修改文件清单

### 6.1 已修改

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `src/unilab/control/gravity_comp_controller.py` | 更新 docstring（符号已恢复原样） | ✅ |
| `src/unilab/control/coriolis_comp_controller.py` | 更新 docstring（符号已恢复原样） | ✅ |
| `src/unilab/control/contact_comp_controller.py` | 更新 docstring（符号已恢复原样） | ✅ |
| `src/unilab/control/imu_gc_controller.py` | 重写：disturbance → gravity_factor | ✅ |
| `src/unilab/control/resolver.py` | 移除 disturbance_scale 参数 | ✅ |
| `src/unilab/envs/locomotion/g1/joystick.py` | 重写 `_compute_imu_signals`，移除 disturbance | ✅ |
| `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_imu_gc_s96.yaml` | max_iterations=10000, 移除 disturbance 参数, 加 imu_modulated_gravity | ✅ |
| `docs/improve/g1_imu_enhanced_gc_implementation.md` | §8 符号分析已修正 | ✅ |

### 6.2 关键代码变更

**Controller** (`imu_gc_controller.py`)：
- 移除 `disturbance_scale` 参数
- 移除 `tau_disturbance` 输入
- 新增 `imu_accel_net_z` 输入 (num_envs,)
- gravity_factor = clip(1 + a_net_z/9.81, 0, 2)，per-env 乘到 g(q) 上

**Env** (`joystick.py` - G1WalkFlatIMUGCEvn)：
- Config: `disturbance_scale`/`disturbance_threshold` → `imu_modulated_gravity: bool`
- `_compute_imu_signals()`: 返回 `(swing_mask, imu_accel_net_z)` 而非 `(swing_mask, tau_disturbance)`
- 不再计算 `J_pelvis^T @ f_residual`（移除 disturbance Jacobian 计算）
- 保留 swing 检测（gait_phase + IMU cross-validation）

---

## 7 当前训练状态

**日志目录**：`logs/flash_sac/G1WalkFlatIMUGC/2026-06-13_22-45-06_mujoco/`

**配置**：IMU-GC 96×2, 10000 iters, gravity_factor + swing_boost=0.3

**进度**（截至会话结束前）：
- iter ~3200, reward ~209
- 收敛速度约为旧版的 1/3~1/5
- PID: 15998 (可能已被系统回收)

**查看训练进度命令**：
```bash
uv run python -c "
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('logs/flash_sac/G1WalkFlatIMUGC/2026-06-13_22-45-06_mujoco/')
ea.Reload()
events = ea.Scalars('reward/mean')
for e in events[-5:]:
    print(f'iter~{e.step//4096:5d}, reward={e.value:8.2f}')
"
```

---

## 8 下一步建议

1. **等训练完成**（10000 iters），评估最终 reward
2. **对比旧版**：如果新版 10k reward 仍低于旧版 5k（312），说明 gravity_factor 方案需要改进
3. **优先尝试方案 C**（禁用 IMU 调制，只保留 swing_boost），最快验证
4. **方案 A**（EMA 滤波）是中期方案，需要调参
5. **方案 D**（Pinocchio g_eff）是长期正确方案，但需解决 per-env RNEA 计算的性能问题

---

## 9 旧版（原始 IMU-GC）训练记录

| 配置 | Actor 参数 | Reward @5k | Reward @10k |
|------|----------|-----------|------------|
| IMU-GC 128×2 | 285K (100%) | 317.53 | — |
| IMU-GC 96×2 | 165K (58%) | 312.17 | 325.92 |
| IMU-GC 64×2 | 77K (27%) | 289.89 | — |
| IMU-GC 96×1 | 90K (32%) | 300.05 | — |
| Baseline 96×2 | 165K (58%) | 300.65 | — |
| Baseline 128×2 | 285K (100%) | 304.67 | — |

**旧版公式**：`τ = PD + g(q)·(mask + 0.3·swing_mask) + 0.2·τ_disturbance`
- gravity: `+=` ✅ 正确
- disturbance: `+=` ⚠️ 方向有矛盾（见 §3）
- 旧版 reward 高但跟踪差，disturbance `+=` 对正常行走有帮助是原因之一

---

## 10 符号验证总结

**关键验证**：从 MuJoCo 动力学推导

```
Mq̈ = τ_ctrl - qfrc_bias + qfrc_constraint
目标: Mq̈ = PD
需要: τ_ctrl = PD + qfrc_bias - qfrc_constraint
     = PD + g(q) - qfrc_constraint
```

这验证了 **gravity 用 `+=`，contact 用 `-=`** — 与原始代码一致。六方案 A/B 对比的结论基本正确（GC > CC > Baseline），不需要重跑。
