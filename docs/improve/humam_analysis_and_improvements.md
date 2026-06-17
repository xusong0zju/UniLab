# HuMam 论文分析 与 Mamba 多技能策略改进方案

> 日期：2026-06-17 | 基于 HuMam (arXiv:2509.18046) 详细分析

## 1. 论文概述

**HuMam** (arXiv:2509.18046, CIS-RAM 2026) 是首个将 Mamba 作为 end-to-end 人形机器人 RL 控制骨干的工作，在 JVRC-1 人形机器人上验证。使用 PPO 训练，单层 Mamba encoder。

### 关键指标
- 比 MLP baseline 少 42.5% 样本达到同等回报
- 跨 seed 方差降低 35.4%
- 站立功耗降低 39%
- 行走能效从 673 J/m 降至 421 J/m

---

## 2. 架构对比：HuMam vs 我们的 Mamba 实现

| 维度 | HuMam | 我们 (MambaActor) |
|------|-------|-------------------|
| Mamba 层数 | **1 层** | 2 层 |
| hidden_dim | **128** | 256 |
| Token 数量 | **2 个** | 4 个 |
| Token 内容 | **body_state ⊗ task_state**（语义不同） | 同一 obs 复制 4 份（语义相同）|
| Token 宽度 | 41 | 98 |
| Reward 项数 | **6 项** | ~15 项 |
| Reward 形式 | 全部 exp(-error²) | 混合 L2 / exp / 线性 |
| 外部引导 | 足部落点 T1, T2 | velocity command |
| 相位表示 | sin/cos 连续时钟 | [left_phase, right_phase] |
| Episode 长度 | **400 步 (8s)** | 1000 步 (20s) |
| 算法 | PPO | FlashSAC |
| 关节数 | 12 (仅腿) | 29 (全身) |
| 技能数 | 1 (行走) | 7 (walk/stand/flamingo/push/fall/transition) |

### 核心差异

#### Token 设计 —— 最关键的差距

```
HuMam:
  s_t = [s_robot, s_external]           # 拼接
  z_robot, z_ext = f_proj(s_t)          # 各自 embedding → d=41
  tokens = [z_robot, z_ext]             # 2 tokens, 语义不同
  SSM 看到: "身体怎么样" vs "任务是什么"

我们:
  x = token_proj(obs)                   # Linear(98→1024)
  x = reshape(B, 4, 256)                # 4 tokens
  SSM 看到: 同一个 obs 的 4 份线性投影, 语义相同
```

**问题**：4 个 token 之间没有信息差异，SSM 的选择性门控 (Δ, B, C) 无从"选择"——所有 token 都告诉它同样的东西。

#### Reward 冲突

HuMam 6 项 reward 全部用 exp(-error²)，梯度平滑，无正值/负值冲突。

我们 15 项 reward 混合正负值（alive=+5, orientation=-20, foot=-30），正值"活下去"和负值"别倒"在 fallen 状态下产生矛盾信号。

---

## 3. 实验教训总结

### 已验证的有效设计

| 设计 | 效果 | 证据 |
|------|------|------|
| Mamba 替代 MLP | tracking +60%~150%, reward 翻倍 | Phase A/B Mamba vs MLP |
| 选择性 SSM | 多模态自然共存, 无显式 skill ID | Phase B 990ep + foot=-0 |
| keyframe 初始化 | 解决"不愿抬脚"问题 | flamingo keyframe → ep=987 |
| 渐进课程 A→B→C | 每阶段稳定收敛 | 全流程 |
| HoST 自适应 orientation | reward +34% | D v2 vs v1 |
| 1 秒时间窗 | 无效果 | D v2 时序版反而更差 |
| 涌现式摔倒恢复 | **未成功** | D v1/v2 reward 停滞 -47 |

### 尚未解决的问题

1. **Fallen recovery**——20% 倒地 env 始终未学会起身，reward 停滞在 -47
2. **Token 同质化**——4 个 token 语义相同，SSM 选择空间受限
3. **Reward 冲突**——15 项 reward 在多模式下互相干扰
4. **长 episode**——1000 步过长，fallen env 长时间挨罚，探索机会少

---

## 4. 改进方案

### 4.1 2-Token 语义分离（最高优先级）

```
body_state (50-dim):
  gyro(3) + gravity(3) + dof_pos_diff(29) + dof_vel(29) + contact(2)
  → 嵌入 → token_0 (256-dim)

task_state (45-dim):
  last_actions(29) + commands(3) + gait_phase(2)
  → 嵌入 → token_1 (256-dim)

tokens = [token_0, token_1]   # 2 tokens, 语义不同
→ SSM 扫描 → mean pool → action
```

### 4.2 缩短 Episode（400 步 = 8s）

- 更多 reset → fallen env 每周期的"躺平惩罚"缩短
- 更频繁的模式切换机会（配合 resampling_time=3.0）

### 4.3 精简 Reward

| 保留 (核心 6 项) | 移除/合并 |
|------|------|
| tracking_lin_vel | ~~tracking_ang_vel~~ |
| penalty_orientation_adaptive | ~~penalty_orientation~~ |
| penalty_base_height | — |
| pose | ~~upper_body_pose~~ |
| penalty_lifted_foot_contact | ~~penalty_support_foot_contact~~ |
| alive | ~~feet_phase, feet_phase_contrast, feet_phase_contact, feet_double_stance, feet_air_time, penalty_feet_ori, penalty_close_feet_xy, com_over_support, forward_progress, under_speed, lin_vel_z, ang_vel_xy~~ |

### 4.4 exp() 形式替换 L2

```
当前: penalty_orientation = -(gravity_x² + gravity_y²) × scale
改为: penalty_orientation = scale × exp(-10 × (1 - gravity_z²))
```

exp 形式梯度平滑，边界值 0~1，不会出现极大负值压垮 fallen env。

### 4.5 连续相位时钟

```
当前: gait_phase = [left_phase, right_phase]  # 二值
改为: gait_phase = [sin(2πφ), cos(2πφ)]        # 连续光滑
```

更适配 Mamba SSM 的连续值门控特性。

---

## 5. 实施计划

### Phase E（消融实验，5000 iter）

```yaml
改进: 2-token + 400步 + 精简reward + exp形式
warm-start: Phase B model_4000
推力: 50N
其他: 保持 resampling=5.0, rel_fallen=0.2, max_tilt=180, min_height=0
```

预期：reward 从 -47 升到正数，fallen env 开始出现"起身"行为。

### 改动文件

| 文件 | 改动 |
|------|------|
| `mamba_actor.py` | `_encode` 改为 2-token 语义分离 |
| `mamba_phaseE.yaml` | 新配置：400ep, 精简 reward, exp 形式 |
| `multiskill.py` | reward 函数加 exp 版本 |
| `mamba_phaseE.yaml` | 连续相位时钟 |

---

## 6. 训练结果汇总

| 阶段 | 技能 | Ep | Timeout | Tracking | Reward | Foot |
|------|------|-----|---------|----------|--------|------|
| A (MLP) | walk+stand | 970 | 90% | 1.0 | +104 | — |
| **A (Mamba)** | walk+stand | **973** | 90% | **1.6** | **+214** | — |
| B (MLP) | +flamingo | 997 | 100% | 1.1 | +69 | -0.2 |
| **B (Mamba)** | +flamingo | **987** | **100%** | **1.7** | **+272** | **-0.0** |
| C1 (50N) | +push | 865 | 60% | 0.6 | +239 | -0.2 |
| C2 (100N) | +push | 346 | 60% | 1.3 | +96 | -0.1 |
| D v1 | +fall+transition | 999 | — | 1.19 | -71 | -0.2 |
| D v2 (HoST) | +adaptive orient | 999 | — | 1.10 | -47 | — |

> Mamba 在 Phase A/B 显著优于 MLP。C1/C2 抗扰后 tracking 震荡但可恢复。D 阶段 fallen recovery 未成功（reward 停滞）。**2-token + 短 episode + 精简 reward** 是下一步方向。

---

## 7. 参考资料

- HuMam: [arxiv.org/abs/2509.18046](https://arxiv.org/abs/2509.18046)
- HoST (RSS 2025): [github.com/InternRobotics/HoST](https://github.com/InternRobotics/HoST)
- Mamba: [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
