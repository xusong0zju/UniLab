# G1 多技能统一策略——Mamba 架构方案

> 状态：**实现中** | Phase A Mamba 训练运行中 (2026-06-16)
> | 日期：2026-06-16 | 作者：xusong0zju

## 1. 动机

当前 FlashSAC MLP 网络（14.4M）在同时学习 walk + stand + flamingo + 多部位抗扰时卡在 ep=261，walk tracking 退化到 0.1。

**根因分析**：四种技能（walk/stand/flamingo/push-resist）的 reward 信号互相冲突——walk 要前进、stand 要定住、flamingo 要抬脚、push 要抗扰。MLP 是纯前馈点映射 obs→action，无法在不同上下文下产生不同策略。

**解决方案**：用户要求一个**统一 policy**，不使用显式 skill ID 或 gait conditioning。Mamba 的**选择性状态空间机制（Selective SSM）**天然适合——它的隐藏状态更新依赖于输入内容，不同观测模式自动对应不同内部状态，实现隐式多模态切换。

## 2. 文献依据

| 论文 | 方法 | 关键结果 |
|------|------|---------|
| [HuMam](https://arxiv.org/abs/2509.18046) | Mamba backbone + PPO，人形机器人运动控制 | 单层 Mamba encoder，比 MLP baseline 更快收敛、更低能耗、跨 seed 更稳定 |
| [LocoMamba](https://www.sciencedirect.com/science/article/abs/pii/S1474034625011231) | Mamba 融合 proprioception + depth image | PPO 训练，回报提升 48.9%，碰撞减少 48.9%，vs Transformer 收敛更快 |
| [Decision Mamba](https://arxiv.org/abs/2403.19925) | SSM 替代 Transformer 做 RL 序列建模 | 与 Decision Transformer 性能持平，参数更少 |
| [Mamba Policy](https://arxiv.org/abs/2409.07163) | XMamba Block (Mamba+Attention 混合) | IROS 2025，参数减少 80%，长时域鲁棒性更强 |
| [LocoMamba-Quad](https://github.com/allen-quad-robot/locomamba) | 开源 Mamba 策略网络实现 | 代码结构和配置可参考 |

## 3. 核心架构（已实现）

### 3.1 Mamba Actor — 1.0M 参数

```
Observation (98,)                    # gyro+gravity+dof_pos+dof_vel+actions+cmd+phase
    ↓
Token Projection: Linear(98→1024)    # → reshape to (B, 4, 256) — 4 tokens × d_model=256
    ↓
+ Learnable Position Embedding (4, 256)
    ↓
Mamba Block × 2:                    # d_model=256, d_state=16, d_conv=4, expand=2
    ├── RMSNorm
    ├── in_proj: x + z (gate) branches
    ├── Conv1d(depthwise, kernel=4)
    ├── SiLU activation
    ├── SelectiveSSM (核心)
    │     Δ = softplus(Linear(x))       # 输入依赖的步长
    │     A = -exp(learned_log_A)        # 状态转移矩阵 (256, 16)
    │     B = Linear(x)                  # 输入投影
    │     C = Linear(x)                  # 输出投影
    │     h_t = dA·h_{t-1} + dB·x_t     # 选择性循环扫描
    │     y_t = C·h_t + D·x_t           # skip connection
    ├── Gate: x_out * SiLU(z)
    ├── out_proj → residual
    ↓
Mean Pool over 4 tokens → (B, 256)
    ↓
RMSNorm
    ↓
NormalTanhPolicy: mean_head(256→29) + learnable log_std(29)
```

### 3.2 与 MLP 的关键区别

| 特性 | MLP (FlashSACActor) | Mamba (MambaActor) |
|------|---------------------|---------------------|
| 映射方式 | obs→action 点映射 | obs→tokens→SSM扫描→action |
| 状态依赖 | 无 | 选择性隐藏状态，输入依赖 |
| 多模态 | 需显式 skill ID | 隐式通过 SSM 状态区分 |
| 参数量 (actor) | 285K (d=128) / 1.6M (d=256) | 1.0M (d=256, 2 blocks) |
| 计算复杂度 | O(d²) | O(L·d²)，L=4 tokens → 常数 |

### 3.3 Critic

保持 MLP（FlashSACDoubleCritic），2 blocks、hidden=512、ensemble=2。Critic 不需要多模态切换能力——它只需评估 state-action 对的 Q 值。

### 3.4 参数量（实际）

| 组件 | 配置 | 实际参数量 |
|------|------|-----------|
| Token Embedding | Linear(98→1024) | ~100K |
| Position Embedding | (4, 256) | ~1K |
| Mamba Blocks × 2 | d=256, state=16, expand=2 | ~800K |
| Action Head | NormalTanhPolicy(256→29) | ~15K |
| Exploration buffers | zeta_cdf 等 | ~1K |
| **Actor 总计** | | **~1.0M** |
| Critic × 2 | d=512, blocks=3 | ~12.8M |
| **总计** | | **~13.8M** |

## 4. 选择性状态空间如何解决多模态

### 4.1 Mamba 核心公式

```
Δt = softplus(W_Δ·x_t + b_Δ)           # 输入依赖的步长
A_t = -exp(A_log)                        # 学习的状态衰减率
B_t = W_B·x_t                            # 输入投影
C_t = W_C·x_t                            # 输出投影
h_t = exp(Δt·A_t)·h_{t-1} + Δt·B_t·x_t  # 选择性状态更新
y_t = C_t·h_t + D·x_t                    # 输出 + skip
```

其中 **Δt, B_t, C_t 都是输入 x_t 的函数**，使模型根据观测内容动态决定：
- 保留多少历史信息（Δt 控制）
- 哪些输入特征写入状态（B_t 选择）
- 哪些状态特征输出（C_t 选择）

### 4.2 在多技能场景中的效果

- **Walking 模式**：观测中有 gait phase[2] + velocity cmd[3]，Mamba 通过 Δ 保持周期性步态状态
- **Standing 模式**：command=0，双脚着地，Mamba 选择保留 COM 平衡信息
- **Flamingo 模式**：左脚 z>0.1m，contact 状态变化，Mamba 切换到单腿平衡状态
- **Push 扰动**：基座 gyro/gravity 突变 → B_t 快速将扰动信息写入隐藏状态 → 上肢补偿动作

**关键**：所有行为用**同一个网络**、**同一组参数**，无显式 skill ID。选择性机制自动在隐藏状态空间完成软切换。

## 5. 多部位持续推力增强

推力的 env 层实现在 `src/unilab/envs/locomotion/g1/multiskill.py` 的 `MultiSkillDRProvider` 中：

- **12 个身体部位**随机选择：pelvis, torso, 左右肩/肘/髋/膝/踝
- **持续力**：每次推力持续 50~100 步（1~2s），通过 `_active_pushes` dict 管理
- **可叠加**：不同身体部位可以同时有活跃推力（`body_ids` + `body_force` 同时下发）
- **推力大小**：渐进课程——Phase C 从 200N 开始，后续可升至 800N

## 6. 训练策略

### 6.1 三阶段渐进课程

| 阶段 | 技能 | rel_flamingo | push | iters | 状态 |
|------|------|-------------|------|-------|------|
| Phase A | Walk + Stand | 0.0 | 无 | 3000 | **训练中** |
| Phase B | + Flamingo (15%) | 0.15 | 无 | 2000 | 待 Phase A 完成 |
| Phase C | 全技能 + 轻推力 | 0.15 | 200N, 4 部位 | 5000 | 待 Phase B 完成 |

### 6.2 训练配置

```
算法：FlashSAC (off-policy, double buffer)
策略：MambaActor (纯 PyTorch, 1.0M params)
Critic：FlashSACDoubleCritic (MLP, 12.8M params)
环境：4096 envs, 50Hz, max 20s/episode
大步行走：vx ∈ [0.3, 1.2] m/s, gait_frequency=1.2, swing_height=0.12
全 DR：质量 ±2.5kg, 摩擦 0.4-1.5, PD 0.85-1.15, 重力方向
```

## 7. 实现记录

### 7.1 新建文件

| 文件 | 内容 |
|------|------|
| `src/unilab/algos/torch/flash_sac/mamba_actor.py` | `MambaActor`、`MambaBlock`、`SelectiveSSM`、`RMSNorm` |
| `conf/offpolicy/task/flashsac/g1_multiskill/mamba_phaseA.yaml` | Phase A Mamba 训练配置 |

### 7.2 修改文件

| 文件 | 变更 |
|------|------|
| `src/unilab/algos/torch/flash_sac/learner.py` | 添加 `use_mamba_actor`、`mamba_d_state`、`mamba_n_tokens` 参数，条件创建 MambaActor |
| `src/unilab/algos/torch/flash_sac/double_buffer.py` | 传递 Mamba 参数到 learner 和 collector actor_kwargs |
| `src/unilab/algos/torch/common/actor_factory.py` | `build_actor()` 添加 Mamba 分支 |
| `src/unilab/envs/locomotion/g1/multiskill.py` | 多技能 env（已在前期实现） |
| `src/unilab/assets/robots/g1/scene_flat.xml` | 添加 `flamingo` keyframe |

### 7.3 集成要点

1. **Weight Sync**：collector 进程通过 `actor_factory.build_actor()` 创建 actor 副本，必须接收 `use_mamba_actor` 等参数，否则 state_dict key 不匹配导致 `KeyError: 'pos_emb'`
2. **API 兼容**：MambaActor 需实现 FlashSACActor 的全部公开方法：`explore(obs, dones, deterministic)`、`normalize_parameters()`、`as_export_module()`
3. **mamba-ssm**：官方 CUDA 库因编译时间长暂未安装，当前使用纯 PyTorch 实现。训练速度约 0.5s/iter (4096 envs)，NVIDIA mamba-ssm 可加速 3-5x
4. **扫描操作**：纯 PyTorch 的 `_selective_scan` 使用 Python for 循环（O(L)），非并行化。安装 `mamba-ssm` 后将自动切换为 CUDA parallel scan

### 7.4 配置参数

```yaml
algo_params:
  use_mamba_actor: true     # 启用 Mamba 替代 MLP actor
  mamba_d_state: 16         # SSM 状态维度
  mamba_n_tokens: 4         # 观测 token 数量
  actor_hidden_dim: 256     # d_model
  actor_num_blocks: 2       # Mamba block 层数
```

## 8. 风险与对策

| 风险 | 状态 | 对策 |
|------|------|------|
| 纯 PyTorch Mamba 训练慢 | **已确认** | 安装 mamba-ssm 后自动切换 CUDA 加速；当前 0.5s/iter 可接受 |
| Collector weight sync key 不匹配 | **已解决** | actor_factory 同步添加 Mamba 分支 |
| API 兼容性（explore/normalize/export） | **已解决** | MambaActor 完整实现所有 API |
| 多技能 reward 冲突仍存在 | **验证中** | Phase A→B→C 渐进课程，Mamba 选择性机制 |
| 1.0M Mamba vs 1.6M MLP 对比不公 | 待验证 | 同 d_model=256 下 Mamba(1.0M) < MLP(1.6M)，Mamba 胜出的话更有说服力 |

## 9. 参考资料

| 来源 | 链接 |
|------|------|
| HuMam (Mamba+PPO 人形机器人) | [arxiv.org/abs/2509.18046](https://arxiv.org/abs/2509.18046) |
| LocoMamba (四足运动控制) | [github.com/allen-quad-robot/locomamba](https://github.com/allen-quad-robot/locomamba) |
| Decision Mamba (SSM for RL) | [arxiv.org/abs/2403.19925](https://arxiv.org/abs/2403.19925) |
| Mamba Policy (混合架构, IROS 2025) | [arxiv.org/abs/2409.07163](https://arxiv.org/abs/2409.07163) |
| Mamba 官方库 | [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba) |
| Gait-Conditioned RL | [arxiv.org/abs/2505.20619](https://arxiv.org/abs/2505.20619) |
| Mamba-1 论文 | [arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752) |
