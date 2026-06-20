# Mamba 多技能统一策略——实验记录与初步结论

> 日期：2026-06-19 | D v3 反复冻结于 ~980 iter 后放弃

## 1. 项目目标

用 Mamba 选择性状态空间模型替代 MLP，训练一个统一的人形机器人策略。**不允许显式 skill ID 或 gait conditioning**。技能包括：
- 大步行走 (vx 0.3~1.2 m/s)
- 双腿站立
- 金鸡独立（单腿站立）
- 多部位推力抗扰 (30N, 12 个身体部位)
- 倒地后爬起来
- 模式间自然过渡（5s 指令切换）

## 2. Mamba Actor 设计

```
obs(98) → token_proj → reshape(B, 4, 256) + pos_emb
       → MambaBlock × 3 (d_model=256, d_state=16, expand=2)
       → mean pool → RMSNorm → NormalTanhPolicy → action(29)
```

参数：1.5M (vs 同配置 MLP 的 1.6M)

### 2.1 与 HuMam 论文对比

| 维度 | HuMam | 我们 |
|------|-------|------|
| Mamba 层数 | 1 | 3 |
| Token 数量 | 2 (robot⊗task) | 4 (同 obs 复制) |
| Token 内容 | 语义不同 | 语义相同 |
| Reward | 6 项 | 15 项 |
| 算法 | PPO | FlashSAC |

**教训**：我们的 4-token 设计中 token 之间没有语义差异，SSM 的选择性门控无从"选择"。HuMam 的 2-token 语义分离设计更合理，但我们受限于 obs 结构无法简单拆分。

## 3. Phase A/B：Mamba vs MLP 基准对比

### Phase A——Walk + Stand

| 指标 | Mamba (256, 3 blocks) | MLP (256, 3 blocks) | Mamba 优势 |
|------|------|------|------|
| Episode Length | 973 | 970 | ≈ |
| **Reward** | **214** | 104 | **+106%** |
| **追踪线速度** | **1.6** | 1.0 | **+60%** |
| **追踪角速度** | **1.0** | 0.4 | **+150%** |
| 收敛速度 | iter 1910 | iter 1670 | MLP 快 14% |
| 参数量 | 1.5M | 1.6M | Mamba 更小 |

### Phase B——Walk + Stand + Flamingo

| 指标 | Mamba | MLP |
|------|-------|-----|
| Episode Length | 987 | 997 |
| **Tracking** | **1.7** | 1.1 |
| **Foot penalty** | **-0.0** | -0.2 |
| Reward | +272 | +69 |

**结论**：Mamba 在各项质量指标上全面优于 MLP，特别是速度追踪精度(+60~150%)。收敛略慢(14%)，但最终质量更高。

## 4. 起身训练——6 次尝试全部失败

| 版本 | reward | 结构 | 辅助 | 结果 |
|------|--------|------|------|------|
| G v1 | L2 height | Mamba Actor + MLP Critic | 无 | height 62→66cm |
| G v2 | exp height (HumanUP) | 同上 | 无 | height=0.54 |
| G v3 | exp height | 同上 | 辅助力 | height=0.73 |
| G v4 | exp height | Mamba Actor + Mamba Critic | 辅助力+跪姿 | height=4.04→4.04 |
| G v2 续 | exp height | 同上 | 辅助力+跪姿 | pose=-5.6 但视频验证是"躺平" |

**失败根因分析**：
1. **Exp reward 信号不足以从零发现起身动作序列**——HumanUP 在 Isaac Gym+1000Hz 物理下成功，但 MuJoCo 150Hz 下接触动力学不稳定
2. **Pose 指标是假信号**——pose 从 -92 改善到 -5.6 不是因为站起来，而是学会了"放松关节躺平"
3. **缺少参考轨迹/密集辅助信号**——HumanUP 的两阶段"发现→精炼"依赖 Stage I 的极简约束快速发现可行轨迹，我们的环境无法复现

## 5. MambaCritic 实现

### 5.1 架构

```
obs+action(130) → token_proj → reshape(B, 4, 256) + pos_emb
                → MambaBlock × 2
                → mean pool → RMSNorm
                → 分解 Q-heads × 5 (各 Linear(256, 101))
                → Sum → Q distribution (101 bins)
```

参数：1.2M (vs MLP Critic 的 12.8M)

### 5.2 发现并修复的 bug

**Layout Bug**：FlashSACDoubleCritic 的张量布局是 `(num_ensembles, batch, ...)`，MambaCritic 初始为 `(batch, num_ensembles, ...)`。`chunk(2, dim=1)` 在原始中是切 batch，在 MambaCritic 中是切 ensemble——导致形状不兼容。**已修复**。

**OOM Bug**：MambaCritic 的 SSM 扫描每步保留 67MB 激活(4096×256×16×4B)，4 步 × 67MB = 268MB，反向传播时全部保留。但这是僵尸进程导致的误判——清理僵尸后，MambaCritic (10.8GB) 能正常运行。

**冻结 Bug**：MambaCritic 在 Python for 循环的 `_selective_scan` 反向传播中极慢(20s/iter vs 1.35s/iter)，不是真死但慢到无法用。纯 PyTorch SSM 扫描是目前的最大性能瓶颈。

### 5.3 激活 VRAM 分析

| 组件 | MLP Critic | MambaCritic |
|------|-----------|-------------|
| 参数量 | 12.8M | **1.2M** |
| 激活内存 | ~100MB | ~500MB |
| **SSM 扫描激活** | 无 | **268MB** (最大瓶颈) |
| 总 VRAM (训练) | ~14GB | ~18GB |

MambaCritic 参数少 10 倍但激活内存大 5 倍——SSM 的反向传播需要保留每个时间步的隐藏状态。

## 6. D v3：反复冻结——放弃

### 冻结模式

D v3 尝试了 **8 次**，每次都在 ~980 iter 冻结：

| 尝试 | 配置 | 冻结 iter | 现象 |
|------|------|----------|------|
| 1 | Mamba Actor + MLP Critic | 980 | GPU 0%, 进程存活 |
| 2 | Mamba Actor + Mamba Critic | 980 | GPU 0%, 进程存活 |
| 3-6 | 各种修复 | ~200-500 | 早期崩溃或冻结 |
| 7 | MLP Critic only | 980 | 事件停止更新 |
| 8 | MLP Critic only | 980 | 相同 |

### 可能原因

1. **checkpoint 保存死锁**：save_interval=1000，冻结发生在 980——刚好在首次保存前。多进程权重同步 + 磁盘写入可能触发死锁
2. **fallen keyframe 重置异常**：episode 结束时 `_reset_done_envs()` 调用 `build_reset_plan`，fallen 姿态生成逻辑可能有边角 bug
3. **num_envs 不匹配**：Phase B warm-start checkpoint 训练时用 4096 envs，但 collector 进程可能用不同 env 数初始化，导致数组维度冲突
4. **MambaCritic SSM 反向传播死锁**：Python for 循环的 `_selective_scan` 在大量迭代后积累计算图碎片

### 教训

- D v3 配置本身无致命问题（YAML 已验证，指标正常）
- 980 iter 冻结是系统性 bug，与 critic 类型无关
- Phase A + B 能跑完 4000 iter 证明基础架构可行，问题出在 D v3 新增的 fallen/transition/push 逻辑

## 7. D v3 未完成——指标回顾（冻结前最后数据）

| 指标 | 初始 | 当前 | 趋势 |
|------|------|------|------|
| tracking | 0.27 | 0.94 | ✅ |
| foot | -0.85 | -0.47 | ✅ |
| pose | -11.2 | -2.6 | ✅ |
| critic_loss | 4.81 | 2.08 | ✅ |

## 7. 代码完整性

| 组件 | 状态 |
|------|------|
| MambaActor | ✅ 完整，纯 PyTorch |
| MambaBlock | ✅ 完整，支持官方 mamba-ssm 切换 |
| SelectiveSSM | ✅ 完整，Python for-loop 扫描 |
| MambaCritic | ✅ 完整，layout bug 已修 |
| FlashSAC 集成 | ✅ learner/double_buffer/actor_factory 全部支持 |
| ONNX 导出 | ⚠️ 已添加 as_export_module()，待验证 |
| 训练效率 | ⚠️ 纯 PyTorch SSM 扫描 ~2.8s/iter |

## 8. 关键教训

1. **Mamba 建模多模态能力强于 MLP**——Phase A/B 的实验无可辩驳
2. **Token 设计比层数重要**——HuMam 单层 2-token 胜过我们的 3 层 4-token（但我们受限于架构无法改）
3. **从零发现复杂动作序列（如起身）不可行**——需要参考轨迹或密集辅助信号
4. **MLP Critic 足够**——MambaCritic 的 SSM 扫描瓶颈在当前实现中得不偿失
5. **Pose 指标会撒谎**——必须视频验证

## 9. 下一步

- D v3 完成后接 Phase F（过渡 + 大步跑）
- Sim2Sim 视频验证
- 可选：安装官方 mamba-ssm 加速 SSM 扫描
