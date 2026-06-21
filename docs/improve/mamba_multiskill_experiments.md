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

## 6. D v3：反复冻结——真因定位与修复（2026-06-20 更新）

> 历史结论曾记为"放弃"。2026-06-20 用 official mamba kernel 排除 recompile 死锁后，**真因暴露并被修复**：见 §6.4。原 §6.1-6.3 的历史记录保留作为排查脉络。

### 6.1 冻结模式

D v3 尝试了 **8 次**，每次都在 ~980 iter 冻结：

| 尝试 | 配置 | 冻结 iter | 现象 |
|------|------|----------|------|
| 1 | Mamba Actor + MLP Critic | 980 | GPU 0%, 进程存活 |
| 2 | Mamba Actor + Mamba Critic | 980 | GPU 0%, 进程存活 |
| 3-6 | 各种修复 | ~200-500 | 早期崩溃或冻结 |
| 7 | MLP Critic only | 980 | 事件停止更新 |
| 8 | MLP Critic only | 980 | 相同 |

### 6.2 可能原因（历史推测，部分已被 §6.4 证实/推翻）

1. **checkpoint 保存死锁**：save_interval=1000，冻结发生在 980——刚好在首次保存前。多进程权重同步 + 磁盘写入可能触发死锁
2. **fallen keyframe 重置异常**：episode 结束时 `_reset_done_envs()` 调用 `build_reset_plan`，fallen 姿态生成逻辑可能有边角 bug ← **§6.4 证实为此**
3. **num_envs 不匹配**：Phase B warm-start checkpoint 训练时用 4096 envs，但 collector 进程可能用不同 env 数初始化，导致数组维度冲突
4. **MambaCritic SSM 反向传播死锁**：Python for 循环的 `_selective_scan` 在大量迭代后积累计算图碎片

### 6.3 教训

- D v3 配置本身无致命问题（YAML 已验证，指标正常）
- 980 iter 冻结是系统性 bug，与 critic 类型无关
- Phase A + B 能跑完 4000 iter 证明基础架构可行，问题出在 D v3 新增的 fallen/transition/push 逻辑

### 6.4 真因定位与修复（2026-06-20）

#### 排除 recompile 死锁——暴露真 bug

历史长期怀疑 980 冻结是 `torch.compile + 纯 PyTorch Mamba SSM` 的 recompile 死锁。2026-06-20 实施**方案C**排除该因素：

| 方案C 措施 | 位置 | 作用 |
|---|---|---|
| `use_compile: false` | `mamba_phaseD_v3.yaml` L34 | learner 不编译 actor 前向，无 recompile |
| `use_official: true` | `mamba_actor.py` L114 | SSM 走 official CUDA kernel，不经 inductor |
| collector actor 上 cuda | `worker.py` L341/423/424/431/439 | official kernel 要 cuda 输入 |

mamba-ssm 2.2.6 装好后 `_HAS_OFFICIAL_MAMBA=True`。重跑后 **iter 940 处仍然崩**——但这次崩因不是死锁，是 env reset 抛出明确异常，见下。

#### 真因：`build_reset_plan` 的 `np.where` shape 不匹配

崩点堆栈：
```
src/unilab/envs/locomotion/g1/multiskill.py:217, in build_reset_plan
    qpos_all[is_fallen] = np.where(kneeling_idx[:, None], kneel_qpos, qpos_all[is_fallen])
ValueError: operands could not be broadcast together with shapes (778,1) (389,36) (778,36)
```

**根因**：fallen env 的"半跪/全倒"姿态分配逻辑里，`np.where` 三个操作数 shape 不一致：

| 操作数 | shape | 含义 |
|---|---|---|
| `kneeling_idx[:, None]` | (n_fallen, 1) = (778,1) | 哪些 fallen env 走半跪 |
| `kneel_qpos` | (**n_kneel**, 36) = (389,36) | 半跪姿态，只 tile 了 kneeling 数量 |
| `qpos_all[is_fallen]` | (n_fallen, 36) = (778,36) | 原姿态 |

`np.where(cond, a, b)` 要求 `a` 能 broadcast 到 `b` 的 shape。`kneel_qpos` 是 (389,36)，`b` 是 (778,36)——**389 ≠ 778，无法 broadcast**。

**为何偏偏在 ~940 iter 触发**（而非更早）：`n_fallen = int(np.sum(is_fallen))`，fallen env 数随训练推进（机器人开始倒）增多。当 fallen 数恰好使 `n_kneel (= n_fallen//2)` 与 `n_fallen` 不等时（即 n_fallen ≥ 2 起就一直不等），每次 reset 都会崩。但**只有 fallen env 数量显著增多后，该代码路径被频繁命中**，叠加 recompile 死锁的掩护，长期被误判为"980 冻结"。

> 即原 §6.2 第 2 条推测"fallen keyframe 重置异常"被证实，且这正是 940/980 反复"冻结"的真因之一（另一部分是 recompile 死锁，已由方案C 解决）。

#### 修复

把 `kneel_qpos` 也 tile 到 `(n_fallen, 36)`，使三个操作数同 shape，kneeling 行填 kneel 姿态、其余行保留 `qpos_all[is_fallen]`（stand）：

```python
# 修复前（崩）：
kneel_qpos = np.tile(self._kneeling_qpos, (np.sum(kneeling_idx), 1))   # (389,36)
qpos_all[is_fallen] = np.where(kneeling_idx[:, None], kneel_qpos, qpos_all[is_fallen])  # (778,1)(389,36)(778,36) → 崩

# 修复后（三操作数同 (n_fallen,36)）：
fallen_block = np.where(
    kneeling_idx[:, None],
    np.tile(self._kneeling_qpos, (n_fallen, 1)),   # (778,36)
    qpos_all[is_fallen],                            # (778,36)
)
qpos_all[is_fallen] = fallen_block
```

**验证**：用崩溃时的真实 shape（n_fallen=778, qpos_dim=36）复现——旧写法精确复现 `(778,1) (389,36) (778,36)` 崩溃；新写法成功，且语义校验通过（kneeling 行均值=0.5 即 kneel 姿态，非 kneeling 行均值=0.0 即 stand 姿态）。

#### 教训补充

- **"冻结"≠"死锁"**：iter 卡在某点 + GPU 掉零 + events 停写，可能是**崩溃被吞**（异常未冒泡到顶层，进程挂着不退）。必须看日志尾部有没有 Python traceback 才能区分。
- **被一个 bug 掩盖的另一个 bug**：recompile 死锁（已治）长期掩盖了 reset shape bug。排除前者后，后者才暴露。复杂系统里"卡在某点"可能多个根因叠加。
- **关键修复要用真实 shape 复现验证**：数值/shape bug 不能只靠肉眼看改对了，要用崩溃现场的真实数值构造最小复现，确认改前崩、改后通且语义对。

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

## 9. 下一步（2026-06-20 更新）

- ✅ D v3 reward hacking 已根除（见 §10），机器人能稳站 6 秒 + 起身见效
- ⏳ 阶段4（降 alive 白给 + 强 pose）→ 阶段5（resampling_time 训技能过渡）
- 待验证：行走能力（前进命令下跟踪）、技能间过渡
- 详见 `docs/improve/dv3_reward_redesign.md`

## 10. Reward Hacking 诊断与分阶段修复（2026-06-20）

### 10.1 现象：训练 reward 高但 play 站不住

D v3 用 §6.4 修复后的代码训到 iter 6000，训练曲线漂亮（reward/mean +255、critic_loss 1.24、terminated_rate 0），但 **sim2sim play 视频里机器人站不住**——初始 0.75m 站姿，0.5 秒坍塌到 0.22m，之后稳定趴着不动。训练 reward 高但行为失败，是典型 **reward hacking**。

诊断工具：`unilab_mamba/diag_play.py`（复用 play_offpolicy 的 env+actor 构建，每步记录 base_z/cmd/action/reward，不录视频）。实测数据（model_6000）：base_z 0.339→0.137m，全程 terminated=False、reward 正。

### 10.2 根因：三个约束被同时放开

对照 flamingo 成功配置（`conf/.../g1_flamingo_stand/`）和 UniLab 原版 G1 walk（`conf/.../g1_walk_flat/mujoco.yaml`）：

| 约束参数 | flamingo 成功 | UniLab 原版 walk | **D v3（hacking）** |
|---|---|---|---|
| `max_tilt_deg` | 60 | 65 | **180** |
| `min_base_height` | 0.35 | 0.3 | **0.0** |
| `penalty_base_height` | -200 | 无 | **无** |

D v3 为训起身把终止阈值 `max_tilt 60→180` + `min_base_height 0.35→0`，并删了 `penalty_base_height`。后果：`tilt>π` 永不成立 + `base_z<0` 永不成立 → **terminated 永远 False** → `alive=+10` 每步白给 → policy 学到"趴低不摔拿分"。

源码核实（`joystick.py:357-362`）：
```python
max_tilt_rad = deg2rad(max_tilt_deg)  # 180° = π
tilt = arccos(clip(gravity[:,2], -1, 1))  # 永远 ≤ π
terminated = (tilt > max_tilt_rad) | (base_z < min_base_height)  # 永远 False
```
`rewards.alive`（`rewards.py:182`）= 无条件 `np.ones`，不终止就给。

### 10.3 关键避坑（本次踩过/差点踩的）

1. **"冻结"≠"死锁"**：iter 卡在某点 + GPU 掉零 + events 停写，可能是**崩溃被吞**（异常未冒泡到顶层，进程挂着不退）。必须看日志尾部有没有 traceback 才能区分。§6.4 的 reset bug 就是被 recompile 死锁掩盖的崩溃。
2. **被一个 bug 掩盖的另一个 bug**：recompile 死锁（已用 official kernel 治）长期掩盖了 reset shape bug；reset bug 修后，reward hacking 才暴露。复杂系统"卡在某点"可能多个根因叠加。
3. **reward 参数不能拍脑袋，要对齐已验证的 baseline**：初版我设计 `min_base_height=0.55/max_tilt=45`（比 UniLab 原版 walk 的 0.3/65 还严），差点训不出。**校准到原版 walk 的 0.3/65 才成功**——已验证的 baseline 是最可靠的参数锚点。
4. **reward key 要查实际注册**：flamingo 用 `penalty_base_height`（flamingo_stand.py 专有注册），但 multiskill 没注册这个 key，加了会无效。multiskill 继承的是 `base_height`（joystick 注册，`rewards.base_height=(base_z-0.754)²`）。改 reward 前必须 grep 确认 key 在当前 env 的 `_reward_fns` 里注册了。
5. **reward 改动用真实数值复现验证**：和 shape bug 一样，不能肉眼看改对了，要用崩溃/异常现场的真实数值构造最小复现确认。

### 10.4 修复：分阶段循序渐进（不一股脑改）

详见 `docs/improve/dv3_reward_redesign.md` 第七节。核心：每阶段只改 1-2 项，靠 play 诊断（base_z 曲线）验证后再进下一阶段。实测进展：

| 阶段 | 改动 | sim2sim 结果（model_2000） | terminated_rate |
|---|---|---|---|
| 基线 | 旧 reward | 秒坍趴 0.14m | 0（假象） |
| 阶段1 | max_tilt 180→65 + min_z 0.0→0.3（对齐原版 walk） | 站满 6 秒，base_z ±6cm | 0.065 |
| 阶段2 | +base_height: -100（强度惩罚逼站直） | 站满 6 秒，base_z ±4.5cm，更贴目标 | 0.0 |
| 阶段3 | fallen 起始 z 0.35→0.55（练起身） | 站满 6 秒，base_z ±4.3cm + 起身见效 | 0.025 |

**阶段1 是命门**——终止漏洞不堵（max_tilt/min_base_height），后面全白费（趴着仍拿分）。先单独验证终止生效（terminated_rate 从 0 升到 >0），再加强度惩罚（阶段2）、再练起身（阶段3）。

### 10.5 教训补充（接 §8）

6. **训练 reward 高 ≠ 行为好**：reward hacking 下 reward 曲线漂亮但 play 崩。**必须 sim2sim 验证**，且不只看 reward/mean，要看 `terminated_rate`（=0 可能是"站太稳"也可能是"终止阈值失效"的假象）、`base_height` 惩罚等。
7. **reward 改动要循序渐进验证**：一次改多项，失败无法归因。分阶段每阶段验证一个假设。
8. **诊断脚本要复用已验证的 env 构建路径**：自己拼 `registry.make()` 会缺注册（`ensure_registries` 没调）、缺 `env_cfg_override`（缺 reward_config）。直接 import `scripts.train_offpolicy` 的函数复用 play_offpolicy 的 env+actor 构建，避免重复踩坑。
9. **续训 reward 改动有适应期**：改 reward 后 critic 要重新适应，terminated_rate 会暂时回到 1.0、reward 掉，是**预期波动不是失败**。reward/mean 和 critic_loss 朝对的方向走（升/降）即正常，需等 ~1500-2000 iter 拐点。


