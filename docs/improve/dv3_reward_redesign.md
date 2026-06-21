# D v3 多技能 Reward 诊断与新设计方案

> 编写：2026-06-20。基于 D v3（mamba_phaseD_v3）跑到 iter 6000 的训练数据 + play 诊断数据 + flamingo 成功经验对照。
>
> **状态**：D v3 训练已停（iter 6000，reward/mean 254.9 为 hacking 假象）。本文是设计方案，**尚未改任何配置**，待审阅。

---

## 一、问题陈述

D v3 训练曲线漂亮（reward/mean 从 -43 涨到 +255，critic_loss 4.3→1.24 收敛），但 **sim2sim play 视频里机器人站不住**——初始 0.75m 站姿，25 步（0.5s）跌到 0.48m，50 步跌到 0.25m，之后稳定趴在 ~0.22m 不动。训练 reward 高但行为失败，是典型的 **reward hacking**。

---

## 二、诊断数据（play 实测，model_5000.pt）

用 `unilab_mamba/diag_play.py`（复用 play_offpolicy 的 env+actor 构建，每步记录 base_z/cmd/action/reward）跑 300 步：

| 时刻 | base_z(m) | cmd | reward | terminated |
|---|---|---|---|---|
| t=0 | **0.751** | [0.57,-0.18,0.28] | 0.14 | False |
| t=25 | 0.482 | 同上 | -0.14 | False |
| t=50 | 0.247 | 同上 | 0.19 | False |
| t=75~300 | **0.21~0.24** | 同上 | 0.16~0.19 | **False** |

**关键观察**：
1. 初始能站（0.751m）→ policy 不是完全坏的
2. 0.5 秒内坍塌到 0.25m → 站立不稳定
3. 趴在 0.22m **不终止**（terminated=False）→ 终止阈值有漏洞
4. 趴着仍拿**正 reward（+0.16）** → reward 在奖励"趴着不摔"
5. command 恒定（诊断脚本未加 resampling，与训练分布一致）→ 排除"play 环境失配"

CSV 原始数据：`/tmp/dv3_play_diag.csv`。诊断脚本：`unilab_mamba/diag_play.py`。

---

## 三、Reward 各项实际计算逻辑（源码核实）

D v3 reward = Σ scaleᵢ × fnᵢ。各 fn 实现（`src/unilab/envs/locomotion/common/rewards.py` + `g1/joystick.py`）：

| reward key | 函数 | 计算 | D v3 scale |
|---|---|---|---|
| `alive` | `rewards.alive` | **无条件常数 1.0**（`np.ones`） | **+10** |
| `tracking_lin_vel` | `rewards.tracking_lin_vel` | `exp(-‖cmd_xy − vel_xy‖² / σ²)`，σ=0.25 | +2.0 |
| `tracking_ang_vel` | `rewards.tracking_ang_vel` | `exp(-(cmd_yaw − gyro_yaw)² / σ²)` | +1.5 |
| `pose` | `rewards.weighted_pose` | `Σ wᵢ·(dof_posᵢ − defaultᵢ)²`（**惩罚**偏离默认姿态） | -0.5 |
| `penalty_orientation` | `rewards.orientation` | `gₓ² + gᵧ²`（重力 xy 分量平方，偏离直立） | -10 |
| `penalty_action_rate` | `rewards.action_rate` | `‖aₜ − aₜ₋₁‖²` | -5.0 |
| `penalty_feet_ori` | 自定义 | 脚朝向惩罚 | -25 |
| `feet_phase` | `joystick._reward_feet_phase` | 步态相位跟踪 | +5.0 |
| `penalty_lifted_foot_contact` | multiskill | 摆动脚误触地 | -30 |
| `penalty_support_foot_contact` | multiskill | 支撑脚乱触地 | -15 |
| `com_over_support` | multiskill | 质心在支撑脚上方 | +2.0 |

**终止条件**（`joystick.py:357-362`）：
```python
max_tilt_rad = deg2rad(max_tilt_deg)   # D v3 = 180° = π
tilt = arccos(clip(gravity[:,2], -1, 1))
terminated = (tilt > max_tilt_rad) | (base_z < min_base_height)
```

---

## 四、根因：三个约束被同时放开 → hacking

对照 **flamingo 成功配置**（`conf/offpolicy/task/flashsac/g1_flamingo_stand/`，所有 phase 都站住了）：

| 约束参数 | **flamingo 成功** | **D v3** | 后果 |
|---|---|---|---|
| `max_tilt_deg` | **60** | **180** | D v3：tilt 永远 ≤π，`tilt>π` 永不成立 → 倾倒不终止 |
| `min_base_height` | **0.35** | **0.0** | D v3：base_z 永远 ≥0，`base_z<0` 永不成立 → 趴地不终止 |
| `penalty_base_height` | **-200** | **（无）** | D v3：删了强高度惩罚 → 没有力逼站直 |
| `alive` | 10→5（phase2 降） | 10 | D v3：白给更多 |

**hacking 机制**：D v3 为了让机器人"能摔倒、训练起身"（`rel_fallen_envs:0.2`），把终止阈值 `max_tilt 60→180` + `min_base_height 0.35→0`，并删了 `penalty_base_height -200`。结果：
- 机器人**趴到 0.14m 都不判摔**（tilt 不到 180°，base_z≥0）
- `alive=+10` 每步白给（不终止就给）
- `pose` 惩罚小（趴着时部分关节可能接近默认）
- → policy 学到"**趴低不摔 + 拿 alive**"比"站直走路"更省力
- 训练 reward 255 是这个捷径堆出来的假象

**flamingo 当初能站住**，正因 `max_tilt=60`（倾 60° 判摔）+ `min_base_height=0.35`（低于 35cm 判摔）+ `penalty_base_height=-200`（强逼站直）三重约束。

---

## 五、疑点清单（不局限 hacking，供全面复查）

1. **`max_tilt_deg=180`**：终止阈值失效。**必改**。
2. **`min_base_height=0.0`**：终止阈值失效。**必改**。
3. **缺 `penalty_base_height`**：没有强高度惩罚。**应补**。
4. **`alive=10` 无条件给**：白给太多，且与终止脱钩。**应改**（见方案）。
5. **`pose` 权重失衡**：手臂等关节权重 50，tracking 仅 2.0——但 pose 是惩罚项，影响次于 alive。
6. **`resampling_time=0.0`**：训练时命令不切换，**无技能过渡训练**——这是"看不到过渡"的另一根因（play 恒定命令，policy 没学过切换）。
7. **`rel_fallen_envs=0.2` + 放宽终止**：意图是训起身，但放宽终止让"趴着"也可行，与起身目标冲突。
8. **`temperature` 降到 0.001、entropy -10.7**：策略确定性过高，过渡处可能僵硬（次要）。

---

## 六、新 Reward 设计方案

### 6.1 设计原则

1. **终止阈值必须收紧**（恢复 flamingo 的 60°/0.35m），否则 hacking 无解。
2. **alive 不能无条件给**——要么与"站直"挂钩，要么大幅降低。
3. **保留起身训练能力**——fallen env 仍 reset，但"趴着"不能成为稳态（靠 `penalty_base_height` 逼起来，而非靠终止）。
4. **技能过渡要训**——`resampling_time>0`，让命令 mid-episode 切换。
5. **尽量续训**——不破坏 checkpoint 兼容（reward 改不影响 actor 结构，可从 model_6000 续）。

### 6.2 核心改动（YAML）

```yaml
reward:
  scales:
    tracking_lin_vel: 2.0        # 不变
    tracking_ang_vel: 1.5        # 不变
    penalty_ang_vel_xy: -1.0     # 不变
    penalty_orientation: -10.0   # 不变
    penalty_action_rate: -5.0    # 不变
    pose: -1.0                   # 0.5→1.0 加强姿态跟踪(flamingo phase2 经验)
    penalty_feet_ori: -25.0      # 不变
    feet_phase: 5.0              # 不变
    alive: 5.0                   # 10→5 降白给(flamingo phase2 经验)
    penalty_base_height: -100.0  # 【新增】强逼站直, -100(flamingo 介于 -200 和 -10 之间)
    penalty_lifted_foot_contact: -30.0  # 不变
    penalty_support_foot_contact: -15.0 # 不变
    com_over_support: 2.0        # 不变
  tracking_sigma: 0.25
  base_height_target: 0.754      # 不变
  min_base_height: 0.55          # 【0.0→0.55】低于 55cm 判摔(站姿 0.75,半跪 0.4)
  max_tilt_deg: 45.0             # 【180→45】倾 45° 判摔(flamingo 60 更严,起身难时放宽)
```

### 6.3 关键参数取舍说明

| 参数 | 旧→新 | 理由 |
|---|---|---|
| `max_tilt_deg` | 180→**45** | 比 flamingo 的 60 更严？否——D v3 有起身需求，45° 让"快倒时终止"但给起身留空间（flamingo 纯站立用 60）。**首版用 45，若起身训不出再放宽到 60**。 |
| `min_base_height` | 0.0→**0.55** | 站姿 0.75m，半跪 0.4m。0.55 让"半跪以下判摔"，但 kneeling(0.4) 仍可短暂经过。**注意**：fallen reset 初始 z=0.35 会立即判摔——需把 fallen 起始 z 提到 0.55 以上，或给 fallen env 一个"起身宽限期"（见 6.4）。 |
| `penalty_base_height` | 无→**-100** | 逼站直的核心力。flamingo 用 -200（纯站立），D v3 用 -100（兼顾起身，不至于把刚 reset 的 fallen env 一开始就罚死）。 |
| `alive` | 10→**5** | 降白给，但保留（完全去掉会失去"活下去"信号）。 |
| `pose` | -0.5→**-1.0** | flamingo phase2 经验，加强姿态跟踪。 |

### 6.4 起身训练的矛盾与解法

**矛盾**：收紧终止（max_tilt 45 / min_z 0.55）后，fallen env reset 到 z=0.35 会**立即判摔**，根本没机会起身。

**解法二选一**：

- **方案 A（推荐）：fallen env 起始 z 抬高**。改 `multiskill.py:228` `fallen_qpos[:,2]=0.35` → `0.55`（半跪高度），让 fallen env 从"半跪"而非"全倒"开始，既保留起身训练又不立即触发终止。
- **方案 B：起身宽限期**。reset 后 N 步内禁用 base_height 终止。代码改动大，不推荐首版。

### 6.5 技能过渡训练

**问题**：`resampling_time=0.0`，训练时命令恒定，policy 没学过切换 → play 看不到过渡。

**改法**：
```yaml
env:
  commands:
    resampling_time: 8.0   # 【0.0→8.0】每 8s 切命令(mid-episode 过渡)
```
但 `rel_standing_envs/flamingo_envs` 是 reset 时分配的初始技能，**运行中切换的是速度命令**，技能身份（walk/stand/flamingo）切换需 `_sample_commands` 支持——multiskill.py:615 的 resample 只切速度命令，技能身份不变。**要真正训技能过渡，需额外改 `_sample_commands` 让它也重抽技能身份**（中等改动，建议作为 v2，首版只切速度命令验证过渡能力）。

### 6.6 续训考量

**好消息**：reward 改动**不影响 actor 网络结构**（obs_dim/action_dim/层结构都不变），checkpoint model_6000.pt 的 actor 权重**可直接加载续训**。

**但**：reward 变了 → reward normalizer 统计失效（`reward_scale_std` 需重新估计）→ critic 估值需重新校准。两种选择：

| 方式 | 操作 | 风险 |
|---|---|---|
| **续训 6k→10k** | load model_6000, 新 reward, critic 重新适应 ~1k iter | critic 短期估值偏，但省 6k iter 算力 |
| **从零 0→10k** | 新 reward 从头 | 干净，但 6k iter 白跑（~8 小时） |

**建议**：先续训（6k→10k），观察 critic_loss 是否在 ~1k iter 内稳定 + reward 是否重新爬升 + play 是否站住。若续训 1-2k iter 后 reward 反而崩（critic 估值失配严重），再从零。**注意 FlashSAC 续训的已知风险**（skills §9.2：replay buffer 不保存，续训 buffer 空导致分布偏移）——但这次 buffer 是新 reward 下新收集的，不存旧 buffer 偏移问题。

---

## 七、循序渐进的训练流程（分阶段，每阶段只动少量变量）

> **方法论**：不一股脑改 6 项再训。reward 调参是高维空间搜索，一次改多项，失败无法归因。分阶段：每阶段只动 1-2 个变量，靠 play 诊断（`diag_play.py` 的 base_z 曲线）验证后再进下一阶段。失败可回退到上一阶段配置，不丢已验证的进展。
>
> 每阶段都**从 model_6000.pt 续训**（reward 改不影响 actor 结构，权重兼容），训 ~1-2k iter 后跑 play 诊断判定。

### 阶段 0：基线复现（不动任何配置，先确认诊断可复现）

- **目的**：用现 model_6000.pt 跑 play 诊断，确认 hacking 可稳定复现（base_z 坍到 0.22m），作为后续对比基线。
- **改动**：无。
- **验证**：`diag_play.py` 输出 base_z 曲线，应与本文第二节一致（0.75→0.22）。
- **耗时**：~5 分钟（只跑 play，不训）。

### 阶段 1：堵住终止漏洞（最关键，单独验证）

- **只改 2 项**：`max_tilt_deg: 180→45` + `min_base_height: 0.0→0.55`
- **不动**：alive、pose、penalty_base_height、resampling_time、fallen 起始 z
- **目的**：先让"趴着"不再是稳态——终止阈值生效后，趴地的 env 会摔并 reset，policy 被迫重新探索站立。
- **风险**：fallen env reset 到 z=0.35 会**立即判摔**（min_base_height 0.55 > 0.35）→ 这部分 env 反复摔、学不到东西。**这是本阶段已知副作用，阶段 3 再修**。
- **续训**：1-2k iter
- **验证（play 诊断）**：
  - ✅ 成功：base_z 不再稳定在 0.22m，要么站住（0.6+）要么反复摔（terminated_rate>0）
  - ❌ 失败：base_z 仍趴 0.22m → 终止阈值没真生效（查代码路径），回退检查
- **判定**：terminated_rate > 0 即说明终止漏洞已堵（这是新旧 reward 的核心区别标志）。

### 阶段 2：加逼站直的力（在阶段 1 基础上 +1 项）

- **前置**：阶段 1 验证终止已生效。
- **只加 1 项**：`penalty_base_height: -100`（新增）
- **目的**：光靠终止不够（终止是离散的，policy 可能学会"快摔前抢救"），加连续高度惩罚逼全程站直。
- **续训**：从阶段 1 的 checkpoint 续，1-2k iter
- **验证**：
  - ✅ 成功：base_z 稳定 0.6-0.75m，reward/penalty_base_height 显著负但不主导
  - ⚠️ 部分：base_z 跌到 0.4-0.5m 半跪稳态 → penalty 不够强，进阶段 2b 加大
  - 阶段 2b（备选）：penalty_base_height -100→-200（flamingo 值）

### 阶段 3：修起身训练（fallen 起始 z，在阶段 2 基础上）

- **前置**：阶段 2 验证站直能力恢复。
- **问题**：阶段 1 已知副作用——fallen env reset z=0.35 < min_base_height 0.55，立即摔。
- **改动**：`multiskill.py:228` `fallen_qpos[:,2]=0.35→0.55`（方案 A，半跪起始）
- **目的**：让 fallen env 从"半跪"(0.55m) 而非"全倒"(0.35m) 开始，既保留起身训练又不立即触发终止。
- **续训**：1-2k iter
- **验证**：fallen env 的 episode 长度变长（不再 reset 即摔），且能从 0.55m 起身到 0.75m。

### 阶段 4：降白给 + 强姿态（微调，在阶段 3 基础上）

- **前置**：阶段 1-3 让站立 + 起身基本 work。
- **改 2 项**：`alive: 10→5` + `pose: -0.5→-1.0`
- **目的**：减少 alive 白给（逼靠 tracking 拿分），加强姿态跟踪。这两项是"锦上添花"，在站立已稳后调，避免过早收紧导致训练不稳。
- **续训**：1-2k iter
- **验证**：reward/mean 不应因 alive 降而崩（说明 tracking 真在起作用），play 站姿更直。

### 阶段 5：技能过渡（最后，在站立稳固后）

- **前置**：阶段 1-4 单技能站立 + 起身都稳。
- **改 1 项**：`resampling_time: 0.0→8.0`（mid-episode 切速度命令）
- **目的**：训过渡能力。**放在最后**——过渡要在单技能稳了之后训，否则站都站不住谈不上过渡。
- **v2（可选）**：改 `_sample_commands` 连技能身份也切（中等代码改动，首版不做）。
- **续训**：1-2k iter
- **验证**：play 时命令切换处不摔、平滑过渡。

### 阶段总览

| 阶段 | 改动 | 续训起点 | 核心验证 | 失败回退 |
|---|---|---|---|---|
| 0 | 无（基线） | model_6000 | hacking 可复现 | — |
| 1 | max_tilt 45 + min_z 0.55 | model_6000 | terminated_rate>0 | 回阶段 0 查代码 |
| 2 | +penalty_base_height -100 | 阶段1 ckpt | base_z 稳 0.6+ | 2b 加大到 -200 |
| 3 | fallen 起始 z 0.55 | 阶段2 ckpt | fallen env 能起身 | 回阶段 2 |
| 4 | alive 5 + pose -1.0 | 阶段3 ckpt | reward 不崩 | 回阶段 3 |
| 5 | resampling_time 8.0 | 阶段4 ckpt | 过渡不摔 | 回阶段 4 |

**总耗时估算**：每阶段 1-2k iter × ~5s/iter ≈ 1.5-3 小时/阶段，5 阶段约 8-15 小时。可中途暂停（checkpoint 续训）。

### 为什么不一股脑改

- reward 是高维搜索，一次改 6 项，若 play 仍崩，无法定位是"终止没堵住"还是"penalty 太强把训练搞崩"还是"alive 降太多失去信号"。
- 分阶段后，每阶段只验证一个假设，失败可精确定位回退。
- **阶段 1 是命门**——终止漏洞不堵，后面所有改动都白费（趴着仍拿分）。所以先单独验证它。

---

## 八、训练启动命令（按阶段）

### 8.1 续训流程（阶段 1 示例，后续阶段类推只改对应项）

```bash
tmux new -s train
cd /root/UniLab_WS/UniLab
export PATH=/root/miniconda3/envs/unilab/bin:$PATH
uv run scripts/train_offpolicy.py \
  task=flashsac/g1_multiskill/mamba_phaseD_v3 \
  algo=flashsac \
  algo.load_run=2026-06-20_13-02-39_mujoco \
  2>&1 | tee /tmp/dv3_v2_train.log
```
（reward 改动直接写进 mamba_phaseD_v3.yaml，load_run 指定旧 run 目录续训）

### 8.2 验收标准（每阶段续训后 play 诊断）

续训 ~1-2k iter 后跑 `unilab_mamba/diag_play.py`，看 base_z 曲线：
- ✅ **成功**：base_z 稳定在 0.6~0.75m，不坍塌
- ⚠️ **部分**：base_z 跌到 0.4-0.5m 但不继续倒（半跪稳态）→ 放宽 min_base_height
- ❌ **失败**：base_z 仍坍塌到 0.2m → max_tilt/min_base_height 还需收紧，或从零训

### 8.3 关键监控指标（盯坎时新增）

除原有 iter/reward/critic_loss，新增盯：
- `reward/penalty_base_height`：应显著负（说明在逼站直），但不应主导总 reward
- `episode/terminated_rate`：应 >0（说明终止阈值生效，有摔倒发生）—— **这是新旧 reward 的核心区别标志**

---

## 九、待审阅决策点

请你拍板（结合分阶段流程，决策更聚焦）：

1. **是否采用分阶段方案**：分 5 阶段循序渐进，还是仍想一次性改？【建议分阶段】
2. **阶段 1 的 max_tilt**：45（严）还是 60（flamingo 值）？【建议 45 首版，失败再放】
3. **阶段 1 的 min_base_height**：0.55 还是 0.45？【建议 0.55】
4. **阶段 2 的 penalty_base_height**：-100 起步，失败加到 -200？【建议 -100 起步】
5. **阶段 3 的 fallen 起始 z**：改 0.35→0.55（方案 A）？【建议改】
6. **阶段 5 的技能过渡**：首版只切速度命令，v2 再切技能身份？【建议首版只切速度】

> 分阶段后，决策不再是"一次性赌 6 项"，而是"每阶段一个小决策"，风险可控。

---

## 十一、阶段性实测结果（2026-06-20，持续更新）

> 分阶段方案的实测验证。每阶段 sim2sim 用 `unilab_mamba/diag_play.py`（300 步 = 6 秒，单环境）。

### 11.1 阶段1：堵终止漏洞 ✅

- **改动**：`max_tilt 180→65` + `min_base_height 0.0→0.3`（校准后对齐 UniLab 原版 walk，**非初版的 45/0.55**——初版过严差点训不出，见避坑）
- **续训**：从 model_6000（旧 reward）续
- **结果**：iter 2000 sim2sim 站满 6 秒（base_z 0.691-0.810，±6cm），terminated_rate 1.0→0.065
- **关键**：terminated_rate 从 0（假象）升到 >0 再降到 0.065，证明终止漏洞已堵 + policy 学会站

### 11.2 阶段2：加逼站直 ✅

- **改动**：+`base_height: -100`（`rewards.base_height=(base_z-0.754)²`，注意用 `base_height` key 不是 `penalty_base_height`，后者 multiskill 没注册）
- **续训**：从阶段1 model_2000 续
- **结果**：iter 2000 sim2sim 站满 6 秒 + **更稳**（base_z 0.683-0.772，±4.5cm，比阶段1 ±6cm 更稳），terminated_rate 0.0
- **关键**：`reward/base_height` 惩罚从 -0.32 降到 -0.013（base_z 贴目标 1.1mm），强度惩罚在逼站直

### 11.3 阶段3：练起身 ✅

- **改动**：fallen 起始 z 0.35→0.55（`multiskill.py:235`，半跪高度，高于 min_base_height 0.3 不立即摔）
- **续训**：从阶段2 model_2000 续
- **结果**：iter 2000 sim2sim 站满 6 秒（base_z 0.691-0.777，±4.3cm），terminated_rate 0.025，起身见效（base_height 惩罚 -0.014，fallen env 起身后站得直）

### 11.4 三阶段对比

| 阶段 | 站满6秒 | base_z 波动 | terminated_rate | 关键改动 |
|---|---|---|---|---|
| 基线 | ❌ 秒坍 0.14m | — | 0（假象） | — |
| 阶段1 | ✅ | ±6cm | 0.065 | 堵终止 |
| 阶段2 | ✅ 更稳 | ±4.5cm | 0.0 | +强度惩罚 |
| 阶段3 | ✅ 更稳 | ±4.3cm | 0.025 | +起身训练 |

### 11.5 实测避坑（补充 §五疑点之外的）

1. **初版参数过严**：阶段1 初版设 max_tilt 45/min_z 0.55（比原版 walk 的 65/0.3 还严），训不出。**校准到原版 walk 值才成功**——已验证 baseline 是最可靠锚点。
2. **reward key 注册差异**：`penalty_base_height` 只在 flamingo_stand.py 注册，multiskill 用 `base_height`。改前必 grep 确认。
3. **续训适应期**：每阶段改 reward 后 critic 重新适应，terminated_rate 暂回 1.0、reward 掉是预期，需 ~1500-2000 iter 拐点。
4. **诊断脚本复用已验证路径**：自己拼 `registry.make()` 缺 `ensure_registries`/`env_cfg_override` 会报 "not registered"。直接 import `scripts.train_offpolicy` 复用 play_offpolicy 的 env+actor 构建。
5. **sim2sim 命令随机性**：单环境 play reset 命令随机，一次诊断可能只测到站立（cmd=0）没测到行走。要多跑几次或强制命令才能覆盖各技能。

### 11.6 待验证（阶段4-5 后）

- 行走能力（前进命令下跟踪）——尚未专门验证
- 技能间过渡（resampling_time>0，阶段5 才训）
- 多次 reset 撞到 fallen 初始，直接拍到"从 0.55 起身到 0.75"完整过程

---

## 十二、全程总结（2026-06-21，全程完成）

> 从"reward hacking 导致 play 站不住"到"能站/能走/能起身/能过渡"，分 5 阶段循序渐进，每阶段 sim2sim 验证。

### 12.1 各阶段成果

| 阶段 | 改动 | sim2sim 结果 | terminated_rate | 判定 |
|---|---|---|---|---|
| 基线 | 旧 reward（max_tilt180/min_z0） | 秒坍趴 0.14m | 0（假象） | ❌ hacking |
| 阶段1 | max_tilt 65 + min_z 0.3（对齐原版 walk） | 站满 6 秒，base_z ±6cm | 0.065 | ✅ 站住 |
| 阶段2 | +base_height -100（强度惩罚） | 站满 6 秒，base_z ±4.5cm | 0.0 | ✅ 更稳 |
| 阶段3 | fallen 起始 z 0.35→0.55（练起身） | 站满 6 秒 + 起身见效，reward+138 | 0.025 | ✅ 起身 |
| 阶段4a | pose -0.5→-1.0 | 撑 57 步（退化） | 1.0 | ❌ 退化 |
| 阶段4b | pose -0.5→-0.7 | 撑 66 步（退化） | 1.0 | ❌ 退化 |
| 阶段4c | pose -0.5 + alive 10→5 | 撑 59 步（退化） | 1.0 | ❌ 退化 |
| **阶段5** | resampling_time 0→8（训过渡） | **站满 6 秒 + 行走 + 过渡** | **0.017** | ✅✅ 成功 |

### 12.2 最终能力验证（阶段5 model_5000，iter 5170，关机前最终）

sim2sim 两次（各 300 步 = 6 秒，model_5000.pt）：
- 第1次：前进命令(0.42m/s)下站满 6 秒，base_z 0.721-0.773（±2.6cm），0 次快倒 ✅ **行走验证**
- 第2次：站立命令下站满 6 秒，base_z 0.704-0.747（±2.2cm），0 次快倒 ✅ **站立验证**
- 起身：从 fallen(0.544m) 成功起身到 0.754m ✅（阶段5 iter1680 时验证）
- 过渡：terminated_rate 0.04（iter 5170），resampling 切命令几乎不摔 ✅

**四大能力全部达成且优化**：站立 / 行走 / 起身 / 技能过渡。reward/mean +267（真实，非 hacking）。

### 12.3 最终 checkpoint

- **阶段5 model_5000**（最终成果，最佳）：`logs/flash_sac/G1MultiSkill/2026-06-20_22-42-08_mujoco/model_5000.pt`
  - iter 5170，reward +267.2，terminated_rate 0.04，能站+行走+起身+过渡
- 阶段5 其他 checkpoint：model_1000/2000/3000/4000 同目录
- **阶段3 model_2000**（备选，已验证）：`logs/flash_sac/G1MultiSkill/2026-06-20_19-08-07_mujoco/model_2000.pt`

### 12.4 关键教训（全程沉淀）

1. **reward hacking 的识别**：训练 reward 高（+255）但 play 崩，是终止阈值失效（max_tilt180/min_z0）导致 alive 无限白给。**terminated_rate=0 可能是假象**——要看终止阈值是否合理。
2. **参数对齐已验证 baseline**：阶段1 初版拍脑袋设 0.55/45 过严差点训废，校准到 UniLab 原版 walk 的 0.3/65 才成功。**已验证 baseline 是最可靠锚点**。
3. **reward key 要查注册**：`penalty_base_height` 只 flamingo 注册，multiskill 用 `base_height`。改前必 grep。
4. **分阶段验证 + 退化可回退**：一次只改 1-2 项，靠 sim2sim 撑住步数验证，退化即回退。阶段4 三轮证伪（pose 加强 / alive 降都退化）正是这个机制生效——没让退化版本继续。
5. **alive=10 对维持站立必要**：阶段4c 证伪"降 alive 白给"假设——alive 降一半后 policy 不积极维持站立（撑 59 步 vs 150 步）。alive 不是白给，是关键 reward。
6. **pose 处于临界点**：-0.5 工作，-0.7 就退化。D v3 的 pose 已在平衡临界点，加强就破坏站立。
7. **续训 reward 改动有适应期**：每阶段改 reward 后 terminated_rate 暂回 1.0、reward 掉，是预期波动，需 ~1500-2000 iter 拐点。reward 升 + critic 降 + base_height 收敛 = 方向对。
8. **诊断脚本复用已验证路径**：自己拼 `registry.make()` 缺 `ensure_registries`/`env_cfg_override` 会报错，直接 import `scripts.train_offpolicy` 复用 play_offpolicy 的 env+actor 构建。
9. **sim2sim 单次有随机性**：reset 命令随机，一次诊断可能只测到站立没测到行走。要多跑几次或看多次结果综合判读。

### 12.5 待改进（阶段5 之后）

1. **过渡稳定性**：阶段5 iter 2290 terminated_rate 0.017 已很好，但 resampling 切命令处偶有失稳（iter 1680 时 2/3 摔，iter 2290 改善到几乎不摔）。继续训或调 resampling_time 可更稳。
2. **行走质量**：当前"行走"是"前进命令下站住不摔"，未专门验证步态/速度跟踪精度。可加 `tracking_lin_vel` 的 sim2sim 量化。
3. **技能身份切换**：当前 resampling 只切速度命令，技能身份（walk/stand/flamingo）切换需改 `_sample_commands`（v2 工作）。
4. **sim2sim→真机**：未做。当前全在 MuJoCo 仿真。

### 12.6 全程时间线

- 2026-06-20 白天：诊断 hacking（play 站不住）→ 查根因（终止阈值失效）→ 设计分阶段方案
- 下午~晚上：阶段1（站住）→ 阶段2（更稳）→ 阶段3（起身，reward+138）
- 晚~深夜：阶段4a/b/c（三轮证伪）→ 阶段5（过渡，01:30 收尾）
- 关机前完成全程总结

> 全程由 reward 校准（对齐原版）+ 分阶段验证（每阶段 sim2sim）+ 退化回退机制（阶段4 三轮）三机制保证质量。最终策略能站、能走、能起身、能过渡，reward hacking 根除。

---

## 十三、附录：相关文件

- 诊断脚本：`unilab_mamba/diag_play.py`（play 每步运动学数据）
- 诊断数据：`/tmp/dv3_play_diag.csv`
- reward 函数：`src/unilab/envs/locomotion/common/rewards.py`、`g1/joystick.py`、`g1/multiskill.py`
- D v3 配置：`conf/offpolicy/task/flashsac/g1_multiskill/mamba_phaseD_v3.yaml`
- flamingo 成功配置：`conf/offpolicy/task/flashsac/g1_flamingo_stand/mujoco*.yaml`
- 终止逻辑：`src/unilab/envs/locomotion/g1/joystick.py:357-362`
- reset bug 修复（前序）：`src/unilab/envs/locomotion/g1/multiskill.py:216-217`（见 mamba_multiskill_experiments.md §6.4）

> 本文是**设计草案**，未改动任何配置。审阅通过后，按第六节改 YAML +（可选）改 multiskill.py fallen 起始 z，再按第七节启动续训。
