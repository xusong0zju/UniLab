# G1 多技能统一策略 — 本轮:Reward Routing 修复 + Phase H 重训

## Context

model_5000 在行走命令下学成了右脚单脚跳(诊断:左脚触地率 0.02、双脚同时触地 0.00、交替频率 0.003),起身(fallen)44% 摔倒。根因是 reward 泄漏 + flag 机制 bug + gait_phase 不前进三重问题。

目标技能(按速度命令路由):vx=0 双脚站 / 低速小步走 / 中速大步走 / 高速跑步(有腾空期)/ ω 转弯 / 金鸡独立(仅 flamingo 标记 env)/ 起身→双脚站(fallen 标记)。去除单脚跳着走。模式间过渡:站↔走↔跑合理(速度命令 8s 切),身份(flamingo/fallen)整 episode 锁定避免不合理过渡。

**本轮范围:只做 Part A(reward routing 修复 + Phase H 重训),不上世界模型。** TD-MPC2+Mamba 作为后续项目(调研已完成,见文末 Part B 草案),待 Part A 验证 env 正确后再启动,避免 reward bug 与 world model bug 混淆归因。

## 硬约束:不删除/不破坏原成果

现有成果一律保留,只读保护,绝不删除或覆盖:

- **所有 checkpoint 目录**(`logs/flash_sac/G1MultiSkill/2026-06-*/`):model_5000 等是数据文件,任何改动都不碰。
- **FlashSAC 代码**(`src/unilab/algos/torch/flash_sac/`):不覆盖 learner/actor/runner。
- **multiskill.py 改动策略**:reward routing 修复必须改此文件,但向后兼容——新 reward 行为通过新 yaml(Phase H)启用,改完用 model_5000 play 验证旧 checkpoint 不崩。
- **新代码用新文件**:Phase H yaml、flight reward、flag 修复都新增,不删旧 phase yaml、不删 MambaCritic/MambaActor。

---

# Part A — Reward Routing + Flag 修复(确定部分,不依赖世界模型)

> 这部分修完后 env 层就正确,无论上层用 FlashSAC 还是 TD-MPC2 都受益。先修 env 再上世界模型,降低归因难度。

## A1. 三个 bug 的代码事实(已核实)

| Bug | 位置 | 现象 |
|---|---|---|
| reward 泄漏 | multiskill.py:476-488 | 3 个 flamingo reward(lifted/support/com_over_support)对所有 env 生效无 mask |
| flag shape 错位 | multiskill.py:175,182 + 616 | `_is_flamingo`/`_is_fallen` 在 `_sample_commands(env, num_reset)` 被覆盖成 num_reset 长;reward 时 99% shape=(2~7,) vs num_envs=128;命令重采样(616)还重洗身份 |
| gait_phase 不前进 | base.py:91 apply_action | G1MultiSkillEnv 继承的 apply_action 不更新 gait_phase,feet_phase reward 追踪静态相位目标=鼓励固定单脚抬起 |

## A2. Flag 机制修复(`src/unilab/envs/locomotion/g1/multiskill.py`)

- **A2.1** 第 469-470 行:`_is_flamingo`/`_is_fallen` 初始化为 `np.zeros(num_envs, dtype=bool)`(当前是 `np.zeros(0)`)。
- **A2.2** 新增 `_sample_skill_flags(self, env, env_ids)`,替代 `_sample_commands` 里的 flag 逻辑(158-186 行)。逻辑:`super()._sample_commands` + standing 零命令;**互斥采样** `u<fallen_prob` → fallen,`fallen_prob≤u<fallen_prob+flamingo_prob` → flamingo,其余 walk;**scatter-write** `env._is_flamingo[env_ids]=is_flamingo; env._is_fallen[env_ids]=is_fallen`;返回 local commands。`_sample_commands` 简化为只采速度+standing(删 175/182 行 flag 赋值)。
- **A2.3** `build_reset_plan`(188-290):第 270 行 `info_updates["commands"]` 改调 `_sample_skill_flags(env, env_ids)` 并**移到方法开头**(先写 flag 再读 flag 选 keyframe);删 191-194 / 203-206 的 fallback,直接 `env._is_flamingo[env_ids]` 索引全长数组。
- **A2.4** step 命令重采样(610-617):新增 `_resample_velocity_commands()` 替代 `self._dr_manager._provider._sample_commands(self, self._num_envs)`——调简化后的 `_sample_commands`(不再写 flag)采速度,再 `cmds[self._is_flamingo|self._is_fallen]=0` 锁定身份 env。保留 8s 切 vx(站↔走↔跑过渡),身份整 episode 不变。

## A3. Reward Routing(`src/unilab/envs/locomotion/g1/multiskill.py`)

- **A3.1** 三个 flamingo reward(476-488)乘 `self._is_flamingo.astype(dtype)` mask。新增 `_flamingo_mask()` helper。
- **A3.2** 起身 reward(height_exp/delta_height/stand_feet/uprightness_exp/soft_symmetry/penalty_orientation_adaptive)乘 `self._is_fallen` mask,避免在正常 env 上给常数 bonus 压过 tracking。
- **A3.3** 行走 reward(feet_phase/feet_phase_contrast/feet_phase_contact/feet_double_stance/feet_air_time)在 G1MultiSkillEnv 覆盖,加 `~_is_flamingo & ~_is_fallen` mask(原速度门控保留)。

## A4. gait_phase 前进(关键,与 reward 泄露同级 bug)

- **A4.1** G1MultiSkillEnv 覆盖 `apply_action`(参考 joystick.py:1041-1054 GC 版):每步 `gait_phase[:,0/1] = (gait_phase + self._gait_phase_delta) % 2π`。不修这个 feet_phase 永远鼓励静态单脚抬起。

## A5. 跑步腾空期 reward(新增)

- **A5.1** 新增 `_reward_feet_flight`:双脚同时离地 `~lc & ~rc`,门控 `cmds[:,0] >= flight_speed_threshold(默认1.0)` & walk mask。参考 joystick.py:574 `_reward_feet_double_stance` 写法。
- **A5.2** 覆盖 `_reward_feet_double_stance` 加高速上界 `cmds[:,0] < flight_speed_threshold`,与 flight 速度区间不重叠(vx∈(0,1.0) 鼓励双支撑走,vx≥1.0 鼓励腾空跑)。
- **A5.3** terminate 复核:min_base_height=0.3 + max_tilt=65° 宽松,跑步腾空不触发(已复核)。

## A6. Phase H 配置(`conf/offpolicy/task/flashsac/g1_multiskill/mamba_phaseH.yaml`,基于 phaseD_v3)

- `vel_limit: [[0.0,-0.3,-0.8],[2.0,0.3,0.8]]`(扩到 vx 0-2.0 覆盖站/走/跑)
- `resampling_time: 8.0`(站↔走↔跑过渡),`rel_standing_envs:0.20`,`rel_flamingo_envs:0.10`,`rel_fallen_envs:0.15`
- `reward.flight_speed_threshold: 1.0`
- reward scales:通用(tracking/penalty/pose/alive/base_height)+ 行走(feet_phase 5.0 / feet_double_stance 1.5 / feet_flight 2.0)+ flamingo 专属(lifted -30 / support -15 / com +2)+ 起身专属(height_exp 5.0 / delta_height 2.0 / stand_feet 2.5 / uprightness 0.25 / soft_sym -1.0)

## A7. 验证(不依赖世界模型,改完 env 先跑)

- **A7.1** flag shape:`env.reset(128)` 后 `assert _is_flamingo.shape==(128,) and not (is_fl&is_fa).any()`;step 500 次后仍全长。
- **A7.2** 步态诊断 `/tmp/diag_gait.py`:左脚触地率>0.4、交替频率>0.02、高速 env 腾空期>0.02。
- **A7.3** reward 日志:`reward/penalty_lifted_foot_contact` 均值应在 -1~-3(10% flamingo),若≈-30×P(左脚触地)说明 mask 没生效。

## A8. 几何预处理(SE3/SO3 务实折中,和 Part A 一起设计)

> 决策(2026-06-21):严格 SO(3)-equivariant Mamba 是开放研究问题(selective SSM 的 input-dependent Δ/B/C 门控破坏等变性,无成熟实现)。本轮走**务实折中**:对 obs 做几何预处理变换到规范坐标系 + 提取 SE(3) 不变量,再喂 Mamba。Mamba 不必严格等变,但输入已消除大部分姿态冗余,拿到等变大半收益。严格等变网络留作后续随世界模型一起研究。

obs 98 维几何量分析:
- **gravity(3)**:world→body 的 SO(3) 作用量。tilt=arccos(g_z)、heading=atan2(g_y,g_x) 是部分不变量
- **gyro(3)**:body 角速度,SO(3) 旋转向量
- **dof_pos/dof_vel/last_actions(各29)**:关节量,左右对称性作用对象
- **commands(3)**:[vx,vy,ω],已在 body frame

几何预处理设计(`multiskill.py` `_compute_obs` 里做,复用已有 `_geo_features` 思路):
- **A8.1** yaw 规范化:用 base yaw 把 world-frame 量(gravity/linvel)旋到 forward-aligned 规范系,消除绝对朝向。复用 `np_quat_mul/np_yaw_to_quat`(src/unilab/envs/common/rotation.py)。
- **A8.2** 显式几何不变量(启用并扩展 `_geo_features`,当前未启用):tilt、heading sin/cos、COM-support 距离。这些是 SO(3) 部分不变量,给 Mamba 直接几何信号。
- **A8.3** 左右对称不变量:对左右成对关节,提取对称分量 `(l+r)/2` 和反对称分量 `(l-r)/2`。对称分量在左右镜像下不变,反对称分量变号——Mamba 据此隐式学对称性。参考 `G1SymmetryAugmentation`(src/unilab/envs/locomotion/g1/symmetry.py)的 mirror 映射定义。
- **A8.4** 对称增广(低成本叠加):Phase H 训练时开 `G1SymmetryAugmentation`,obs+action 左右镜像增广,作为几何预处理的补充(软约束)。

**注意**:A8 改变 obs 维度(加 geo 不变量 + 对称分量),需同步更新 `obs_groups_spec` 和 MambaActor 的 obs_dim。之前启用 `_geo_features` 导致维度 mismatch crash(98→102),这次要同步 buffer 预分配。

## A9. 实施顺序(每步独立验证)

1. A2 flag 修复 → A7.1 验证
2. A4 gait_phase 前进 → 单独测(IMU-GC 的 swing 检测依赖它,必须先修对)
3. A3 reward mask → A7.3 验证
4. A5 flight reward
5. A8 几何预处理(obs 维度同步)→ 重建 MambaActor
6. A10 动力学补偿(IMU-GC)接入
7. A6 Phase H yaml → fresh start 训练 → A7.2 diag_gait 验证 → model_5000 play 确认不崩

## A10. 动力学补偿:显式 g(q) + 隐式 MLP 残差(混合前馈)

> 决策(2026-06-21,客观判断):用户指出 CTC 在躺地/爬起时多点动态非刚性接触不可建模(qfrc_constraint/J_c 失准,CTCP reward 313 反不如简单 CC 317 已验证)。纯 GC 没顾接触力有前馈偏差。分阶段开关 controller 太粗暴(fallen 阶段完全无补偿)。**正确方案:显式+隐式混合前馈**——显式 g(q)(纯 Pinocchio 重力,任何阶段都可靠,不依赖接触建模)+ 隐式小 MLP 学残差(接触力/未建模动力学,躺地爬起也能学)。这是 residual dynamics / learned inverse dynamics 标准范式,文献成熟。
>
> 隐式训练:监督学习,τ_implicit 学 `τ* - PD - g(q)`(τ* 为 simulator 真实 actuator force)。backbone:小 MLP(起步,快稳易调;Mamba 留后续)。需扩展 backend 暴露真实 actuator force(类似 CTCP 读 qfrc_constraint 的老问题,红线验证)。

**公式**:`τ = PD + gravity_scale·g(q) + τ_implicit`
- g(q):Pinocchio 重力扭矩(全程,纯构型,不依赖接触——躺地也准)
- τ_implicit = MLP_θ(obs):学接触残差 + 未建模动力学
- 训练:监督 `τ_implicit → τ* - PD - gravity_scale·g(q)`,τ* = simulator qfrc_actuator(需 backend 暴露)

**为什么比纯 CTC/CTCP 好**:
- g(q) 全程可靠(不依赖接触假设),fallen 阶段也有重力兜底
- 接触残差由 MLP 从 IMU/接触传感器/gait_phase 隐式学,不假设刚性接触,躺地爬起适用
- 无 force sensor 内部力问题(CTC(force) 的坑)

**接入**(`multiskill.py` + 新 controller):
- A10.1 新建 `ResidualDynamicsController`(`src/unilab/control/residual_dynamics_controller.py`):PD + g(q)(复用 PinocchioDynamicsModel)+ MLP_θ 残差。
- A10.2 backend 扩展暴露真实 actuator force(MuJoCo `qfrc_actuator` 或 actuator force)——**红线:先验证可拿,拿不到则退方案 c(只显式 g(q))**。
- A10.3 MLP_θ 训练:off-policy 从 replay 采 batch,监督 loss `||MLP_θ(obs) - (τ* - PD - g(q))||²`,和策略训练分离(独立 optimizer)或联合。
- A10.4 obs 加 IMU 残余加速度 + 接触传感器(给 MLP_θ 输入,和 A8 几何特征一起同步 obs 维度)。
- A10.5 Phase H yaml 配 control_config:gravity_scale + MLP 残差开关/scale。

**关键耦合/风险**:
- g(q) 依赖 Pinocchio model 初始化(参考 joystick.py:895-900 已有)
- backend 暴露真实 force 是前提,不可拿则退 c 方案(只显式 g(q),不学残差)
- MLP_θ 监督信号质量依赖 τ* 精度;躺地多点接触时 τ* 本身含接触力(正是要学的),合理
- 显式 g(q) 全程不 mask(可靠);隐式残差自然在 fallen 阶段学接触(不需分阶段开关)

---

## A11. 地基性质疑:Mamba 对多技能合一的优势是否真实体现?(待决)

> 用户核心质疑(2026-06-21):Mamba 对多技能合一(单 policy)到底有没有优势?优势如何体现?当前设计是否体现?

### 客观判断(不迎合)

**一、现有实验证据不足以下结论。** Phase A/B 的 Mamba vs MLP 数据(reward +106%、tracking +60~150%)看似漂亮,但:
- Phase A/B 只 2-3 技能(Walk+Stand[+Flamingo]),**未到真正多技能合一**(目标 6+ 技能:站/走/跑/转弯/金鸡/起身)
- Episode Length 几乎一样(973/987 vs 970/997),生存无差别
- 优势可能来自单任务建模能力/参数效率,**不是**多技能统一带来的
- 缺 Phase D 规模下 Mamba vs MLP 对照(D v3 只跑了 Mamba)

**二、当前 MambaActor 设计根本没发挥 SSM 优势(文档教训已记但未改):**
1. 4-token 是同一 obs 复制,token 间无语义差异 → SSM 选择性门控无从选择
2. mean pooling 抹平时序 → 序列建模被削弱
3. 无跨 step hidden state 传递 → 每 step 独立 forward,Mamba 退化成"单步带门控 MLP"
- → 当前 Mamba 可能只是参数效率略高的 MLP,Phase A/B 优势更可能来自容量而非架构

**三、Mamba 理论优势(若设计对):** 时序选择性——根据观测历史自动激活技能子回路(无需 skill ID)、跨步状态传递(起身多步序列)。这些当前设计都没体现。

### 决策需求

这是地基问题。若 Mamba 设计不改正,"用 Mamba 做多技能合一"名不副实。需在动手 Part A 前定方向(见用户确认)。可能方向:
- (i) 改正 Mamba 设计:token 语义分离(HuMam 式 robot/task 拆分)+ 跨步状态传递(真正 SSM)+ 去 mean pooling。先验证 Mamba 优势再做 reward routing。
- (ii) 暂搁 Mamba 优势争议,先用 MLP 跑通 reward routing(修 env 正确性),Mamba 设计后续单独攻。
- (iii) 加 A/B 对照:Phase H 同 config 跑 Mamba vs MLP,用数据证明 Mamba 在多技能规模是否有优势,再决定。

**用户决策(2026-06-21):先改正 Mamba 设计(方向 i),正本清源。**

### A11.1 当前 MambaActor 三个硬伤(代码实证)

1. **token 语义分离写了没接**:`token_body`/`token_task`(200-203行,HuMam 式 2-token 拆分)是死代码,`_encode`(251行)用 `token_proj`(legacy,同 obs 复制成 4 token)。SSM 选择性无语义可选。
2. **mean pooling 抹平时序**:`x.mean(dim=1)`(256行)把 token 序列平均成单向量,序列建模被削。
3. **无跨步 hidden state**:`_encode` 每步从零 forward,hidden state 不跨步保留 → Mamba 退化成"单步带门控 MLP"。

### A11.2 管线事实(决定可行性)

- **collector 推理**:`actor.explore(obs, dones=...)`(worker.py:69)已传 dones,但当前只用于重置探索噪声(mamba_actor.py:294-302),**没用于重置 hidden state**。→ 推理端支持跨步状态,只差 actor 自己维护 + done reset。
- **learner 训练**:off-policy 从 replay 采**随机 batch(非连续序列)**,无法用跨步 hidden state。这是 off-policy + RNN 经典难题(TD-MPC2 用 H-step rollout 采连续片段解决)。

**关键结论**:Mamba 跨步状态优势在 collector(推理)可用,在 learner(训练)需改采样为序列片段。**这和 Part B(TD-MPC2 H-step rollout)天然耦合**——若后续上世界模型,序列采样基础设施可复用。

### A11.3 Mamba 设计改正方案

**Step 1 — token 语义分离(接上死代码)**:
- `_encode` 改用 `token_body`(body 状态:joint pos/vel/gravity/gyro)+ `token_task`(task:commands/gait_phase/skill 信号)。
- token 数从 4(同 obs 复制)降到 2(语义分离),让 SSM 选择性有语义可选。
- 需按 obs 98 维实际结构确定 body/task 拆分边界(查 _compute_obs 各组分的 obs 偏移)。

**obs 98 维组分(joystick.py:318 注释确认):`gyro(3)+gravity(3)+dof_pos_diff(29)+dof_vel(29)+last_action(29)+cmd(3)+phase(2)=98`**
- **body token 边界**:gyro+gravity+dof_pos_diff+dof_vel+last_action = 3+3+29+29+29 = **93 维**(索引 0:93)
- **task token 边界**:cmd+phase = 3+2 = **5 维**(索引 93:98)
- 当前代码 `split=2*obs_dim//3=65`(201行)是错的拆分点,需改为 93。`token_body: Linear(93, d_model)`,`token_task: Linear(5, d_model)`。
- **注意**:若 A8 启用 `_geo_features`(加 geo 不变量到 obs 末尾),obs 变 >98,task 边界要相应调整或 geo 归入 body token。A8 和 A11 要协同定 obs 结构。

**Step 2 — 去 mean pooling,用有效 token**:
- 不再 mean(pool),改取 task token 或 [CLS] 式聚合。或保留序列输出给跨步状态。

**Step 3 — 跨步 hidden state(推理端先行)**:
- MambaActor 维护 `_hidden` (B, n_layers, d_model),每步 forward 用上一步 hidden,done 时 reset。
- `explore` 里 dones 触发 hidden reset(复用现有 done_mask 逻辑)。
- learner 端先**不用**跨步状态(单步 forward,和现在一致),等序列采样基础设施(随 Part B)再加。

**Step 4 — A/B 验证 Mamba 优势**:
- 同 Phase H config 跑 Mamba(改正后) vs MLP,在多技能规模(6 技能)对比 tracking/episode/skill 切换质量。
- **若 Mamba 无明显优势,客观承认,回退 MLP**——不强行用 Mamba。

### A11.4 调整后的实施顺序

Mamba 设计改正(Step1-3)→ A/B 验证(Step4)→ 若通过再做 Part A reward routing。**Mamba 优势是地基,先证再建上层。**

**注意**:Step3 learner 端跨步状态受 off-policy 随机 batch 限制,本轮只做推理端跨步 + token 分离。完整时序优势留待 Part B 序列采样。本轮目标是"证明 token 语义分离 + 推理跨步 是否带来多技能优势",若否则客观回退。

---

# Part B — TD-MPC2 + Mamba 世界模型(后续项目,本轮不实施)

> 调研已完成(TD-MPC2 arXiv:2310.16828),技术事实多源验证。待 Part A 验证 env 正确后启动。

核心:Mamba 替换 TD-MPC2 的 latent dynamics(原为无状态单步 MLP,多步靠 for-loop 喂回;Mamba 带隐藏状态跨步递推,理论上是 GRU/RSSM 的现代升级)。5 组件世界模型(encoder/dynamics-Mamba/reward/Q-ensemble/policy_prior),训练 loss=20*consistency+0.1*reward+0.1*value+termination,MPPI 推理。新建 `src/unilab/algos/torch/td_mpc2/`,复用 FlashSAC 的 MambaBlock + OffPolicyRunner。详见上文调研结论(已从 deep research 获取完整架构)。**本轮不做。**
