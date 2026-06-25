# 人形起身 MoE 调研 — Deep Research 原始报告

> 调研规模: 101 agents, deep research harness

> 原始问题: 人形机器人(humanoid)起身/跌倒恢复(get-up/fall recovery)的参考轨迹引导 RL 训练方法。重点:1) HumanUP/HoST(RSS 2025)如何用参考轨迹做人形起身,两阶段发现→精炼的具体机制,参考轨迹从哪来(mocap/关键帧/教师);2) 人形起身RL的开源github实现(如基于IsaacGym/legged_gym的getup/recovery训练,能跑...


## 调研注意点 (Caveats)

1) The research question's premise conflated HumanUP and HoST as a single 'two-stage discovery/refinement guided by reference trajectories from mocap/keyframes/teacher' method. This is incorrect on two counts: (a) HumanUP and HoST are separate papers (arXiv:2502.12152 vs 2502.08378), and (b) neither uses mocap/keyframe/teacher reference trajectories — HumanUP's reference is RL-self-discovered, HoST uses no reference at all. 2) The MuJoCo-150Hz feasibility conclusion is an inference (150 < 200 < 1000), not a directly tested result; no source empirically tested get-up tracking at 150 Hz in MuJoCo. The contact-heavy nature of get-up makes lower frequencies riskier, but this is a physics-grounded caution, not a proven failure. 3) AMASS access is gated (registration on 3 sites, manual download, non-commercial license) — not scriptable, a friction point for automated RL pipelines. 4) Two newer papers (arXiv:2603.08619 from 2026-03, Stubborn arXiv:2606.12814 from 2026-06) are only ~3 months old with little independent third-party corroboration, though both are primary arXiv sources. 5) The refuted DeepMimic-imitation claim (0-3) does not invalidate the surviving RSI/ET claim [12] — DeepMimic's RSI/ET techniques remain valid and reusable regardless of whether get-up work 'builds on' DeepMimic's imitation paradigm. 6) UniLab's existing NPZ motion-tracking loader was not directly inspected in this research (no UniLab-internal claim survived verification); the reuse recommendation is inferred from HumanUP's comparable .pkl loader design. 7) No source addressed whether keyframe interpolation or trajectory optimization (as opposed to mocap or RL-discovery) can generate viable get-up trajectories — this method was not found in the literature surveyed.


## 核心发现 (Findings)


### Finding 1: HumanUP (arXiv:2502.12152, RSS 2025, UIUC+SFU, He/Dong/Chen/Gupta) is the canonical two-stage discover-then-refine RL framework for humanoid get-up. Stage I ('Discovery Policy Training') learns a getting-up/rolling-over trajectory via RL with sparse task rewards and weak regularization under minimal deployment constraints, with NO reference motion. Stage II ('Deployable Policy Training') logs the 

- **置信度**: high

- **投票**: 3-0 (merged from claims [0],[1],[3],[6],[8])

- **证据**: Verified verbatim against the primary arXiv paper (v2), the project page (humanoid-getup.github.io), and the GitHub repo README across multiple independent fetches. Stage I 'discovers a good getting-up trajectory under minimal constraints'; Stage II 'refines the discovered motions into deployable (smooth and slow) motions.' Full-text counts confirm teacher=0, keyframe=0, distill=0. Ablation: 'HumanUP w/o Two-Stage Learning cannot solve the task.'

- **来源**: https://arxiv.org/abs/2502.12152, https://humanoid-getup.github.io/, https://github.com/RunpeiDong/HumanUP, https://raw.githubusercontent.com/RunpeiDong/HumanUP/master/simulation/README.md


### Finding 2: HumanUP's reference trajectory is stored as pickle .pkl files (dof_pos_all.pkl, head_height_all.pkl, and projected_gravity_all.pkl for rollover) under simulation/legged_gym/logs/env_logs/{getup_traj,rollover_traj}/, and is resampled to a fixed 8-second length via np.interp linear interpolation (target_traj_length = int(8/dt)+1). The authors publicly release their own RL-discovered trajectories on 

- **置信度**: high

- **投票**: 3-0 (merged from claims [4],[9])

- **证据**: Verified verbatim in g1waist_track.py:114-116 (opens dof_pos_all.pkl, head_height_all.pkl), g1waistroll_track.py:113-117 (adds projected_gravity_all.pkl), and simulation/README.md:54-64 (Google Drive release + 'Feel free to download it to directly train Stage II policy'). The 8-second np.interp resampling is in g1waist_track.py. Repo-wide grep for mocap/CMU returned no trajectory-sourcing hits.

- **来源**: https://github.com/RunpeiDong/HumanUP, https://raw.githubusercontent.com/RunpeiDong/HumanUP/master/simulation/README.md


### Finding 3: HumanUP's open-source implementation (github.com/RunpeiDong/HumanUP, Apache-2.0, master branch) is built on legged_gym (IsaacGym-based, rsl_rl PPO) and targets the Unitree G1 humanoid (23 DoF, wrists removed, custom collision mesh URDF g1_29dof_fixedwrist_custom_collision.urdf). It provides separate Stage II tracking entrypoints run_track.sh / eval_track.sh that take a traj_name argument to select

- **置信度**: high

- **投票**: 3-0 (merged from claims [7],[10])

- **证据**: Verified by direct repo inspection: README states '23 DoFs G1 with wrists DoFs removed', 'customized the original G1 collision mesh', and 'simulation training is based on Isaac Gym... based on legged_gym... rsl_rl'. run_track.sh source confirms traj_name=${4} -> --traj_name. Project page confirms 'Sim2Sim Transfer to MuJoco' section and 'directly deployed in the real world' on G1.

- **来源**: https://github.com/RunpeiDong/HumanUP, https://raw.githubusercontent.com/RunpeiDong/HumanUP/master/simulation/README.md, https://humanoid-getup.github.io/


### Finding 4: HumanUP requires simulation at 1 kHz (dt=0.001) in Isaac Gym for a real-world-working get-up policy, with the low-level PD position controller running at 50 Hz. The 200 Hz frequency common for locomotion is explicitly stated to produce a get-up policy that 'will not work in the real world' (though it trains reasonably in simulation). The high frequency is justified by the need to accurately model 

- **置信度**: high

- **投票**: 3-0 / 2-1 (merged from claims [2],[5],[11])

- **证据**: Verified verbatim in simulation/README.md: 'For the getting up policy learning, we use a higher frequency of 1k Hz (dt=0.001). Although you can train a reasonable policy in simulation under 200Hz, but it will not work in the real world.' Paper confirms 50 Hz PD control and 1000 Hz sim 'to accurately model the numerous contacts between the humanoid and the ground.' The 150 Hz MuJoCo inference is a physics-grounded caution (150 < 200 < 1000), not a directly tested result.

- **来源**: https://arxiv.org/abs/2502.12152, https://raw.githubusercontent.com/RunpeiDong/HumanUP/master/simulation/README.md


### Finding 5: HoST (arXiv:2502.08378, RSS 2025 Best Systems Paper Finalist, OpenRobotLab) learns humanoid standing-up control FROM SCRATCH via RL and does NOT use reference trajectories, motion capture, keyframes, retargeting, AMASS, or a teacher policy — the full-text HTML contains zero occurrences of all these terms. Its actual mechanisms are: multi-critic RL (independent reward groups), a 4-terrain curriculu

- **置信度**: high

- **投票**: 3-0 (merged from claims [18],[19],[20])

- **证据**: Verified against arxiv.org/html/2502.08378v1 (103K chars): case-insensitive regex counts = 0 for 'reference motion', 'reference trajectory', 'motion capture', 'mocap', 'keyframe', 'retarget', 'AMASS', 'teacher'. Abstract: 'learns standing-up control from scratch' and contrasts with 'predefined ground-specific motion trajectories'. Verbatim: 'We use Isaac Gym simulator with 4096 parallel environments and the 23-DoF Unitree G1 robot to train standing-up control policies with the PPO algorithm.' Official repo (github.com/OpenRobotLab/HoST) has only a main branch, requires Isaac Gym/rsl_rl/legged_gym, zero MuJoCo dependency; GitHub search for 'HoST humanoid standing mujoco' returned total_count: 0.

- **来源**: https://arxiv.org/abs/2502.08378, https://github.com/OpenRobotLab/HoST


### Finding 6: AMASS (Archive of Motion Capture as Surface Shapes, arXiv:1904.03278, ICCV 2019, Mahmood et al.) is the primary public mocap source usable as reference-trajectory data for humanoid get-up. It unifies 15 different optical-marker mocap datasets into a single common framework and SMPL parameterized representation, making it readily useful for animation, visualization, and generating deep-learning tra

- **置信度**: high

- **投票**: 3-0 / medium accessibility (merged from claims [16],[17])

- **证据**: AMASS abstract (via arXiv API): 'unifies 15 different optical marker-based mocap datasets by representing them within a common framework and parameterization... here we use SMPL... readily useful for animation, visualization, and generating training data for deep learning.' Accessibility confirmed via BlenderProc official docs noting the 3-site registration + manual download requirement (confidence on accessibility is medium due to inability to directly fetch the gated official site).

- **来源**: https://amass.is.tue.mpg.de/, https://arxiv.org/abs/1904.03278


### Finding 7: Stubborn (arXiv:2606.12814, SUSTech, Ren/Yang/Weng/Liu/Kong, 2026-06) trains a SINGLE unified RL policy (asymmetric Actor-Critic: privileged-information value net + proprioception-only policy net, with yaw-aligned tracking representation) that JOINTLY performs motion tracking AND fall recovery on the 29-DoF Unitree G1, explicitly avoiding the multi-stage training and separate recovery policies use

- **置信度**: high

- **投票**: 3-0 / 2-1 (merged from claims [21],[22])

- **证据**: Verified verbatim against arxiv.org/html/2606.12814v1: 'Stubborn uses a unified single policy for tracking highly dynamic motions while maintaining robustness under strong perturbations and achieving recovery from fallen states... asymmetric Actor-Critic architecture... value network uses privileged simulation information, while the policy network relies only on proprioceptive observations.' Motions: 'Simulation experiments are conducted in IsaacLab/MuJoCo using motions from the public LAFAN1 dataset... retargeted motions from LAFAN1 and AMASS.' Contribution: 'synthesizing emergent recovery behaviors without auxiliary recovery-specific reward design.'

- **来源**: https://arxiv.org/abs/2606.12814


### Finding 8: DeepMimic (arXiv:1804.02717, Peng et al., SIGGRAPH 2018) provides two reusable reference-tracking RL techniques directly applicable to UniLab get-up training: (1) Reference State Initialization (RSI) — sampling each episode's start state from the reference motion data rather than a fixed initial state, increasing exploration coverage and robustness to diverse initial lying postures; (2) Early Term

- **置信度**: high

- **投票**: 3-0 (claim [12]); related imitation-paradigm claim refuted 0-3

- **证据**: Confirmed against the primary source: RSI samples initial state from reference motion data each episode; ET aborts on failure-state deviation. Ablation study confirms both are 'key mechanisms for accelerating convergence.' xbpeng/DeepMimic GitHub implementation corroborates. The refuted imitation-reward claim does not affect RSI/ET validity.

- **来源**: https://arxiv.org/abs/1804.02717


### Finding 9: Humanoid-Gym (github.com/roboterax/humanoid-gym, arXiv:2404.05695) is NOT a usable base for humanoid get-up training: its only registered task is humanoid_ppo (PPO, multi-frame low-level control, XBotL robot), it is locomotion-only by design, and it contains NO get-up/fall-recovery/stand-up policy, reward, or env file (grep for getup/recover/standup/fall/mocap returned no functional matches; no .n

- **置信度**: high

- **投票**: 3-0 (merged from claims [13],[14])

- **证据**: Direct inspection of cloned repo (commit ae46e20): humanoid/envs/__init__.py:42 is the sole task_registry.register call (humanoid_ppo); grep for stand_up|get_up|fall_down|recover = zero matches; compute_ref_state() uses sinusoidal gait phase, not mocap. sim2sim.py loads torch.jit.load(args.load_model) and runs pure mj_step forward rollout — no training logic.

- **来源**: https://github.com/roboterax/humanoid-gym


### Finding 10: An anti-reference-trajectory result: humanoid recovery (stand-up + push recovery + compliant falling) can be learned by a SINGLE unified RL policy WITHOUT any reference trajectories, scripted contacts, or motion-capture priors, by embedding classical balance metrics (capture point, center-of-mass state, centroidal momentum) as privileged critic inputs and shaping rewards directly around these quan

- **置信度**: high

- **投票**: 3-0 (claim [15])

- **证据**: Verified verbatim against arxiv abstract: 'Without reference trajectories or scripted contacts, a single policy spans the full recovery spectrum: ankle and hip strategies for small disturbances, corrective stepping under large pushes, and compliant falling with multi-contact stand-up using the hands, elbows, and knees.' 93.4% recovery on H1-2; ablation confirms balance-informed structure is necessary.

- **来源**: https://arxiv.org/abs/2603.08619



---


# MoE 是否引入 — 最大 effort 综合斟酌(2026-06-24)

> 用户要求用最大 effort 重新斟酌 MoE。基于已完成的 deep research(`docs/improve/moe_research_getup.md`, 101 agents) + 针对性核实 + 当前起身任务的实际情况。

## 结论(不变,但附明确边界)

**当前起身任务(0.7-3.9M + 单 GPU + 缺参考轨迹)不引入 MoE。** 但与之前报告比,明确"何时该重新考虑"的边界。

## 重新斟酌的核心逻辑

### 之前结论的潜在弱点(诚实复盘)

1. **小模型 MoE 并非"零证据"**:
   - Quadruped Parkour MoE (arXiv:2604.19344): **2 倍成功率**,dense 需 +14.3% 算力达同等 → MoE 在足式控制**有参数效率收益**(同等性能省算力)
   - SAC-MoE (arXiv:2511.12361): **6 倍零样本泛化**(地形)
   - 这两个是 RL 正面案例,之前报告可能低估

2. **但这些正面案例不适用起身**:
   - 跑酷/地形泛化是**连续稳态任务**(周期步态),MoE 按地形切专家合理
   - 起身是**接触丰富的瞬态任务**(翻身→撑起→站立,接触点时刻变),专家按什么切?
   - 起身的核心瓶颈不是"多模态动作分布不可表达",是"缺参考轨迹发现序列"(HumanUP/HoST 共识)

### G1 上的决定性负面证据(维持判断)

- **CMoE (arXiv:2603.03067, Unitree G1)**: "vanilla MoE 路由在不同地形近乎均匀激活,专家特化失败"——**同款机器人**
- **CoRe-MoE (arXiv:2606.04718, G1)**: 独立确认,需两阶段+contrastive gating
- → G1 上用 MoE **必须 contrastive gating**,NLP 的 load-balancing 不迁移

### 起身任务的独特性(为什么 MoE 在其他足式任务有用,在起身没用)

| 任务类型 | MoE 是否有用 | 原因 |
|---|---|---|
| 多地形行走 | ✅ 有用(SAC-MoE 6x 泛化) | 按地形切专家,稳态周期 |
| 跑酷 | ✅ 有用(Parkour MoE 2x) | 按障碍类型切,动作模态明确 |
| 全身模仿 | ✅ 有用(FARM residual MoE) | 按身体部位/动作切 |
| **起身** | ❌ 证据不支持 | 接触瞬态,无明确可切模态;瓶颈是参考轨迹非表达力 |

起身文献(HumanUP/HoST/H1-2/ANYmal)**全部单策略+phase reward**,无一用 MoE。

## 何时该重新考虑 MoE(明确边界)

重新考虑 MoE 的**触发条件**(任一满足):

1. **scale 到 >50M 参数 + 多技能数据充足**:MoE 优势在大模型大数据显现。我们 0.7-3.9M 太小,路由器学不好。
2. **验证过单策略+课程确实卡在"动作分布多模态不可表达"**:
   - 诊断:同一状态下策略需输出多个截然不同动作(如翻身 vs 不翻身),单高斯策略无法表达
   - 若诊断到此,用 **GMM-MoE / Soft MoE**(PMOE arXiv:2104.09122)针对性解决,非稀疏 top-k
3. **泛化成瓶颈**(sim2sim/real 到未见模态):SAC-MoE 6x 泛化是唯一正向 RL 数据点
4. **多技能合一规模上去**(6+ 技能,单策略梯度冲突严重):此时 MoE-Loco 式梯度隔离有用

## 当前任务的正确方向(替代 MoE)

1. **分阶段 reward 课程**(Phase N,已实现):翻身+起身靠状态门控 reward 分工,文献支持(H1-2)
2. **Mamba 隐式选择**(已用):input-dependent selective 门控,虽 L=4 优势有限,但免费
3. **若阶段间干扰**:层级参数共享(Tree Learning arXiv:2604.12909)比 MoE 轻,100% 技能保留

## 最终判断

**起身用 MoE = 进无人区(零先例)+ G1 已证伪(vanilla)+ scale 不够(<4M)+ 瓶颈不匹配(缺轨迹非表达力)。** 

不引入,走分阶段 reward + Mamba。但记录上述"重新考虑边界",未来若多技能 scale 上去且验证卡在多模态表达,再回来用 GMM-MoE。

## 与之前报告的关系

之前 deep research(`docs/improve/moe_research_getup.md`)结论一致(不引入)。本文档补充:
- 承认小模型 MoE 有参数效率正面案例(Parkour/SAC-MoE),但都不适用起身
- 明确"何时重新考虑"的边界,避免未来重复调研
- 起身任务独特性分析(为何 MoE 在足式有用,在起身没用)
