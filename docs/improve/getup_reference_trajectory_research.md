# 人形起身参考轨迹引导 RL — Deep Research 调研报告

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
