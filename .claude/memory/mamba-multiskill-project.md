---
name: mamba-multiskill-project
description: "Mamba多技能统一策略项目总览——G1行走/站立/金鸡独立/起身/推力抗扰,不用skill ID"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

G1 多技能统一策略项目:用一个 Mamba SSM actor(无显式 skill ID/gait conditioning)统一行走(大步 vx 0.3-1.2)、双腿站立、金鸡独立、多部位推力抗扰(12部位 30N)、倒地起身、模式间过渡(8s 切命令)。算法 FlashSAC,Mamba Actor + MLP Critic。

**当前最佳 checkpoint**(2026-06-21):
- `logs/flash_sac/G1MultiSkill/2026-06-20_22-42-08_mujoco/model_5000.pt`
- 收紧口径评估(min_z0.5/tilt40°):全体摔倒率 12%,行走/站立 95% 存活,金鸡独立 87%,起身 56% 存活(最弱)
- config: `conf/offpolicy/task/flashsac/g1_multiskill/mamba_phaseD_v3.yaml`(全技能+resampling_time=8 过渡+fallen+push)
- 弃用 model_3000(续训退化,摔倒率 43%)

**关键技术点**:
- 980 冻结已修复(`build_reset_plan` 的 np.where shape bug,见 docs §6.4)
- Mamba actor 用 official mamba-ssm CUDA kernel(`use_official=True`),纯 PyTorch for-loop 扫描是瓶颈
- token 设计:4-token 同 obs 复制(HuMam 是 2-token 语义分离,更优但我们受限于 obs 结构)
- env: `src/unilab/envs/locomotion/g1/multiskill.py`,keyframe 在 `scene_flat.xml`(stand/flamingo/kneeling)

**剩余问题**:
- 起身最弱(44% 摔倒),从零发现起身动作不可行,需参考轨迹/密集辅助(与 G 阶段 6 次失败一致)
- 续训后期易退化,checkpoint 选择必须收紧口径评估

**相关记忆/文档**:
- [[mamba-multiskill-eval-strict]] — 真实评估口径与 5000 vs 3000 对比
- [[osmesa-render-env-limit]] — 录视频方法(UNILAB_RENDER_PROCESSES=1 串行渲染)
- [[g1-flamingo-stand-project]] — 独立的单技能 flamingo 项目(已 64/64 完成,不同于本多技能项目)
- docs/improve/mamba_multiskill_experiments.md — 完整实验记录(§6.4 冻结修复/§10 reward hacking/§11 续训退化)
- docs/improve/g1_mamba_multiskill_design.md — 设计文档
- docs/improve/humam_analysis_and_improvements.md — HuMam 论文分析
