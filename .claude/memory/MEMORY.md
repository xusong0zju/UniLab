# Memory Index

- [G1 Gravity Compensation Implementation](g1-gravity-comp-impl.md) — 训练侧实现完成，待 A/B 对比
- [User Profile](user-profile.md) — xusong0zju，中文沟通，关注收敛速度
- [G1 Gravity Comp Pitfalls](g1-gravity-comp-pitfalls.md) — 实现中踩过的 6 个关键坑
- [G1 Dynamics Compensation Full Results](g1-coriolis-comp-impl.md) — 六方案A/B完成：GC>CTC(IMU)>CC>CTCP>Baseline
- [G1 IMU-Enhanced GC](g1-imu-gc-impl.md) — IMU-GC 96×2 @10k=325.9, sim2sim鲁棒
- [IMU-GC Sign & Redesign](imu-gc-sign-redesign.md) — A/B证实gravity_factor无影响，已安全禁用，IMU专注swing检测
- [G1 Flamingo Stand Project](g1-flamingo-stand-project.md) — Phase 1✅(双腿站→990steps), Phase 2⬜(单腿), Phase 3⬜(推力), 续训命令
- [G1 Flamingo Stand Impl Notes](g1-flamingo-stand-impl-notes.md) — 7个实现踩坑: DR baselines/contact切片/Phase1 penalty禁能/config字段对齐
- [Skills & Workflows](../skills-and-workflows.md) — §9 Sim2Sim完整工作流: 降参训练→录制→扰动评估
- [Mamba多技能项目总览](mamba-multiskill-project.md) — G1多技能统一策略(行/站/金鸡/起身/推力),最佳ckpt=model_5000(12%摔倒)
- [Mamba多技能真实评估口径](mamba-multiskill-eval-strict.md) — 用base_z+tilt自判(非state.terminated);5000远好于3000(12%vs43%),续训致退化
- [osmesa渲染env上限](osmesa-render-env-limit.md) — EGL坏,osmesa软渲染:16envs可行/24envs死锁,靠延长steps不加env
- [Mamba行走学成单脚跳](mamba-multiskill-gait-bug.md) — model_5000行走左脚触地率0.02;flamingo的support/lifted reward在行走下也生效致语义冲突
- [Phase H reward routing修复](mamba-multiskill-reward-routing-fix.md) — flamingo/起身/行走reward按身份mask;行走0.02→0.38恢复双脚走/跑,腾空0.15
- [fallen姿态bug](mamba-multiskill-fallen-pose-bug.md) — 半跪0.55非真倒地,起身瞬间完成看不到;决策起身后续专门训
- [A8几何预处理](mamba-multiskill-geo-preprocess-a8.md) — obs 98→130(+geo3+sym29);Phase I训练terminated 10%vs21%,样本效率提升
