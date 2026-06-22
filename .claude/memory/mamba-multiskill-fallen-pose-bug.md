---
name: mamba-multiskill-fallen-pose-bug
description: "Phase H fallen env 半跪0.55非真倒地,起身瞬间完成看不到过程;决策起身后续专门训"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

2026-06-22 诊断 model_8000 发现:视频里看不到起身,因为 fallen env **根本没真倒地**。

**根因**:`build_reset_plan` 的 fallen 姿态 base_z=0.55(半跪高度),加 80° roll/prone/side 倾斜四元数,但 base 在 0.55m 悬空被接触力撑住——**不是平躺地面**。代码注释"设 0.55 为了不触发 min_base_height=0.3 终止"是错误权衡:为避免终止把 fallen 抬到半跪,导致根本没倒。

**实测**(model_8000, 1024 envs):
- fallen env reset 瞬间 base_z=0.55 ✅(姿态设置生效)
- 但 600 步后 base_z=0.736(第1步就站起来了)
- base_z 全程稳定 0.73 不动 → 无起身过程可观察
- "起身成功率100%"是假象(本来就没倒)

**真倒地应 base_z<0.3**(平躺躯干贴地),但计划 §4 记 G 阶段 6 次失败教训:从真倒地起身在 MuJoCo 150Hz 下极难,HumanUP 两阶段法都失败。

**决策(2026-06-22)**:起身后续专门训。当前 Phase H 接受"半跪起身"( fallen env 从半跪0.55站起,非平躺起身)。行走/站立/金鸡独立/跑步已修复(reward routing),起身弱点后续用参考轨迹或 HoST 两阶段法单独攻,不混入多技能训练。

**How to apply**:
- 当前用 model_8000(起身是半跪起身,非倒地起身),不要误判"起身100%成功"
- 后续训起身:改 fallen 姿态 base_z 0.2-0.25(真倒地)+ 对 fallen env 单独放宽 min_z 终止 + 参考轨迹/HoST 两阶段(发现→精炼)。详见 [[mamba-multiskill-reward-routing-fix]]、[[g1-flamingo-stand-project]] G阶段教训。
