---
name: mamba-multiskill-phasen-getup-result
description: "Phase N起身训练成果: 翻身100%成功(roll_to_supine reward), 起身0%失败(需参考轨迹); aux撤光后stand_feet稳定3.0"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

2026-06-25 Phase N 起身训练完成(model_8000),**翻身成功起身失败**的分阶段成果。

**配置**(`mamba_phaseN.yaml`):
- 推倒真倒地起步(推shoulder随机部位+方向, base_z0.12 tilt84°, 肘部伸直, 朝向多样仰/俯/侧)
- roll_to_supine reward: g_target=[-1,0,0](背=-x方向, 仰卧背朝下), 状态门控base_z<0.40
- alive=0(关白给, phaseL教训), uprightness_exp 3.0
- aux衰减step500-5000(贴地轻托gain150/thr0.40)
- 大Mamba 3.9M + 几何obs130

**sim2sim评估(aux=0, 256envs×400步)**:
| 阶段 | model_5000 | model_8000 |
|---|---|---|
| 翻身(任意倒地→曾仰卧) | 100% | 100% ✅ |
| 完整起身(任意→曾站起) | 0% | 0% ❌ |
| 曾仰卧env起身率 | 0% | 0% |

**关键修正历程**(踩坑后改对):
1. 推pelvis力臂小只平移不翻→改推shoulder(力臂大产生翻转力矩)→真倒地100%
2. g_target初始[0,0,1]错(真实仰卧g_z≈0)→改[0,-1,0]又错(是侧翻非仰卧, 用户看图指出real/side都侧卧)→改[-1,0,0]对(背=-x方向)
3. PD控用cur_dof保留摔时肘弯→改_sample_lie_joints肘部强制0°伸直
4. push阶段ctrl=stand锁僵硬被推倒(非ctrl=0松散平移)

**结论**:
- ✅ 翻身阶段训成(roll_to_supine reward有效, G阶段+phaseJ/K/L八次失败后首次)
- ❌ 起身阶段(仰卧→站立)纯reward训不出, 与HumanUP调研结论一致: 需参考轨迹做Stage II tracking
- aux撤光后stand_feet稳定3.0(不靠辅助), 但sim2sim起身0%(stand_feet reward有假象, 在非真倒地env拿分)

**视频狂闪根因**: 非动作抖(帧间差0.034), 是env autoreset——起身失败→terminated→重置到新倒地姿态, 16env频繁重置视觉狂闪。

**How to apply**:
- 起身攻Stage II tracking(参考轨迹引导): 下载HumanUP G1起身轨迹(GDrive 1kRSGkMDnqsX6OLr7-8OM5R6bF9mn84sK, 23DoF补手腕对齐29)或自造(仰卧→半跪→站立关键帧插值)
- 翻身阶段已可用(model_8000), 不必重训
- checkpoint: `logs/flash_sac/G1MultiSkill/2026-06-24_21-57-06_mujoco/model_8000.pt`
详见 [[mamba-multiskill-getup-failed-lessons]]、[[mamba-multiskill-fallen-pose-bug]]。
