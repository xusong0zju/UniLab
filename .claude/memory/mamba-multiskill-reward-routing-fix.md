---
name: mamba-multiskill-reward-routing-fix
description: "Phase H reward routing 修复完成——flamingo/起身/行走 reward 按身份mask,行走从单脚跳(0.02)恢复成双脚走/跑(0.38),腾空0.15"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

2026-06-22 完成 Part A reward routing 修复 + Phase H fresh start 训练验证。

**根因(用户洞察)**:`mamba_phaseD_v3` 的 flamingo 专属 reward(lifted_foot_contact -30 / support_foot_contact -15 / com_over_support +2)**对所有 env 生效无 mask**,把行走/站立/起身都"金鸡独立化"——行走变单脚跳(左脚触地率0.02)、起身被逼直接进单脚支撑(44%摔倒)、站立偏单脚。

**修复(multiskill.py + joystick.py + phaseH.yaml)**:
1. **flag 机制**:`_sample_skill_flags` 互斥采样(单u分 fallen/flamingo/walk)+ scatter-write 全长数组;step 命令重采样(8s切vx)锁定身份 env vx=0。整 episode 身份不变。
2. **gait_phase 前进**:覆盖 `apply_action` 每步 `gait_phase += delta % 2π`(原继承 LocomotionBaseEnv 不前进,feet_phase 追踪静态目标=鼓励单脚抬起)。
3. **reward routing mask**:
   - 3 flamingo reward × `self._is_flamingo`
   - 4 起身 bonus(height_exp/delta_height/stand_feet/uprightness_exp)× `self._is_fallen`
   - 行走 reward(feet_phase via `_gait_reward_gate` / double_stance / air_time)× `_walk_mask`
   - `penalty_orientation_adaptive` 和 `soft_symmetry` **保留通用**(直立/对称是通用约束,不限 fallen——纠正了计划 A3.2 的误判)
4. **跑步腾空**:新增 `feet_flight`(vx≥flight_speed_threshold=1.0 且双脚离地)+ double_stance 加 `vx<threshold` 上界(走/跑区间不重叠)。
5. `G1RewardConfig` 加 `flight_speed_threshold: float = 1.0` 字段(struct 必需)。

**验证(Phase H fresh start 8000 iter, model_8000)**:
| 指标 | 旧5000(泄漏) | 新8000(修复) |
|---|---|---|
| 左脚触地率 | 0.02 | 0.38 |
| 腾空率 | — | 0.15 |
| penalty_lifted_foot_contact | ≈-30 | -0.2(mask生效) |
| episode_length | 988 | 856 |
| terminated | 4% | 19%(fresh start未充分训) |

**How to apply**:用 `mamba_phaseH.yaml` 训练,checkpoint 在 `logs/flash_sac/G1MultiSkill/2026-06-21_21-04-14_mujoco/model_8000.pt`。诊断脚本 `/tmp/diag_gait.py`(task 改 phaseH)。reward 大改不能续训旧 checkpoint,必须 fresh start。详见 [[mamba-multiskill-gait-bug]](根因)、[[mamba-multiskill-eval-strict]](评估口径)。
