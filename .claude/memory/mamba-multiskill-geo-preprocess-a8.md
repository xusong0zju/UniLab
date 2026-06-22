---
name: mamba-multiskill-geo-preprocess-a8
description: "A8几何预处理完成——obs 98→130(+geo3 +sym29),Phase I训练terminated 10% vs 21%,样本效率提升"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

2026-06-22 完成 A8 几何预处理(SE3/SO3 务实折中)+ Phase I fresh start 训练验证。

**设计**(multiskill.py):
- obs 98 → 130 维:base 98 + geo 3(tilt, heading_sin, heading_cos)+ sym 29(13 对左右关节 (l+r)/2 + (l-r)/2 + 3 中线 waist)
- `_geo_features(gravity)` (B,3):纯 gravity 派生,reset/step 都 batch-safe(去掉了 com_dist,因为它用 backend 取全 num_envs 会和 reset 路径的 num_reset 不匹配)
- `_sym_features(dof_pos)` (B,29):用 `G1_LEFT_RIGHT_PAIRS`(13 对,基于 actuator 顺序 0-5左腿/6-11右腿/12-14 waist/15-21左臂/22-28右臂)
- `obs_groups_spec`: {obs:130, critic:133}
- `_actor_symmetry_obs_layout`:末尾追加 ("geo",3),("sym",29) 让 mirror_obs dim 校验通过
- MambaActor/buffer 不改(从 obs_groups_spec 动态读 obs_dim)
- A8.4 对称增广**不做**:FlashSAC double_buffer_runner 未接入 augment(只 log),只 fast_sac/runner.py 接入

**验证(Phase I fresh start 8000 iter, model_8000 vs Phase H model_8000)**:

训练过程 A/B(同 iter 对比,核心优势):
| 指标 | Phase H | Phase I | 判断 |
|---|---|---|---|
| terminated @iter4000 | 21% | 10% | ✅ 少摔一半 |
| episode_length @iter2330 | 645 | 737 | ✅ +14% |
| tracking @iter2330 | 0.766 | 0.822 | ✅ 反超 |

最终步态(model_8000,都到天花板):
| 指标 | Phase H | Phase I | 判断 |
|---|---|---|---|
| 左脚触地率 | 0.38 | 0.37 | 持平 |
| 腾空率 | 0.15 | 0.16 | ✅ 略升 |

**结论**:A8 几何预处理积极作用主要体现在**训练过程样本效率**(同 iter 少摔一半 + tracking 反超),而非最终步态质量(Phase H 训够也能达到)。tilt/heading 几何不变量 + 左右对称分量确实帮 Mamba 更快找到稳定策略。

**checkpoint**: `logs/flash_sac/G1MultiSkill/2026-06-22_10-36-42_mujoco/model_8000.pt`(obs 130 维,不能续训旧 98 维 checkpoint)

**How to apply**:用 `mamba_phaseI.yaml` 训练(obs 130)。诊断时 diag_gait.py 的 task 改 phaseI、gravity 在 obs[3:6] 位置不变(geo/sym 追加在末尾)。详见 [[mamba-multiskill-reward-routing-fix]]、[[mamba-multiskill-fallen-pose-bug]]。
