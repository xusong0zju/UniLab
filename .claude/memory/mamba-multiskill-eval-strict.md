---
name: mamba-multiskill-eval-strict
description: "Mamba多技能真实评估口径——收紧终止条件(min_z0.5/tilt40°)用base_z+tilt自判,不能用state.terminated"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

评估 Mamba 多技能策略(D v3)的正确方法:用独立脚本 deterministic play,自己按 `base_z < 0.5 或 tilt > 40°` 判摔,**不能用 `state.terminated`**(play 路径下该字段语义/传递有问题,会给出 69% 的虚假高摔倒率)。

正确口径下的真实结果(2048 envs × 800 步):

| 技能 | model_5000 摔倒率 | model_3000 摔倒率 |
|------|-----------------|-----------------|
| 行走/站立 | 5% | 40% |
| 金鸡独立 | 13% | 46% |
| 起身(fallen) | 44% | 54% |
| 全体 | 12% | 43% |

**关键结论**:
1. model_5000 质量其实不错(全体摔倒率 12%,行走/站立 95% 存活),之前以为"指标造假"是评估脚本 bug 误判,env 终止条件宽松(min_z0.3/tilt65°)会让 episode_length 偏高但不至于完全失真。
2. **续训(5000→3000)导致退化**:行走摔倒率 5%→40%,全体 12%→43%。续训时 episode_length 在 iter 1750-2250 从 979 掉到 323 不是震荡而是真实退化,后续反弹是宽松口径假象。
3. 起身始终最弱(44-54% 摔倒),与 G 阶段 6 次失败一致。

**How to apply**:用 model_5000,不要用 model_3000。续训 off-policy 后期易退化,checkpoint 选择必须用收紧口径的独立评估,不能信训练日志的 episode_length。评估脚本 /tmp/eval_strict.py(脚本不持久,核心逻辑见下)。

**model_5000 路径**:`logs/flash_sac/G1MultiSkill/2026-06-20_22-42-08_mujoco/model_5000.pt`(原 run,完好无损)
**model_3000 路径**(退化,弃用):`logs/flash_sac/G1MultiSkill/2026-06-21_08-34-53_mujoco/model_3000.pt`

**评估脚本核心逻辑**(可重建):
- 加载方式同 play:flashsac 需显式传 mamba kwargs(`use_mamba_actor=True, mamba_d_state=16, mamba_n_tokens=4`)给 `build_actor`,否则建 MLP actor 导致 state_dict key 不匹配
- 判摔:`base_z = env._backend.get_base_pos()[:,2]`;tilt 从 obs 取(projected gravity 在 obs[3:6],`obs[:,5]` 是 gravity_z,`tilt=degrees(arccos(abs(grav_z)))`)。**不要用 `env._backend.get_gravity()`**(返回世界重力 (3,),非 per-env)
- 技能标记:`env._is_flamingo` / `env._is_fallen`(reset 后就有,数组长度=num_envs);standing = 非flamingo非fallen
- 收紧口径:`min_z=0.5, max_tilt=40°`(env 训练用宽松的 `min_z=0.3/tilt65°`,评估时自己收紧)

修正了之前误写的"指标造假"结论——是评估脚本用了 `state.terminated` 的 bug,不是 env 终止条件造假。详见 [[osmesa-render-env-limit]] 录视频方法。
