---
name: mamba-multiskill-gait-bug
description: model_5000行走学成单脚跳——flamingo reward对所有env生效+flag机制shape错位双重bug
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

**症状**:model_5000 在行走命令(vx 0.3-1.2)下不是双脚交替走,而是右脚单脚跳/单脚拖行。用户视频看到"单脚站立与单脚跳",行走没看到。起身(fallen)也难成功(44%摔倒)。

**诊断数据**(model_5000, 365 行走 env, 300 步, /tmp/diag_gait.py):
- 左脚触地率 0.02(几乎不落地),右脚触地率 0.81
- 双脚同时触地 0.00(正常走必有双支撑期)
- 支撑脚交替频率 0.003(正常双脚走应 0.02-0.06)
- → 左脚全程悬空,右脚单脚支撑,从不双脚交替

**双重根因**:

1. **reward 泄漏**:`mamba_phaseD_v3.yaml` 三个 flamingo 足部 reward 对所有 env 生效无 mask:
   - `penalty_lifted_foot_contact: -30`(左脚着地就罚)
   - `penalty_support_foot_contact: -15`(右脚没着地就罚)
   - `com_over_support: 2.0`(COM压右脚才奖)
   `run_reward_dispatch` 统一算所有 reward 项,无 per-env mask。行走/站立/起身 env 都被拉向单脚。起身最严重:爬起后被逼直接进金鸡独立而非双脚站,难度暴涨。

2. **flag 机制 shape 错位(更隐蔽)**:`_is_flamingo`/`_is_fallen` 在 `_sample_commands(env, num_reset)` 里按 num_reset 长度赋值,但 reward 函数对全 num_envs 计算。实测(/tmp/probe_reward_time.py):reward 调用时 99% 的步 `_is_flamingo.shape=(2~7,)` 而 num_envs=128——**shape 严重错位**。根因:
   - 部分 env done reset 时 `_sample_commands(env, num_reset)` 把 `env._is_flamingo` 覆盖成 num_reset 长
   - 命令重采样(step 616行 `_sample_commands(self, self._num_envs)`)每 resampling_time(8s)把所有 env 的 flamingo/fallen 身份**重新随机洗牌**
   - flamingo 和 fallen 还是各自独立随机不互斥(实测重叠3个)
   - 所以即使想做 `* self._is_flamingo` mask,flag 本身就是坏的

**修复必须先修 flag 机制**(reward routing 的前提):
- `_is_flamingo`/`_is_fallen` 必须全长 num_envs、整 episode 稳定、只在各自 env reset 时更新、不被全局重采样洗牌
- flamingo/fallen 互斥采样
- 修复后 reward 才能安全按 flag mask

**重训目标(用户2026-06-21确认)**:按速度命令路由技能——
- vx=0:双脚站立 | 低速:小步走 | 中速:大步走 | 高速:跑步(有双脚离地腾空期)
- ω≠0:按角速度转弯
- 金鸡独立:仅 flamingo 标记 env(特殊输入触发),保留单脚站能力
- 起身:爬起→双脚站(非单脚)
- 去除:单脚跳着走(不实用)
- 起点:**从 Phase B 重新训**(不用model_5000 warm-start)

**How to apply**:先修 flag 机制→reward routing(flamingo reward 乘 _is_flamingo)→用 diag_gait.py 验证(左脚触地率回升>0.4、交替频率>0.02、跑步腾空期>0)→从Phase B重训。详见 [[mamba-multiskill-eval-strict]]。
