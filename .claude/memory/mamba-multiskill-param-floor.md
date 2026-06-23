---
name: mamba-multiskill-param-floor
description: "Mamba多技能走步的参数下限~2M: 0.73M/1.04M(d128)学不会走, 3.9M(d256)会走, d128宽度是硬下限深度救不了"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

2026-06-23 验证 Mamba 多技能走步的**网络容量下限**。

**实验**(均为 fresh start 8000 iter, Phase I 配置带 A8 几何, G1MultiSkill env):

| 配置 | d_model | n_layers | 参数 | 实际vx(走步) | 结论 |
|---|---|---|---|---|---|
| 小Mamba v1 | 128 | 2 | 0.73M | ≈0 | ❌ 学不会走 |
| 小Mamba v3(+alive路由) | 128 | 2 | 0.73M | ≈0 | ❌ 路由也救不了 |
| 中B | 128 | 3 | 1.04M | ≈0 | ❌ 加深度救不了 |
| 大Mamba(Phase I) | 256 | 3 | 3.9M | >0.3 | ✅ 会走 |

**核心结论**:
1. **d128 宽度是硬下限**——无论 2层还是 3层,d128 都学不会走(实际 vx≈0)。深度(3层)无法弥补宽度(d128)不足。
2. **多技能基础模型不能小于 ~2M 参数**。走步(步态生成+平衡+腾空+落地)需要足够容量,0.73-1.04M 不足以同时学稳+走。
3. **alive 指令路由(v3)无法突破容量下限**——小网络无论怎么调 reward/路由,都选择"站着不动"(它能学会的最复杂动作),因为走步超出容量。
4. **tracking 均值会撒谎**——低速/站立 env(命令vx≈0)站着不动 tracking 就高,拉高均值掩盖"没学会走"。必须用 diag_tracking.py 按命令速度分档看实际 vx。

**诊断方法**:用 `/tmp/diag_tracking.py` 拆分——按命令 vx 分档看实际 vx。命令 vx∈[0.5,1.5) 的 env 实际 vx>0.3 才算真学会走。tracking 均值不可信。

**How to apply**:
- 多技能基础模型规模下限 ~2M(d≥192 或 d256×2层起)
- 不要用 d128 训多技能(无论几层)
- 评估走步必须看实际 vx 分档,不能只看 tracking 均值
- 小网络(<2M)即使稳定(episode 高、terminated 低)也可能是"站着装样子",实际 vx≈0

详见 [[mamba-multiskill-geo-preprocess-a8]](几何预处理)、[[mamba-multiskill-eval-strict]](评估口径)。
