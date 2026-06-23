---
name: mamba-multiskill-a11-token-split-fail
description: "A11 token语义分离+去meanpool失败: 只取task token丢失body信息, 走步能力退化(原版能走,改正版vx≈0)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

2026-06-23 A11 Step1+2(token 语义分离 + 去 mean pool)实验,**负面结论**。

**改动**(`src/unilab/algos/torch/flash_sac/mamba_actor.py`):
- Step1: token 从 4(同obs复制)改为 2(body 125维 + task 5维,语义分离)
- Step2: 去 mean pool,改取 task token 输出(index 1)
- 参数 3.9M→3.77M

**结果**(fresh start, Phase I config 大Mamba+几何, 8000 iter):
| checkpoint | 实际vx(命令0.65档) | tracking均值 |
|---|---|---|
| model_3000 | -0.004 | 0.245 |
| model_5000 | 0.004 | 0.097(退化) |

**对比原版 Phase I(4-token同语义+mean pool)**:能学会走(实际vx>0.3)。

**结论**:
1. **A11 Step1+2 改正破坏了原版走步能力**——改正版 vx≈0,原版能走
2. **根因:只取 task token 丢失了 body token 的 SSM 处理结果**。原版 mean pool 融合 body+task,改正版只取 task(body 信息经 SSM 后没被用到)
3. token 语义分离方向需要重新设计——不能简单丢 body token。可选:
   - (a) 两 token 都过 SSM 后 **concat** 而非取 task
   - (b) 加 [CLS] token 聚合
   - (c) body token 放最后取(因 SSM 因果性,后 token 能看到前 token)

**How to apply**:
- A11 Step1+2(取 task token)**不可用**,已回退代码
- token 分离若再做,用 concat/CLS 聚合,不能丢 body token
- 走步瓶颈不在 token 设计,在别处(容量?reward?或原版 mean pool 已够好)
- 原 Phase I(4-token+mean pool)仍是最佳大Mamba配置

详见 [[mamba-multiskill-param-floor]](容量下限)、[[mamba-multiskill-geo-preprocess-a8]](几何)。
