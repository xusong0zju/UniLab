# Mamba SSM Actor 结构与意义——3层深度分析

> 基于 `src/unilab/algos/torch/flash_sac/mamba_actor.py` 实际代码
> 背景:G1 多技能统一策略,验证"网络深度 vs 宽度"对学会走步的影响

## 1. 整体数据流

```
obs(130维: base98 + geo3 + sym29)
  → token_proj: Linear(130 → d_model × n_tokens)        # 升维
  → reshape: (B, n_tokens=4, d_model)                    # 4个token
  → + pos_emb (可学习位置编码)
  → MambaBlock × N (N=2 或 3)                            # 串联N层
  → mean pool over tokens: (B, d_model)                  # 跨token平均
  → post_norm (RMSNorm)
  → predictor (NormalTanhPolicy): d_model → action(29)   # 策略头
```

**关键点**:
- 4 个 token 是**同一 obs 的不同投影**(不是 4 个不同观测),靠 token_proj 把 130 维映射到 `4×d_model` 再 reshape。位置编码区分 4 个 token 的"角色"。
- N 层 MambaBlock **串联**,每层输出 = 输入 + 残差(残差连接)。
- 最后 mean pool 把 4 个 token 融合成单个 d_model 向量,再出动作。

## 2. 单个 MambaBlock 结构

每层 MambaBlock 的内部数据流(`forward`):

```
x (B, L=4, d_model)
  │
  ├─ residual = x                          # 残差备份
  │
  ├─ RMSNorm(d_model)                      # 归一化(比LayerNorm省参数)
  │
  ├─ in_proj: Linear(d_model → 2×inner_dim)   # 投影+门控分支
  │   ├─ x_in  (inner_dim = d_model × expand=2)   # 主分支
  │   └─ z     (inner_dim)                          # 门控分支
  │
  ├─ 主分支 x_in:
  │   ├─ Conv1d(depthwise, kernel=4)       # 局部时序卷积(因果padding)
  │   ├─ SiLU 激活
  │   └─ SelectiveSSM(inner_dim, d_state=16)  # 核心:选择性状态空间
  │         h_t = Δ_t · A · h_{t-1} + Δ_t · B · x_t    # 状态递归
  │         y_t = C · h_t                                # 读出
  │         (Δ, B, C 都是 input-dependent,这就是"selective")
  │
  ├─ 门控: x_out = SSM输出 * SiLU(z)      # 门控融合
  │
  ├─ out_proj: Linear(inner_dim → d_model)  # 降回 d_model
  │
  └─ return out_proj(x_out) + residual      # 残差连接
```

**SelectiveSSM 的"选择性"**:Δ(步长)、B(输入权重)、C(读出权重)都由当前输入 `x_t` 经投影决定,不是固定参数。这让 SSM 能**根据输入内容动态决定"记住多少、遗忘多少"**——这是 Mamba 区别于传统 RNN(LSTM/GRU 固定门控)的核心。

## 3. 3层 vs 2层:结构差异

### 3层配置(中B: d_model=128, n_layers=3, 1.04M 参数)
```
obs → token_proj → [Block1] → [Block2] → [Block3] → mean_pool → action
                    128→256     256→256     256→128
```
每层:RMSNorm + in_proj(128→512,因expand=2×2分支) + Conv1d + SSM(d_state=16) + out_proj(256→128) + 残差。

### 2层配置(小Mamba: d_model=128, n_layers=2, 0.73M 参数)
```
obs → token_proj → [Block1] → [Block2] → mean_pool → action
```
少一层 SSM 递归。

### 参数量来源(3层比2层多 0.31M)
每多一层 MambaBlock 增加:
- in_proj: 128 × 512 = 65,536
- out_proj: 256 × 128 = 32,768
- Conv1d: 128 × 4 = 512
- SSM (Δ/B/C/A): ~128 × 16 × 3 ≈ 6,144
- 合计 ~0.105M/层

3层比2层多 1 层 ≈ +0.105M,但 token_proj/d_model 相关项也让 3层总参数升到 1.04M(2层 0.73M)。

## 4. 3层的意义:为什么深度可能比宽度重要

### 4.1 SSM 递归深度 = 时序建模能力
Mamba 的核心优势是**跨时间步的隐藏状态递归**(h_t = f(h_{t-1}, x_t))。每经过一个 MambaBlock,隐藏状态被"精炼"一次:
- 第1层:从原始 token 提取初级时序模式
- 第2层:在初级模式上提取高阶模式
- 第3层:进一步组合,捕捉更长程依赖

走步是**强时序任务**:步态周期(swing→stance→swing)、腾空相位、落地缓冲,这些需要网络"记住"过去几步的状态来决定当前动作。3层 SSM 递归比 2 层能建模更长的时序依赖。

### 4.2 残差连接让深度可训练
每层 `return x + residual`,梯度能直接回传到早期层,避免深网络梯度消失。所以 3 层虽深但可训(虽然收敛比 2 层慢,你观察到的"3层不易收敛"是正常的)。

### 4.3 深度 vs 宽度的权衡(本实验核心)
- **宽度(d_model 大)**:每层表达力强,但层数少,时序递归浅
- **深度(n_layers 大)**:每层窄,但 SSM 递归次数多,时序建模强

对走步这种时序任务,**深度可能更关键**——因为步态的"周期性+相位"本质是时序结构,需要深递归来捕捉。这也是为什么大Mamba(3层×d256)能学会走,而小Mamba(2层×d128)学不会——不仅是参数量,更是层数(3 vs 2)。

中B 实验(3层×d128, 1.04M)正是验证:**在参数接近时,3层窄是否优于2层宽**。如果中B 学会走而小Mamba(2层)学不会,证明深度是关键;如果中B 也学不会,说明 d128 太窄(宽度也有下限)。

## 5. 当前实验配置对比

| 配置 | d_model | n_layers | 参数 | 能否走 | 说明 |
|---|---|---|---|---|---|
| 小Mamba v1 | 128 | 2 | 0.73M | ❌ vx≈0 | 浅窄,容量不足 |
| 小Mamba v3 | 128 | 2 | 0.73M | ❌ vx≈0 | +alive路由,仍不行 |
| 中B | 128 | **3** | 1.04M | ⏳训练中 | 同d128但深一层 |
| 中A(已停) | 192 | 2 | 1.54M | 未验证 | 浅宽,被中B替换 |
| 大Mamba | 256 | 3 | 3.9M | ✅ 会走 | 基准(Phase I) |

## 6. 已知设计局限(诚实)

1. **4 token 是同语义复制**:token_proj 把同一 obs 投影成 4 份,token 间无语义差异(不像 HuMam 的 robot⊗task 2-token 分离)。SSM 的选择性门控无从"选择"不同信息源。这是 memory 里记的 A11 硬伤。
2. **mean pool 丢失时序**:最后 `x.mean(dim=1)` 把 4 token 平均,丢失了 SSM 在 token 序列上的递归成果。A11 建议改用最后 token 或保留 hidden state。
3. **token 内 SSM 递归只有 4 步**:L=4(token数),SSM 只递归 4 步,时序建模能力远不如序列任务(语言模型 L=上千)。这是把 Mamba 当 MLP 用的局限。

这些局限意味着:当前 MambaActor 的"深度"优势可能没完全发挥——因为 token 序列太短(L=4),3 层 SSM 递归的累积效果有限。真正释放 Mamba 时序优势需要长序列(如 TD-MPC2 的 latent dynamics 多步 rollout,见 Part B 调研)。

## 7. 结论(待中B 验证)

3层 Mamba 的结构意义:
- **SSM 递归深度增加** → 时序建模能力提升,理论上利于走步等时序任务
- **残差连接保证可训** → 深度不会导致梯度消失,但收敛会慢(正常)
- **参数效率** → 3层×d128(1.04M)比2层×d192(1.54M)参数少 32%,但深度多一层

如果中B 学会走,证明**深度比宽度更重要**——多技能基础模型应优先加深度而非宽度,这对控制参数规模(你的多技能路线)有指导意义。如果中B 也学不会走,说明 d128 宽度本身不足,需加宽。

无论结果如何,当前 MambaActor 的 4-token 同语义复制 + mean pool 设计限制了 Mamba 优势的发挥(A11),后续应结合 Part B(TD-MPC2 latent dynamics 用 Mamba 做长序列递归)才能真正释放 Mamba 的时序建模能力。
