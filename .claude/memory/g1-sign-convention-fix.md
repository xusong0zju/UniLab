---
name: g1-sign-convention-fix
description: G1 dynamics compensation sign convention fix (2026-06-13)
metadata: 
  node_type: memory
  type: project
  originSessionId: e13282b2-6267-4105-81e2-7c26d13944bb
---

## 符号修正（commit 4559d482 "临时保存"）

MuJoCo 动力学：`Mq̈ = ctrl - qfrc_bias + qfrc_constraint`
正确的前馈补偿：`ctrl = PD - g(q) + τ_contact`

### 修正内容

| 控制器 | 项 | 修正前 | 修正后 | 原因 |
|--------|-----|--------|--------|------|
| GravityComp | gravity | `+= g(q)` | `-= g(q)` | 抵消 qfrc_bias |
| CoriolisComp | gravity | `+= g(q)` | `-= g(q)` | 同上 |
| CoriolisComp | coriolis | `+= C(q,q̇)q̇` | `-= C(q,q̇)q̇` | 同上 |
| ContactComp | gravity | `+= g(q)` | `-= g(q)` | 同上 |
| ContactComp | coriolis | `+= C(q,q̇)q̇` | `-= C(q,q̇)q̇` | 同上 |
| ContactComp | contact | `-= τ_contact` | `+= τ_contact` | GRF 支撑身体，抵消 qfrc_constraint |
| IMU-GC | gravity | `+= g(q)·mask` | `-= g(q)·mask` | 同 GravityComp |
| IMU-GC | disturbance | `+= τ_disturbance` | `-= τ_disturbance` | 抵消残余力 |

### 状态

- ✅ 代码已修改并 commit（4559d482）
- ⚠️ **尚未重新训练验证** — 修正后需要重新跑 A/B 对比
- 之前所有 A/B 结果（见 [[g1-coriolis-comp-impl]]）是基于错误符号跑的，但 GC "过补偿"效应依然成立（符号错误导致 +=g 等于 -=(-g)，方向一致只是物理含义不同）

### 待做

- 用修正后的符号重新训练各方案，验证 reward 变化
- 特别关注 GC 是否依然最优（理论上修正后补偿更精确，reward 应更好）
- [[g1-coriolis-comp-impl]] 中记录的数值结果需更新
