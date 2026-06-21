---
name: g1-coriolis-comp-impl
description: G1 dynamics compensation final results (GC+CC+CTC+IMU+CTCP)
metadata: 
  node_type: memory
  type: project
  originSessionId: e3e7c873-dd4c-4902-8120-22b949b4bed9
---

## 最终 A/B 结果（5000 iters, FlashSAC, 4096 envs）

```
GC 323.68 > CTC(IMU) 319.84 > CC 317.24 > CTCP 313.38 > Baseline 308.23 >> CTC(force) 302.10
```

**GC 是最优方案**（简单有效），CTC(IMU) 紧随其后且提供 sim-to-real 保障。

## 六方案总结

| 方案 | reward | 特点 |
|------|--------|------|
| GC | 323.68 | 重力补偿，"过补偿"效应有益 |
| CTC(IMU) | 319.84 | IMU残余+pelvis Jacobian，sim-to-real直接可用 |
| CC | 317.24 | 重力+Coriolis，ang_vel弱点 |
| CTCP | 313.38 | 特权qfrc_constraint，qacc估计误差限制 |
| Baseline | 308.23 | 纯PD |
| CTC(force) | 302.10 | force sensor含内部力，不可用 |

## IMU 辅助方案

`_compute_contact_torque()`：用 IMU 加速度计检测残余加速度（R·a_sensor - [0,0,9.81]），
乘以总质量得到残余力，通过 pelvis site Jacobian 映射到关节空间。
站立时残余≈0，动态时提供微弱但关键的修正。

## CTCP 特权方案

用 Pinocchio RNEA 反推 `qfrc_constraint = RNEA(q,q̇,q̈) - τ_actuator`。
qacc 从 qvel 差分估计（控制步率 + EMA 滤波），精度不足以完全发挥特权信息优势。
**要真正发挥优势，需扩展 BatchEnvPool 支持读取 qfrc_constraint。**

## 关键文件

- ContactCompController: `src/unilab/control/contact_comp_controller.py`
- CTC(IMU) env: `src/unilab/envs/locomotion/g1/joystick.py` G1WalkFlatCTCEvn
- CTCP env: 同上 G1WalkFlatCTCPvn
- PinocchioDynamicsModel.constraint_force(): `src/unilab/control/pinocchio_model.py`
- CTC config: `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_ctc.yaml`
- CTCP config: `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_ctc_priv.yaml`
- 文档: `docs/improve/g1_dynamics_compensation_implementation.md` §16-18
