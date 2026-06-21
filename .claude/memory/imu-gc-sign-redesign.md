---
name: imu-gc-sign-redesign
description: IMU-GC sign analysis, gravity_factor A/B test, and final conclusion — gravity_factor makes no difference
metadata: 
  node_type: memory
  type: project
  originSessionId: e3e7c873-dd4c-4902-8120-22b949b4bed9
---

## IMU-GC 符号分析与重设计（2026-06-14 更新）

**核心结论**：gravity `+=` 正确。**gravity_factor 对训练无影响**，可以安全禁用。真正加速的是 τ_disturbance。

### gravity_factor A/B 对比结果（2026-06-14）

| iter | gf=true | gf=false | Δ |
|------|---------|----------|---|
| 500 | 5.75 | 5.26 | -0.5 |
| 1000 | 9.35 | 10.92 | +1.6 |
| 3000 | 234.5 | 232.5 | -1.9 |
| 5000 | 310.5 | 319.0 | +8.5 |
| 10000 | 323.8 | 327.7 | +3.9 |

**关键发现**：
- gravity_factor 对收敛无负面影响，iter 0-3000 两条曲线重合
- 晚期新版微弱领先 ~2-8 点（<3%）
- 之前 iter 1000=3.75 的慢速来自删除 τ_disturbance，不是 gravity_factor 的错
- 当前配置：imu_modulated_gravity=false，IMU 仅用于 swing 检测

### 三控制器符号不需要改
- GC: `+= g(q)` ✅
- CC: `+= g(q)` + `+= C(q,q̇)q̇` ✅
- Contact: `+= g(q)` + `+= C(q,q̇)q̇` + `-= τ_contact` ✅

### 关键文件
- 物理分析: `docs/analy/imu_torque_feedforward_physics_analysis.md`
- 实现文档: `docs/improve/g1_imu_enhanced_gc_implementation.md` §8.12
- Controller: `src/unilab/control/imu_gc_controller.py`
- Env: `src/unilab/envs/locomotion/g1/joystick.py`
- Config: `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_imu_gc_s96.yaml`

[[g1-imu-gc-impl]]
[[g1-coriolis-comp-impl]]
[[g1-gravity-comp-impl]]