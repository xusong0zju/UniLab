---
name: g1-imu-gc-impl
description: G1 IMU-Enhanced GC implementation — swing boost + disturbance correction
metadata: 
  node_type: memory
  type: project
  originSessionId: e3e7c873-dd4c-4902-8120-22b949b4bed9
---

## IMU-Enhanced GC (IMU-GC) 实现

**公式**：`τ = PD + gravity_scale·g(q)·(mask + swing_boost·swing_mask) + disturbance_scale·τ_disturbance`

**核心创新**：保持 GC 的有益过补偿（stance 不变），利用 IMU 残余加速度检测 swing 腿并增强补偿。

### 配置参数
- swing_boost: 0.3（swing 腿 GC 增强 30%）
- disturbance_scale: 0.2（扰动修正系数）
- disturbance_threshold: 5.0 N（激活阈值）

### Swing 检测
- gait_phase 先验：phase > π → swing
- IMU 交叉验证：f_residual_z < -mg/2 → stance
- 最终：swing = gait_says_swing AND NOT imu_says_stance

### 关键文件
- IMUGravityCompController: `src/unilab/control/imu_gc_controller.py`
- G1WalkFlatIMUGCEvn: `src/unilab/envs/locomotion/g1/joystick.py`
- Config: `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_imu_gc.yaml`
- 文档: `docs/improve/g1_imu_enhanced_gc_implementation.md`

### 训练状态
- ✅ 4096 envs × 5000 iters，训练完成
- **最终 reward: 317.53**（与 GC 316.51 持平，略高）
- **收敛速度最快**：iter 2000 时 reward 59.4，远超 GC 46.4（+28%）

### 降参实验结果
| 配置 | Actor参数 | Reward |
|------|----------|--------|
| IMU-GC 128×2 | 285K (100%) | 317.53 |
| **IMU-GC 96×2** | **165K (58%)** | **312.17** |
| IMU-GC 64×2 | 77K (27%) | 289.89 |
| IMU-GC 96×1 | 90K (32%) | 300.05 |
| Baseline 96×2 | 165K (58%) | 300.65 |

**核心结论**：IMU-GC 96×2 是最佳性价比（165K 参数，312 reward），超过全量 Baseline (304.67)+7.5
**前馈补偿支撑降参**：同样 165K，IMU-GC 312 vs Baseline 301，差 +11.5

### 设计决策
- 不使用 qacc 估计（CTCP 已证明精度不够）
- 不使用特权信息（足底力传感器不可 sim-to-real）
- IMU 只用于：1) swing 检测（加速度 → 接触判断）2) 扰动修正（残余力 → Jacobian → 关节空间）
- [[g1-coriolis-comp-impl]] 六方案对比中 GC 最优 → IMU-GC 在 GC 基础上增强

### 10k iters 延长训练
- IMU-GC 96x2 从 5k→10k: 312.17 → **325.92** (+13.75)
- 10k 时 96x2 已超过 5k 时 128x2 (317.53)
- 小网络只是收敛更慢，天花板未降低
- **建议: IMU-GC 96x2 训练 10k iters**
