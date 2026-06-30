# Step 1 力矩补偿: 最终结论与模块化设计

> 2026-06-28 最终验证完成。 elbow=0.3, 手悬空, g(q)补偿正确性已验证.

## 核心结论

**g(q) = rnea(q,0,0)[6:] 是正确的静态力矩补偿**

验证方法：`forwardDynamics(tau=full_rnea, Jc, a0=0) → ddq=0`

- rnea(q,0,0) 返回完整力矩（含 base 6 维 + actuated 29 维）
- forwardDynamics 解 KKT 系统，tau=rnea=h 时 ddq=0（静态平衡）
- **g(q) = rnea[6:] 对 actuated joints 是完整正确的补偿**

## 六项验证结果（elbow=0.3 最终姿态）

| 验证项 | 结果 |
|--------|------|
| 1. g(q) = qfrc_bias | diff=0 ✓ |
| 2. 静态平衡 (actuator+g+qconst-qbias=0) | max=0 ✓ |
| 3. err = -qfrc_constraint/kp | 所有关节完全匹配 ✓ |
| 4. forwardDynamics ddq=0 (Pinocchio硬接触) | ddq=0 ✓ |
| 5. 手悬空 | 左0.078m 右0.077m ✓ |
| 6. 左右对称 | hip diff=0.0006, shoulder diff=0.0022, elbow diff=0.0003 ✓ |

## 最终姿态参数（elbow=0.3）

| 关节 | 目标值 | 说明 |
|------|--------|------|
| hip_pitch L/R | -0.4 | 腿屈 |
| knee L/R | 0.9 | 膝弯 |
| ankle L/R | 0.0 | 默认 |
| waist (yaw/roll/pitch) | 不控制 | 自由（仅 g(q)+kd 阻尼） |
| shoulder_pitch L/R | 0.67 | 双肩往后（对称） |
| shoulder_roll L/R | 0.0 | 对称（不外展） |
| **elbow L/R** | **0.3** | 更弯，确保手悬空（余量 0.078m） |
| wrist | 0.0 | 默认 |

## settle 后误差表（MuJoCo 软接触, elbow=0.3）

| 关节 | 目标 | 实际 | 误差° | kp | 说明 |
|------|------|------|--------|-----|------|
| hip_pL/R | -0.400 | -0.544 | ±8.2° | 40.2 | MuJoCo软接触残余 |
| knee L/R | 0.900 | 0.872 | ±1.6° | 99.1 | kp高误差小 |
| ank_p L/R | 0.000 | -0.005 | 0.3° | 28.5 | 小 |
| wst_pitch | 自由 | 自然态 | — | 0 | 不控制 |
| shldr_p L/R | 0.670 | 0.521/0.523 | ±8.5° | 14.3 | kp最低 |
| shldr_r L/R | 0.000 | ±0.064 | ±3.7° | 14.3 | |
| shldr_y L/R | 0.000 | ±0.075 | ±4.3° | 14.3 | |
| elbow L/R | 0.300 | 0.311/0.310 | -0.6° | 14.3 | 余量大 |
| wrist | 0.000 | ±0.028 | ±1.6° | 25 | 小 |

**残余误差来源**：err = qfrc_constraint / kp（已逐关节验证完全匹配）

- MuJoCo 软接触（solref=[0.02, 1.0]）有穿透 → qfrc_constraint ≠ 0
- Pinocchio 硬接触无穿透 → qfrc_constraint = 0 → err = 0
- 真机（硬接触）上 g(q) 补偿应接近无静差

## 力臂与手部接触分析

elbow=0.5 时手最低边=0.001m（几乎着地），手部接触力虽小(1N)但力臂大，导致额外 qfrc_constraint 使 shoulder 误差增大。

**elbow=0.3 后手最低边=0.078m（悬空，余量大）**，消除手部接触力，姿态正确。

## 力矩补偿的明确定义

**静态补偿 = 各部件重力导致的力矩与相互间的正确传递**

- 用 Pinocchio RNEA（rnea）正确计算
- rnea 从叶子到根，逐关节累积子树重力矩
- g(q) = rnea[6:]（actuated joints 的重力补偿力矩）
- 方向正确（= qfrc_bias，验证 diff=0）
- 含完整的重力传递（各连杆重力通过运动链正确传递到各关节）

## 分阶段控制

### 阶段1：肘撑静态保持（当前）
- **腿（hip/knee/ankle）**：PD + g(q)（位置控制 + 重力补偿）
- **waist（yaw/roll/pitch）**：仅 g(q) + kd 阻尼（无位置目标，物理自然平衡）
- **肩/肘/腕**：PD + g(q)（位置控制 + 重力补偿）
- elbow = 0.3（手悬空，余量 0.078m）

### 阶段2：蹬腿起身（后续）
- **waist** 恢复位置控制，**从实际状态值开始**（不从预设目标突变）
- 平滑过渡：`target = actual + alpha * (goal - actual)`，alpha 渐增

## 模块化控制器

`src/unilab/control/modular_torque_controller.py` — `ModularTorqueController`

- τ = PD(q_d, q, q̇) + g(q)
- waist 分阶段控制（free/PD + 平滑过渡）
- 向后兼容（不传 gc_weights 回退到 g(q) only）
- 可扩展（Stage 2 起身、Stage 3 接触修正、Stage 4 RL）

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/unilab/control/modular_torque_controller.py` | 模块化力矩补偿控制器 |
| `src/unilab/control/pinocchio_model.py` | Pinocchio 接口（gravity/rnea/forwardDynamics） |
| `docs/improve/getup_step1_final_conclusion.md` | 本文档 |
| `logs/getup_keyframes/step1_verified_4view.png` | 四视角验证图 |
| `logs/getup_keyframes/step1_verified_side.png` | 侧视图 |
| `logs/getup_keyframes/step1_verified_front.png` | 前视图 |
| `logs/getup_keyframes/step1_verified_top.png` | 俯视图 |
| `logs/getup_keyframes/step1_verified_front3q.png` | 3/4视角 |
