# Step 1 最终结论: 残余误差 = qfrc_constraint / kp

> 2026-06-28 最终分析

## 残余误差根因(数学证明)

静态平衡方程(完全验证):
```
actuator + g(q) + qfrc_constraint - qfrc_bias = 0
```
g(q) = qfrc_bias (验证 diff=0) → actuator = -qfrc_constraint
actuator = kp × (target - actual) → **err = -qfrc_constraint / kp**

验证(完全吻合):
| joint | qfrc_constraint | kp | 预测err | 实际err |
|-------|----------------|-----|---------|---------|
| hip_pL | -5.72 | 40.2 | +0.142 | +0.142 ✓ |
| waist | +5.18 | 28.5 | -0.182 | -0.182 ✓ |
| shoulder_pL | -1.72 | 14.3 | +0.120 | +0.120 ✓ |

## 核心矛盾

- g(q) 正确补偿了重力(qfrc_bias) ✓
- 接触力矩(qfrc_constraint)由PD扛, 有限增益→静差
- **Pinocchio硬接触的Jc^T·lambda≈0**(base吸收), 不等于MuJoCo的qfrc_constraint
- **contactInverseDynamics** cc=0→lambda=0(硬接触无穿透→无力)
- 改kp(你说不改)、PI(耦合振荡)、kp=0(更差)都不行

## 用户提示
- waist在肘撑时不需要给kp/kd(让物理自然平衡) — 试了, 其他关节仍有误差
- 接触力有切向(不只法向) — 验证了, 但Jc^T·f_mujoco也≈0
- 不用PI控制
- policy不输出接触力/支撑点

## 待解决
如何在不改kp、不用PI、不依赖MuJoCo接触力的前提下, 消除PD静差?
