# Step 1: 静态姿势力矩补偿验证

> 2026-06-28. 模块化设计, 基于Pinocchio g(q) + MuJoCo接触.
> 脚本: `scripts/verify_getup_torque_comp.py`

## 目标
验证 supine_elbow_sp067_settled 静态姿势, 用 PD + g(q) 补偿, 达到无静差(<2°).

## 模块化设计(GravityCompensator类)
```python
class GravityCompensator:
    """模块化重力补偿, 可扩展:
    Stage 1: g(q) RNEA (当前)
    Stage 2: + 接触力修正 (g_foot/g_pelvis)
    Stage 3: + pin.forwardDynamics (接触力求解)
    Stage 4: + RL policy 驱动
    """
    def compute_gravity(self, qpos) -> np.ndarray:
        # RNEA(q,0,0), 和MuJoCo qfrc_bias一致
```

## 验证结果

### g(q) 符号确认(正确)
g(q)(Pinocchio RNEA) = MuJoCo qfrc_bias(完全一致, diff=0):
- hip_pL: g(q)=-18.21, qfrc_bias=-18.21 ✓
- waist_p: g(q)=+20.94, qfrc_bias=+20.94 ✓
- shoulder_pL: g(q)=-5.04, qfrc_bias=-5.04 ✓

动力学: M·qdd = tau_actuator + qfrc_applied - qfrc_bias
静态(qdd=0): qfrc_applied = g(q) = qfrc_bias → tau_actuator=0 → 无静差 ✓

### PD only vs PD+g(q) 对比(kp/kd不改, 原始值)
| 方案 | max err | 接触力Fz |
|------|---------|---------|
| PD only | 30.0° | 1.9N |
| PD + g(q) | 10.4° | 0.0N |
| 改善 | 2.9x | - |

### 残余误差分析(10.4°)
g(q) 补偿了无接触时的重力, 但残余10.4°来自**接触力未补偿**:
- robot 仰卧背着地, 地面支撑力(被动接触)改变了各关节实际负载
- g(q) 假设base自由(无接触), 不含地面支撑力
- 接触力(地面支撑背)造成的额外力矩没被抵消

关键关节残余误差:
- shoulder_pL: err=0.12(7°) — 手臂重力大, kp=14.25低
- waist_p: err=-0.18(10°) — 腰部, 接触力改变了腰部负载
- hip_pL: err=0.14(8°) — 腿部

## 当前问题: 接触检测失效(robot穿地)
独立MuJoCo脚本和UniLab env都出现robot穿地:
- ncon=0 或 接触力≈0(robot重330N但接触力0-2N)
- base_z从0.30沉到0.078(穿地)
- contype/conaffinity正确(1,1), 但接触力不产生

可能原因:
- switch_to_motor_actuators 改了actuator但不影响contype
- MuJoCo软接触参数(solref/solimp)太软, 穿透深力小
- 多env backend的接触检测可能有特殊处理

## 下一步
1. **修接触检测**(让robot不穿地, 接触力正常~330N)
2. **g(q) + 接触力修正**: 用MuJoCo contact force或pin.forwardDynamics算接触力,
   从g(q)扣减接触力造成的关节力矩: tau = g(q) - J_c^T·lambda
3. **验证无静差**(目标<2°)
4. **模块化扩展**: Stage 2(接触力修正) → Stage 3(forwardDynamics) → Stage 4(RL)

## 关键文件
- `scripts/verify_getup_torque_comp.py`: 模块化验证脚本(GravityCompensator类)
- `src/unilab/control/gravity_comp_controller.py`: 现有GC控制器(已实施三基准)
- `src/unilab/control/pinocchio_model.py`: Pinocchio接口(gravity/forwardDynamics可用)
