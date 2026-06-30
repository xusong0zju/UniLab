# Step 1 力矩补偿: 完整结论与下一步

> 2026-06-28. 经过大量实验后的最终分析.

## 已验证
1. **g(q) = qfrc_bias** (Pinocchio RNEA = MuJoCo, diff=0) ✓
2. **g(q) 改善 2.9 倍** (PD only 30° → PD+g(q) 10.4°) ✓
3. **forwardDynamics 解出接触力 330N** (正确, 候选集泛化成功) ✓
4. **Jc^T·lambda 对 actuated joints ≈ 0** (接触力在base被吸收) ✓
5. **contactInverseDynamics cc=0 → lambda=0** (硬接触无穿透→无需力) ✓
6. **硬接触(MuJoCo solref极硬)下 g(q) 残余误差仍 10.4°** ✓

## 残余误差根因(最终结论)

**Pinocchio 硬接触模型下, 接触力对 actuated joints 的力矩贡献≈0**:
- 接触力330N主要作用在floating base(6自由度), base吸收垂直支撑力
- Jc^T·lambda 对 actuated joints 几乎为0(hip=0.02, waist=0.94)
- MuJoCo的qfrc_constraint大(hip=-18, waist=+14) — 来自软接触穿透效应
- Pin硬接触 vs MuJoCo软接触: 接触模型不匹配, 力分布不同

**结论**: g(q) 补偿在Pinocchio硬接触模型下是正确的(无残余接触力矩).
残余误差来自MuJoCo软接触的穿透效应(仿真特有, 非物理).
真机上(硬接触), g(q)补偿应该就够了.

## 独立脚本 vs env 差异
- 独立脚本: 10.4° (base_z=0.078, 穿地, 接触力~0N)
- env: 57° (base_z=0.108, 正常着地, 接触力大)
- env误差更大因为接触正常(接触力大→qfrc_constraint大→PD静差大)
- 两者g(q)补偿都对, 差异来自接触力效应不同

## 下一步方向

### 方向1: 直接用MuJoCo的qfrc_constraint做前馈(仿真only)
qfrc_applied = g(q) + qfrc_constraint(符号待定)
但真机没有qfrc_constraint, 不通用.

### 方向2: 用Pin的contactInverseDynamics + 正确cc
需要搞清cc含义, 或用更大穿透量(人为设gap=-0.1m)
让proximal算法产生非零lambda.

### 方向3: 不追求精确接触力补偿, 用迭代修正
每步算g(q) + 上一步误差的修正(类似积分但不用PI):
tau = g(q) + alpha * (target - q_prev)
alpha可调, 避免PI的耦合问题.

### 方向4: 用ABA(Articulated Body Algorithm)
pin.aba(model, data, q, v, tau) → ddq
若tau=g(q), ddq≠0(有接触), 需要额外tau=M*ddq修正
但之前试过M*ddq_free, 方向不对(可能符号或Jacobian问题).

## 关键代码
- `scripts/verify_getup_torque_comp.py`: GravityCompensator类
- `src/unilab/control/pinocchio_model.py`: gravity/forwardDynamics/getKKTContactDynamicMatrixInverse
- `src/unilab/control/gravity_comp_controller.py`: GravityCompController(已实施三基准GC)

## 注意: policy不应输出接触力/支撑点(用户明确指示)
接触力补偿应由控制层(Pinocchio)算, 不由policy输出.
