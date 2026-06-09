---
name: g1-gravity-comp-impl
description: G1 + FlashSAC Pinocchio gravity compensation training-side implementation (completed)
metadata: 
  node_type: memory
  type: project
  originSessionId: 8bd6825e-9f4d-4a51-b690-2b467fbb06bf
---

## 状态：训练侧实现已完成，待跑 A/B 训练对比

### 已完成的工作

1. **`src/unilab/control/` 模块**（8个文件）：
   - `base.py` — MotorController ABC
   - `pd_controller.py` — 纯 PD 力矩控制
   - `gravity_comp_controller.py` — PD + 重力前馈补偿 τ = kp(q_d-q) - kd·q̇ + g(q)
   - `ctc_controller.py` — CTC 桩（NotImplementedError）
   - `pinocchio_model.py` (~310行) — 从 MuJoCo model 程序化构建 Pinocchio 模型，批量重力计算
   - `actuator_switch.py` — position actuator → motor actuator 运行时切换，返回 MotorActuatorInfo
   - `resolver.py` — 控制器工厂 resolve_controller()

2. **Backend 扩展**：
   - `base.py` 新增 `get_full_qpos()` / `get_full_qvel()` 抽象方法
   - `mujoco/backend.py` 实现上述方法

3. **G1WalkFlatGC 环境**（`src/unilab/envs/locomotion/g1/joystick.py`）：
   - 继承 G1BaseEnv（非 G1WalkEnv，因为需在 materialize 前切换 actuator）
   - G1WalkGCControlConfig: action_scale, gravity_comp_mask, gravity_scale
   - G1WalkGCDomainRandomizationProvider: kp/kd DR 迁移到 env 中（与 Go2W 一致）
   - `_pre_step_motor_control()` 回调：读 qpos/qvel → Pinocchio g(q) → τ = PD + gravity
   - `_init_action_space()` 返回 [-1,1]^29（policy 输出目标角，不是力矩）

4. **训练配置**：`conf/offpolicy/task/flashsac/g1_walk_flat/mujoco_gc.yaml`
   - gravity_comp_mask: 腿+腰补偿，手臂跳过
   - 超参数与 baseline (mujoco.yaml) 完全一致

5. **依赖**：`pyproject.toml` 添加 `pin>=3.1.0`

6. **文档**：`docs/improve/g1_gravity_compensation_implementation.md`（487行，8章节）

### 验证结果

- Pinocchio 重力向量与 MuJoCo RNE 对齐：max abs diff ≈ 1e-16
- G1WalkFlatGC 4 envs 运行 50 步：100% 存活，力矩输出合理
- 回归测试：1121 passed, 60 skipped，无退化

### 关键设计决策

- Pinocchio 模型从 MuJoCo model 程序化构建（非 URDF），保证 1:1 对齐
- 选择性重力补偿 mask：腿+腰(index 0-14)补偿，手臂(index 15-28)跳过（forcerange 窄 ±5Nm，截断引入非线性）
- G1 的力矩限制在 actuator_forcerange（非 ctrlrange，G1 ctrlrange=[0,0]）
- 切换 motor actuator 后设 ctrlrange=forcerange
- 继承 G1BaseEnv 而非 G1WalkEnv（需在 super().__init__ 前切换 actuator）

### 待做

- **A/B 训练对比**：G1WalkFlat vs G1WalkFlatGC，关注收敛速度与指标质量
- Coriolis 补偿、CTC 控制器、numba 加速、部署侧 C++ 导出

### 启动命令

```bash
# Baseline
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco
# Gravity Compensation
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco_gc
```
