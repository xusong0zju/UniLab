# UniLab 技能与工作流速查

> 本文件由 Claude Code 维护，记录项目常用命令、开发工作流、关键模式与调试技巧，供新 session 快速上手。

---

## 1 常用命令

### 1.1 环境与依赖

```bash
uv sync                          # 同步依赖（首次 / pyproject.toml 变更后）
uv sync --extra motrix           # 含 Motrix 后端
make sync-rocm                   # AMD ROCm 环境
```

### 1.2 代码质量

```bash
make format                      # ruff format + ruff check --fix
make type                        # mypy + pyright
make check                       # format + type（提交前必跑）
make test                        # 非慢速测试
make test-cov                    # 测试 + 覆盖率报告
make test-slow                   # 慢速集成/训练冒烟测试
make test-all                    # check + test-cov（PR gate 必过）
make clean                       # 清理缓存与临时文件
```

### 1.3 训练启动

```bash
# PPO (RSL-RL)
uv run python scripts/train_rsl_rl.py --config-name=ppo task=<task_config>

# MLX PPO (Apple Silicon)
uv run python scripts/train_mlx_ppo.py --config-name=ppo_mlx task=<task_config>

# APPO (异步 PPO)
uv run python scripts/train_appo.py --config-name=appo task=<task_config>

# FlashSAC / SAC / TD3 (off-policy)
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/<task>

# G1 Walk Baseline
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco

# G1 Walk + Gravity Compensation
uv run python scripts/train_offpolicy.py --config-name=offpolicy task=flashsac/g1_walk_flat/mujoco_gc
```

### 1.4 可视化与回放

```bash
# 交互式回放
uv run python -m unilab.visualization.playback --logdir=<path>

# 批量渲染
uv run python -m unilab.visualization.render_many --logdir=<path>
```

### 1.5 Git & PR

```bash
git checkout -b feat/xxx         # 创建功能分支
make test-all                    # 提交前验证
git add -A && git commit -m "feat: xxx"
git push -u origin feat/xxx
gh pr create --title "feat: xxx" --body "..." --base main
```

PR Gate：工作树干净 + `make test-all` 通过（除非用户 override）。

---

## 2 项目结构速览

```
UniLab/
├── conf/                         # Hydra 配置
│   ├── ppo/                      #   PPO 算法配置
│   ├── appo/                     #   APPO 算法配置
│   ├── offpolicy/                #   Off-policy (SAC/TD3/FlashSAC)
│   │   ├── algo/                 #     算法超参 (flashsac.yaml, sac.yaml, td3.yaml)
│   │   └── task/flashsac/        #     任务配置
│   │       ├── g1_walk_flat/     #       G1 步行 (mujoco.yaml, mujoco_gc.yaml)
│   │       └── go2_joystick_flat/
│   └── ppo_him/                  #   PPO-HiM 配置
├── scripts/                      # 训练入口
│   ├── train_rsl_rl.py           #   PPO
│   ├── train_mlx_ppo.py          #   MLX PPO
│   ├── train_appo.py             #   APPO
│   ├── train_offpolicy.py        #   FlashSAC / SAC / TD3
│   └── train_hora_distill.py     #   HORA 蒸馏
├── src/unilab/
│   ├── base/                     # 核心抽象
│   │   ├── np_env.py             #   Env contract: NpEnvState.obs=dict, reset→(obs,info)
│   │   └── backend/
│   │       ├── base.py           #   SimBackend 抽象接口
│   │       ├── mujoco/           #   MuJoCo 后端
│   │       └── motrix/           #   Motrix 后端
│   ├── control/                  # 力矩控制模块
│   │   ├── base.py               #   MotorController ABC
│   │   ├── pd_controller.py      #   纯 PD
│   │   ├── gravity_comp_controller.py  # PD + 重力前馈
│   │   ├── ctc_controller.py     #   CTC (stub)
│   │   ├── pinocchio_model.py    #   Pinocchio 刚体动力学
│   │   ├── actuator_switch.py    #   Position→Motor actuator 切换
│   │   └── resolver.py           #   控制器工厂
│   ├── envs/locomotion/          # 运动控制环境
│   │   ├── common/               #   共享工具 (rotation.py, math.py)
│   │   ├── g1/                   #   G1 人形
│   │   ├── go2/                  #   Go2 四足
│   │   └── go2w/                 #   Go2W (motor actuator 参考)
│   ├── algos/                    # 算法实现
│   ├── dr/                       # Domain Randomization
│   │   ├── provider.py           #   LocomotionDRProvider
│   │   ├── manager.py            #   DR 管理器
│   │   └── types.py              #   ResetPlan 等
│   ├── ipc/                      # 异步 IPC
│   │   └── async_runner.py       #   异步 Runner
│   ├── training/                 # 训练辅助
│   │   └── run.py                #   训练运行 helper
│   ├── visualization/            # 可视化与回放
│   └── structured_configs.py     # Hydra 配置 schema
├── tests/                        # 测试套件
├── docs/                         # 文档
│   ├── improve/                  #   改进文档 (含 gravity compensation)
│   └── sphinx/                   #   Sphinx 文档源
└── Makefile                      # 构建/测试/清理
```

---

## 3 关键开发模式

### 3.1 pre_step_control 回调模式（Go2W 验证）

用于在 MuJoCo step 前介入 ctrl 计算，实现 motor actuator + 自定义力矩：

```python
# 1. Env __init__ 中注册回调
self._backend.set_pre_step_control(self._pre_step_motor_control)

# 2. 回调签名
def _pre_step_motor_control(self, backend, policy_ctrl) -> np.ndarray:
    """每个 substep 前被 backend 调用。返回实际施加的 ctrl (力矩)。"""
    joint_pos = backend.get_dof_pos()
    joint_vel = backend.get_dof_vel()
    full_qpos = backend.get_full_qpos()
    full_qvel = backend.get_full_qvel()
    tau = self._controller.compute(policy_ctrl, joint_pos, joint_vel,
                                    full_qpos=full_qpos, full_qvel=full_qvel)
    return tau
```

**适用场景**：重力补偿、Coriolis 补偿、CTC、自定义力矩控制。

### 3.2 Actuator 切换模式

将 MuJoCo position actuator → motor actuator，使 env 接管力矩计算：

```python
from unilab.control import switch_to_motor_actuators

# 在 backend.materialize() 之前调用
actuator_info = switch_to_motor_actuators(backend._model)
# → MotorActuatorInfo(kp, kd, force_lower, force_upper)

# 之后创建自定义 controller
from unilab.control import GravityCompController
controller = GravityCompController(
    dynamics_model=..., kp=actuator_info.kp, kd=actuator_info.kd,
    force_lower=actuator_info.force_lower, force_upper=actuator_info.force_upper,
    gravity_comp_mask=..., gravity_scale=1.0,
)
```

**注意**：
- 必须在 `backend.materialize()` 之前调用（model 修改后 pool 编译使用新参数）
- G1 的 `ctrlrange=[0,0]`，力矩限制在 `forcerange`，`switch_to_motor_actuators` 会自动设置 `ctrlrange=forcerange`
- 切换后 kp/kd DR 必须在 env 中实现（MuJoCo model 不再持有增益）

### 3.3 DR 迁移模式（kp/kd 从 backend 迁移到 env）

当使用 motor actuator 时，MuJoCo model 的 gainprm 已被改为 1.0，backend DR 无法处理 kp/kd：

```python
class CustomDRProvider(LocomotionDRProvider):
    def _get_base_actuator_gains(self, env):
        return None, None  # 不走 backend DR

    def build_reset_plan(self, env, env_ids):
        # kp/kd 在 env 中处理
        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)

        # backend 只处理 mass/gravity/friction 等
        return ResetPlan(
            randomization=_build_backend_reset_randomization(env, num_reset),
        )
```

### 3.4 Pinocchio 模型构建（冷路径）

从 MuJoCo model 程序化构建，无需 URDF：

```python
from unilab.control import PinocchioDynamicsModel

dynamics_model = PinocchioDynamicsModel(mj_model)
# 一次性构建，后续调用 gravity()/rnea()/coriolis()
gravity = dynamics_model.gravity(qpos_batch, qvel_batch)  # shape (num_envs, nv_actuated)
```

**关键转换**：
- 四元数：MuJoCo (w,x,y,z) ↔ Pinocchio (x,y,z,w)
- 惯性：MuJoCo `diaginertia` + `iquat` → Pinocchio 完整 3×3 矩阵
- armature：`model.armature[:] = mj.dof_armature`

### 3.5 Config 组合模式

Hydra 配置通过 task 路径选择 owner YAML：

```bash
# task 路径 = conf/<entry>/task/<algo>/<task_name>/<backend>.yaml
task=flashsac/g1_walk_flat/mujoco        # → G1WalkFlat
task=flashsac/g1_walk_flat/mujoco_gc     # → G1WalkFlatGC

# 后端切换通过 owner YAML，不是 override training.sim_backend
# ✗ training.sim_backend=motrix  (不能单独 override)
# ✓ task=flashsac/g1_walk_flat/motrix   (选对应的 owner YAML)
```

### 3.6 Backend 扩展模式

添加 backend 专有方法时，必须先在 `SimBackend` 基类声明：

```python
# src/unilab/base/backend/base.py
class SimBackend(ABC):
    def new_method(self) -> ...:
        raise NotImplementedError  # 先在基类声明

# src/unilab/base/backend/mujoco/backend.py
class MuJoCoBackend(SimBackend):
    def new_method(self) -> ...:
        ...  # 然后在子类实现
```

**禁止**在 env 里直接调用 backend 子类私有方法（feature leakage）。

---

## 4 新增环境 Checklist

创建新 env（如 G1WalkFlatGC）时的关键步骤：

1. **继承选择**：确认是否需要在 `super().__init__` 前操作 model → 选择合适的基类
2. **Actuator 切换**：如需 motor actuator，在 materialize 前调用 `switch_to_motor_actuators()`
3. **Action space**：motor actuator 的 ctrlrange=forcerange，policy 输出是目标角而非力矩 → `_init_action_space()` 返回 `[-1, 1]^n`
4. **Controller 创建**：选择 PD / GravityComp / CTC
5. **pre_step_control 注册**：`self._backend.set_pre_step_control(self._pre_step_motor_control)`
6. **DR 迁移**：如用 motor actuator，kp/kd DR 必须自定义 provider 迁移到 env
7. **obs_groups_spec**：确保与 baseline 一致（A/B 对比公平性）
8. **训练配置**：创建对应的 YAML，超参数与 baseline 一致
9. **Registry 注册**：`@registry.env("EnvName", sim_backend="mujoco")`

---

## 5 调试技巧

### 5.1 Pinocchio 对齐验证

```python
import pin, mujoco
# 对比重力向量
mj_model = mujoco.MjModel.from_xml_path("...")
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)
mj_gravity = np.zeros(mj_model.nv)
mujoco.mj_rne(mj_model, mj_data, 0, mj_gravity)  # 注意: result 参数必须传入

from unilab.control import PinocchioDynamicsModel
pin_model = PinocchioDynamicsModel(mj_model)
pin_gravity = pin_model.gravity(qpos[None], qvel[None])
print("Max diff:", np.max(np.abs(pin_gravity[0] - mj_gravity[6:])))
# 期望 ≈ 1e-16
```

### 5.2 MuJoCo Actuator 诊断

```python
import mujoco
mj = mujoco.MjModel.from_xml_path("...")
print("nu:", mj.nu)                              # actuator 数量
print("gainprm[:,0]:", mj.actuator_gainprm[:mj.nu, 0])    # kp
print("biasprm[:,2]:", -mj.actuator_biasprm[:mj.nu, 2])   # kd
print("ctrlrange:", mj.actuator_ctrlrange[:mj.nu])         # 控制范围
print("forcerange:", mj.actuator_forcerange[:mj.nu])       # 力矩限制
print("biastype:", mj.actuator_biastype[:mj.nu])           # 偏置类型
```

### 5.3 环境快速冒烟测试

```python
from unilab.envs.locomotion.g1.joystick import G1WalkFlatGCEvn
env = G1WalkFlatGCEvn(cfg, num_envs=4, backend_type="mujoco")
obs, info = env.reset()
for i in range(50):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(f"Step {i}: reward={reward.mean():.4f}, alive={1 - terminated.mean():.2%}")
```

### 5.4 DR 验证

```python
# 检查 kp/kd 随机化是否生效
env.reset()
print("Motor kp:", env._motor_kp[0])  # 应与 base_kp 不同（DR 后）
print("Motor kd:", env._motor_kd[0])
```

---

## 6 关键约束速查

| 约束 | 细节 |
|------|------|
| 运行命令 | **必须用 `uv run`**，不用 `python` |
| Env contract | `NpEnvState.obs` 是 dict；`reset()` → `(obs_dict, info_dict)` |
| Backend 隔离 | env 只调用 `SimBackend.base.py` 声明的方法 |
| 后端切换 | 通过 `task=<task>/<backend>` 选 YAML，不 override `sim_backend` |
| keyframe 位置 | 放 task-level XML，**不放 robot.xml** |
| 冷路径访问 | asset/XML/metadata 只在 init/materialization/cache 路径访问 |
| PR gate | 工作树干净 + `make test-all` 通过 |
| 提交规范 | Conventional Commits: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:` |
| G1 forcerange | 力矩限制在 `actuator_forcerange`，非 `ctrlrange`（G1 ctrlrange=[0,0]） |
| Motor actuator ctrlrange | 切换后必须设 `ctrlrange=forcerange`，否则 MuJoCo 不做力矩限幅 |

---

## 7 文档与参考

| 资源 | 路径 |
|------|------|
| 架构标准 | `docs/sphinx/source/zh_CN/4-developer_guide/0-index.md` |
| 协作流程 | `docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md` |
| 开发者入口 | `CONTRIBUTING.md` |
| 动力学补偿实现文档 | `docs/improve/g1_dynamics_compensation_implementation.md` |
| 项目记忆 | `.claude/memory/` |

---

## 8 动力学补偿经验教训

> 来自 G1 六方案 A/B 对比（GC/CC/CTC-IMU/CTC-force/CTCP/Baseline）的实战经验。

### 8.1 MuJoCo 力矩补偿陷阱

1. **`<force>` 传感器不是纯 GRF**：MuJoCo force sensor 测量的是体上约束力（含内力），不是地面反力。直接用于补偿会严重过度补偿（CTC(force) reward 302 vs Baseline 308）
2. **`qfrc_constraint` 是关节空间约束力**：已包含接触+关节限位力的关节空间投影，不需要 Jacobian 映射。站立时精确平衡 `qfrc_smooth`（误差 <0.001）
3. **`BatchEnvPool.get_field()` 仅支持 model 级字段**：不支持 `qfrc_constraint`、`qacc` 等 data 级字段。要读取需逐 env 调用（4096 envs 效率不够）或扩展 pool

### 8.2 qacc 估计经验

4. **qvel 差分估 qacc 必须用控制步率**：仿真步率（6.67ms）下 qvel 噪声被质量矩阵放大，RNEA 算出的 qfrc_constraint 误差巨大。控制步率（20ms）+ EMA(α=0.2) 大幅改善
5. **qacc 精度是 RNEA 反推法的瓶颈**：即使用控制步率+滤波，CTCP reward 仍低于 CC——qacc 误差是根本限制
6. **EMA 滤波双刃剑**：平滑噪声但也延迟动态响应，冲击/步态切换时的接触力变化被平滑掉

### 8.3 IMU 使用经验

7. **IMU 姿态 R 来自 qpos（仿真真值）**：不是从 IMU 传感器融合估计的。sim-to-real 时需替换为 IMU 姿态估计
8. **IMU 残余加速度 + pelvis site Jacobian 是稳定组合**：脚底 Jacobian 因长力臂导致不稳定（16.6×g(q)），pelvis Jacobian 短力臂更稳定
9. **IMU 的真正价值在 sim-to-real**：仿真中模型完美时残余≈0，真机 URDF 参数有偏差时能在线修正

### 8.4 补偿策略经验

10. **GC 的"过补偿"有益**：g(q) 补偿全部重力但不减去 GRF 贡献，PD 看到净向上力，有利于学习。减去 GRF 反而移除了这个好处
11. **contact_scale 要保守**：接触力信号（特权或 IMU）幅值大，0.2-0.5 是安全范围。过大导致过度补偿、训练崩溃
12. **补偿越完整不代表 reward 越高**：GC(323) > CTC-IMU(319) > CC(317) > CTCP(313)。原因：补偿误差会抵消信息增益

### 8.5 前馈补偿与参数缩减

13. **前馈补偿支撑参数缩减 42%**：IMU-GC 96×2 (165K) @10k reward 325.9，超过全量 Baseline 128×2 (285K) @5k 的 304.7。同样 165K 参数，有/无前馈差距 +11.5 reward
14. **小网络收敛更慢但天花板不低**：96×2 在 5k 时 312.2（低于全量 317.5），但 10k 时 325.9（超过全量 5k）。降参后需增加训练轮数
15. **网络深度比宽度关键**：96×1 比 96×2 差 12 reward，2 层残差 block 对步态学习至关重要

### 8.6 Off-policy 续训陷阱

16. **FlashSAC 续训不可靠**：checkpoint 只保存 learner state 不保存 replay buffer，续训时 buffer 为空导致 off-policy 分布偏移（312→287）。**从头跑 10k 比续训 5k→10k 更可靠**
17. **指数衰减模型仅适合短期预测**：拟合 `dr/dt = k·(r_max - r)` 模型对 5k→10k 外推偏差 >15%（预测 341.5 vs 实际 325.9），但短期（2-3k iters 内）较准
