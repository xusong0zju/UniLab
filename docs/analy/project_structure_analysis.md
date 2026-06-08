# UniLab 工程结构与控制器架构深度分析

> 生成日期：2026-06-08
> 对应版本：`main` 分支 @ `43c2c77c`
> 核心动机：理解工程结构 + PD 控制器分布 + 为引入 Pinocchio 高级控制器（动力学补偿 / 自抗扰）做架构准备

---

# 第一部分：工程结构

## 1. 项目概览

**UniLab**（Universal Lab for Robot Learning）是一个高性能、模块化、contract 驱动的强化学习基础设施仓库，专注于机器人的 locomotion、motion tracking、manipulation 以及 manip-loco 等任务。

| 维度 | 说明 |
|------|------|
| 语言 | Python ≥ 3.10, < 3.14 |
| 物理引擎 | MuJoCo（`mujoco-uni`）+ Motrix（`motrixsim-core`，可选） |
| 深度学习 | PyTorch 2.7.0 + Apple MLX（macOS 可选） |
| 配置系统 | Hydra / OmegaConf |
| 包管理 | uv |
| 许可证 | Apache-2.0 |

### 核心设计原则

1. **Contract First** — 不为了一次通过绕过 env / backend / runner contract
2. **Fix at Owner Layer** — `scripts/` 只组装流程，不承载长期业务规则
3. **Config First** — task / reward / backend 优先通过 Hydra + registry 表达
4. **Backend Isolation** — MuJoCo / Motrix 差异留在 backend 适配层和配置层
5. **Cold-path Asset Access** — asset/XML/model 元数据只允许在 init / materialization / cache 等低频路径处理

---

## 2. 顶层目录结构

```
UniLab/
├── conf/                   # Hydra 配置文件（84 个 YAML）
├── docs/                   # 文档
│   ├── analy/              # 分析文档（本文件所在目录）
│   └── sphinx/             # Sphinx 文档源码，双语（en/zh_CN）
├── scripts/                # 训练 / 评估 / 工具脚本入口
├── src/
│   └── unilab/             # 核心 Python 包
├── tests/                  # 测试套件
├── .github/                # CI 工作流 + Issue/PR 模板
├── benchmark/              # Benchmark 脚本
├── notebook/               # Jupyter notebook
├── logs/                   # 训练日志输出（gitignored）
├── pyproject.toml          # 项目元数据 + 依赖
├── pyproject.rocm.toml     # AMD ROCm 变体
├── Makefile                # 常用命令快捷入口
├── Dockerfile              # Docker 镜像定义
├── CLAUDE.md               # AI Agent 开发指引
├── CONTRIBUTING.md         # 贡献者指南
├── README.md / README_zh.md # 项目说明（英/中）
└── AGENTS.md               # Agent 开发规范
```

---

## 3. 核心架构：Contract 驱动设计

UniLab 的核心架构围绕三大 contract 构建：

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Training    │────▶│  NpEnv       │────▶│  SimBackend │
│  Scripts     │     │  (Env Contract)│    │ (Backend    │
│  / Runners   │     │              │     │  Contract)  │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       │              Registry (注册)         MuJoCo / Motrix
       │                    │                  两种实现
       └────────────────────┘
           Hydra Config (配置驱动)
```

### 3.1 Env Contract（`src/unilab/base/np_env.py`）

- `NpEnvState.obs` 必须是 `dict[str, np.ndarray]`
- `reset()` 返回 `(obs_dict, info_dict)`
- `obs_groups_spec` 属性影响 wrapper 和 learner 维度
- 支持 Domain Randomization、Symmetry Augmentation、NaN Guard

### 3.2 Backend Contract（`src/unilab/base/backend/base.py`）

`SimBackend` 抽象基类定义了统一的仿真后端接口，包括：

| 功能组 | 关键方法 |
|--------|---------|
| 模型属性 | `num_envs`, `num_actuators`, `num_dof_vel`, `model` |
| 状态查询 | `get_base_pos/quat/lin_vel/ang_vel`, `get_dof_pos/vel` |
| Body 运动学 | `get_body_pos/quat/lin_vel/ang_vel_w/b`（世界系/基座系） |
| 关节范围 | `get_actuator_ctrl_range`, `get_joint_range` |
| 传感器 | `get_sensor_data`, `get_sensor_data_batch` |
| 仿真控制 | `step(ctrl)`, `set_state(env_ids, qpos, qvel)` |
| 初始化 | `get_keyframe_qpos`, `get_default_qpos`, `get_init_qvel` |
| Jacobian | `get_site_jacobian_w` |
| Domain Randomization | `get_dr_capabilities`, `apply_init_randomization`, `apply_interval_randomization` |
| Play/Render | `init_renderer`, `render`, `capture_video_frame`, `run_playback` |
| 地形 | `create_hfield_scanner` |

**关键规则**：env 层只能调用 `SimBackend` 中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法。

### 3.3 Registry（`src/unilab/base/registry.py`）

- `@envcfg("name")` 装饰器注册环境配置类
- `EnvMeta` 记录 `env_cfg_cls` + 各后端对应的 `env_cls`
- `support_sim_backend(name)` 检查后端支持
- 自动发现路径：`unilab.envs.locomotion`, `unilab.envs.manipulation`, `unilab.envs.motion_tracking`
- 默认后端优先级：`mujoco` > `motrix`

---

## 4. 源码模块详解

### 4.1 `src/unilab/base/` — 基础抽象层

```
base/
├── base.py                 # ABEnv 抽象基类, EnvCfg, EnvPlayCapabilities
├── np_env.py               # NpEnv (核心 Env Contract), NpEnvState
├── registry.py             # 环境注册表 @envcfg 装饰器
├── scene.py                # SceneCfg, TerrainSceneCfg
├── observations.py         # 观测空间构建工具
├── augmentation.py         # SymmetryAugmentation 对称性增强
├── curriculum.py           # 课程学习
├── final_observation.py    # 终末观测处理
└── backend/
    ├── base.py             # SimBackend 抽象基类（Backend Contract）
    ├── playback_common.py  # Playback 通用逻辑
    ├── mujoco/             # MuJoCo 后端实现
    │   ├── backend.py      # MuJoCoSimBackend
    │   ├── playback.py     # MuJoCo playback
    │   └── xml.py          # XML 场景组装 + materialization
    └── motrix/             # Motrix 后端实现
        ├── backend.py      # MotrixSimBackend
        ├── playback.py     # Motrix playback
        └── scene.py        # Motrix 场景构建
```

**关键数据流**：

```
SceneCfg (model_file + fragment_files + terrain)
  → materialize (冷路径)
    → SimBackend 创建 (MuJoCo/Motrix)
      → NpEnv(cfg, backend, num_envs)
        → reset / step / observe
```

### 4.2 `src/unilab/envs/` — 环境实现层

```
envs/
├── common/                 # 跨任务共享工具
│   ├── math.py             # 数学工具函数
│   └── rotation.py         # 旋转工具函数
├── locomotion/             # 运动控制任务
│   ├── common/             # Locomotion 共享逻辑
│   │   ├── base.py         # LocomotionBaseEnv
│   │   ├── commands.py     # 速度命令生成
│   │   ├── domain_rand.py  # 领域随机化配置
│   │   ├── dr_provider.py  # DR Provider 实现
│   │   ├── height_scan.py  # 高度图扫描
│   │   ├── rewards.py      # Reward 函数
│   │   └── terrain_spawn.py # 地形生成
│   ├── g1/                 # G1 人形机器人
│   │   ├── base.py         # G1LocomotionBaseEnv
│   │   ├── joystick.py     # G1JoystickFlat/Walk
│   │   └── symmetry.py     # 对称性增强
│   ├── go1/                # Go1 四足机器人
│   ├── go2/                # Go2 四足机器人
│   │   ├── base.py, joystick.py, rough.py
│   │   ├── footstand.py    # 单脚站立
│   │   └── handstand.py    # 倒立
│   ├── go2w/               # Go2-W（轮腿）
│   └── go2_arm/            # Go2 + 机械臂（操作+运动）
│       ├── base.py
│       └── manip_loco.py   # ManipLocoEnv
├── manipulation/           # 灵巧操作任务
│   ├── allegro_inhand/     # Allegro 手灵巧操控
│   │   ├── base.py, grasp_gen.py, rotation.py
│   └── sharpa_inhand/      # Sharpa Wave 手灵巧操控
│       ├── base.py, grasp_gen.py, rotation.py
└── motion_tracking/        # 运动追踪任务
    └── g1/                 # G1 全身运动追踪
        ├── tracking.py     # G1MotionTrackingEnv
        ├── tracking_obs.py # 观测定义
        ├── tracking_sac.py # SAC 变体
        ├── flip_tracking.py      # 翻跟头追踪
        ├── flip_tracking_sac.py  # 翻跟头 SAC 变体
        ├── box_tracking.py       # Box 追踪
        ├── motion_loader.py      # 运动数据加载
        └── motion_box_loader.py  # Box 运动数据加载
```

**继承关系示意**：

```
ABEnv → NpEnv → LocomotionBaseEnv → G1LocomotionBaseEnv → G1JoystickFlat
                                         → Go1JoystickFlat
                                         → Go2JoystickFlat/Rough
                                         → Go2Footstand / Go2Handstand
                                         → Go2wJoystickFlat/Rough
                   → ManipLocoEnv (Go2 + Arm)
                   → AllegroInhandEnv / SharpaInhandEnv
                   → G1MotionTrackingEnv / G1FlipTrackingEnv / G1BoxTrackingEnv
```

### 4.3 `src/unilab/algos/` — 算法实现层

```
algos/
├── torch/                  # PyTorch 算法
│   ├── rsl_rl_ppo.py       # PPO（基于 rsl-rl-lib）
│   ├── rsl_rl_runtime.py   # PPO 运行时解析
│   ├── appo/               # APPO（异步 PPO）
│   │   ├── learner.py, worker.py, runner.py, runtime.py, staging.py
│   ├── offpolicy/          # Off-policy 通用框架
│   │   ├── runner.py, double_buffer_runner.py, multi_gpu_runner.py, worker.py, runtime.py
│   ├── fast_sac/            # FastSAC（高性能 SAC）
│   ├── fast_td3/            # FastTD3
│   ├── flash_sac/           # FlashSAC（分布式 SAC）
│   ├── him_ppo/             # HIM-PPO（混合优势估计）
│   ├── hora/                # HORA（层次化策略蒸馏）
│   └── common/             # 算法共享组件
└── mlx/                    # Apple MLX 算法
    └── ppo/                # MLX-PPO
```

**算法矩阵**：

| 算法 | 类型 | 框架 | 多进程 | 入口脚本 |
|------|------|------|--------|---------|
| PPO (rsl-rl) | On-policy | PyTorch | ✗ | `scripts/train_rsl_rl.py` |
| APPO | On-policy (async) | PyTorch | ✓ | `scripts/train_appo.py` |
| MLX-PPO | On-policy | Apple MLX | ✗ | `scripts/train_mlx_ppo.py` |
| HIM-PPO | On-policy (hybrid) | PyTorch | ✗ | `scripts/train_him_ppo.py` |
| SAC | Off-policy | PyTorch | ✓ | `scripts/train_offpolicy.py` |
| TD3 | Off-policy | PyTorch | ✓ | `scripts/train_offpolicy.py` |
| FastSAC | Off-policy | PyTorch | ✓ | `scripts/train_offpolicy.py` |
| FlashSAC | Off-policy (distributed) | PyTorch | ✓ | `scripts/train_offpolicy.py` |
| HORA | Distillation | PyTorch | ✓ | `scripts/train_hora_distill.py` |

### 4.4 `src/unilab/training/` — 训练辅助层

```
training/
├── run.py                  # 日志路径、checkpoint 解析、playback 决策
├── backend_adapter.py      # BackendAdapter：根据 cfg 创建 SimBackend + NpEnv
├── common.py               # 训练共享工具
├── experiment.py           # ExperimentTracker（TensorBoard/WandB）
├── monitoring.py           # 训练监控
├── reward.py               # Reward 注入/覆盖
├── rsl_rl.py               # rsl-rl 适配器（RslRlVecEnvWrapper）
├── seed.py                 # 种子管理
└── __init__.py             # 导出 ensure_registries, create_env, BackendAdapter 等
```

### 4.5 `src/unilab/ipc/` — 进程间通信与异步训练

```
ipc/
├── async_runner.py             # AsyncRunner 基类（多进程生命周期管理）
├── replay_buffer.py            # Replay Buffer（off-policy）
├── rollout_ring_buffer.py      # Ring Buffer（on-policy rollout）
├── shared_buffer.py            # 共享内存缓冲区
├── shared_obs_stats.py         # 跨进程观测统计同步
├── weight_sync.py              # 跨进程权重同步
├── collector_error.py          # 采集进程错误传播
├── memory_budget.py            # 内存预算管理
└── replay_pipelines/           # Replay 数据传输管线
```

### 4.6 `src/unilab/dr/` — Domain Randomization

```
dr/
├── manager.py              # DomainRandomizationManager（编排 DR 生命周期）
├── provider.py             # DomainRandomizationProvider（用户实现 DR 策略）
├── types.py                # DR 类型定义（Capabilities, Plan, Payload）
└── dr_utils.py             # DR 工具函数
```

**DR 生命周期**：

1. `apply_init_randomization()` — 冷路径：模型/材质随机化（env 创建时）
2. `reset()` — 每次 reset 时应用随机化
3. `apply_interval_randomization_if_due()` — 按步数间隔的扰动（推力、速度等）

### 4.7 其他模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 可视化 | `src/unilab/visualization/` | `playback.py`, `interactive_playback.py`, `viser_scene.py`, `render_many.py` |
| 地形 | `src/unilab/terrains/` | `terrain_generator.py`, `heightfield_terrains.py`, `config.py`, `utils.py` |
| 工具 | `src/unilab/tools/` | `completion.py`, `export_scene.py`, `import_robot.py`, `render_teaser.py`, `viz_nan.py` |
| 工具类 | `src/unilab/utils/` | `device.py`, `tensor.py`, `nan_guard.py`, `support_matrix.py` |
| 日志 | `src/unilab/logging/` | `common.py`, `onpolicy.py`, `offpolicy.py`, `trace_event.py` |

---

## 5. 配置系统

UniLab 使用 Hydra + OmegaConf 管理所有训练配置，配置文件位于 `conf/` 目录。

### 5.1 目录结构

```
conf/
├── ppo/                    # PPO 训练配置
│   ├── config.yaml         # 全局默认 + 默认 task
│   ├── config_mlx.yaml     # MLX-PPO 变体
│   └── task/               # 任务配置（每个任务 × 每个后端 = 一个 YAML）
├── appo/                   # APPO 训练配置
├── offpolicy/              # Off-policy (SAC/TD3/FlashSAC) 配置
├── ppo_him/                # HIM-PPO 配置
└── hora_distill/           # HORA 蒸馏配置
```

### 5.2 配置组成模式

**关键规则**：后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。

### 5.3 结构化配置（`src/unilab/structured_configs.py`）

所有算法的配置由 dataclass 定义，确保类型安全：

| 配置类 | 对应算法 |
|--------|---------|
| `PPOConfig` / `PPOPolicyConfig` / `PPOAlgorithmConfig` | PPO (rsl-rl) |
| `APPOConfig` / `APPOActorConfig` / `APPOCriticConfig` / `APPOAlgorithmConfig` | APPO |
| `SACConfig` / `SACAlgoParams` | SAC |
| `TD3Config` / `TD3AlgoParams` | TD3 |
| `FlashSACConfig` / `FlashSACAlgoParams` | FlashSAC |

---

## 6. 资产体系

```
src/unilab/assets/
├── robots/                 # 机器人模型（全部 MJCF 格式，无 URDF）
│   ├── g1/                 # G1 人形
│   ├── go1/                # Go1 四足
│   ├── go2/                # Go2 四足
│   ├── go2_arm/            # Go2 + 机械臂
│   ├── go2w/               # Go2-W 轮腿
│   ├── allegro_hand/       # Allegro 灵巧手
│   ├── sharpa_wave/        # Sharpa Wave 手
│   └── hfields/            # 全局高度场
├── scenes/                 # 全局场景资源
├── motions/                # 运动数据（.npz）
├── objects/                # 操作对象
└── caches/                 # 缓存
```

**关键规则**：

- `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml`），**禁止放进 robot.xml**
- robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关
- `ASSETS_ROOT_PATH`、`model_file` 等元数据只允许在 init / materialization / cache 等低频路径访问

---

## 7. 测试体系

```
tests/
├── base/                   # 基础层测试（NpEnv, SimBackend, Registry）
├── algos/                  # 算法层测试（PPO, APPO, MLX, Off-policy, HORA）
├── envs/                   # 环境层测试（DR, obs_noise, motion_loader）
├── ipc/                    # IPC 层测试（async_runner, replay_buffer, weight_sync）
├── config/                 # 配置测试
├── integration/            # 集成测试
├── dr/                     # Domain Randomization 测试
├── terrains/               # 地形测试
├── training/               # 训练辅助测试
├── utils/                  # 工具测试
├── visualization/          # 可视化测试
├── scripts/                # 脚本测试
├── benchmark/              # 性能基准测试
└── cli/                    # CLI 入口测试
```

**测试命令**（via Makefile）：`make test` / `make test-cov` / `make test-slow` / `make test-all`

---

## 8. CI/CD

```
.github/
├── workflows/
│   ├── ci.yml              # 持续集成（lint + type + test）
│   └── docs.yml            # 文档构建与部署
├── CODEOWNERS
├── pull_request_template.md
└── ISSUE_TEMPLATE/
```

---

# 第二部分：PD 控制器架构分析

## 9. 核心发现：PD 控制存在于两个层次

UniLab 中的 PD 控制分布在**两个完全不同的层次**，理解这一区分是替换工作的前提：

```
┌─────────────────────────────────────────────────────────────────┐
│ 层次 1: Env 层 apply_action() — 策略输出 → 目标位置             │
│   actions * action_scale + default_angles → ctrl (目标位置)      │
│   这是 "policy → target" 的映射，PD 增益不在此层                │
├─────────────────────────────────────────────────────────────────┤
│ 层次 2: Backend 层 / pre_step_control — 目标位置 → 力矩         │
│   Kp * (target - qpos) - Kd * qvel → torque                    │
│   这是 "target → torque" 的映射，PD 增益在此层生效              │
└─────────────────────────────────────────────────────────────────┘
```

**关键区别**：
- 层次 1 输出的是**目标位置**（position setpoint），交给 MuJoCo/Motrix 的 position actuator
- 层次 2 是**真正的 PD 计算**，把目标位置转为力矩——这正是需要替换的部分

---

## 10. 层次 1：Env 层 `apply_action()` — 目标生成

### 10.1 基类：`LocomotionBaseEnv.apply_action()`

文件：`src/unilab/envs/locomotion/common/base.py:91-102`

```python
def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
    state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
    state.info["current_actions"] = actions
    exec_actions = (
        state.info["last_actions"]
        if self._cfg.control_config.simulate_action_latency
        else actions
    )
    ctrl: np.ndarray = (
        exec_actions * self._cfg.control_config.action_scale + self.default_angles
    )
    return ctrl
```

**作用**：将策略输出 `actions`（通常 ∈ [-1, 1]）通过 `action_scale` 缩放后加上 `default_angles`，得到**目标关节位置** `ctrl`。

### 10.2 各子类的 `apply_action()` 变体

| 环境类 | 文件 | 行为特点 |
|--------|------|---------|
| `LocomotionBaseEnv` | `common/base.py:91` | 标准模式：`actions * action_scale + default_angles` |
| `G1JoystickWalkTask` | `g1/joystick.py:629` | 同标准模式 |
| `Go2RoughTask` | `go2/rough.py:357` | 增加 `clip_actions`，用独立 `_action_scale` 数组（hip/非hip 不同缩放） |
| `Go2FootStandTask` | `go2/footstand.py:384` | 累积式：`_motor_targets += exec_actions * action_scale`，目标逐帧累加 |
| `Go2WJoystickFlatTask` | `go2w/joystick.py:363` | 腿部目标位置 + 轮部目标速度，分离处理 |
| `G1MotionTrackingEnv` | `motion_tracking/g1/tracking.py:602` | 同标准模式 + `default_dof_pos_bias` |
| `AllegroInhandEnv` | `manipulation/allegro_inhand/base.py:107` | 累积式：`prev_ctrl + action_scale * clipped_actions` |
| `SharpaInhandEnv` | `manipulation/sharpa_inhand/rotation.py` | 类似 Allegro，声明了 `torque_control` 但当前强制为 `False` |

### 10.3 控制配置继承体系

```
ControlConfigBase (common/base.py:23)
  ├── action_scale: float = 0.25
  ├── simulate_action_latency: bool = False
  │
  └── PdControlConfig (common/base.py:29)  ← Go1/Go2/Go2W 用这个
        ├── Kp: float = 35.0
        └── Kd: float = 0.5

各机器人子类覆写：
  G1: ControlConfig (g1/base.py:26) → 直接继承 ControlConfigBase，无 Kp/Kd
  Go2: ControlConfig (go2/base.py:21) → 继承 PdControlConfig
  Go2W: ControlConfig (go2w/base.py:64) → 继承 PdControlConfig + wheel_Kd
  Go2 Rough: RoughControlConfig (go2/rough.py:69) → + hip/non_hip_action_scale, clip_actions
  Footstand: FootstandControlConfig (go2/footstand.py:56) → + clip_actions
  Sharpa: SharpaControlConfig (sharpa_inhand/base.py:80) → + p_gain, d_gain, torque_control
  Allegro: (内联在 base.py:22) → kp=1.0, kd=0.1
```

---

## 11. 层次 2：Backend 层 — PD 力矩计算（真正需要替换的部分）

PD 力矩计算有**两种实现路径**：

### 11.1 路径 A：MuJoCo Position Actuator（内建 PD）

**这是大多数环境使用的方式。**

MuJoCo XML 中定义 position actuator：
```xml
<actuator>
  <position name="hip" joint="hip_joint" kp="35"/>
</actuator>
```

MuJoCo 内部自动计算：`torque = kp * (target - qpos) - kd * qvel`

**`position_actuator_gains` 覆盖机制**：

文件：`src/unilab/base/backend/mujoco/backend.py:166-178`

```python
def _apply_position_actuator_gains_to_mj_model(
    model, *, kp, kd, actuator_ids=slice(None),
) -> None:
    model.actuator_gainprm[actuator_ids, 0] = kp_arr
    model.actuator_biasprm[actuator_ids, 1] = -kp_arr
    model.actuator_biasprm[actuator_ids, 2] = -kd_arr
```

各环境在创建 backend 时传入 `position_actuator_gains`：

| 环境类 | 文件 | 传入方式 |
|--------|------|---------|
| `Go2JoystickFlatTask` | `go2/joystick.py:115` | `{"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd}` |
| `Go1JoystickFlatTask` | `go1/joystick.py:105` | 同上 |
| `Go2HandStandTask` | `go2/handstand.py:141` | 同上 |
| `Go2ArmManipLocoEnv` | `go2_arm/manip_loco.py:303` | `build_go2_arm_position_gains(cfg)` — 腿/臂不同增益 |
| `AllegroInhandEnv` | `allegro_inhand/rotation.py:272` | `{"kp": cfg.control_config.kp, "kd": cfg.control_config.kd}` |

**G1 环境没有传入**：G1 的 `ControlConfig` 不继承 `PdControlConfig`，没有 `Kp/Kd` 字段。G1 直接使用 XML 中的默认 actuator 增益。

### 11.2 路径 B：`pre_step_control` 回调（显式 PD 计算）

**Go2W 轮腿机器人使用这种方式**，因为它需要对腿部和轮部使用不同的控制逻辑。

文件：`src/unilab/envs/locomotion/go2w/base.py:93-116`

```python
def compute_go2w_motor_ctrl(
    policy_ctrl, joint_pos, joint_vel,
    leg_kp, leg_kd, wheel_kd,
    ctrl_lower, ctrl_upper, out,
) -> np.ndarray:
    leg_out = out[:, :NUM_LEG_ACTIONS]
    np.subtract(policy_ctrl[:, :NUM_LEG_ACTIONS], joint_pos[:, :NUM_LEG_ACTIONS], out=leg_out)
    np.multiply(leg_out, leg_kp, out=leg_out)
    leg_out -= leg_kd * joint_vel[:, :NUM_LEG_ACTIONS]
    ...
```

**注册机制**：`self._backend.set_pre_step_control(self._pre_step_motor_control)`

**`pre_step_control` 的调用链**：

```
NpEnv.step(actions)
  → apply_action(actions) → ctrl (目标位置)
  → backend.step(ctrl, nsteps)
      → 如果有 _pre_step_control_fn:
          → _step_with_pre_step_control(ctrl, nsteps)
              → 每个 substep:
                  native_ctrl = self._apply_pre_step_control(ctrl)  ← 在这里做 PD 计算
                  pool.step(physics_state, nstep=1, control=native_ctrl)
```

### 11.3 路径 C：Env 层显式 PD 估算（仅用于 reward，不影响仿真）

一些环境在 `update_state()` 中估算 PD 力矩，**仅用于 reward 计算**，不影响实际仿真：

| 环境 | 文件位置 | 方法 |
|------|---------|------|
| `Go2RoughTask` | `go2/rough.py:523-539` | `_estimate_pd_torques()` |
| `Go2FootStandTask` | `go2/footstand.py:659-666` | `_estimate_pd_torques()` |

---

## 12. 控制流全景图

以 Go2 Joystick Flat 为例（**路径 A** — 最常见的模式）：

```
策略网络输出 actions ∈ [-1, 1]^{12}
  │
  ▼
LocomotionBaseEnv.apply_action()          ← Env 层
  ctrl = actions * 0.25 + default_angles   ← 目标关节位置
  │
  ▼
NpEnv.step() → backend.step(ctrl, 3)      ← 3 个 sim substeps
  │
  ▼
MuJoCo Position Actuator (内建 PD)        ← Backend 层（MuJoCo 自动）
  torque = Kp * (ctrl - qpos) - Kd * qvel  ← 这是 PD 力矩
  │
  ▼
MuJoCo 物理仿真 mj_step()
```

以 Go2W Joystick Flat 为例（**路径 B** — pre_step_control 模式）：

```
策略网络输出 actions ∈ [-1, 1]^{16}
  │
  ▼
Go2WJoystickFlatTask.apply_action()       ← Env 层
  leg_targets = actions[:,:12] * 0.25 + default_angles
  wheel_vel_targets = actions[:,12:] * 10.0
  ctrl = concat(leg_targets, wheel_vel_targets)
  │
  ▼
NpEnv.step() → backend.step(ctrl, nsteps)
  │
  ▼
_step_with_pre_step_control()             ← Backend 层（每个 substep 调用）
  native_ctrl = _pre_step_motor_control(backend, ctrl)
    │
    ▼
  compute_go2w_motor_ctrl()               ← 显式 PD 计算
    leg_torque = leg_kp * (leg_target - leg_pos) - leg_kd * leg_vel
    wheel_torque = wheel_kd * (wheel_vel_target - wheel_vel)
  │
  ▼
MuJoCo 物理仿真 mj_step()（使用 native_ctrl 作为力矩）
```

---

## 13. 替换高级控制器的切入点

### 13.1 推荐切入点：`pre_step_control` 回调

**`pre_step_control` 是替换 PD 控制器的最佳切入点**，理由：

1. **已在 Go2W 中验证**：`compute_go2w_motor_ctrl()` 就是这个模式的生产级实例
2. **每个 substep 调用**：可以在每个物理步读取当前状态（qpos, qvel, sensor data）并计算补偿力矩
3. **不影响 Env Contract**：`apply_action()` 仍然返回目标位置，PD → 高级控制的替换完全在 backend 层完成
4. **Backend 无关**：MuJoCo 和 Motrix 都支持 `pre_step_control`
5. **可读取 backend 状态**：回调签名 `fn(backend, ctrl)` 可以访问 backend 的全部传感器和运动学接口

### 13.2 需要修改的层次

| 修改点 | 说明 | 影响范围 |
|--------|------|---------|
| **新增控制器模块** | `src/unilab/control/` | 新增代码，不影响现有 |
| **Env 层 `apply_action()`** | 可能需要输出更丰富的信息（如前馈力矩） | 仅影响使用新控制器的 env |
| **`ControlConfig` 配置** | 新增 `controller_type` / `controller_params` 字段 | 仅影响使用新控制器的 env |
| **`pre_step_control` 回调** | 将 PD 替换为高级控制器 | 仅影响使用新控制器的 env |
| **MuJoCo actuator 模式** | 如果用 `pre_step_control` 做力矩控制，需要将 XML actuator 改为 `motor`（纯力矩） | 仅影响使用新控制器的 env |

---

# 第三部分：引入 Pinocchio 高级控制器方案

## 14. 为什么必须引入 Pinocchio（而非用 MuJoCo 内建动力学 API）

| 维度 | MuJoCo 内建 API | Pinocchio |
|------|----------------|-----------|
| **sim2real 一致性** | ✗ 真机无法调用 MuJoCo API | ✓ 真机和仿真共用同一套动力学计算 |
| **部署可行性** | ✗ MuJoCo 仅限仿真 | ✓ Pinocchio 可部署在嵌入式端 (C++ / 纯头文件) |
| **建模来源** | MuJoCo 内部模型（与 XML 参数耦合） | 独立 URDF/MJCF，可与真机 SDK 共享 |
| **动力学量精度** | 与 MuJoCo 仿真自身一致，但与真机有偏差 | 独立计算，可校准到与真机一致 |
| **计算力矩控制 (CTC)** | 需要提取 `mj_fullM` + `qfrc_bias`，代码侵入 MuJoCo 内部 | `pinocchio.rnea(q, v, a)` 一行搞定 |
| **自抗扰 (ADRC)** | 同上 | 同上 |

**结论**：如果目标是在真机部署时用 Pinocchio 做动力学补偿 + ADRC，那仿真中**必须也用 Pinocchio**，否则补偿项不一致会直接引入 sim2real gap。

---

## 15. 现状盘点：缺少什么

### 15.1 没有可用的 URDF

当前资产全部是 MJCF（MuJoCo XML）格式，**没有 URDF 文件**。Pinocchio 需要 URDF 来构建动力学模型。**这是第一个要解决的阻塞项**。

### 15.2 没有控制器抽象层

当前 PD 控制是硬编码的两种方式：
- MuJoCo position actuator（内建 PD）
- Go2W 的 `compute_go2w_motor_ctrl()`（硬编码在 env 里）

### 15.3 Backend 没有暴露动力学量查询接口

`SimBackend` 目前不暴露 `qfrc_bias`（Coriolis+重力）、质量矩阵等。虽然用 Pinocchio 后不需要从 MuJoCo 读这些，但 `qpos`/`qvel` 的读取是 Pinocchio 计算的输入——这些已有。

---

## 16. URDF 获取路径

### 16.1 路径 A：从厂商/社区获取现成 URDF

| 机器人 | 可能来源 |
|--------|---------|
| G1 | Unitree 官方 SDK (`unitree_ros`)，GitHub 上有 `g1_description` URDF |
| Go2 | Unitree 官方 `go2_description` URDF，`unitree_ros` 仓库 |
| Go2W | 需要从 Go2 URDF 衍生 + 添加轮关节 |
| Go2 Arm | 需要从 Go2 URDF 衍生 + 添加臂关节 |

**注意**：URDF 中的惯性参数必须与仿真 MJCF 中的一致，否则 Pinocchio 的动力学计算与仿真对不上。

### 16.2 路径 B：MJCF → URDF 转换

UniLab 已有 `unilab-import-robot` 工具做 URDF → MJCF 转换，但**没有反向转换**。

### 16.3 路径 C：Pinocchio 直接从 MJCF 加载（推荐探索）

Pinocchio 3.x 支持 `pinocchio.MjcfModel`，可以直接加载 MJCF：

```python
import pinocchio as pin
model = pin.buildModelFromMjcf("go2.xml")
```

**这是最优路径**——不需要额外维护 URDF，直接复用现有 MJCF，且惯性参数天然一致。

> ⚠️ 需要验证：Pinocchio 的 MJCF 加载器是否完整支持所有 MJCF 特性（如 `freejoint`、`armature`、多重几何体等）。如果部分支持不完整，可能仍需回退到 URDF。

---

## 17. 控制器架构设计

### 17.1 核心抽象：`MotorController`

```python
# src/unilab/control/base.py (提议)
from __future__ import annotations
import abc
import numpy as np
from unilab.base.backend import SimBackend


class MotorController(abc.ABC):
    """低层级电机控制器抽象。

    在 pre_step_control 回调中被调用，每个物理 substep 前执行一次。
    输入：策略输出的目标（来自 apply_action）+ 当前机器人状态
    输出：实际施加的力矩（或修改后的 ctrl）
    """

    @abc.abstractmethod
    def compute_torque(
        self,
        backend: SimBackend,
        target: np.ndarray,       # apply_action 输出的目标 (num_envs, nu)
        dof_pos: np.ndarray,      # 当前关节位置 (num_envs, nv)
        dof_vel: np.ndarray,      # 当前关节速度 (num_envs, nv)
    ) -> np.ndarray:
        """计算实际输出力矩。返回 (num_envs, nu) 数组。"""

    @abc.abstractmethod
    def reset(self, env_ids: np.ndarray) -> None:
        """重置指定环境的控制器内部状态（如 ESO 状态）。"""


class PDController(MotorController):
    """标准 PD 控制器（与现有行为等价）。"""

    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self._kp = kp
        self._kd = kd

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        return self._kp * (target - dof_pos) - self._kd * dof_vel

    def reset(self, env_ids):
        pass  # PD 无状态


class CTCController(MotorController):
    """计算力矩控制 (Computed Torque Control)。

    τ = M(q) * [Kp*(q_d - q) + Kd*(q̇_d - q̇)] + c(q, q̇) + g(q)

    使用 Pinocchio 计算 M, c+g，实现精确的反馈线性化。
    """

    def __init__(self, model, kp, kd):
        self._model = model          # pinocchio.Model
        self._data = model.createData()
        self._kp = kp
        self._kd = kd

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        # 1. 从 backend 读取 generalized coordinates
        # 2. Pinocchio 计算质量矩阵 + bias force
        # 3. CTC 公式计算力矩
        ...


class ADRCController(MotorController):
    """自抗扰控制器。

    在 CTC 基础上添加扩张状态观测器 (ESO)，
    估计并补偿未建模动态和外部扰动。

    τ = M(q) * [Kp*(q_d - q) + Kd*(q̇_d - q̇) - z_3/b0] + c(q, q̇) + g(q)
                                                         ↑ ESO 估计的扰动
    """

    def __init__(self, model, kp, kd, eso_bandwidth, b0):
        self._model = model
        self._data = model.createData()
        self._kp = kp
        self._kd = kd
        self._eso_bandwidth = eso_bandwidth
        self._b0 = b0
        self._eso_states = None     # 扩张状态

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        ...

    def reset(self, env_ids):
        self._eso_states[env_ids] = 0.0
```

### 17.2 与 `pre_step_control` 的集成

```python
# 在 env 的 __init__ 中：
self._motor_controller = resolve_controller(cfg.control_config, robot_model)
self._backend.set_pre_step_control(self._pre_step_motor_control)

# pre_step_control 回调：
def _pre_step_motor_control(self, backend, policy_ctrl):
    dof_pos = backend.get_dof_pos()
    dof_vel = backend.get_dof_vel()
    return self._motor_controller.compute_torque(backend, policy_ctrl, dof_pos, dof_vel)
```

### 17.3 配置集成

```python
# 在 ControlConfig 中扩展
@dataclass
class AdvancedControlConfig(PdControlConfig):
    controller_type: str = "pd"    # "pd" | "ctc" | "adrc" | "impedance"
    # CTC 参数
    use_gravity_comp: bool = True
    use_coriolis_comp: bool = True
    # ADRC 参数
    adrc_eso_bandwidth: float = 10.0
    adrc_b0: float = 1.0
```

```yaml
# YAML 配置示例
env:
  control_config:
    controller_type: "adrc"
    Kp: 35.0
    Kd: 0.5
    use_gravity_comp: true
    use_coriolis_comp: true
    adrc_eso_bandwidth: 10.0
    adrc_b0: 1.0
```

---

## 18. MuJoCo Actuator 模式的必要修改

当前所有机器人的 XML 使用 **position actuator**。如果用 `pre_step_control` 做 CTC/ADRC，力矩已经被 Python 层计算好了，**MuJoCo 不应该再做一次 PD**。

### 18.1 方案：将 position actuator 替换为 motor actuator

```xml
<!-- go2.xml — 当前 -->
<position class="abduction" name="FR_hip" joint="FR_hip_joint"/>

<!-- go2.xml — 修改后 -->
<motor class="abduction" name="FR_hip" joint="FR_hip_joint"/>
```

### 18.2 向后兼容：保留 `controller_type: "pd"` 的场景

当 `controller_type == "pd"` 时，仍然使用 position actuator + 不设置 `pre_step_control`，行为与现有完全一致。

当 `controller_type != "pd"` 时，需要使用 motor actuator 变体。**推荐运行时修改 `model.actuator_biastype[:] = 0`**，在 init 阶段把 position 改为 motor——无需维护额外 XML，且 MuJoCo 允许运行时修改。

---

## 19. Pinocchio 与 MuJoCo 的状态对齐

Pinocchio 独立计算动力学，但输入（q, q̇）来自 MuJoCo backend。需要注意：

### 19.1 坐标系对齐

| 量 | MuJoCo | Pinocchio | 对齐方式 |
|----|--------|-----------|---------|
| 关节位置 q | `data.qpos` — free joint 用 7D (pos+quat) | `q` — floating base 用 7D (quat+pos) 或 6D (SE3) | **四元数顺序不同**：MuJoCo `wxyz`，Pinocchio `xyzw` |
| 关节速度 v | `data.qvel` — free joint 用 6D (3 lin + 3 ang) | `v` — body velocity 6D | 顺序和参考系需确认 |
| 质量矩阵 M | `mj_fullM` | `pinocchio.crba(q)` | 需要验证在相同 q 下数值一致 |
| Coriolis + 重力 | `data.qfrc_bias` | `pinocchio.rnea(q, v, 0)` | 应该一致 |

### 19.2 关节顺序对齐与 `PinocchioDynamicsModel` 封装

MuJoCo 和 Pinocchio 的关节顺序可能不同。需要一个**关节映射表**：

```python
class PinocchioDynamicsModel:
    """封装 Pinocchio 模型，处理与 MuJoCo backend 的状态对齐。"""

    def __init__(self, mjcf_or_urdf_path: str, backend: SimBackend):
        self._model = pin.buildModelFromMjcf(mjcf_or_urdf_path)  # 或 buildModelFromURDF
        self._data = self._model.createData()
        self._build_joint_mapping(backend)

    def _build_joint_mapping(self, backend):
        """建立 MuJoCo → Pinocchio 关节索引映射。"""
        ...

    def compute_dynamics(self, qpos: np.ndarray, qvel: np.ndarray):
        """从 MuJoCo 状态计算 Pinocchio 动力学量。"""
        q_pin = self._mj_to_pin_qpos(qpos)   # 坐标变换
        v_pin = self._mj_to_pin_qvel(qvel)
        M = pin.crba(self._model, self._data, q_pin)     # 质量矩阵
        bias = pin.rnea(self._model, self._data, q_pin, v_pin, np.zeros_like(v_pin))  # c+g
        return M, bias
```

---

## 20. 自抗扰 (ADRC) 在此架构下的实现要点

### 20.1 ADRC 的核心结构

```
                    ┌──────────┐
  q_d (target) ──▶│  TD      │──▶ v_1 (过渡过程)
                    │ 跟踪微分器 │──▶ v_2 (过渡微分)
                    └──────────┘
                          │
                    ┌─────▼──────┐
  q, q̇ ──────────▶│   ESO      │──▶ z_1 (q 估计)
                    │ 扩张状态观测器│──▶ z_2 (q̇ 估计)
                    │            │──▶ z_3 (扰动估计) ← 核心创新
                    └────────────┘
                          │
                    ┌─────▼──────┐
                    │  NLSEF     │──▶ u_0 (误差控制量)
                    │ 非线性状态误差│
                    │   反馈律    │
                    └────────────┘
                          │
                    ┌─────▼──────────────────────┐
                    │  动力学补偿                 │
                    │  τ = M(q)*(u_0 - z_3/b0) + c(q,q̇) + g(q) │
                    └─────────────────────────────┘
```

### 20.2 ESO 的 per-env 状态管理

ESO 维护 per-env 内部状态（z_1, z_2, z_3），需要在 `reset()` 时清零：

```python
class ADRCController(MotorController):
    def __init__(self, model, kp, kd, eso_bw, b0, num_envs, nu):
        ...
        self._eso_z = np.zeros((num_envs, 3, nu))  # (num_envs, 3 states, nu)
        # z[:, 0] = z_1 (q 估计), z[:, 1] = z_2 (q̇ 估计), z[:, 2] = z_3 (扰动估计)

    def compute_torque(self, backend, target, dof_pos, dof_vel):
        dt = backend._sim_dt
        w = self._eso_bandwidth

        # ESO 更新 (线性 ESO, 三阶)
        e = dof_pos - self._eso_z[:, 0]
        self._eso_z[:, 0] += dt * (self._eso_z[:, 1] + 3*w*e)
        self._eso_z[:, 1] += dt * (self._eso_z[:, 2] + 3*w**2*e)
        self._eso_z[:, 2] += dt * (w**3*e)

        # 误差反馈
        e1 = target - self._eso_z[:, 0]      # 位置误差
        e2 = 0.0 - self._eso_z[:, 1]         # 速度误差（目标速度 = 0，或从 TD 获取）
        u0 = self._kp * e1 + self._kd * e2

        # 动力学补偿 + 扰动补偿
        M, bias = self._pin_model.compute_dynamics(...)
        tau = M @ (u0 - self._eso_z[:, 2] / self._b0) + bias

        return tau

    def reset(self, env_ids):
        self._eso_z[env_ids] = 0.0
```

### 20.3 ADRC 降低 sim2real gap 的机理

| 来源 | PD 无法处理 | ADRC 如何处理 |
|------|------------|--------------|
| 未建模摩擦 | 产生稳态误差 | ESO 估计为扰动 z_3，前馈补偿 |
| 负载变化 | Kp/Kd 不匹配 → 超调/振荡 | ESO 自适应估计等效质量变化 |
| 外部扰动（推力） | 无抵抗力 | z_3 实时估计并补偿 |
| 关节柔性 | 模型不匹配 | ESO 将柔性效应纳入扰动估计 |
| 电机饱和 | PD 积分饱和 | ADRC 无积分项，天然抗饱和 |

---

## 21. 性能考量

### 21.1 Pinocchio 的计算开销

| 操作 | 典型耗时 (Go2, 12 DoF) | 备注 |
|------|----------------------|------|
| `pin.crba(q)` (质量矩阵) | ~10-30 μs | 可优化为稀疏 Cholesky |
| `pin.rnea(q, v, a)` (逆动力学) | ~5-15 μs | |
| ESO 更新 (12 joints) | ~2-5 μs | 纯 numpy 运算 |
| **总增量** | **~20-50 μs / env / step** | |

对比 MuJoCo 一个 substep 约 100-500 μs，Pinocchio 开销约 5-10%。

### 21.2 向量化：多 env 的批量计算

当前 `num_envs` 通常为 2048-4096。Pinocchio 的 `crba`/`rnea` 是单 env 计算的。

**优化策略**：

1. **批量循环**：对每个 env 独立调用 Pinocchio。优点：实现简单。缺点：Python 循环开销（4096 次 × 20μs ≈ 80ms，太慢）
2. **提取动力学参数，批量 numpy 计算**（推荐）：初始化时从 Pinocchio 提取 `model.inertias`、`model.jointPlacements` 等，每步用 numpy 批量计算 M(q) 和 bias(q, q̇)。关节链较短（如四足 12 DoF）时，可手动展开 RNEA 递推公式
3. **Pinocchio batch API**（Pinocchio 3.x）：有限支持批量，需要确认是否支持 (num_envs, nv) 形状的批量输入

**推荐**：初期用策略 1 验证正确性，后续优化用策略 2。

---

## 22. 实施路线图

### Phase 0：前置准备
- [ ] 确认 Pinocchio 3.x 的 MJCF 加载能力，若不支持则获取/编写 URDF
- [ ] 验证 Pinocchio 计算的动力学量与 MuJoCo 在相同状态下的数值一致性
- [ ] 将 `pinocchio` 加入 `pyproject.toml` 依赖

### Phase 1：控制器抽象层
- [ ] 创建 `src/unilab/control/` 模块
- [ ] 实现 `MotorController` 抽象基类
- [ ] 实现 `PDController`（与现有行为等价）
- [ ] 在 `ControlConfig` 中添加 `controller_type` 字段
- [ ] 选一个 env（如 Go2 Joystick Flat）作为试点

### Phase 2：Pinocchio 动力学模型
- [ ] 实现 `PinocchioDynamicsModel`（封装 Pinocchio 模型 + MuJoCo 状态对齐）
- [ ] 实现 `CTCController`（计算力矩控制）
- [ ] 验证 CTC 在仿真中的跟踪性能

### Phase 3：ADRC 控制器
- [ ] 实现 `ADRCController`（含 ESO）
- [ ] 验证 ESO 的扰动估计能力（引入外部扰动测试）
- [ ] 对比 PD / CTC / ADRC 的 sim2real gap

### Phase 4：Actuator 模式切换
- [ ] 实现运行时 position → motor actuator 切换
- [ ] 确保所有现有 env 在 `controller_type: "pd"` 时行为不变
- [ ] 在 CI 中添加控制器切换的集成测试

### Phase 5：扩展到更多机器人和后端
- [ ] G1 人形机器人 CTC / ADRC
- [ ] Go2W 轮腿混合控制器
- [ ] Motrix 后端的 `pre_step_control` 对接

---

## 23. 关键风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Pinocchio MJCF 加载不完整 | 需要额外维护 URDF | 先验证；若失败，用 `unilab-import-robot` 的反向流程 |
| 四元数顺序不一致 (MuJoCo wxyz vs Pinocchio xyzw) | 动力学计算错误 | `PinocchioDynamicsModel` 中做自动转换 |
| Pinocchio 批量计算性能不足 | 训练速度下降 | 先用循环验证，再优化为手动批量 RNEA |
| Position → Motor actuator 切换影响仿真稳定性 | 仿真发散 | 保留 `controller_type: "pd"` 作为 fallback |
| ADRC 参数 (eso_bandwidth, b0) 敏感 | 控制器不稳定 | 在仿真中系统性扫描参数空间 |

---

## 附录 A：关键文件速查

| 用途 | 路径 |
|------|------|
| PPO 训练 | `scripts/train_rsl_rl.py` |
| APPO 训练 | `scripts/train_appo.py` |
| Off-policy 训练 | `scripts/train_offpolicy.py` |
| Env Contract | `src/unilab/base/np_env.py` |
| Backend Contract | `src/unilab/base/backend/base.py` |
| 环境注册 | `src/unilab/base/registry.py` |
| 场景配置 | `src/unilab/base/scene.py` |
| 训练辅助 | `src/unilab/training/run.py` |
| 结构化配置 | `src/unilab/structured_configs.py` |
| 异步 Runner | `src/unilab/ipc/async_runner.py` |
| DR Manager | `src/unilab/dr/manager.py` |
| 可视化 | `src/unilab/visualization/` |

## 附录 B：控制器相关文件速查

| 用途 | 文件路径 |
|------|---------|
| 基类 apply_action | `src/unilab/envs/locomotion/common/base.py` |
| ControlConfigBase / PdControlConfig | `src/unilab/envs/locomotion/common/base.py:23-33` |
| SimBackend pre_step_control 接口 | `src/unilab/base/backend/base.py:247-265` |
| MuJoCo _step_with_pre_step_control | `src/unilab/base/backend/mujoco/backend.py:819-863` |
| MuJoCo position_actuator_gains 注入 | `src/unilab/base/backend/mujoco/backend.py:166-178` |
| Go2W 显式 PD (pre_step_control 示例) | `src/unilab/envs/locomotion/go2w/base.py:93-116` |
| Go2W 注册 pre_step_control | `src/unilab/envs/locomotion/go2w/joystick.py:273` |
| Go2W _pre_step_motor_control | `src/unilab/envs/locomotion/go2w/joystick.py:391-405` |
| NpEnv.step() 控制流 | `src/unilab/base/np_env.py:104-166` |
| DR 中 kp/kd 随机化 | `src/unilab/dr/dr_utils.py:151-160` |
| Motrix position_actuator kp/kd 覆盖 | `src/unilab/base/backend/motrix/backend.py:1385-1399` |
| Go2 Rough _estimate_pd_torques (reward) | `src/unilab/envs/locomotion/go2/rough.py:523-539` |
| Footstand _estimate_pd_torques (reward) | `src/unilab/envs/locomotion/go2/footstand.py:659-666` |
| Sharpa torque_control 声明 | `src/unilab/envs/manipulation/sharpa_inhand/base.py:88` |
| create_backend position_actuator_gains | `src/unilab/base/backend/__init__.py:38-78` |
| Go2 Arm 分段增益 | `src/unilab/envs/locomotion/go2_arm/base.py:141` |
| Go2 XML actuator 定义 | `src/unilab/assets/robots/go2/go2.xml:188-201` |
| G1 XML actuator 定义 | `src/unilab/assets/robots/g1/g1.xml:328-361` |
