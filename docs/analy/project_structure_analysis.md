# UniLab 工程结构分析

> 生成日期：2026-06-07
> 对应版本：`main` 分支 @ `43c2c77c`

---

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
│   │   ├── learner.py      # APPO Learner
│   │   ├── worker.py       # APPO Worker（数据采集）
│   │   ├── runner.py       # APPO Runner（多进程编排）
│   │   ├── runtime.py      # APPO 运行时解析
│   │   └── staging.py      # 数据暂存
│   ├── offpolicy/          # Off-policy 通用框架
│   │   ├── runner.py       # 单 GPU runner
│   │   ├── double_buffer_runner.py  # 双缓冲 runner
│   │   ├── multi_gpu_runner.py      # 多 GPU runner
│   │   ├── worker.py       # 采集 worker
│   │   └── runtime.py      # 运行时解析
│   ├── fast_sac/            # FastSAC（高性能 SAC）
│   │   ├── learner.py, runner.py
│   ├── fast_td3/            # FastTD3
│   │   ├── learner.py, runner.py
│   ├── flash_sac/           # FlashSAC（分布式 SAC）
│   │   ├── learner.py, runner.py, network.py, layers.py
│   │   ├── double_buffer.py, update.py
│   ├── him_ppo/             # HIM-PPO（混合优势估计）
│   │   ├── actor_critic.py, algorithm.py, estimator.py
│   │   ├── runner.py, storage.py
│   ├── hora/                # HORA（层次化策略蒸馏）
│   │   ├── appo.py / appo_learner.py / appo_runner.py / appo_worker.py
│   │   ├── sac.py / sac_learner.py / sac_models.py
│   │   ├── distill.py / distill_config.py
│   │   ├── models.py, observations.py, runtime.py
│   │   ├── ppo.py, rsl_rl.py, rsl_rl_compat.py
│   └── common/             # 算法共享组件
│       ├── networks.py     # 神经网络基础模块
│       ├── normalization.py # 归一化
│       ├── actor_factory.py # Actor 工厂
│       ├── ane_actor.py / ane_inference.py / ane_wrapper.py  # ANE 加速
│       ├── base_collector.py # 数据采集基类
│       ├── device.py       # 设备管理
│       └── stability.py    # 训练稳定性工具
└── mlx/                    # Apple MLX 算法
    └── ppo/                # MLX-PPO
        ├── model.py, ppo.py, runner.py
        └── common/         # MLX 共享组件
            ├── activations.py, distributions.py
            ├── mlp.py, normalization.py
            ├── rotation.py, rollout_storage.py
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

**关键流程**（以 PPO 为例）：

```
train_rsl_rl.py
  → ensure_registries()            # 自动发现并注册所有环境
  → BackendAdapter.create_env()    # 根据 cfg 创建 SimBackend + NpEnv
  → RslRlVecEnvWrapper(env)        # 包装为 rsl-rl 兼容接口
  → OnPolicyRunner(env, cfg)       # rsl-rl runner
  → runner.learn()                 # 训练循环
  → should_run_playback()          # 判断是否执行 playback
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
    ├── base.py                 # 管线基类
    ├── cpu_pinned_double_buffer.py  # CPU Pinned 双缓冲
    ├── native_h2d.py           # 原生 Host-to-Device
    └── transfer/               # 传输后端
        ├── base.py, factory.py
        ├── torch_copy.py       # PyTorch Copy
        ├── cuda_like.py        # CUDA-like
        └── xpu.py              # Intel XPU
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
| 可视化 | `src/unilab/visualization/` | `playback.py`（通用回放）, `interactive_playback.py`（交互式）, `viser_scene.py`（Viser 3D 可视化）, `render_many.py`（批量渲染） |
| 地形 | `src/unilab/terrains/` | `terrain_generator.py`（程序化地形生成）, `heightfield_terrains.py`（高度场地形）, `config.py`, `utils.py` |
| 工具 | `src/unilab/tools/` | `completion.py`（Shell 补全）, `export_scene.py`（场景导出）, `import_robot.py`（机器人导入）, `render_teaser.py`（演示渲染）, `viz_nan.py`（NaN 可视化） |
| 工具类 | `src/unilab/utils/` | `device.py`（设备检测）, `tensor.py`（张量工具）, `nan_guard.py`（NaN 防护）, `support_matrix.py`（支持矩阵） |
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
│       ├── g1_walk_flat/
│       │   ├── mujoco.yaml # G1 平地行走 (MuJoCo)
│       │   └── motrix.yaml # G1 平地行走 (Motrix)
│       ├── go2_joystick_flat/
│       │   ├── mujoco.yaml
│       │   └── motrix.yaml
│       └── ...
├── appo/                   # APPO 训练配置
│   ├── config.yaml
│   └── task/...
├── offpolicy/              # Off-policy (SAC/TD3/FlashSAC) 配置
│   ├── config.yaml
│   ├── algo/               # 算法级配置
│   │   ├── sac.yaml
│   │   ├── td3.yaml
│   │   └── flashsac.yaml
│   └── task/               # 任务配置（按 algo/task/backend 三级）
│       ├── sac/g1_walk_flat/mujoco.yaml
│       └── ...
├── ppo_him/                # HIM-PPO 配置
│   ├── config.yaml
│   └── task/...
└── hora_distill/           # HORA 蒸馏配置
    ├── config.yaml
    └── task/...
```

### 5.2 配置组成模式

每个训练入口通过 Hydra `@hydra.main` 装饰器加载配置。典型组成：

```yaml
# conf/ppo/config.yaml
defaults:
  - _self_
  - task: go1_joystick_flat/mujoco   # 默认任务

algo:           # 算法超参（对应 structured_configs.py 中的 PPOConfig）
  algo: ppo
  num_envs: 4096
  algorithm:
    clip_param: 0.2
    ...

training:       # 训练控制（设备、日志、playback 等）
  task_name: Go1JoystickFlat
  sim_backend: mujoco
  device: null
  ...

env:            # 环境覆盖（control_config, noise_config 等）
  ...

reward:         # Reward 权重覆盖
  ...

interactive:    # 交互式 playback 配置
  ...

hydra:          # Hydra 自身配置
  run:
    dir: .
```

**关键规则**：后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。

### 5.3 任务 YAML 示例（`conf/ppo/task/g1_walk_flat/mujoco.yaml`）

```yaml
# @package _global_
training:
  task_name: G1WalkFlat
  sim_backend: mujoco
algo:
  num_envs: 2048
  max_iterations: 2200
  obs_groups:
    actor:
      - actor
env:
  control_config:
    action_scale: 0.25
  noise_config:
    level: 1.0
  ...
reward:
  scales:
    tracking_lin_vel: 2.0
    tracking_ang_vel: 0.2
    feet_phase: 1.0
    ...
```

### 5.4 结构化配置（`src/unilab/structured_configs.py`）

所有算法的配置由 dataclass 定义，确保类型安全：

| 配置类 | 对应算法 |
|--------|---------|
| `PPOConfig` / `PPOPolicyConfig` / `PPOAlgorithmConfig` | PPO (rsl-rl) |
| `APPOConfig` / `APPOActorConfig` / `APPOCriticConfig` / `APPOAlgorithmConfig` | APPO |
| `SACConfig` / `SACAlgoParams` | SAC |
| `TD3Config` / `TD3AlgoParams` | TD3 |
| `FlashSACConfig` / `FlashSACAlgoParams` | FlashSAC |

---

## 6. 训练入口脚本

| 脚本 | 算法 | 说明 |
|------|------|------|
| `scripts/train_rsl_rl.py` | PPO | 基于 rsl-rl-lib 的 On-policy PPO，单进程 |
| `scripts/train_appo.py` | APPO | 异步 PPO，多进程（Collector + Learner） |
| `scripts/train_mlx_ppo.py` | MLX-PPO | Apple MLX 框架上的 PPO |
| `scripts/train_offpolicy.py` | SAC/TD3/FastSAC/FlashSAC | 统一 Off-policy 入口 |
| `scripts/train_him_ppo.py` | HIM-PPO | 混合优势估计 PPO |
| `scripts/train_hora_distill.py` | HORA | 层次化策略蒸馏 |
| `scripts/play_interactive.py` | — | 交互式回放 |
| `scripts/play_viser.py` | — | Viser 3D 可视化回放 |
| `scripts/visualize_task_env.py` | — | 可视化任务环境 |

**训练脚本统一模式**：

```python
@hydra.main(config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig):
    ensure_registries()                         # 1. 注册环境
    env = create_env(cfg)                       # 2. 创建 Env + Backend
    runner = resolve_runtime(env, cfg)           # 3. 解析运行时
    runner.learn()                               # 4. 训练
    if should_run_playback(...):                 # 5. Playback
        ...
```

---

## 7. 资产体系

```
src/unilab/assets/
├── robots/                 # 机器人模型
│   ├── g1/                 # G1 人形
│   │   ├── g1.xml          # 机器人 URDF/MJCF 描述
│   │   ├── g1_sphere_hand.xml
│   │   ├── locomotion_task.xml   # 任务级 keyframe（stand/home 等）
│   │   ├── scene_flat.xml  # 场景 fragment
│   │   ├── scene_rough.xml
│   │   ├── scene_climb_*.xml
│   │   ├── hfields/        # 高度场
│   │   ├── assets/         # 网格/纹理
│   │   └── textures/
│   ├── go1/                # Go1 四足
│   ├── go2/                # Go2 四足
│   ├── go2_arm/            # Go2 + 机械臂
│   ├── go2w/               # Go2-W 轮腿
│   ├── allegro_hand/       # Allegro 灵巧手
│   ├── sharpa_wave/        # Sharpa Wave 手
│   └── hfields/            # 全局高度场
├── scenes/                 # 全局场景资源
├── motions/                # 运动数据
│   └── g1/                 # G1 运动轨迹（.npz）
├── objects/                # 操作对象
│   └── sharpa_cylinder/    # Sharpa 圆柱
└── caches/                 # 缓存
```

**关键规则**：

- `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml`），**禁止放进 robot.xml**
- robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关
- Motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML
- `ASSETS_ROOT_PATH`、`model_file` 等元数据只允许在 init / materialization / cache 等低频路径访问

---

## 8. 测试体系

```
tests/
├── conftest.py             # 全局 fixture
├── base/                   # 基础层测试
│   ├── test_np_env.py      # NpEnv Contract 测试
│   ├── test_sim_backend.py # SimBackend Contract 测试
│   ├── test_registry.py    # Registry 测试
│   ├── test_backend_imports.py
│   ├── test_backend_pre_step_control.py
│   ├── test_motrix_backend_options.py
│   ├── test_mujoco_batch_env_*.py
│   └── test_reward_override.py
├── algos/                  # 算法层测试
│   ├── test_rsl_rl_ppo.py / test_rsl_rl_runner.py
│   ├── test_appo_*.py      # APPO (learner, runner, worker, staging, metrics)
│   ├── test_mlx_ppo.py
│   ├── test_offpolicy_*.py # Off-policy (runner, worker, runtime, bootstrap)
│   ├── test_fast_sac_*.py
│   ├── test_fast_td3_learner.py
│   ├── test_flash_sac_learner.py
│   ├── test_hora_*.py
│   └── test_him_ppo.py (implied by storage)
├── envs/                   # 环境层测试
│   ├── test_env_configs.py
│   ├── test_*_domain_randomization.py
│   ├── test_*_obs_noise.py
│   ├── test_motion_loader.py
│   └── test_sharpa.py
├── ipc/                    # IPC 层测试
│   ├── test_async_runner.py
│   ├── test_replay_buffer.py / test_replay_pipeline_double_buffer.py
│   ├── test_rollout_ring_buffer.py
│   ├── test_shared_obs_stats.py / test_shared_weight_sync.py
├── config/                 # 配置测试
│   ├── test_config_system.py
│   ├── test_locomotion_params.py
│   └── test_reward_injection.py
├── integration/            # 集成测试
│   ├── test_appo_rsl_reward.py
│   └── test_reward_injection_integration.py
├── dr/                     # Domain Randomization 测试
│   └── test_manager.py
├── terrains/               # 地形测试
│   └── test_terrain_generator.py
├── training/               # 训练辅助测试
│   ├── test_training_helpers.py
│   ├── test_resume_logger_state.py
│   └── test_seed_contract.py
├── utils/                  # 工具测试
├── visualization/          # 可视化测试
├── scripts/                # 脚本测试（hygiene, doc checks, 配置验证）
├── benchmark/              # 性能基准测试
└── cli/                    # CLI 入口测试
```

**测试命令**（via Makefile）：

```bash
make test          # pytest -m "not slow"
make test-cov      # pytest -m "not slow" --cov
make test-slow     # pytest -m "slow" -v
make test-all      # format + type + test-cov
```

---

## 9. 文档体系

```
docs/
├── README.md               # 文档索引
├── analy/                  # 分析文档
└── sphinx/                 # Sphinx 文档源码
    ├── Makefile
    ├── README.md           # 构建与部署说明
    ├── AGENTS.md           # Agent 写文档规范
    ├── requirements.txt
    └── source/
        ├── index.md        # Sphinx 入口
        ├── conf.py         # Sphinx 配置
        ├── glossary.md     # 术语表（共享）
        ├── changelog.md    # 变更日志（共享）
        ├── en/             # 英文文档
        │   ├── 1-getting_started/   # 快速开始
        │   ├── 2-user_guide/        # 用户指南（训练、算法、后端、任务、DR、地形、工具、操作）
        │   ├── 3-deployment/        # 部署（Sim-to-Real、Sim-to-Sim、框架迁移）
        │   ├── 4-developer_guide/   # 开发者指南（架构、Contract、扩展、贡献）
        │   └── 5-reference/         # 参考（API、术语表、Changelog、ADR、支持矩阵）
        ├── zh_CN/          # 中文文档（与 en/ 结构平行）
        ├── adr/            # Architecture Decision Records
        │   ├── ADR-0001-runtime-model-and-layer-boundaries.md
        │   ├── ADR-0002-backend-capability-boundary-for-play-and-snapshot.md
        │   ├── ADR-0003-task-owner-and-config-compose-contract.md
        │   ├── ADR-0004-registry-bootstrap-contract.md
        │   └── ADR-0005-unified-obs-critic-env-and-ipc-contract.md
        └── api_reference/  # API 自动文档（autodoc）
```

---

## 10. CI/CD

```
.github/
├── workflows/
│   ├── ci.yml              # 持续集成（lint + type + test）
│   └── docs.yml            # 文档构建与部署
├── CODEOWNERS
├── pull_request_template.md
└── ISSUE_TEMPLATE/
    ├── bug_report.yml
    ├── config.yml
    └── work_item.yml
```

---

## 附录：关键文件速查

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
| 旋转工具 | `src/unilab/envs/common/rotation.py` |
| 数学工具 | `src/unilab/envs/common/math.py` |
| MLX 旋转 | `src/unilab/algos/mlx/common/rotation.py` |
| 可视化 | `src/unilab/visualization/` |
| 开发指南 | `docs/sphinx/source/zh_CN/4-developer_guide/0-index.md` |
| 贡献流程 | `docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md` |
| ADR | `docs/sphinx/source/adr/` |
