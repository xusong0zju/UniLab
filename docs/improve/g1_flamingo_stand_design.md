# G1 金鸡独立（单腿站立）抗干扰策略 — 设计方案

> 状态：设计阶段 | 日期：2026-06-14 | 作者：xusong0zju

## 目标

为 Unitree G1 人形机器人训练一个"金鸡独立"（单腿站立 / Flamingo Stand）策略，能够在受到外部推力干扰时保持平衡不倒。

## 参考文献

### 核心论文
- **HuB**: Zhang et al., "Learning Extreme Humanoid Balance," CoRL 2025. [arXiv:2505.07294](https://arxiv.org/abs/2505.07294) — 最直接相关的工作，在 G1 上实现单腿平衡（"燕式平衡"、"李小龙踢"）
- **Classical Balance + RL**: Poddar et al., "Embedding Classical Balance Control Principles in RL," 2025. [arXiv:2603.08619](https://arxiv.org/abs/2603.08619) — 将 Capture Point、Centroidal Momentum 等经典平衡指标嵌入 RL reward/privileged critic
- **Gait-Conditioned Curriculum**: "Gait-Conditioned RL with Multi-Phase Curriculum," 2025. [arXiv:2505.20619](https://arxiv.org/abs/2505.20619) — G1 上的多阶段课程学习（站→走→跑）
- **HoST**: Huang et al., "Humanoid Standing-up Control," RSS 2025. [Project](https://taohuang13.github.io/humanoid-standingup.github.io/) — G1 从任意姿态站起，含抗扰恢复

### 开源参考
- Unitree RL Gym: [github.com/unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- Legged Gym (ETH): [github.com/leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
- Isaac Lab (NVIDIA): [github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

---

## 1. 总体策略

### 1.1 选择 Pose-Based Reward，不使用参考运动追踪

**理由**：
- 金鸡独立是一个**静态姿态**，不需要动态运动轨迹
- 用 target joint angles 定义姿态比准备 motion data 更简单、更可控
- UniLab 现有的 motion tracking 基础设施（motion_loader, 全身追踪 reward）是为动态舞蹈动作设计的，对静态姿态过于复杂
- HuB 论文也指出：对于静态平衡任务，relaxed tracking + COM/contact reward 比 exact pose matching 更有效

### 1.2 训练流程：三阶段课程

| 阶段 | 内容 | 迭代数（估算） |
|------|------|---------------|
| **Phase 1** | 双腿站立 + 基础平衡（零速度命令、无扰动） | ~500 iters |
| **Phase 2** | 单腿站立姿态（target pose 约束、无扰动） | ~1000 iters |
| **Phase 3** | 单腿站立 + 外部推力扰动 + 全域随机化 | ~1500+ iters |

**说明**：UniLab 使用 PPO 配置 2048 envs，每 iter 约 2048×20s/50Hz ≈ 819k steps。以上迭代数为保守估计。

### 1.3 算法选择

首选 **PPO**（与 UniLab 现有 walk 任务一致），理由：
- UniLab PPO 基础设施成熟，有 symmetry augmentation、empirical normalization
- 已有 2048 envs + 2200 iters 的 walk flat 成功经验
- PPO 对于这种连续控制 + 平衡任务已经在多篇论文中验证有效

备选：FlashSAC（如果 PPO 收敛不佳），使用了 `penalty_` 前缀 reward + walk profile 的配置模式。

---

## 2. Reward 设计

### 2.1 Reward 项总览

基于 UniLab 现有 `src/unilab/envs/locomotion/common/rewards.py` 和文献调研：

| Reward 键 | 权重 | 类型 | 公式/说明 |
|-----------|------|------|-----------|
| `orientation` | -5.0 | 惩罚 | `gravity_x² + gravity_y²`（保持躯干直立） |
| `base_height` | -200.0 | 惩罚 | `(height - target)²`（保持 COM 高度 0.754m） |
| `ang_vel_xy` | -0.5 | 惩罚 | `gyro_x² + gyro_y²`（抑制 roll/pitch 旋转） |
| `pose` | -0.2 | 惩罚 | `Σ w_i * (dof_pos_i - target_i)²`（追踪 flamingo 姿态） |
| `support_foot_contact` | +2.0 | 奖励 | 支撑脚必须着地（接触力 > 阈值） |
| `lifted_foot_contact` | -3.0 | 惩罚 | 抬起脚不得着地（接触力 > 阈值） |
| `com_over_support` | +1.0 | 奖励 | COM 水平投影在支撑脚多边形内 |
| `action_rate` | -0.02 | 惩罚 | `‖a_t - a_{t-1}‖²`（动作平滑） |
| `joint_torques` | -1e-6 | 惩罚 | `‖τ‖²`（能效） |
| `alive` | +5.0 | 奖励 | 常数存活奖励 |
| `termination` | -200.0 | 惩罚 | 跌倒时一次性惩罚 |

### 2.2 关键 Reward 详解

#### Pose Reward（姿态追踪）

```python
# target_joint_angles: 29-dim, 定义 flamingo 姿态
# 支撑腿（如右腿）：接近 default_angles（直立）
# 抬起腿（如左腿）：hip_pitch=-0.8, knee=1.2, ankle_pitch=-0.4（弯曲抬高）
# 手臂：自然下垂或略微张开以辅助平衡
# pose_weights: 对抬起腿关节、支撑腿踝关节给更高权重

pose_error = torch.sum(
    pose_weights * (dof_pos - target_joint_angles) ** 2,
    dim=-1
)
```

**Target Pose 定义**（以右腿支撑为例，所有值相对于 default_angles 的偏移）：

| 关节组 | 关节 | 目标值 (rad) | 权重 | 说明 |
|--------|------|-------------|------|------|
| 右腿（支撑） | hip_pitch/roll/yaw | default | 2.0 | 保持直立 |
| 右腿（支撑） | knee | default | 3.0 | 锁膝支撑 |
| 右腿（支撑） | ankle_pitch/roll | default | 5.0 | 脚掌平放 |
| 左腿（抬起） | hip_pitch | default - 1.0 | 10.0 | 大腿前抬 |
| 左腿（抬起） | hip_roll | default | 3.0 | 不外展 |
| 左腿（抬起） | knee | default + 1.5 | 8.0 | 膝关节弯曲 |
| 左腿（抬起） | ankle | default - 0.5 | 5.0 | 脚踝微屈 |
| 双臂 | 全部 | default | 0.5 | 辅助平衡 |
| 躯干 | waist | default | 1.0 | 躯干直立 |

#### Foot Contact Reward

```python
# 利用 MuJoCo contact sensor 或 contact force
# support_foot: 必须是 ground contact（鼓励着地）
# lifted_foot: 必须不是 ground contact（惩罚着地）
# threshold: 10N，区分真实接触和噪声

support_contact = contact_force[support_foot_idx] > threshold
support_reward = torch.where(support_contact, 1.0, -1.0)

lifted_contact = contact_force[lifted_foot_idx] > threshold
lifted_penalty = torch.where(lifted_contact, -1.0, 0.0)
```

#### COM-over-Support 奖励

```python
# COM 在支撑脚上方的投影
# support_foot_pos: 支撑脚踝关节的世界坐标
# com_xy: COM 的水平位置
# sigma: 0.02 (约 0.14m 容忍半径)

dist_com_to_foot = torch.norm(com_xy - support_foot_xy, dim=-1)
com_in_support = torch.exp(-dist_com_to_foot**2 / 0.02)
```

### 2.3 终止条件

| 条件 | 阈值 | 说明 |
|------|------|------|
| 倾斜角度 | `arccos(clip(gravity_z, -1, 1)) > 45°` | 比 walk 的 25° 宽松，允许更大的倾斜恢复 |
| 基座高度 | `base_height < 0.45m` | 比 walk 的 0.55m 更低，给跌倒恢复留空间 |
| 膝盖触地 | 支撑腿膝关节 contact > 阈值 | 防止跪地 |
| 最大步数 | 20s = 1000 steps @ 50Hz | 成功完成 |

---

## 3. 观测空间

### Actor 观测（约 95 维）

| 组分 | 维度 | 说明 |
|------|------|------|
| gyro | 3 | 陀螺仪（含噪声） |
| gravity | 3 | 投影重力向量（含噪声） |
| dof_pos | 29 | 关节位置 - default_angles（含噪声） |
| dof_vel | 29 | 关节速度（含噪声） |
| last_actions | 29 | 上一帧动作 |
| contact_states | 2 | 双脚接触状态（0/1，含噪声） |
| com_vel | 3 | COM 线速度（可选项，privileged） |

对比 G1WalkFlat actor obs：98 维（含 command 3 + gait_phase 2）。

**设计要点**：
- 去掉了 `command`（金鸡独立是静态任务，不需要速度指令）
- 去掉了 `gait_phase`（不需要周期步态）
- 加入了 `contact_states`（双脚接触信息对单腿站立至关重要）
- 可选加入 `com_vel`（帮助 policy 感知自身的运动趋势）

### Critic 观测（privileged，约 107 维）

Actor 观测的全部 +：

| 组分 | 维度 | 说明 |
|------|------|------|
| linvel | 3 | base 线速度（真值） |
| mass_offset | 1 | 基座质量偏移 |
| friction | 1 | 地面摩擦系数 |
| push_force | 3 | 当前施加的外部推力 |
| contact_forces | 4 | 双脚真实接触力（连续值） |

---

## 4. 扰动机制

### 4.1 利用现有 Push 基础设施

UniLab 已有完整的 push 机制（`DomainRandConfig.push_robots`），直接配置：

```yaml
env.domain_rand:
  push_robots: true
  push_interval: 150        # 每 3 秒推一次（walk: 750=15s, Go2 rough: 625）
  max_force: [2.5, 2.5, 1.0]  # x/y/z 方向最大推力 (N)
  push_body_name: "pelvis"  # 默认基座
```

| 对比 | push_interval | max_force | 说明 |
|------|--------------|-----------|------|
| Go2 rough | 625 (12.5s) | [1.0, 1.0, 0.5] | 四足 rough terrain |
| G1 deploy | 750 (15s) | [1.0, 1.0, 0.5] | 全身运动部署 |
| **金鸡独立** | **150 (3s)** | **[2.5, 2.5, 1.0]** | 单腿，需更频繁更强的推力 |

### 4.2 推力课程

| 训练阶段 | push_interval | max_force | 说明 |
|----------|--------------|-----------|------|
| Phase 1（双腿站立） | 禁用 | — | 先学会站 |
| Phase 2（单腿姿态） | 禁用 | — | 先学会抬腿 |
| Phase 3 早期 | 300 (6s) | [1.0, 1.0, 0.5] | 渐进引入扰动 |
| Phase 3 中期 | 200 (4s) | [2.0, 2.0, 0.8] | 增大扰动 |
| Phase 3 后期 | 150 (3s) | [2.5, 2.5, 1.0] | 最终目标强度 |

> **注意**：`push_interval` 和 `max_force` 需要作为 Hydra config 字段来支持课程调度。如果 DR provider 不支持动态调整，可以通过多次训练 + checkpoint 续训实现阶段切换。

### 4.3 额外扰动形式（可选增强）

UniLab 的 `IntervalRandomizationPlan` 还支持：
- **Body linear velocity delta**：直接给基座一个速度脉冲（类似 Isaac Gym/legged_gym 的 push_vel，更接近 HuB 论文的扰动方式）
- **Per-body force**：对特定身体部位（如躯干、抬起腿）施加定向力

可以考虑在 Phase 3 后期加入速度脉冲作为额外扰动形式。

---

## 5. 域随机化（Domain Randomization）

全开 DR 以提高 sim-to-real 鲁棒性：

```yaml
env.domain_rand:
  # 质量和惯量
  randomize_base_mass: true
  added_mass_range: [-2.0, 2.0]        # kg，比 walk 范围更大
  
  randomize_body_mass: true
  body_mass_multiplier_range: [0.85, 1.15]
  
  # 质心偏移（模拟负载不均衡）
  random_com: true
  com_offset_x: [-0.03, 0.03]          # m
  
  # 地面摩擦
  randomize_ground_friction: true
  ground_friction_multiplier_range: [0.4, 1.5]  # 比 walk [0.8,1.2] 更宽
  
  # PD 增益（模拟电机差异）
  randomize_kp: true
  kp_multiplier_range: [0.85, 1.15]
  randomize_kd: true
  kd_multiplier_range: [0.85, 1.15]
  
  # 重力方向微调（模拟地面不平）
  randomize_gravity: true
  gravity_range: [[0, 0, -9.81], [0.1, 0.1, -9.71]]
  
  # 推力扰动
  push_robots: true
  push_interval: 150
  max_force: [2.5, 2.5, 1.0]
```

---

## 6. 观测噪声

使用比 walk 更低的噪声水平（站立任务对精度要求更高）：

```yaml
env.noise_config:
  level: 1.0
  scale_joint_angle: 0.005     # walk: 0.01
  scale_joint_vel: 0.05        # walk: 0.3（PPO）
  scale_gyro: 0.05             # walk: 0.1（PPO）/ 0.2（SAC）
  scale_gravity: 0.02          # walk: 0.05
  scale_linvel: 0.05           # walk: 0.1
```

---

## 7. 代码实现计划

### 7.1 新建文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/g1/flamingo_stand.py` | `G1FlamingoStandCfg`、`G1FlamingoStandEnv`、reward 函数、DR provider |
| `conf/ppo/task/g1_flamingo_stand/mujoco.yaml` | PPO MuJoCo 训练配置 |
| `docs/improve/g1_flamingo_stand_design.md` | 本设计文档 |

### 7.2 修改文件

| 文件 | 变更 |
|------|------|
| `src/unilab/envs/locomotion/g1/__init__.py` | 添加 `from .flamingo_stand import ...` |

### 7.3 核心类结构

```python
# ─── 配置 dataclass ───────────────────────────────

@registry.envcfg("G1FlamingoStand")
@dataclass
class G1FlamingoStandCfg(G1BaseCfg):
    scene: SceneCfg = SceneCfg(model_file="scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = InitState(pos=[0.0, 0.0, 0.754])
    reward_config: FlamingoStandRewardConfig
    domain_rand: G1FlamingoStandDomainRandCfg
    noise_config: NoiseConfig
    # 新增 flamingo-specific 配置
    support_leg: str = "random"  # "right" | "left" | "random"
    target_pose: FlamingoPoseConfig  # target joint angles

# ─── 环境类 ───────────────────────────────────────

class G1FlamingoStandEnv(G1BaseEnv):
    """G1 金鸡独立环境。继承 G1BaseEnv，自定义 standing-specific reward 和 termination。"""
    
    def _init_reward_functions(self):
        """注册 standing-specific reward 函数。"""
        ...
    
    def _compute_reward(self):
        """计算总 reward。"""
        ...
    
    def _check_termination(self):
        """检查终止条件（倾斜、高度、膝盖触地）。"""
        ...
    
    def _init_obs_buf(self):
        """初始化观测 buffer（无 gait_phase 和 command，有 contact_states）。"""
        ...
```

### 7.4 与现有架构的集成点

| 功能 | 位置 | 复用方式 |
|------|------|---------|
| Base env (noise, control, sensor) | `src/unilab/envs/locomotion/g1/base.py` | 继承 `G1BaseEnv` |
| Symmetry augmentation | `src/unilab/envs/locomotion/g1/symmetry.py` | 复用 `G1SymmetryAugmentation` |
| DR (mass, friction, etc.) | `src/unilab/dr/dr_utils.py` | 复用 `build_common_reset_randomization` |
| Push / perturbation | `src/unilab/envs/locomotion/common/domain_rand.py` | 复用 `DomainRandConfig.push_robots` |
| Reward dispatch | `src/unilab/envs/locomotion/common/rewards.py` | 复用 `RewardDispatch.run_reward_dispatch` |
| Task registry | `src/unilab/base/registry.py` | `@registry.envcfg` + `registry.register_env` |

---

## 8. 训练命令

```bash
# Phase 1-3 统一训练
uv run scripts/train_rsl_rl.py \
  task=g1_flamingo_stand/mujoco \
  training.sim_backend=mujoco

# 训练后 MuJoCo 可视化
uv run scripts/play_interactive.py \
  --algo ppo \
  --task g1_flamingo_stand \
  --sim mujoco \
  interactive.action_mode=policy

# Web 可视化
uv run scripts/play_viser.py \
  task=g1_flamingo_stand/mujoco \
  interactive.action_mode=policy
```

---

## 9. 验证方案

### 9.1 仿真内验证

| # | 验证项 | 通过标准 |
|---|--------|---------|
| 1 | 静态稳定性 | Policy 能在无扰动下维持单腿站立 ≥ 20s |
| 2 | 推力恢复 | 在随机方向 2.5N 推力下，恢复率 > 90% |
| 3 | DR 鲁棒性 | 在不同质量/摩擦/PD 参数下，成功率 > 80% |
| 4 | 左右对称 | 左右腿都能独立执行金鸡独立 |
| 5 | 连续推力 | 承受连续 3 次间隔 3s 的推力不倒 |

### 9.2 验证方法

```bash
# 1. 训练
uv run scripts/train_rsl_rl.py task=g1_flamingo_stand/mujoco

# 2. MuJoCo viewer 回放 + 手动推力测试
uv run scripts/play_interactive.py --algo ppo --task g1_flamingo_stand --sim mujoco

# 3. 批量评估（如工具可用）
uv run scripts/eval.py task=g1_flamingo_stand/mujoco checkpoint=<path>
```

---

## 10. 潜在风险与对策

| 风险 | 可能性 | 影响 | 对策 |
|------|--------|------|------|
| 单腿站立 RL 找不到可行解 | 中 | 高 | Phase 1 双腿站立先行，确保基础平衡后再过渡；降低 target pose 的抬腿高度 |
| 对称性问题（偏好一条腿） | 中 | 中 | `support_leg: random` + symmetry augmentation |
| 抬起腿抖动/不自然 | 高 | 低 | 增加 pose/action_rate weight；对抬起腿关节加大 penalty |
| Policy 学会"跳步"而非"站稳" | 中 | 中 | 增加 foot_contact penalty；限制推力上限 |
| 过拟合特定扰动方向 | 低 | 中 | 随机推力方向 + DR 多样性覆盖 |
| MuJoCo contact sensor 缺失/不可靠 | 低 | 高 | 优先检查 G1 XML 中 foot 的 contact sensor 配置；如缺失则从 contact force 间接获取 |
| 推力课程动态调整不被 DR provider 支持 | 中 | 中 | 使用多次训练 + 从 checkpoint 续训模拟阶段切换 |
| Phase 1/2 到 Phase 3 衔接时 policy 崩溃 | 中 | 高 | 在 Phase 3 使用较小的初始推力，或从 Phase 2 checkpoint warm-start |

---

## 11. 后续扩展方向

1. **Sim-to-real 部署**：冻结 policy，通过 `scripts/deploy/export_deploy_config.py` 导出部署配置
2. **更多姿态**：扩展为多姿态金鸡独立（燕式平衡、侧抬腿等），通过 `support_leg` + `target_pose` 参数化
3. **动态过渡**：双腿站立 ↔ 单腿站立的动态切换
4. **地形泛化**：在斜坡、粗糙地面上训练金鸡独立
5. **扰动升级**：加入旋转推力、连续推力、推力 + 视觉干扰的组合扰动

---

## UniLab 内部参考

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/g1/joystick.py` | G1 Walk Flat 完整实现（环境、reward、DR、课程） |
| `src/unilab/envs/locomotion/g1/base.py` | G1 Base Env（noise、control、sensor 配置） |
| `src/unilab/envs/locomotion/common/base.py` | LocomotionBaseEnv（action apply、reset 逻辑） |
| `src/unilab/envs/locomotion/common/rewards.py` | 通用 reward 函数库 |
| `src/unilab/envs/locomotion/common/domain_rand.py` | DomainRandConfig + Push 配置 |
| `src/unilab/dr/dr_utils.py` | DR provider 构建工具（含 push_plan） |
| `src/unilab/base/backend/mujoco/backend.py` | MuJoCo 后端 push_robots() 实现 |
| `src/unilab/envs/locomotion/g1/symmetry.py` | G1 symmetry augmentation |
| `src/unilab/envs/locomotion/go2/footstand.py` | Go2 双足站立参考（reward 设计借鉴） |
| `conf/ppo/task/g1_walk_flat/mujoco.yaml` | PPO 训练配置模板 |
