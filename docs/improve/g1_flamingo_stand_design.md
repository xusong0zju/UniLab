# G1 金鸡独立（单腿站立）抗干扰策略 — 设计方案

> 状态：设计阶段 | 日期：2026-06-14 | 作者：xusong0zju
>
> **算法**：FlashSAC | **抗扰核心**：手臂 + 腰部补偿

## 目标

为 Unitree G1 人形机器人训练一个"金鸡独立"（单腿站立 / Flamingo Stand）策略。核心思路是：**通过手臂摆动和腰部旋转来主动补偿外部推力**，在仅单脚着地的情况下维持平衡。这模拟了人类单腿站立时的自然策略——用上肢的大幅度运动来抵消 COM 偏移，而非仅仅依赖支撑腿的踝/膝微调。

## 参考文献

### 核心论文
- **HuB**: Zhang et al., "Learning Extreme Humanoid Balance," CoRL 2025. [arXiv:2505.07294](https://arxiv.org/abs/2505.07294) — 在 G1 上实现单腿平衡（"燕式平衡"、"李小龙踢"），relaxed tracking + 高频推力
- **Classical Balance + RL**: Poddar et al., "Embedding Classical Balance Control Principles in RL," 2025. [arXiv:2603.08619](https://arxiv.org/abs/2603.08619) — Capture Point、Centroidal Momentum 嵌入 RL reward/privileged critic
- **Gait-Conditioned Curriculum**: "Gait-Conditioned RL with Multi-Phase Curriculum," 2025. [arXiv:2505.20619](https://arxiv.org/abs/2505.20619) — G1 上的多阶段课程（站→走→跑）

### 开源参考
- Unitree RL Gym: [github.com/unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- Legged Gym (ETH): [github.com/leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
- Isaac Lab (NVIDIA): [github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

---

## 1. 总体策略

### 1.1 算法选择：FlashSAC

选择 **FlashSAC**（而非 PPO），理由：

- **更好的 exploration**：SAC 的 entropy-regularized 策略在平衡任务中探索更充分，不容易陷入"僵硬站住→轻微扰动就倒"的局部最优
- **Off-policy 样本效率**：replay buffer 复用历史经验，对稀有事件（被推后恢复）的学习更高效
- **UniLab 已有成熟配置**：`conf/offpolicy/task/flashsac/g1_walk_flat/` 下有 13 个变体（含 GC、CC、CTC、IMU-GC 等），均已在 G1 上验证
- **PenaltyCurriculum 内置**：FlashSAC 配置天然使用 `penalty_` 前缀 reward + `PenaltyCurriculum`，可以根据 episode 长度自动调节惩罚强度，天然适合"先学会站、再学会抗扰"的课程

### 1.2 抗扰核心思路：手臂 + 腰部补偿

人类单腿站立被推时，不是靠支撑腿的脚踝微调（那只能应对极小扰动），而是：

1. **腰部旋转**：扭转躯干，利用上半身的转动惯量产生反力矩
2. **手臂大幅度摆动**：像走钢丝的人一样，手臂向 COM 偏移的反方向挥动，产生补偿力矩
3. **支撑腿微调**：踝关节和髋关节做小幅度角度调整

因此，我们的设计原则是：
- **手臂关节的 pose 权重极低或为零**——给 policy 最大的自由度去"挥舞"手臂
- **腰部关节的 pose 权重较低**——允许一定范围的躯干旋转来调整 COM
- **支撑腿的 pose 权重较高**——结构稳定性要求，踝/膝不能乱动
- **抬起腿的 pose 权重较高**——维持 flamingo 姿态的外观

### 1.3 训练流程：三阶段课程（通过 PenaltyCurriculum 自动调度）

| 阶段 | 内容 | 说明 |
|------|------|------|
| **Phase 1** | 双腿站立 + 基础平衡 | `rel_standing_envs=1.0`（全部零速度命令），无推力 |
| **Phase 2** | 引入单腿姿态 | 从双腿站立 warm-start，逐步增大 pose reward 权重 |
| **Phase 3** | 外部推力 + 全域随机化 | 推力渐进增强，PenaltyCurriculum 自动调节 |

**FlashSAC 默认配置**：4096 envs，5000 iters，每 iter 约 4096×20s/50Hz ≈ 1.6M steps。

> **课程实现方式**：FlashSAC 的 `PenaltyCurriculum` 根据 episode 平均长度自动调节 negative reward 的 scale。训练初期 penalty 减轻（允许更多探索），随着 episode 变长，penalty 逐渐恢复到全量。Phase 1→2 的过渡通过从 checkpoint warm-start 实现；Phase 2→3 的推力渐进通过多次训练 + checkpoint 续训实现。

---

## 2. Reward 设计

### 2.1 Reward 项总览（FlashSAC `penalty_` 命名惯例）

| Reward 键 | 权重 | 类型 | 公式/说明 |
|-----------|------|------|-----------|
| `penalty_orientation` | -20.0 | 惩罚 | `gravity_x² + gravity_y²`（保持躯干直立，高权重是关键） |
| `base_height` | -200.0 | 惩罚 | `(height - 0.754)²`（保持 COM 高度） |
| `penalty_ang_vel_xy` | -2.0 | 惩罚 | `gyro_x² + gyro_y²`（抑制 roll/pitch 旋转） |
| `pose` | -0.5 | 惩罚 | `Σ w_i * (dof_pos_i - target_i)²`（**分区分权重**，见 §2.2） |
| `penalty_support_foot_contact` | -5.0 | 惩罚 | 支撑脚离地时惩罚 |
| `penalty_lifted_foot_contact` | -10.0 | 惩罚 | 抬起脚着地时惩罚（高权重防止"作弊"） |
| `com_over_support` | +2.0 | 奖励 | COM 水平投影在支撑脚多边形内 |
| `penalty_action_rate` | -5.0 | 惩罚 | `‖a_t - a_{t-1}‖²`（动作平滑） |
| `alive` | +10.0 | 奖励 | 常数存活奖励 |
| `penalty_feet_ori` | -25.0 | 惩罚 | 脚掌着地时保持水平（复用 walk 的 reward） |
| `penalty_close_feet_xy` | -10.0 | 惩罚 | 双脚距离不能太近（防止抬起脚蹭地） |

**对比 FlashSAC G1WalkFlat 的 reward**：去掉了 `tracking_lin_vel`、`tracking_ang_vel`、`feet_phase`（不需要步态），加入了 `com_over_support`、`penalty_support_foot_contact`、`penalty_lifted_foot_contact`。

### 2.2 核心创新：分区分权重的 Pose Reward

这是本设计的核心。不同身体部位的关节被赋予完全不同的 pose 追踪权重，反映它们在抗扰策略中的不同角色：

| 区域 | 关节（索引范围） | 角色 | Pose 权重 | 说明 |
|------|-----------------|------|----------|------|
| **手臂** | 12-23（shoulder/elbow/wrist） | **主动抗扰执行器** | **0.0 ~ 0.1** | 极低约束，给 policy 最大自由度去挥舞手臂产生反力矩 |
| **腰部** | 0-2（waist joints） | **辅助抗扰执行器** | **1.0** | 允许一定范围的躯干旋转来调整 COM |
| **支撑腿** | 对应侧的 6 个 leg joint | **结构支柱** | **10.0 ~ 20.0** | 高约束，踝/膝不能乱动，必须保持稳定支撑 |
| **抬起腿** | 对应侧的 6 个 leg joint | **姿态定义** | **20.0 ~ 30.0** | 最高约束，维持 flamingo 姿态外观 |

**设计理念**：

```
外部推力 → COM 偏移 → policy 感知到重力向量变化
                                    ↓
                    ┌─ 手臂大幅度反向挥动（低 pose 约束允许大动作）
                    ├─ 腰部旋转产生反力矩（中 pose 约束允许适度旋转）
                    └─ 支撑腿踝/膝微调（高 pose 约束，仅小幅调整）
                                    ↓
                            COM 回归支撑脚上方
```

**与 Walk 任务的 pose_weights 对比**：

| 区域 | Walk FlashSAC | Flamingo Stand |
|------|--------------|----------------|
| 腿部（索引 0-11） | 0.01 ~ 5.0 | 10.0 ~ 30.0（区分支撑/抬起） |
| 手臂（索引 12-23） | 50.0（极严格） | **0.0 ~ 0.1**（极宽松） |
| 其他（索引 24-28） | 50.0 | 1.0 |

Walk 任务中手臂权重 50.0 是为了让走路时手臂保持自然姿态；Flamingo Stand 中手臂权重近乎零是为了让手臂成为**主动平衡器官**。

### 2.3 Target Pose 定义（以右腿支撑为例）

所有值为相对于 `default_angles` 的偏移：

| 关节组 | 关节 | 目标偏移 (rad) | 说明 |
|--------|------|---------------|------|
| 右腿（支撑） | hip_pitch/roll/yaw | 0.0 | 保持直立 |
| 右腿（支撑） | knee | 0.0 | 锁膝支撑 |
| 右腿（支撑） | ankle_pitch/roll | 0.0 | 脚掌平放 |
| 左腿（抬起） | hip_pitch | -1.0 | 大腿前抬 |
| 左腿（抬起） | hip_roll | 0.0 | 不外展 |
| 左腿（抬起） | knee | +1.5 | 膝关节弯曲 |
| 左腿（抬起） | ankle | -0.5 | 脚踝微屈 |
| 双臂 | 全部 | 0.0（但权重 ~0） | 名义 default，实际 policy 自由偏离 |
| 腰部 | waist | 0.0（但权重低） | 名义 default，实际允许旋转 |

### 2.4 Foot Contact Reward

```python
# 复用现有 contact sensor 机制（joystick.py 已有 LEFT_FOOT_CONTACT_SENSORS 等）
# 支撑脚接触：reward
# 抬起脚接触：heavy penalty（防止 policy 靠两只脚站着作弊）

support_contact = compute_aggregated_foot_contact(backend, SUPPORT_FOOT_SENSORS)
lifted_contact = compute_aggregated_foot_contact(backend, LIFTED_FOOT_SENSORS)

penalty_support = torch.where(support_contact, 0.0, -1.0)   # 支撑脚离地→惩罚
penalty_lifted = torch.where(lifted_contact, -1.0, 0.0)     # 抬起脚着地→惩罚
```

### 2.5 COM-over-Support 奖励

```python
# COM 水平投影 vs 支撑脚位置
# 鼓励 policy 主动将 COM 保持在支撑脚上方
dist = torch.norm(com_xy - support_foot_xy, dim=-1)
com_in_support = torch.exp(-dist**2 / 0.02)  # sigma ≈ 0.14m
```

### 2.6 终止条件

| 条件 | 阈值 | 说明 |
|------|------|------|
| 倾斜角度 | `arccos(clip(gravity_z, -1, 1)) > 60°` | FlashSAC walk 用 65°，standing 用 60° |
| 基座高度 | `base_height < 0.35m` | 比 walk 的 0.3m 略高 |
| 膝盖触地 | 支撑腿膝关节 contact > 阈值 | 防止跪地 |
| 最大步数 | 20s = 1000 steps @ 50Hz | 成功完成 |

---

## 3. 观测空间

### Actor 观测（约 95 维）

沿用 FlashSAC walk profile 惯例（`penalty_` 前缀触发 walk observation profile 的 gyro/dof_vel 缩放）：

| 组分 | 维度 | 说明 |
|------|------|------|
| gyro | 3 | 陀螺仪（walk profile: scale × 0.25） |
| gravity | 3 | 投影重力向量（取反） |
| dof_pos | 29 | 关节位置 - default_angles |
| dof_vel | 29 | 关节速度（walk profile: scale × 0.05） |
| last_actions | 29 | 上一帧动作 |
| contact_states | 2 | 双脚聚合接触状态（0/1） |

对比 G1WalkFlat actor obs（98 维）：去掉了 `command`（3）和 `gait_phase`（2），加入了 `contact_states`（2）。

### Critic 观测（privileged，约 107 维）

Actor 观测的全部 +：

| 组分 | 维度 | 说明 |
|------|------|------|
| linvel | 3 | base 线速度（walk profile: scale × 2.0） |
| mass_offset | 1 | 基座质量偏移 |
| friction | 1 | 地面摩擦系数 |
| push_force | 3 | 当前施加的外部推力（真值） |
| contact_forces | 4 | 双脚真实接触力（连续值，左右各 2） |

### Obs Groups Spec

```python
# obs: gyro(3) + gravity(3) + diff(29) + dof_vel(29) + action(29) + contact(2) = 95
# critic: obs(95) + linvel(3) = 98
# 可通过扩展 privileged info 增加 critic 维度
```

---

## 4. 扰动机制

### 4.1 利用现有 Push 基础设施

UniLab 已有完整的 push 机制（`DomainRandConfig.push_robots`），直接配置：

```yaml
env.domain_rand:
  push_robots: true
  push_interval: 150        # 每 3 秒推一次
  max_force: [3.0, 3.0, 1.5]  # x/y/z 方向最大推力 (N)
  push_body_name: "pelvis"
```

| 对比 | push_interval | max_force | 说明 |
|------|--------------|-----------|------|
| Go2 rough | 625 (12.5s) | [1.0, 1.0, 0.5] | 四足 |
| G1 deploy | 750 (15s) | [1.0, 1.0, 0.5] | 全身运动 |
| **金鸡独立** | **150 (3s)** | **[3.0, 3.0, 1.5]** | 单腿，高频大推力 |

**推力方向是随机的**（uniform 在 `[-max_force, max_force]` 各轴），迫使 policy 学会应对各方向的推力。这正是手臂 + 腰部补偿发挥价值的场景——不同方向的推力需要不同的上肢补偿策略。

### 4.2 推力课程（多次训练 + checkpoint 续训）

| 训练轮次 | push_interval | max_force | 说明 |
|----------|--------------|-----------|------|
| Round 0（Phase 1） | 禁用 | — | 先学会双腿站 |
| Round 1（Phase 2） | 禁用 | — | 先学会单腿站 |
| Round 2（Phase 3 早期） | 300 | [1.0, 1.0, 0.3] | 小推力热身 |
| Round 3（Phase 3 中期） | 200 | [2.0, 2.0, 0.8] | 增大推力 |
| Round 4（Phase 3 后期） | 150 | [3.0, 3.0, 1.5] | 目标推力 |

每次从上一轮的 best checkpoint 续训。

### 4.3 额外扰动形式（可选增强）

- **速度脉冲**（`body_linear_velocity_delta`）：复制 HuB 论文的扰动方式，每次给基座 0.3~0.5 m/s 的速度脉冲
- **定向推力**：除了 random push，还可以周期性地施加朝向特定方向的推力，测试 policy 的定向抗扰

---

## 5. 域随机化（Domain Randomization）

参照 FlashSAC G1WalkFlat 的 DR 配置，全开：

```yaml
env.domain_rand:
  # 质量
  randomize_base_mass: true
  added_mass_range: [-2.5, 2.5]       # kg
  randomize_body_mass: true
  body_mass_multiplier_range: [0.85, 1.15]

  # 质心
  random_com: true
  com_offset_x: [-0.03, 0.03]         # m

  # 地面
  randomize_ground_friction: true
  ground_friction_multiplier_range: [0.4, 1.5]

  # PD 增益（FlashSAC 惯例：开启）
  randomize_kp: true
  kp_multiplier_range: [0.85, 1.15]
  randomize_kd: true
  kd_multiplier_range: [0.85, 1.15]

  # 重力
  randomize_gravity: true
  gravity_range: [[0.0, 0.0, -9.81], [0.1, 0.1, -9.71]]

  # 推力
  push_robots: true
  push_interval: 150
  max_force: [3.0, 3.0, 1.5]
```

---

## 6. 观测噪声

FlashSAC 惯例是 noise level 较低（off-policy 不需要 on-policy 那么强的噪声做 exploration）：

```yaml
env.noise_config:
  level: 0.0             # FlashSAC walk flat 也用 0.0
  scale_gyro: 0.0        # 无噪声（或训练后期逐步加入）
  scale_gravity: 0.0
  scale_joint_angle: 0.005
  scale_joint_vel: 0.05
  scale_linvel: 0.0
```

> 训练初期 noise=0 帮助快速学会基础平衡；后期可以通过 `level: 1.0` + 非零 scale 加入噪声提升 sim-to-real 鲁棒性。

---

## 7. 代码实现计划

### 7.1 新建文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/g1/flamingo_stand.py` | `G1FlamingoStandCfg`、`G1FlamingoStandEnv`、reward 函数 |
| `conf/offpolicy/task/flashsac/g1_flamingo_stand/mujoco.yaml` | FlashSAC MuJoCo 训练配置 |
| `docs/improve/g1_flamingo_stand_design.md` | 本设计文档 |

### 7.2 修改文件

| 文件 | 变更 |
|------|------|
| `src/unilab/envs/locomotion/g1/__init__.py` | 添加 `from .flamingo_stand import ...` |

### 7.3 核心类结构

```python
# ─── 配置 dataclass ───────────────────────────────

@dataclass
class FlamingoStandRewardConfig:
    """金鸡独立专用 reward 配置。"""
    scales: dict[str, float]            # reward 权重
    base_height_target: float = 0.754
    min_base_height: float = 0.35
    max_tilt_deg: float = 60.0
    pose_weights: list[float]           # 29-dim 分区分权重（见 §2.2）
    target_pose: list[float]            # 29-dim target joint angles
    support_leg: str = "random"         # "right" | "left" | "random"
    close_feet_threshold: float = 0.15
    contact_force_threshold: float = 10.0  # N

@registry.envcfg("G1FlamingoStand")
@dataclass
class G1FlamingoStandCfg(G1BaseCfg):
    scene: SceneCfg = SceneCfg(model_file="scene_flat.xml")
    max_episode_seconds: float = 20.0
    init_state: InitState = InitState(pos=[0.0, 0.0, 0.754])
    reward_config: FlamingoStandRewardConfig
    domain_rand: G1FlamingoStandDomainRandCfg
    noise_config: NoiseConfig
    curriculum: CurriculumConfig

# ─── 环境类 ───────────────────────────────────────

class G1FlamingoStandEnv(G1BaseEnv):
    """G1 金鸡独立环境。"""

    def _init_reward_functions(self):
        """注册 reward: orientation, base_height, ang_vel_xy,
        pose(分区权重), support_foot_contact, lifted_foot_contact,
        com_over_support, action_rate, alive, feet_ori, close_feet_xy."""

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        """观测：无 command、无 gait_phase、有 contact_states。"""

    def _check_termination(self):
        """倾斜 60°、高度 0.35m、膝盖触地。"""

    def _get_support_and_lifted_indices(self):
        """根据 support_leg 返回支撑脚/抬起脚的 contact sensor 列表。"""
```

### 7.4 与现有架构的集成点

| 功能 | 位置 | 复用方式 |
|------|------|---------|
| Base env (noise, control, sensor) | `src/unilab/envs/locomotion/g1/base.py` | 继承 `G1BaseEnv` |
| Foot contact 检测 | `joystick.py`:`compute_aggregated_foot_contact` | 直接调用 |
| Symmetry augmentation | `src/unilab/envs/locomotion/g1/symmetry.py` | 复用 `G1SymmetryAugmentation` |
| DR (mass, friction, kp/kd, push) | `src/unilab/dr/dr_utils.py` | 复用 `build_common_reset_randomization` |
| Push / perturbation | `src/unilab/envs/locomotion/common/domain_rand.py` | 复用 `DomainRandConfig.push_robots` |
| PenaltyCurriculum | `src/unilab/base/curriculum.py` | 复用 `PenaltyCurriculum` |
| Reward dispatch | `src/unilab/envs/locomotion/common/rewards.py` | 复用 `run_reward_dispatch` |
| Task registry | `src/unilab/base/registry.py` | `@registry.envcfg` + `registry.register_env` |
| Contact sensors | `joystick.py`:`LEFT_FOOT_CONTACT_SENSORS`, `RIGHT_FOOT_CONTACT_SENSORS` | 直接引用 |

---

## 8. 训练命令

```bash
# Round 0: Phase 1 双腿站立（无推力）
uv run scripts/train_offpolicy.py \
  task=flashsac/g1_flamingo_stand/mujoco \
  training.sim_backend=mujoco

# Round 1: Phase 2 单腿姿态（无推力，从 Round 0 checkpoint warm-start）
uv run scripts/train_offpolicy.py \
  task=flashsac/g1_flamingo_stand/mujoco \
  training.sim_backend=mujoco \
  algo.checkpoint=<round0_best>

# Round 2-4: Phase 3 推力渐进（每轮增大 max_force，缩短 push_interval）
uv run scripts/train_offpolicy.py \
  task=flashsac/g1_flamingo_stand/mujoco \
  training.sim_backend=mujoco \
  env.domain_rand.push_robots=true \
  env.domain_rand.push_interval=300 \
  env.domain_rand.max_force=[1.0,1.0,0.3] \
  algo.checkpoint=<round1_best>

# 训练后可视化
uv run scripts/play_interactive.py \
  --algo sac --task g1_flamingo_stand --sim mujoco \
  interactive.action_mode=policy
```

---

## 9. 验证方案

### 9.1 仿真内验证

| # | 验证项 | 通过标准 |
|---|--------|---------|
| 1 | 静态单腿站立 | 无扰动下维持 ≥ 20s |
| 2 | 推力恢复（2D） | 随机水平方向 3.0N 推力，恢复率 > 90% |
| 3 | DR 鲁棒性 | 不同质量/摩擦/PD 参数，成功率 > 80% |
| 4 | 左右对称 | 左右腿各作为支撑腿均能站稳 |
| 5 | 连续推力 | 连续 3 次推力（间隔 3s）不倒 |
| 6 | **手臂参与度** | 可视化确认手臂在推力后有明显补偿运动 |
| 7 | **腰部参与度** | 可视化确认腰部在推力后有旋转补偿 |

### 9.2 验证方法

```bash
# 训练
uv run scripts/train_offpolicy.py task=flashsac/g1_flamingo_stand/mujoco

# MuJoCo viewer 回放（观察手臂/腰部补偿行为）
uv run scripts/play_interactive.py --algo sac --task g1_flamingo_stand --sim mujoco

# Web 可视化
uv run scripts/play_viser.py task=g1_flamingo_stand/mujoco
```

---

## 10. 潜在风险与对策

| 风险 | 可能性 | 影响 | 对策 |
|------|--------|------|------|
| 手臂权重为 0 导致不自然的乱挥 | 中 | 中 | 保留极小权重（0.05），或加 arm_symmetry reward（鼓励双臂对称运动） |
| 单腿站立 RL 找不到可行解 | 中 | 高 | 从双腿站立 warm-start；降低 target pose 抬腿高度；先用低抬腿再渐进抬高 |
| 抬起腿着地"作弊" | 高 | 中 | `penalty_lifted_foot_contact` 给极高权重（-10.0），让作弊代价 > 收益 |
| Policy 过度依赖腿部而不用手臂 | 中 | 高 | 通过消融实验对比（手臂高权重 vs 低权重），确认手臂权重设计的有效性 |
| 对称性问题 | 中 | 中 | `support_leg: random` + symmetry augmentation |
| 推力方向不平衡（policy 只适应某个方向） | 低 | 中 | random push 方向均匀采样 + DR 多样性 |

---

## 11. 消融实验计划

为验证"手臂+腰部补偿"的有效性，建议做以下对比实验：

| 实验 | 手臂权重 | 腰部权重 | 预期结果 |
|------|---------|---------|---------|
| A | 50.0（walk 默认） | 50.0 | 抗扰能力弱，手臂僵硬，被推即倒 |
| B | 1.0 | 5.0 | 一定的抗扰能力 |
| C | **0.05**（推荐） | **1.0** | **最佳抗扰，手臂腰部主动补偿** |
| D | 0.0（完全自由） | 0.0 | 手臂可能乱挥，外观不自然 |

---

## 12. 后续扩展

1. **多姿态**：燕式平衡（躯干水平）、侧抬腿等不同单腿姿态
2. **Sim-to-real 部署**：冻结 policy → `scripts/deploy/export_deploy_config.py`
3. **动态过渡**：双腿 ↔ 单腿的动态切换，模拟武术动作
4. **地形泛化**：在斜坡/粗糙地面/软地面上训练
5. **组合扰动**：旋转推力 + 接触力 + 视觉干扰

---

## UniLab 内部参考

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/g1/joystick.py` | G1 Walk 完整实现（env、reward、DR provider、contact detection） |
| `src/unilab/envs/locomotion/g1/base.py` | G1 Base Env（noise、control、sensor 配置） |
| `src/unilab/envs/locomotion/common/rewards.py` | 通用 reward 函数库（可复用的 orientation、base_height 等） |
| `src/unilab/envs/locomotion/common/domain_rand.py` | `DomainRandConfig` + push 配置 |
| `src/unilab/dr/dr_utils.py` | DR provider 构建工具（`build_common_reset_randomization`、`build_interval_push_plan`） |
| `src/unilab/base/curriculum.py` | `PenaltyCurriculum`（FlashSAC 自动课程调度） |
| `src/unilab/envs/locomotion/g1/symmetry.py` | G1 symmetry augmentation（26 对关节镜像映射） |
| `conf/offpolicy/task/flashsac/g1_walk_flat/mujoco.yaml` | FlashSAC G1 配置模板 |
| `scripts/train_offpolicy.py` | Off-policy 训练入口（SAC/TD3/FlashSAC） |
