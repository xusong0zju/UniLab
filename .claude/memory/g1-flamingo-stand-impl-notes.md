---
name: g1-flamingo-stand-impl-notes
description: G1金鸡独立实现过程中的踩坑记录与技术要点
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

## 实现要点

### 1. 继承 G1BaseEnv 而非 G1WalkEnv
FlamingoStandEnv 直接继承 `G1BaseEnv`，不需要 `G1WalkEnv` 的 gait_phase、commands、walk-specific reward。G1BaseEnv 提供 noise、control、sensor、default_angles 等基础能力。

### 2. DR Provider 需要提供 baselines
`LocomotionDRProvider` 的 `build_reset_plan` 会调用 `build_common_reset_randomization`，该函数需要 `base_body_mass`（body mass 随机化）、`base_geom_friction`（摩擦随机化）等 baseline 数据。必须在 DR Provider 中实现 `_get_reset_randomization_baselines()` 并返回这些数据。

### 3. Contact sensor 在 reset 路径需要切片
`_compute_obs` 在 reset 路径被 `_compute_reset_obs` 调用时，传入的数据是 subset（仅 `env_ids`），但 `compute_aggregated_foot_contact` 返回全部 env 的数据。需要用 `dof_pos.shape[0]` 做切片对齐。

### 4. Phase 1 必须禁用单腿相关 penalty
Phase 1（双腿站立）如果启用了 `penalty_lifted_foot_contact` 和 `penalty_support_foot_contact`，会惩罚正确的站立行为。Phase 1 中两者权重设为 0.0，Phase 2 再启用。

### 5. Config 字段必须与 dataclass 对齐
Hydra 的 `build_task_env_cfg_override` 会将整个 `env:` section 作为 override dict 传入 `apply_cfg_overrides`。如果 YAML 中有 `reset_base_qvel_limit` 等字段，对应的 dataclass 必须声明该属性，否则会报 `has no attribute`。

### 6. 训练命令格式
Off-policy 训练必须同时指定 task 和 algo：
```bash
uv run scripts/train_offpolicy.py task=flashsac/g1_flamingo_stand/mujoco algo=flashsac
```
`task=flashsac/...` 中的 `flashsac` 是 task group 前缀，必须与 `algo=flashsac` 一致，否则 `assert_offpolicy_task_choice_matches_algo` 会报错。

### 7. Checkpoint warm-start
通过 `algo.load_run=<绝对路径>` 加载 checkpoint。可以传 `.pt` 文件路径或 run 目录路径。

**Why**: 以上 7 点是实际调试过程中遇到的阻塞问题，已全部解决。

**How to apply**: 后续开发类似任务时参考这些要点，避免重复踩坑。
