---
name: g1-flamingo-stand-project
description: G1金鸡独立训练+Sim2Sim全部完成，Phase1/2/3均收敛，64/64通过Full DR+3N推力
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

# G1 Flamingo Stand (金鸡独立) 项目状态

## 最终结果 ✅ 全部完成

| 阶段 | 收敛 ep len | Timeout | 说明 |
|------|------------|---------|------|
| Phase 1 双腿站立 | 990 | — | 3500 iters, DR全开 |
| Phase 2 单腿金鸡独立 | 991 | 99.1% | 3500 iters, lifted_foot_contact=0 |
| Phase 3 轻推 (1N/6s) | 1000 | 99.2% | 2500 iters |
| **Sim2Sim** | **1000** | **100% (64/64)** | **3N/3s + Full DR: mass±2.5kg, friction 0.4-1.5, PD 0.85-1.15, com offset, gravity** |

## Sim2Sim 验证结果

- 64 envs, 3N/3s 随机推力, 全DR → **64/64 存活 20s**
- Push 后 peak velocity 最高 2.1 m/s — 模型通过手臂+腰部补偿恢复
- 单 DR 消融 (mass/friction/com/gravity/kp-kd) — 全部通过
- 5N/2s 极限推力 — 也通过

## 最佳 Checkpoint

```
/root/UniLab_WS/UniLab/logs/flash_sac/G1FlamingoStand/2026-06-15_09-43-15_mujoco/model_2500.pt
```

## 可视化命令

```bash
uv run scripts/play_interactive.py --algo sac --task g1_flamingo_stand --sim mujoco
```

## 关键设计参数

- **算法**：FlashSAC, 4096 envs, action_scale=1.0
- **观测**：95 维 actor (gyro+gravity+dof_pos+dof_vel+last_actions+contact_states)
- **Reward**：penalty_orientation(-20) + penalty_base_height(-200) + pose(-0.5, 分区权重) + com_over_support(+2) + alive(+10) + foot_contact penalties
- **Pose 分区权重**：支撑腿 15-30, 抬起腿 20-30, 手臂 0.05, 腰部 1.0
- **DR**：质量 ±2.5kg, 摩擦 0.4-1.5, PD 0.85-1.15, 重力方向扰动, body_mass ×0.85-1.15

## 未来方向

- **观测改进**：Actor 加入 linvel(3维) 可进一步提升推力感知速度（当前 95→98 维），但因旧 checkpoint shape 不兼容需重训
- **跳跃恢复**：降低 lifted_foot_contact penalty，允许大推力时支撑脚跳跃/抬腿脚短暂触地恢复

## 新增文件（不影响现有代码）

- `src/unilab/envs/locomotion/g1/flamingo_stand.py`
- `conf/offpolicy/task/flashsac/g1_flamingo_stand/mujoco.yaml` (Phase 1)
- `conf/offpolicy/task/flashsac/g1_flamingo_stand/mujoco_phase2.yaml`
- `conf/offpolicy/task/flashsac/g1_flamingo_stand/mujoco_phase3.yaml`

**Why**: 三个Phase全部收敛，Sim2Sim 64/64通过，项目完成。

**How to apply**: 可直接使用 Phase 3 checkpoint 进行部署或进一步实验。
