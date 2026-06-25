---
name: mamba-multiskill-getup-failed-lessons
description: "起身训练Phase J/K/L失败教训: 纯reward+辅助力从真倒地起步训不出, G阶段6次+K+L复现, 需参考轨迹/MPC"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

2026-06-24 起身训练 Phase J/K/L 全部失败（含 G 阶段 6 次），核心教训记录。

**Phase J(半跪起步, aux衰减)**: stand_feet 4.85 看似成功, 但诊断发现 55% env reset 后没真倒(tilt 31°), 是"半跪/微倾起身"假象。aux 衰减课程有效但起步姿态非真倒地。

**Phase K(缓存版推倒真倒地起步, aux gain400)**: 真倒地起步达成(base_z0.165 tilt81°), 但训练失败——aux=1.0 时辅助力(gap×400, 纯z+)把机器人"举"到 base_z 0.88m 半空(超站立0.754), 躯干倾斜脚没着地(stand_feet 0.3)。撤辅助就掉。stand_feet aux 0.95→0.47 一直 0.27 停滞。

**Phase L(贴地轻托 aux_gain150/thr0.30 + mid_height reward + uprightness3.0)**: dry-run 验证 aux=1 时 base_z 0.184(不举半空✅), uprightness 1.16(比K的0.39好)。但 stand_feet 仍停滞 0.30→0.22(随aux撤降), mid_height 0.08(半跪中间态没探索到)。同样失败。

**核心结论(诚实)**:
1. **纯 reward + 辅助力, 从真倒地起步, 训不出起身**——G阶段6次 + K + L = 8次复现
2. 根因不是 aux 力大小/课程(都试过: 举到半空/贴地轻托/慢撤), 是**纯 reward 不足以发现起身动作序列**(HumanUP/RSS2025 教训一致)
3. **需要参考轨迹引导**——HumanUP 两阶段(发现→精炼)依赖参考轨迹; 纯 reward 引导在 MuJoCo 150Hz 接触动力学下不够
4. MPC/轨迹优化生成起身参考轨迹是可行方向(MPC 在线优化绕开 RL 发现序列难题)

**How to apply**:
- 起身不要再试纯 reward + aux 力调整(已 8 次失败)
- 走参考轨迹引导: MPC/优化生成轨迹 + motion_tracking 追踪, 或 mocap 参考轨迹
- UniLab 有 motion_tracking 基础设施(motion_loader NPZ), 可复用
- 详见 [[mamba-multiskill-fallen-pose-bug]](姿态诊断)、[[mamba-multiskill-reward-routing-fix]](reward routing)
- deep research 调研 HumanUP/HoST/开源起身实现(2026-06-24 启动)
