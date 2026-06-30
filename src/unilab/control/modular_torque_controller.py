"""模块化力矩补偿控制器 — 分阶段静态补偿.

基于 Pinocchio RNEA 的静态力矩补偿:
  g(q) = rnea(q, 0, 0)[6:]  (actuated joints 重力补偿)

分阶段控制:
  Stage 1 (静态保持): 腿+肩+肘 PD+g(q), waist 仅g(q)+kd(无位置目标)
  Stage 2 (蹬腿起身): waist 恢复PD, 从实际值平滑过渡到目标

接触力修正 (2026-06-29):
  经验修正 τ_contact = kp * err_settled (从PD+g(q)稳态误差测得).
  适用于真机 (测量PD误差→补偿), 绕过Pinocchio模型动力学bug
  (非stand姿态下g_pin ≠ qbias_MJ, 当前版本Pinocchio模型有此问题).

设计原则:
  - 不依赖 MuJoCo 的 qfrc_constraint (仿真only, 真机没有)
  - 不让 policy 输出接触力/支撑点 (控制层自己算)
  - 模块化: 补偿模块可组合 (g(q) / g(q)+接触修正 / 完整inverse dynamics)
  - 向后兼容: contact_comp 默认 None → 纯 g(q) 补偿

后续扩展:
  Stage 3: 模型化接触力修正 (修复pinocchio模型动力学 → KKT逆映射)
  Stage 4: RL policy 驱动 (policy 输出关节目标, 控制器补偿)
"""
from __future__ import annotations

import numpy as np

from unilab.control.base import MotorController
from unilab.control.pinocchio_model import PinocchioDynamicsModel


class ModularTorqueController(MotorController):
    """模块化力矩补偿控制器.

    τ = PD(q_d, q, q̇) + g(q) + 可选接触修正

    分阶段 waist 控制:
    - waist_free: waist 不给位置目标(kp=0), 仅 g(q)+kd 阻尼
    - waist 控制时: 从实际值平滑过渡到目标(不突变)

    参数:
        dynamics_model: PinocchioDynamicsModel
        kp, kd: PD 增益 (num_envs, num_actions)
        force_lower, force_upper: 力矩限制
        waist_joint_indices: waist 关节索引 (默认 [12,13,14] for G1)
        waist_free: True=waist 不位置控制(阶段1), False=waist PD(阶段2)
    """

    def __init__(
        self,
        dynamics_model: PinocchioDynamicsModel,
        kp: np.ndarray,
        kd: np.ndarray,
        force_lower: np.ndarray,
        force_upper: np.ndarray,
        waist_joint_indices: list[int] | None = None,
        waist_free: bool = True,
        contact_comp: np.ndarray | None = None,
        use_mj_constraint: bool = False,
    ) -> None:
        self._dynamics_model = dynamics_model
        self._kp_base = np.asarray(kp, dtype=np.float64).copy()
        self._kd_base = np.asarray(kd, dtype=np.float64).copy()
        self._force_lower = np.asarray(force_lower, dtype=np.float64)
        self._force_upper = np.asarray(force_upper, dtype=np.float64)
        # waist 分阶段控制
        self._waist_indices = waist_joint_indices or [12, 13, 14]
        self._waist_free = waist_free
        # 当前生效的 kp/kd (每步可能被 DR 修改)
        self._kp = self._kp_base.copy()
        self._kd = self._kd_base.copy()
        # waist 平滑过渡 (阶段2: 从实际值过渡到目标)
        self._waist_blend = 0.0  # 0=自由, 1=完全PD控制
        self._waist_actual: np.ndarray | None = None  # 记录waist实际值(过渡起点)
        self._out: np.ndarray | None = None
        # 接触力修正 (经验或模型化, 可选)
        # shape: (num_actions,) 或 (num_envs, num_actions)
        self._contact_comp: np.ndarray | None = (
            np.asarray(contact_comp, dtype=np.float64).copy()
            if contact_comp is not None
            else None
        )
        # [验证用] 使能 MuJoCo qfrc_constraint 直接前馈
        # τ = PD + g(q) - qfrc_constraint → 理论零静差
        # 真机不可用, 仅用于仿真验证接触力模型上限
        self._use_mj_constraint = use_mj_constraint
        # IMU→GC基准混合: upvector_z → α → g_blend = (1-α)g_pelvis + α·g_total
        # 起身过程中自动从背支撑(pelvis)过渡到脚支撑(total)
        self._use_imu_blend = False
        self._imu_blend_deadband = (0.3, 0.7)  # up_z<0.3→α=0, up_z>0.7→α=1

    @property
    def kp(self) -> np.ndarray:
        return self._kp

    @kp.setter
    def kp(self, value: np.ndarray) -> None:
        self._kp = np.asarray(value, dtype=np.float64)
        # 保持 waist kp 修改 (DR 随机化后仍应用 waist_free)
        if self._waist_free:
            for idx in self._waist_indices:
                self._kp[..., idx] = 0.0

    @property
    def kd(self) -> np.ndarray:
        return self._kd

    @kd.setter
    def kd(self, value: np.ndarray) -> None:
        self._kd = np.asarray(value, dtype=np.float64)

    @property
    def contact_comp(self) -> np.ndarray | None:
        return self._contact_comp

    @contact_comp.setter
    def contact_comp(self, value: np.ndarray | None) -> None:
        """设置接触力修正值 (τ_contact). 形状: (num_actions,) 或 (num_envs, num_actions).

        经验修正: τ_contact = kp * err_settled (从静态PD+g(q)稳态误差测得).
        None → 禁用接触修正, 仅使用 g(q).
        """
        if value is not None:
            self._contact_comp = np.asarray(value, dtype=np.float64).copy()
        else:
            self._contact_comp = None

    @property
    def use_mj_constraint(self) -> bool:
        """是否使用 MuJoCo qfrc_constraint 直接前馈 (仅仿真验证)."""
        return self._use_mj_constraint

    @use_mj_constraint.setter
    def use_mj_constraint(self, value: bool) -> None:
        """设置 MuJoCo 约束力前馈使能 (True=启用, False=禁用)."""
        self._use_mj_constraint = value

    @property
    def use_imu_blend(self) -> bool:
        """是否启用 IMU→GC基准混合."""
        return self._use_imu_blend

    @use_imu_blend.setter
    def use_imu_blend(self, value: bool) -> None:
        """IMU→GC基准混合: upvector_z 自动在 g_pelvis↔g_total 之间切换."""
        self._use_imu_blend = value

    def set_waist_free(self, free: bool) -> None:
        """切换 waist 控制: True=自由(阶段1), False=PD(阶段2)."""
        self._waist_free = free
        if free:
            for idx in self._waist_indices:
                self._kp[..., idx] = 0.0
        else:
            # 恢复 kp, 记录当前实际值作为过渡起点
            self._waist_blend = 0.0
            self._waist_actual = None  # 下一步记录
            for idx in self._waist_indices:
                self._kp[..., idx] = self._kp_base[..., idx]

    def update_waist_blend(self, alpha: float) -> None:
        """更新 waist 控制混合系数 (0=自由, 1=完全PD)."""
        self._waist_blend = max(0.0, min(1.0, alpha))

    def compute(
        self,
        target_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        *,
        full_qpos: np.ndarray | None = None,
        full_qvel: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """计算 PD + g(q) 力矩.

        τ = kp·(q_d - q) - kd·q̇ + g(q)

        waist_free 时: waist 的 kp=0, 只给 g(q)+kd
        waist 过渡时: target 从 actual 平滑过渡到 goal
        """
        if self._out is None or self._out.shape != target_pos.shape:
            self._out = np.empty_like(target_pos, dtype=np.float64)

        # waist 过渡: 阶段2切换时, target 从实际值平滑过渡
        effective_target = target_pos.copy()
        if not self._waist_free and self._waist_blend < 1.0:
            if self._waist_actual is None:
                self._waist_actual = joint_pos[..., self._waist_indices].copy()
            for i, idx in enumerate(self._waist_indices):
                actual = self._waist_actual[..., i]
                goal = target_pos[..., idx]
                effective_target[..., idx] = (
                    actual * (1.0 - self._waist_blend)
                    + goal * self._waist_blend
                )

        # PD term: τ_pd = kp * (q_d - q) - kd * q̇
        np.subtract(effective_target, joint_pos, out=self._out)
        np.multiply(self._out, self._kp, out=self._out)
        self._out -= self._kd * joint_vel

        # Gravity compensation: g(q) = rnea(q, 0, 0)[6:]
        # forwardDynamics 验证: tau=full_rnea → ddq=0 (静态平衡)
        if full_qpos is not None and full_qvel is not None:
            # IMU→GC基准混合 (起身过程中自动从背支撑过渡到脚支撑)
            if self._use_imu_blend:
                up_z = kwargs.get("upvector_z")
                if up_z is not None:
                    lo, hi = self._imu_blend_deadband
                    alpha = np.clip((up_z - lo) / (hi - lo), 0.0, 1.0)
                    g_pelvis, g_total = self._dynamics_model.gravity_multi_base(
                        full_qpos, full_qvel
                    )
                    tau_gravity = (1 - alpha) * g_pelvis + alpha * g_total
                else:
                    tau_gravity = self._dynamics_model.gravity(full_qpos, full_qvel)
            else:
                tau_gravity = self._dynamics_model.gravity(full_qpos, full_qvel)
            self._out += tau_gravity

        # Contact force compensation (optional)
        # Empirical: τ_contact = kp * err_settled (from static PD+g(q) error)
        # Model-based: τ_contact = -Jc^T·λ (from forwardDynamics + KKT inverse)
        if self._contact_comp is not None:
            # Broadcast to batch dim if needed
            if self._contact_comp.ndim == 1 and self._out.ndim == 2:
                self._out += self._contact_comp[None, :]
            else:
                self._out += self._contact_comp

        # [验证用] MuJoCo qfrc_constraint 直接前馈
        # τ = PD + g(q) - qfrc_constraint → 理论零静差
        # 开关: controller.use_mj_constraint = True/False
        if self._use_mj_constraint:
            qc = kwargs.get("qfrc_constraint")
            if qc is not None:
                # qfrc_constraint shape: (num_envs, nv_act) 或 (nv_act,)
                if qc.ndim == 1 and self._out.ndim == 2:
                    self._out -= qc[None, :]
                else:
                    self._out -= qc

        np.clip(self._out, self._force_lower, self._force_upper, out=self._out)
        return self._out
