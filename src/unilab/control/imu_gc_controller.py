"""IMU-enhanced gravity compensation motor controller.

τ = PD + gravity_scale · g(q) · gravity_factor · modulated_mask

The IMU at the pelvis measures the net acceleration of the base.  In the
non-inertial frame of the pelvis, the effective gravity is::

    g_eff = g - a_pelvis

For purely vertical changes (dominant in walking), this means::

    gravity_factor = (9.81 + a_net_z) / 9.81

Physical examples:
- Standing still (a_net_z ≈ 0):   factor = 1.0 (full compensation)
- Free fall (a_net_z ≈ -9.81):    factor = 0.0 (no gravity felt)
- Pushed up (a_net_z > 0):         factor > 1.0 (body feels heavier)

The gravity_factor is applied to ALL joints (both upper and lower body)
because RNEA computes the gravity torque for the entire kinematic chain
under the modified effective gravity.

For the lower body (legs), ground reaction forces (GRF) also play a role:
- Stance legs: GRF partially counteracts gravity → full g(q) is an
  "over-compensation" that lightens the stance legs (beneficial for learning).
  The gravity_factor preserves this over-compensation ratio across different
  base accelerations.
- Swing legs: No GRF → full g(q) is the correct compensation.  The
  ``swing_boost`` parameter adds extra compensation for swing legs to make
  them lighter and easier to control.

Neither term uses qacc estimation or privileged simulator information.
All signals come from IMU (accelerometer + gyro) + joint encoders + gait phase.
"""

from __future__ import annotations

import numpy as np

from unilab.control.base import MotorController
from unilab.control.pinocchio_model import PinocchioDynamicsModel

# Standard gravity magnitude (m/s^2)
_G = 9.81


class IMUGravityCompController(MotorController):
    """PD control with IMU-modulated gravity compensation.

    τ = kp*(q_d - q) - kd*q̇
        + gravity_scale * g(q) * gravity_factor * modulated_mask

    where:
    - gravity_factor = clip((9.81 + a_net_z) / 9.81, 0, 2) per environment
      (from IMU net vertical acceleration, determines effective gravity)
    - modulated_mask = gravity_comp_mask + swing_boost * swing_mask (for legs)
    """

    def __init__(
        self,
        dynamics_model: PinocchioDynamicsModel,
        kp: np.ndarray,
        kd: np.ndarray,
        force_lower: np.ndarray,
        force_upper: np.ndarray,
        gravity_comp_mask: np.ndarray | None = None,
        gravity_scale: float = 1.0,
        swing_boost: float = 0.0,
    ) -> None:
        self._dynamics_model = dynamics_model
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._force_lower = np.asarray(force_lower, dtype=np.float64)
        self._force_upper = np.asarray(force_upper, dtype=np.float64)
        self._gravity_scale = gravity_scale
        self._swing_boost = swing_boost
        if gravity_comp_mask is not None:
            self._gravity_comp_mask = np.asarray(gravity_comp_mask, dtype=np.float64)
        else:
            self._gravity_comp_mask = None
        self._out: np.ndarray | None = None

    @property
    def kp(self) -> np.ndarray:
        return self._kp

    @kp.setter
    def kp(self, value: np.ndarray) -> None:
        self._kp = np.asarray(value, dtype=np.float64)

    @property
    def kd(self) -> np.ndarray:
        return self._kd

    @kd.setter
    def kd(self, value: np.ndarray) -> None:
        self._kd = np.asarray(value, dtype=np.float64)

    def compute(
        self,
        target_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        *,
        full_qpos: np.ndarray | None = None,
        full_qvel: np.ndarray | None = None,
        swing_mask_per_env: np.ndarray | None = None,
        imu_accel_net_z: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute PD + IMU-modulated gravity compensation torques.

        Args:
            target_pos: Target joint positions ``(num_envs, num_actions)``.
            joint_pos: Current joint positions ``(num_envs, num_actions)``.
            joint_vel: Current joint velocities ``(num_envs, num_actions)``.
            full_qpos: Full MuJoCo qpos ``(num_envs, nq)`` for Pinocchio.
            full_qvel: Full MuJoCo qvel ``(num_envs, nv)`` for Pinocchio.
                Should include the base angular velocity (from IMU gyro in
                deployment, or simulation state in training) so that Coriolis
                compensation is accurate when used with CoriolisCompController.
            swing_mask_per_env: Per-env swing leg mask ``(num_envs, num_actions)``.
                1.0 for joints belonging to a swing leg, 0.0 otherwise.
            imu_accel_net_z: Net vertical acceleration at the IMU/pelvis
                ``(num_envs,)``.  This is the IMU-measured specific force
                rotated to world frame minus gravity: ``a_net_z = R·a_sensor_z - 9.81``.
                Determines the effective gravity for ALL joints:
                - Standing still: a_net_z ≈ 0 → factor = 1.0 (full GC)
                - Free fall: a_net_z ≈ -9.81 → factor = 0.0 (no GC)
                - Pushed up: a_net_z > 0 → factor > 1.0 (more GC)

        Returns:
            Motor torques ``(num_envs, num_actions)``.
        """
        if self._out is None or self._out.shape != target_pos.shape:
            self._out = np.empty_like(target_pos, dtype=np.float64)

        # PD term: τ_pd = kp * (q_d - q) - kd * q̇
        np.subtract(target_pos, joint_pos, out=self._out)
        np.multiply(self._out, self._kp, out=self._out)
        self._out -= self._kd * joint_vel

        # Gravity compensation with IMU modulation
        if full_qpos is not None and full_qvel is not None:
            tau_gravity = self._dynamics_model.gravity(full_qpos, full_qvel)

            # Build effective mask: gravity_comp_mask + swing_boost * swing_mask
            if self._gravity_comp_mask is not None:
                effective_mask = self._gravity_comp_mask.copy()
            else:
                effective_mask = np.ones_like(tau_gravity)

            if self._swing_boost != 0.0 and swing_mask_per_env is not None:
                effective_mask = effective_mask + self._swing_boost * swing_mask_per_env

            # ── IMU gravity modulation ──────────────────────────────
            # Effective gravity: g_eff = g - a_base
            # For vertical component: gravity_factor = (9.81 + a_net_z) / 9.81
            # - Standing still (a_net_z=0):    factor = 1.0 (full compensation)
            # - Free fall (a_net_z=-9.81):     factor = 0.0 (no compensation)
            # - Pushed up (a_net_z>0):          factor > 1.0 (more compensation)
            #
            # Applied to ALL joints because RNEA computes gravity torque for
            # the entire kinematic chain.  For stance legs, the GRF partially
            # counteracts gravity, so full g(q) is an over-compensation that
            # is preserved proportionally by the gravity_factor.
            if imu_accel_net_z is not None:
                gravity_factor = np.clip(
                    1.0 + imu_accel_net_z / _G, 0.0, 2.0
                )  # (num_envs,)
                # Apply per-env factor to all joints
                self._out += self._gravity_scale * tau_gravity * gravity_factor[:, None] * effective_mask
            else:
                self._out += self._gravity_scale * tau_gravity * effective_mask

        np.clip(self._out, self._force_lower, self._force_upper, out=self._out)
        return self._out
