"""Gravity compensation motor controller — PD + gravity feedforward."""

from __future__ import annotations

import numpy as np

from unilab.control.base import MotorController
from unilab.control.pinocchio_model import PinocchioDynamicsModel


class GravityCompController(MotorController):
    """PD control with gravity feedforward compensation.

    τ = kp * (q_d - q) - kd * q̇ + gravity_scale * g(q)

    The gravity term ``g(q)`` is computed by Pinocchio's RNEA algorithm
    (with zero velocity and acceleration).  In the RNEA convention,
    ``g(q)`` is the torque needed to **support** the robot against gravity,
    and adding it to the PD output cancels the gravity component of
    MuJoCo's ``qfrc_bias``, leaving the PD controller to work in a
    partially gravity-free space.

    For stance legs (with ground contact), GRF partially counteracts gravity,
    so full g(q) compensation is an "over-compensation" that effectively
    lightens the stance legs — beneficial for learning to walk.

    An optional ``gravity_comp_mask`` selects which joints receive gravity
    compensation. For the G1 humanoid, this is useful to skip arm joints
    whose narrow torque limits (±5 Nm) would cause clipping artifacts.
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
    ) -> None:
        self._dynamics_model = dynamics_model
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._force_lower = np.asarray(force_lower, dtype=np.float64)
        self._force_upper = np.asarray(force_upper, dtype=np.float64)
        self._gravity_scale = gravity_scale
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
        **kwargs,
    ) -> np.ndarray:
        """Compute PD + gravity compensation torques.

        Args:
            target_pos: Target joint positions ``(num_envs, num_actions)``.
            joint_pos: Current joint positions ``(num_envs, num_actions)``.
            joint_vel: Current joint velocities ``(num_envs, num_actions)``.
            full_qpos: Full MuJoCo qpos ``(num_envs, nq)`` for Pinocchio.
            full_qvel: Full MuJoCo qvel ``(num_envs, nv)`` for Pinocchio.

        Returns:
            Motor torques ``(num_envs, num_actions)``.
        """
        if self._out is None or self._out.shape != target_pos.shape:
            self._out = np.empty_like(target_pos, dtype=np.float64)

        # PD term: τ_pd = kp * (q_d - q) - kd * q̇
        np.subtract(target_pos, joint_pos, out=self._out)
        np.multiply(self._out, self._kp, out=self._out)
        self._out -= self._kd * joint_vel

        # Gravity compensation term: g(q)
        if full_qpos is not None and full_qvel is not None:
            tau_gravity = self._dynamics_model.gravity(full_qpos, full_qvel)
            if self._gravity_comp_mask is not None:
                tau_gravity = tau_gravity * self._gravity_comp_mask
            self._out += self._gravity_scale * tau_gravity

        np.clip(self._out, self._force_lower, self._force_upper, out=self._out)
        return self._out
