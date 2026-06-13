"""Coriolis + centrifugal compensation motor controller — PD + gravity + Coriolis feedforward."""

from __future__ import annotations

import numpy as np

from unilab.control.base import MotorController
from unilab.control.pinocchio_model import PinocchioDynamicsModel


class CoriolisCompController(MotorController):
    """PD control with gravity and Coriolis + centrifugal feedforward compensation.

    τ = kp * (q_d - q) - kd * q̇ + gravity_scale * g(q) + coriolis_scale * C(q,q̇)q̇

    where:
    - ``g(q)`` is the generalized gravity vector (Pinocchio RNEA with zero velocity/accel)
    - ``C(q,q̇)q̇`` is the Coriolis + centrifugal force vector

    Adding g(q) and C(q,q̇)q̇ to the PD output cancels the corresponding
    components of MuJoCo's ``qfrc_bias``, reducing the dynamics burden on
    the policy.

    Compared to ``GravityCompController``, this additionally compensates for
    velocity-dependent coupling forces. At low speeds the Coriolis term is
    small and the controller degrades gracefully to gravity compensation; at
    higher speeds (fast walking, torso rotation) the Coriolis term becomes
    significant and reduces the tracking burden on the policy.

    Optional masks allow selective compensation per joint. For the G1 humanoid,
    arm joints (narrow forcerange ±5 Nm) are typically masked out for both
    gravity and Coriolis terms to avoid clipping artifacts.
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
        coriolis_comp_mask: np.ndarray | None = None,
        coriolis_scale: float = 1.0,
    ) -> None:
        self._dynamics_model = dynamics_model
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._force_lower = np.asarray(force_lower, dtype=np.float64)
        self._force_upper = np.asarray(force_upper, dtype=np.float64)
        self._gravity_scale = gravity_scale
        self._coriolis_scale = coriolis_scale

        if gravity_comp_mask is not None:
            self._gravity_comp_mask = np.asarray(gravity_comp_mask, dtype=np.float64)
        else:
            self._gravity_comp_mask = None

        if coriolis_comp_mask is not None:
            self._coriolis_comp_mask = np.asarray(coriolis_comp_mask, dtype=np.float64)
        elif gravity_comp_mask is not None:
            # Default: follow gravity_comp_mask if not explicitly specified
            self._coriolis_comp_mask = np.asarray(gravity_comp_mask, dtype=np.float64)
        else:
            self._coriolis_comp_mask = None

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
        """Compute PD + gravity + Coriolis compensation torques.

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

        # Dynamics compensation terms (require full state)
        if full_qpos is not None and full_qvel is not None:
            # Gravity compensation: g(q)
            tau_gravity = self._dynamics_model.gravity(full_qpos, full_qvel)
            if self._gravity_comp_mask is not None:
                tau_gravity = tau_gravity * self._gravity_comp_mask
            self._out += self._gravity_scale * tau_gravity

            # Coriolis + centrifugal compensation: C(q,q̇)q̇
            tau_coriolis = self._dynamics_model.coriolis(full_qpos, full_qvel)
            if self._coriolis_comp_mask is not None:
                tau_coriolis = tau_coriolis * self._coriolis_comp_mask
            self._out += self._coriolis_scale * tau_coriolis

        np.clip(self._out, self._force_lower, self._force_upper, out=self._out)
        return self._out
