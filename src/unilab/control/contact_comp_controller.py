"""Contact-aware compensation motor controller.

τ = kp*(q_d - q) - kd*q̇ + gravity_scale*g(q) + coriolis_scale*C(q,q̇)q̇
    - contact_scale * J_c^T · F_contact

This extends CoriolisCompController by additionally compensating for the
joint-space contribution of foot contact forces.  The contact force term
is the largest uncompensated residual after CC compensation — the
diagnostic shows it is ~4× the magnitude of gravity for G1 walking.

The contact Jacobian J_c and foot forces F are obtained from the MuJoCo
backend at each substep.  A per-joint mask allows selective compensation
(e.g., skip arm joints that have narrow forcerange).

.. note::
    Unlike g(q) and C(q,q̇)q̇ which are purely model-based, this term
    depends on the *simulated* contact forces.  On real hardware, foot
    force/torque sensors provide the equivalent signal.
"""

from __future__ import annotations

import numpy as np

from unilab.control.base import MotorController
from unilab.control.pinocchio_model import PinocchioDynamicsModel


class ContactCompController(MotorController):
    """PD + gravity + Coriolis + contact force compensation.

    τ = PD + gravity_scale·g(q) + coriolis_scale·C(q,q̇)q̇
        - contact_scale·J_c^T·F_contact

    The contact force contribution J_c^T·F_contact is computed from
    the MuJoCo backend's site Jacobians and foot force sensors.
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
        contact_comp_mask: np.ndarray | None = None,
        contact_scale: float = 1.0,
    ) -> None:
        self._dynamics_model = dynamics_model
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._force_lower = np.asarray(force_lower, dtype=np.float64)
        self._force_upper = np.asarray(force_upper, dtype=np.float64)
        self._gravity_scale = gravity_scale
        self._coriolis_scale = coriolis_scale
        self._contact_scale = contact_scale

        if gravity_comp_mask is not None:
            self._gravity_comp_mask = np.asarray(gravity_comp_mask, dtype=np.float64)
        else:
            self._gravity_comp_mask = None

        if coriolis_comp_mask is not None:
            self._coriolis_comp_mask = np.asarray(coriolis_comp_mask, dtype=np.float64)
        elif gravity_comp_mask is not None:
            self._coriolis_comp_mask = np.asarray(gravity_comp_mask, dtype=np.float64)
        else:
            self._coriolis_comp_mask = None

        if contact_comp_mask is not None:
            self._contact_comp_mask = np.asarray(contact_comp_mask, dtype=np.float64)
        elif gravity_comp_mask is not None:
            # Default: follow gravity_comp_mask if not specified
            self._contact_comp_mask = np.asarray(gravity_comp_mask, dtype=np.float64)
        else:
            self._contact_comp_mask = None

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
        tau_contact: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute PD + gravity + Coriolis - contact compensation torques.

        Args:
            target_pos: Target joint positions ``(num_envs, num_actions)``.
            joint_pos: Current joint positions ``(num_envs, num_actions)``.
            joint_vel: Current joint velocities ``(num_envs, num_actions)``.
            full_qpos: Full MuJoCo qpos ``(num_envs, nq)`` for Pinocchio.
            full_qvel: Full MuJoCo qvel ``(num_envs, nv)`` for Pinocchio.
            tau_contact: Pre-computed contact torque contribution
                J_c^T · F_contact, shape ``(num_envs, num_actions)``.
                Must be computed externally from the backend's Jacobian
                and force sensors before calling compute().

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
            self._out -= self._gravity_scale * tau_gravity

            # Coriolis + centrifugal compensation: C(q,q̇)q̇
            tau_coriolis = self._dynamics_model.coriolis(full_qpos, full_qvel)
            if self._coriolis_comp_mask is not None:
                tau_coriolis = tau_coriolis * self._coriolis_comp_mask
            self._out -= self._coriolis_scale * tau_coriolis

        # Contact force compensation: +contact_scale * τ_contact
        # After fixing gravity to -= g(q), contact term flips from -= to +=
        # because contact force opposes gravity (GRF supports the body)
        if tau_contact is not None:
            tc = tau_contact
            if self._contact_comp_mask is not None:
                tc = tc * self._contact_comp_mask
            self._out += self._contact_scale * tc

        np.clip(self._out, self._force_lower, self._force_upper, out=self._out)
        return self._out
