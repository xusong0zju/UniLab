"""Computed Torque Controller (CTC) — full dynamics linearization.

CTC: τ = M(q)·[kp*(q_d-q) + kd*(q̇_d-q̇)] + C(q,q̇)·q̇ + g(q)

This fully linearizes the closed-loop dynamics, making the system behave
like a decoupled double integrator. It is more expensive than gravity
compensation alone (requires mass matrix inversion) but provides the
strongest dynamics simplification.

.. note::
    This is a stub implementation. Full CTC will be added after gravity
    compensation is validated.
"""

from __future__ import annotations

from unilab.control.base import MotorController


class CTCController(MotorController):
    """Computed Torque Controller (stub).

    Will implement: τ = M(q)·[kp*(q_d-q) + kd*(q̇_d-q̇)] + C(q,q̇)·q̇ + g(q)
    """

    def compute(
        self,
        target_pos,
        joint_pos,
        joint_vel,
        **kwargs,
    ):
        raise NotImplementedError(
            "CTCController is not yet implemented. "
            "Use GravityCompController for PD + gravity compensation."
        )
