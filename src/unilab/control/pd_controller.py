"""Pure PD motor controller — reproduces MuJoCo position-actuator behavior."""

from __future__ import annotations

import numpy as np

from unilab.control.base import MotorController


class PDController(MotorController):
    """PD torque controller: τ = kp * (q_d - q) - kd * q̇.

    When used with motor actuators (via ``switch_to_motor_actuators``), this
    exactly reproduces the behavior of MuJoCo position actuators, giving a
    baseline for comparing against dynamics-compensation controllers.
    """

    def __init__(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        force_lower: np.ndarray,
        force_upper: np.ndarray,
    ) -> None:
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._force_lower = np.asarray(force_lower, dtype=np.float64)
        self._force_upper = np.asarray(force_upper, dtype=np.float64)
        # Pre-allocate output buffer
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
        **kwargs,
    ) -> np.ndarray:
        """Compute PD torques: τ = kp * (q_d - q) - kd * q̇."""
        if self._out is None or self._out.shape != target_pos.shape:
            self._out = np.empty_like(target_pos, dtype=np.float64)

        # τ = kp * (target - q) - kd * q̇
        np.subtract(target_pos, joint_pos, out=self._out)
        np.multiply(self._out, self._kp, out=self._out)
        # Subtract kd * q̇ (broadcast over env dimension)
        self._out -= self._kd * joint_vel

        np.clip(self._out, self._force_lower, self._force_upper, out=self._out)
        return self._out
