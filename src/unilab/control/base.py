"""Abstract base class for motor controllers used in pre_step_control."""

from __future__ import annotations

import abc

import numpy as np


class MotorController(abc.ABC):
    """Base class for controllers that convert target positions to motor torques.

    Subclasses implement different control strategies:
    - PDController: pure PD torque control
    - GravityCompController: PD + gravity compensation
    - CTCController: computed torque control (full dynamics linearization)

    All controllers follow the same interface used by the ``pre_step_control``
    callback pattern validated in Go2W.
    """

    @abc.abstractmethod
    def compute(
        self,
        target_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute motor torques given target and current state.

        Args:
            target_pos: Target joint positions, shape ``(num_envs, num_actions)``.
            joint_pos: Current joint positions, shape ``(num_envs, num_actions)``.
            joint_vel: Current joint velocities, shape ``(num_envs, num_actions)``.
            **kwargs: Additional arguments (e.g. full qpos/qvel for Pinocchio).

        Returns:
            Motor torques, shape ``(num_envs, num_actions)``.
        """
