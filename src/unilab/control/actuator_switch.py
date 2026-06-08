"""Switch MuJoCo position actuators to motor (torque) actuators.

After switching, the control input is interpreted as direct torque rather
than a target position. The original position-actuator gains (kp, kd) and
force limits are extracted and returned so the env can implement PD control
in its ``pre_step_control`` callback.

This follows the pattern validated in Go2W, where motor actuators give the
env full control over the torque computation, enabling dynamics compensation
strategies like gravity feedforward.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class MotorActuatorInfo:
    """Gains and limits extracted from MuJoCo position actuators before switching.

    Attributes:
        kp: Per-actuator position gains, shape ``(nu,)``.
        kd: Per-actuator velocity gains (damping), shape ``(nu,)``.
        force_lower: Per-actuator torque lower bounds, shape ``(nu,)``.
        force_upper: Per-actuator torque upper bounds, shape ``(nu,)``.
    """

    kp: np.ndarray
    kd: np.ndarray
    force_lower: np.ndarray
    force_upper: np.ndarray


def switch_to_motor_actuators(mj_model) -> MotorActuatorInfo:
    """Switch all MuJoCo position actuators to motor (torque) actuators.

    This modifies the ``mj_model`` in-place:
    1. Reads the current position-actuator gains (kp from gainprm, kd from biasprm).
    2. Reads the force limits from actuator ctrlrange.
    3. Sets ``gainprm[:, 0] = 1.0`` and ``biasprm[:] = 0.0`` so that
       ``ctrl`` is interpreted as direct torque.
    4. Returns the extracted gains and limits.

    Args:
        mj_model: A ``mujoco.MjModel`` instance with position actuators.

    Returns:
        ``MotorActuatorInfo`` containing the original per-actuator gains and
        force limits.

    Raises:
        ValueError: If any actuator is not a position (affine-bias) actuator.
    """
    nu = mj_model.nu

    # Validate: all actuators must be position (affine bias) type
    affine_bias = int(mujoco.mjtBias.mjBIAS_AFFINE)
    biastype = np.asarray(mj_model.actuator_biastype, dtype=np.int32)
    invalid = np.where(biastype != affine_bias)[0]
    if invalid.size > 0:
        names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(idx))
            or str(int(idx))
            for idx in invalid[:8]
        ]
        suffix = "" if invalid.size <= 8 else f", ... ({invalid.size} total)"
        raise ValueError(
            "switch_to_motor_actuators requires all actuators to be position type; "
            f"non-position actuator ids/names: {', '.join(names)}{suffix}"
        )

    # Extract gains before switching
    kp = np.asarray(mj_model.actuator_gainprm[:nu, 0], dtype=np.float64).copy()
    kd = np.asarray(-mj_model.actuator_biasprm[:nu, 2], dtype=np.float64).copy()

    # Extract force limits from forcerange (the actual torque limits).
    # MuJoCo position actuators may have ctrlrange=[0,0] (unlimited ctrl),
    # while forcerange encodes the hardware torque limits from the XML.
    # For motor actuators, we set ctrlrange = forcerange so the ctrl
    # input is directly interpreted as a clamped torque.
    force_lower = np.asarray(mj_model.actuator_forcerange[:nu, 0], dtype=np.float64).copy()
    force_upper = np.asarray(mj_model.actuator_forcerange[:nu, 1], dtype=np.float64).copy()

    # Switch to motor actuator mode: gain = 1, no bias
    mj_model.actuator_gainprm[:nu, 0] = 1.0
    mj_model.actuator_biasprm[:nu, 0] = 0.0
    mj_model.actuator_biasprm[:nu, 1] = 0.0
    mj_model.actuator_biasprm[:nu, 2] = 0.0

    # Set ctrlrange = forcerange for motor actuators (ctrl = torque)
    mj_model.actuator_ctrlrange[:nu, 0] = force_lower
    mj_model.actuator_ctrlrange[:nu, 1] = force_upper

    return MotorActuatorInfo(kp=kp, kd=kd, force_lower=force_lower, force_upper=force_upper)
