"""Controller factory — resolve a controller name to a MotorController instance."""

from __future__ import annotations

from typing import Any

from unilab.control.base import MotorController
from unilab.control.contact_comp_controller import ContactCompController
from unilab.control.coriolis_comp_controller import CoriolisCompController
from unilab.control.gravity_comp_controller import GravityCompController
from unilab.control.imu_gc_controller import IMUGravityCompController
from unilab.control.pd_controller import PDController
from unilab.control.pinocchio_model import PinocchioDynamicsModel


def resolve_controller(
    name: str,
    *,
    dynamics_model: PinocchioDynamicsModel | None = None,
    kp: Any = None,
    kd: Any = None,
    force_lower: Any = None,
    force_upper: Any = None,
    gravity_comp_mask: Any = None,
    gravity_scale: float = 1.0,
    coriolis_comp_mask: Any = None,
    coriolis_scale: float = 1.0,
    contact_comp_mask: Any = None,
    contact_scale: float = 1.0,
    swing_boost: float = 0.0,
    disturbance_scale: float = 0.0,
) -> MotorController:
    """Create a MotorController by name.

    Args:
        name: Controller type. One of ``"pd"``, ``"gravity_comp"``,
            ``"imu_gc"``, ``"coriolis_comp"``, ``"contact_comp"``.
        dynamics_model: PinocchioDynamicsModel (required for gravity_comp+).
        kp: Per-actuator position gains.
        kd: Per-actuator velocity gains.
        force_lower: Per-actuator torque lower bounds.
        force_upper: Per-actuator torque upper bounds.
        gravity_comp_mask: Binary mask for selective gravity compensation.
        gravity_scale: Scaling factor for the gravity compensation term.
        coriolis_comp_mask: Binary mask for selective Coriolis compensation.
            Defaults to gravity_comp_mask if not specified.
        coriolis_scale: Scaling factor for the Coriolis compensation term.
        contact_comp_mask: Binary mask for selective contact compensation.
            Defaults to gravity_comp_mask if not specified.
        contact_scale: Scaling factor for the contact force compensation term.

    Returns:
        A MotorController instance.

    Raises:
        ValueError: If the name is unknown or required parameters are missing.
    """
    import numpy as np

    if name == "pd":
        return PDController(
            kp=np.asarray(kp, dtype=np.float64),
            kd=np.asarray(kd, dtype=np.float64),
            force_lower=np.asarray(force_lower, dtype=np.float64),
            force_upper=np.asarray(force_upper, dtype=np.float64),
        )

    if name == "gravity_comp":
        if dynamics_model is None:
            raise ValueError("gravity_comp controller requires a PinocchioDynamicsModel")
        mask = np.asarray(gravity_comp_mask, dtype=np.float64) if gravity_comp_mask is not None else None
        return GravityCompController(
            dynamics_model=dynamics_model,
            kp=np.asarray(kp, dtype=np.float64),
            kd=np.asarray(kd, dtype=np.float64),
            force_lower=np.asarray(force_lower, dtype=np.float64),
            force_upper=np.asarray(force_upper, dtype=np.float64),
            gravity_comp_mask=mask,
            gravity_scale=gravity_scale,
        )

    if name == "coriolis_comp":
        if dynamics_model is None:
            raise ValueError("coriolis_comp controller requires a PinocchioDynamicsModel")
        gmask = np.asarray(gravity_comp_mask, dtype=np.float64) if gravity_comp_mask is not None else None
        cmask = np.asarray(coriolis_comp_mask, dtype=np.float64) if coriolis_comp_mask is not None else None
        return CoriolisCompController(
            dynamics_model=dynamics_model,
            kp=np.asarray(kp, dtype=np.float64),
            kd=np.asarray(kd, dtype=np.float64),
            force_lower=np.asarray(force_lower, dtype=np.float64),
            force_upper=np.asarray(force_upper, dtype=np.float64),
            gravity_comp_mask=gmask,
            gravity_scale=gravity_scale,
            coriolis_comp_mask=cmask,
            coriolis_scale=coriolis_scale,
        )

    if name == "contact_comp":
        if dynamics_model is None:
            raise ValueError("contact_comp controller requires a PinocchioDynamicsModel")
        gmask = np.asarray(gravity_comp_mask, dtype=np.float64) if gravity_comp_mask is not None else None
        cmask = np.asarray(coriolis_comp_mask, dtype=np.float64) if coriolis_comp_mask is not None else None
        ctmask = np.asarray(contact_comp_mask, dtype=np.float64) if contact_comp_mask is not None else None
        return ContactCompController(
            dynamics_model=dynamics_model,
            kp=np.asarray(kp, dtype=np.float64),
            kd=np.asarray(kd, dtype=np.float64),
            force_lower=np.asarray(force_lower, dtype=np.float64),
            force_upper=np.asarray(force_upper, dtype=np.float64),
            gravity_comp_mask=gmask,
            gravity_scale=gravity_scale,
            coriolis_comp_mask=cmask,
            coriolis_scale=coriolis_scale,
            contact_comp_mask=ctmask,
            contact_scale=contact_scale,
        )

    if name == "imu_gc":
        if dynamics_model is None:
            raise ValueError("imu_gc controller requires a PinocchioDynamicsModel")
        mask = np.asarray(gravity_comp_mask, dtype=np.float64) if gravity_comp_mask is not None else None
        return IMUGravityCompController(
            dynamics_model=dynamics_model,
            kp=np.asarray(kp, dtype=np.float64),
            kd=np.asarray(kd, dtype=np.float64),
            force_lower=np.asarray(force_lower, dtype=np.float64),
            force_upper=np.asarray(force_upper, dtype=np.float64),
            gravity_comp_mask=mask,
            gravity_scale=gravity_scale,
            swing_boost=swing_boost,
            disturbance_scale=disturbance_scale,
        )

    if name == "ctc":
        raise NotImplementedError(
            "CTCController is not yet implemented. Use 'contact_comp' instead."
        )

    raise ValueError(
        f"Unknown controller: {name!r}. "
        f"Available: 'pd', 'gravity_comp', 'imu_gc', 'coriolis_comp', 'contact_comp', 'ctc'."
    )
