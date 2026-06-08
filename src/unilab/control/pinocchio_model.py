"""Build a Pinocchio rigid-body dynamics model from a MuJoCo model.

This module constructs a Pinocchio model that is consistent with the MuJoCo
simulation model, enabling dynamics compensation (gravity, Coriolis, inertia)
using the same library that runs at deployment time on real robots.

Key features:
- Programmatic model construction from MuJoCo model parameters (cold path)
- Quaternion convention conversion: MuJoCo (w,x,y,z) ↔ Pinocchio (x,y,z,w)
- Armature injection for RNEA consistency with MuJoCo
- Batch gravity computation with caching across substeps
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)


class PinocchioDynamicsModel:
    """Pinocchio rigid-body dynamics model built from a MuJoCo model.

    This is constructed once on the cold path (env init) and provides batch
    dynamics queries on the hot path (pre_step_control).

    Usage::

        dynamics = PinocchioDynamicsModel(mj_model)
        gravity = dynamics.gravity(qpos_batch, qvel_batch)  # (num_envs, nv_actuated)
    """

    def __init__(self, mj_model: Any) -> None:
        self._mj_model = mj_model
        self._model: pin.Model | None = None
        self._data: pin.Data | None = None

        # Mapping tables
        self._pin_to_mj_dof: np.ndarray | None = None  # Pinocchio nv → MuJoCo nv

        # Cached gravity for substep reuse
        self._cached_gravity: np.ndarray | None = None
        self._cache_valid = False

        self._build_from_mj_model()

    @property
    def model(self) -> pin.Model:
        """The Pinocchio model."""
        return self._model

    @property
    def data(self) -> pin.Data:
        """The Pinocchio data (pre-allocated)."""
        return self._data

    @property
    def nq(self) -> int:
        """Pinocchio configuration dimension."""
        return self._model.nq

    @property
    def nv(self) -> int:
        """Pinocchio velocity dimension (including floating base)."""
        return self._model.nv

    @property
    def nv_actuated(self) -> int:
        """Number of actuated DoFs (excluding floating base)."""
        return self._model.nv - 6

    # ------------------------------------------------------------------ #
    # Model construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_from_mj_model(self) -> None:
        """Build Pinocchio model from MuJoCo model parameters.

        Traverses the MuJoCo body tree and adds joints + inertias to a
        Pinocchio model. Handles:
        - Free joint → JointModelFreeFlyer
        - Hinge joint → JointModelRX/RY/RZ based on axis
        - Quaternion convention conversion (wxyz → xyzw)
        - Inertia frame rotation (body_iquat)
        - Armature injection
        """
        mj = self._mj_model
        model = pin.Model()

        # MuJoCo joint type constants
        MJ_JNT_FREE = 0
        MJ_JNT_HINGE = 3

        # Map: MuJoCo body index → Pinocchio joint id
        # body 0 = world → Pinocchio joint 0 (universe)
        mj_body_to_pin_joint: dict[int, int] = {0: 0}

        for body_id in range(1, mj.nbody):
            parent_body = int(mj.body_parentid[body_id])
            jnt_adr = int(mj.body_jntadr[body_id])

            if jnt_adr < 0:
                # Body without a joint (fixed to parent) — skip
                # In Pinocchio, this would be a fixed joint, but for G1
                # all bodies have joints. Log and skip.
                logger.warning(
                    "Body %d (parent=%d) has no joint — treating as fixed.",
                    body_id,
                    parent_body,
                )
                mj_body_to_pin_joint[body_id] = mj_body_to_pin_joint.get(parent_body, 0)
                continue

            parent_pin_joint = mj_body_to_pin_joint[parent_body]

            jnt_type = int(mj.jnt_type[jnt_adr])

            # Joint placement: position and orientation of the child body
            # relative to parent, from MuJoCo body_pos / body_quat
            body_pos = np.asarray(mj.body_pos[body_id], dtype=np.float64)
            body_quat_wxyz = np.asarray(mj.body_quat[body_id], dtype=np.float64)
            # Convert quaternion: MuJoCo (w,x,y,z) → Pinocchio (x,y,z,w)
            body_quat_xyzw = np.array(
                [body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]]
            )
            joint_placement = pin.SE3(
                pin.Quaternion(body_quat_xyzw), body_pos
            )

            # Determine Pinocchio joint model
            jnt_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, jnt_adr) or f"jnt_{jnt_adr}"

            if jnt_type == MJ_JNT_FREE:
                joint_model = pin.JointModelFreeFlyer()
            elif jnt_type == MJ_JNT_HINGE:
                axis = np.asarray(mj.jnt_axis[jnt_adr], dtype=np.float64)
                if np.allclose(axis, [1, 0, 0], atol=1e-6):
                    joint_model = pin.JointModelRX()
                elif np.allclose(axis, [0, 1, 0], atol=1e-6):
                    joint_model = pin.JointModelRY()
                elif np.allclose(axis, [0, 0, 1], atol=1e-6):
                    joint_model = pin.JointModelRZ()
                else:
                    raise ValueError(
                        f"Joint {jnt_name} has non-axis-aligned axis {axis.tolist()}. "
                        "Only axis-aligned hinge joints are supported."
                    )
            else:
                raise ValueError(
                    f"Joint {jnt_name} has unsupported type {jnt_type}. "
                    "Only free (0) and hinge (3) joints are supported."
                )

            pin_joint_id = model.addJoint(
                parent_pin_joint, joint_model, joint_placement, jnt_name
            )

            # Add body inertia to this joint
            body_mass = float(mj.body_mass[body_id])
            if body_mass > 0:
                body_ipos = np.asarray(mj.body_ipos[body_id], dtype=np.float64)
                body_iquat_wxyz = np.asarray(mj.body_iquat[body_id], dtype=np.float64)
                # Convert inertia quaternion
                body_iquat_xyzw = np.array(
                    [
                        body_iquat_wxyz[1],
                        body_iquat_wxyz[2],
                        body_iquat_wxyz[3],
                        body_iquat_wxyz[0],
                    ]
                )
                iquat = pin.Quaternion(body_iquat_xyzw)
                iquat.normalize()

                # MuJoCo body_inertia stores the 3 diagonal elements of the
                # principal inertia, rotated by body_iquat into body frame.
                # We need the full 3x3 inertia in body frame.
                diaginertia = np.asarray(mj.body_inertia[body_id], dtype=np.float64)
                # Construct inertia in principal frame, then rotate to body frame
                I_principal = np.diag(diaginertia)
                R = iquat.toRotationMatrix()
                I_body = R @ I_principal @ R.T

                # Pinocchio Inertia: mass, COM position in joint frame, 3x3 inertia
                body_placement = pin.SE3(pin.Quaternion.Identity().toRotationMatrix(), body_ipos)
                inertia = pin.Inertia(body_mass, body_ipos, I_body)

                model.appendBodyToJoint(pin_joint_id, inertia, pin.SE3.Identity())

            mj_body_to_pin_joint[body_id] = pin_joint_id

        # Inject armature from MuJoCo
        # MuJoCo armature is per-DoF; for the floating base it's 0.
        # Pinocchio model.armature has the same layout as nv.
        dof_armature = np.asarray(mj.dof_armature, dtype=np.float64)
        if model.nv == len(dof_armature):
            model.armature = dof_armature.copy()
        else:
            logger.warning(
                "MuJoCo dof_armature length (%d) doesn't match Pinocchio nv (%d). "
                "Armature may not be injected correctly.",
                len(dof_armature),
                model.nv,
            )

        self._model = model
        self._data = model.createData()

        # Build DOF mapping: Pinocchio nv → MuJoCo nv
        # In Pinocchio with a FreeFlyer, the first 6 nv are the base (vx,vy,vz,wx,wy,wz).
        # In MuJoCo, the first 6 nv are also the base.
        # For the actuated joints, both use the same ordering (hinge = 1 DoF).
        # So the mapping is 1:1 for the joint DoFs after the floating base.
        self._pin_to_mj_dof = np.arange(model.nv, dtype=np.int32)

        logger.info(
            "Built Pinocchio model from MuJoCo: nq=%d, nv=%d, njoints=%d",
            model.nq,
            model.nv,
            model.njoints,
        )

    # ------------------------------------------------------------------ #
    # State alignment                                                      #
    # ------------------------------------------------------------------ #

    def _align_state(
        self, qpos_batch: np.ndarray, qvel_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert MuJoCo qpos/qvel to Pinocchio q/v format.

        MuJoCo qpos for a free joint: [x, y, z, qw, qx, qy, qz, q0, q1, ...]
        Pinocchio q for FreeFlyer:    [x, y, z, qx, qy, qz, qw, q0, q1, ...]

        The only difference is the quaternion convention (wxyz vs xyzw)
        for the floating base.
        """
        num_envs = qpos_batch.shape[0]
        nq = self._model.nq
        nv = self._model.nv

        pin_q = np.zeros((num_envs, nq), dtype=np.float64)
        pin_v = np.zeros((num_envs, nv), dtype=np.float64)

        # Base position (same in both)
        pin_q[:, 0:3] = qpos_batch[:, 0:3]
        # Base quaternion: MuJoCo (w,x,y,z) → Pinocchio (x,y,z,w)
        pin_q[:, 3] = qpos_batch[:, 4]  # qx
        pin_q[:, 4] = qpos_batch[:, 5]  # qy
        pin_q[:, 5] = qpos_batch[:, 6]  # qz
        pin_q[:, 6] = qpos_batch[:, 3]  # qw

        # Joint positions (same in both)
        if nq > 7:
            pin_q[:, 7:] = qpos_batch[:, 7:]

        # Velocities are the same layout
        pin_v[:] = qvel_batch[:, :nv]

        return pin_q, pin_v

    # ------------------------------------------------------------------ #
    # Dynamics queries                                                     #
    # ------------------------------------------------------------------ #

    def gravity(self, qpos_batch: np.ndarray, qvel_batch: np.ndarray) -> np.ndarray:
        """Compute the generalized gravity vector g(q) for a batch of states.

        Args:
            qpos_batch: MuJoCo qpos, shape ``(num_envs, nq_mj)``.
            qvel_batch: MuJoCo qvel, shape ``(num_envs, nv_mj)``.

        Returns:
            Gravity vector for actuated joints only, shape ``(num_envs, nv_actuated)``.
            The floating-base components (first 6) are excluded.
        """
        pin_q, _ = self._align_state(qpos_batch, qvel_batch)
        num_envs = pin_q.shape[0]
        nv = self._model.nv
        nv_act = nv - 6  # Exclude floating base

        gravity = np.zeros((num_envs, nv_act), dtype=np.float64)

        for i in range(num_envs):
            pin.computeGeneralizedGravity(self._model, self._data, pin_q[i])
            # Skip floating-base DoFs (first 6), take actuated joints
            gravity[i] = self._data.g[6:].copy()

        return gravity

    def gravity_with_cache(
        self, qpos_batch: np.ndarray, qvel_batch: np.ndarray, *, force: bool = False
    ) -> np.ndarray:
        """Compute gravity with caching — reuse across substeps within a ctrl_dt.

        Call with ``force=True`` at the start of each ctrl_dt to invalidate
        the cache. Subsequent calls within the same ctrl_dt (substeps) return
        the cached result.

        Args:
            qpos_batch: MuJoCo qpos, shape ``(num_envs, nq_mj)``.
            qvel_batch: MuJoCo qvel, shape ``(num_envs, nv_mj)``.
            force: If True, recompute gravity even if cache is valid.

        Returns:
            Gravity vector for actuated joints, shape ``(num_envs, nv_actuated)``.
        """
        if force or not self._cache_valid or self._cached_gravity is None:
            self._cached_gravity = self.gravity(qpos_batch, qvel_batch)
            self._cache_valid = True
        return self._cached_gravity

    def invalidate_cache(self) -> None:
        """Mark the gravity cache as invalid (call at ctrl_dt boundary)."""
        self._cache_valid = False

    def rnea(
        self,
        qpos_batch: np.ndarray,
        qvel_batch: np.ndarray,
        qddot_batch: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute full RNEA: M(q)·q̈ + C(q,q̇)·q̇ + g(q).

        Args:
            qpos_batch: MuJoCo qpos, shape ``(num_envs, nq_mj)``.
            qvel_batch: MuJoCo qvel, shape ``(num_envs, nv_mj)``.
            qddot_batch: Desired accelerations, shape ``(num_envs, nv_mj)``.
                If None, computes g(q) only (zero velocity and acceleration).

        Returns:
            RNEA torques for actuated joints, shape ``(num_envs, nv_actuated)``.
        """
        pin_q, pin_v = self._align_state(qpos_batch, qvel_batch)
        num_envs = pin_q.shape[0]
        nv = self._model.nv
        nv_act = nv - 6

        if qddot_batch is None:
            pin_a = np.zeros((num_envs, nv), dtype=np.float64)
        else:
            pin_a = np.asarray(qddot_batch[:, :nv], dtype=np.float64)

        tau = np.zeros((num_envs, nv_act), dtype=np.float64)

        for i in range(num_envs):
            pin.rnea(self._model, self._data, pin_q[i], pin_v[i], pin_a[i])
            tau[i] = self._data.tau[6:].copy()

        return tau

    def coriolis(
        self, qpos_batch: np.ndarray, qvel_batch: np.ndarray
    ) -> np.ndarray:
        """Compute Coriolis + centrifugal forces C(q,q̇)·q̇.

        Args:
            qpos_batch: MuJoCo qpos, shape ``(num_envs, nq_mj)``.
            qvel_batch: MuJoCo qvel, shape ``(num_envs, nv_mj)``.

        Returns:
            Coriolis forces for actuated joints, shape ``(num_envs, nv_actuated)``.
        """
        pin_q, pin_v = self._align_state(qpos_batch, qvel_batch)
        num_envs = pin_q.shape[0]
        nv = self._model.nv
        nv_act = nv - 6

        coriolis = np.zeros((num_envs, nv_act), dtype=np.float64)

        for i in range(num_envs):
            # RNEA with zero gravity gives C(q,q̇)·q̇ + M(q)·0 = C(q,q̇)·q̇
            # But actually we need to subtract gravity from full RNEA
            # C(q,q̇)·q̇ = rnea(q, q̇, 0) - g(q)
            pin.computeGeneralizedGravity(self._model, self._data, pin_q[i])
            grav = self._data.g.copy()
            pin.rnea(self._model, self._data, pin_q[i], pin_v[i], np.zeros(nv))
            coriolis[i] = (self._data.tau - grav)[6:]

        return coriolis
