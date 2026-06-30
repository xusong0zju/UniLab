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

    def __init__(self, mj_model: Any, use_conjugate_quat: bool = False) -> None:
        self._mj_model = mj_model
        self._model: pin.Model | None = None
        self._data: pin.Data | None = None
        # 2026-06-29: MuJoCo qpos quaternion = R_body→world, but Pinocchio
        # JointModelFreeFlyer expects R_world→body = conj(q_mj).
        # Set True for getup/fallen tasks (non-stand orientations).
        # Default False for backward compat with walk envs (near stand).
        self._use_conjugate_quat = use_conjugate_quat

        # Mapping tables
        self._pin_to_mj_dof: np.ndarray | None = None  # Pinocchio nv → MuJoCo nv

        # Cached gravity for substep reuse
        self._cached_gravity: np.ndarray | None = None
        self._cached_gravity_multi: tuple[np.ndarray, np.ndarray] | None = None  # (g_pelvis, g_total)
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
        # Phase P 三基准GC: collect (pin_joint_id, axis_local) per hinge joint
        # to compute g_total geometrically later (avoid isinstance pitfall).
        _hinge_axes: list[tuple[int, np.ndarray]] = []

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
            # Phase P 三基准GC: record hinge joint axis for g_total computation
            if jnt_type == MJ_JNT_HINGE:
                _hinge_axes.append((pin_joint_id, axis.copy()))

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

        # Phase P 三基准GC: store total mass and per-actuated-joint local axes.
        # Used by gravity_multi_base() to compute g_total geometrically
        # (g_pelvis misses the torso on leg joints when the foot is the support).
        # _hinge_axes collected in build order = Pinocchio joint-id order; the
        # first hinge is the first actuated nv after the 6-dof floating base.
        self._total_mass = float(np.sum(np.asarray(mj.body_mass[1:], dtype=np.float64)))
        # axis_local[j] = local rotation axis of the j-th actuated joint (nv index 6+j)
        self._actuated_axis_local = (
            np.stack([ax for _, ax in _hinge_axes], axis=0) if _hinge_axes else np.zeros((0, 3))
        )
        self._actuated_pin_joint_ids = (
            np.array([jid for jid, _ in _hinge_axes], dtype=np.int32) if _hinge_axes else np.zeros(0, dtype=np.int32)
        )

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
        # MuJoCo stores R_body→world; Pinocchio JointModelFreeFlyer expects R_world→body.
        # R_world→body = (R_body→world)⁻¹ = conj(q_mj) → [-x, -y, -z, +w].
        # For backward compat (walk envs near stand), this is opt-in via use_conjugate_quat.
        if self._use_conjugate_quat:
            pin_q[:, 3] = -qpos_batch[:, 4]  # -qx
            pin_q[:, 4] = -qpos_batch[:, 5]  # -qy
            pin_q[:, 5] = -qpos_batch[:, 6]  # -qz
            pin_q[:, 6] = qpos_batch[:, 3]   # +qw
        else:
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

    def gravity_multi_base(
        self, qpos_batch: np.ndarray, qvel_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase P 三基准GC: return (g_pelvis, g_total) for a batch.

        g_pelvis = current RNEA gravity (base=pelvis free-floating). Leg joints
        only get their own leg's gravity — torso gravity is NOT on the legs.

        g_total = total gravity holding-torque about each actuated joint's axis,
        from ALL links (base-independent). When the foot is the support, the
        torso (and rest of robot) hangs off the legs, so the legs must hold the
        torso: g_foot(joint_on_leg_chain) = g_total - g_pelvis (the torso part).

        g_total(j) = -M_total * (ω_j · cross(p_total - o_j, g_acc))
          ω_j     = world-frame joint axis (oMi.rotation @ axis_local)
          o_j     = world-frame joint origin (oMi.translation)
          p_total = whole-robot COM in world (centerOfMass)
          g_acc   = [0,0,-9.81]

        Returns: (g_pelvis, g_total), each shape (num_envs, nv_act).
        """
        pin_q, _ = self._align_state(qpos_batch, qvel_batch)
        num_envs = pin_q.shape[0]
        nv = self._model.nv
        nv_act = nv - 6
        g_acc = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        g_pelvis = np.zeros((num_envs, nv_act), dtype=np.float64)
        g_total = np.zeros((num_envs, nv_act), dtype=np.float64)
        axis_local = self._actuated_axis_local  # (nv_act, 3)
        jid = self._actuated_pin_joint_ids  # (nv_act,)
        M_total = self._total_mass

        for i in range(num_envs):
            # RNEA gravity (fills data.g and refreshes kinematics data.oMi)
            pin.computeGeneralizedGravity(self._model, self._data, pin_q[i])
            g_pelvis[i] = self._data.g[6:].copy()
            # Whole-robot COM in world frame (refreshes data.com)
            pin.centerOfMass(self._model, self._data, pin_q[i], True)
            p_total = self._data.com[0].copy()  # (3,)
            # Per actuated joint: world axis ω_j and origin o_j
            # data.oMi[joint_id] is the joint frame placement (world); access per-joint.
            omega = np.zeros((nv_act, 3), dtype=np.float64)
            o = np.zeros((nv_act, 3), dtype=np.float64)
            for k in range(nv_act):
                T = self._data.oMi[int(jid[k])]
                omega[k] = T.rotation @ axis_local[k]
                o[k] = T.translation
            cross = np.cross(p_total[None, :] - o, g_acc[None, :])  # (nv_act,3)
            # holding torque = -M_total * (ω · (r × g))
            g_total[i] = -M_total * np.einsum("ka,ka->k", omega, cross)

        return g_pelvis, g_total

    def gravity_multi_base_with_cache(
        self, qpos_batch: np.ndarray, qvel_batch: np.ndarray, *, force: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cached version of gravity_multi_base — reuse across substeps within a ctrl_dt.

        g only depends on q (not on the policy's gc_weights), so the cache is
        valid across substeps. Weight blending happens in the controller.
        """
        if force or not self._cache_valid or self._cached_gravity_multi is None:
            self._cached_gravity_multi = self.gravity_multi_base(qpos_batch, qvel_batch)
            self._cache_valid = True
        return self._cached_gravity_multi

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

    def constraint_force(
        self,
        qpos_batch: np.ndarray,
        qvel_batch: np.ndarray,
        qacc_batch: np.ndarray,
        actuator_force_batch: np.ndarray,
    ) -> np.ndarray:
        """Compute constraint force (contact + joint limits) in actuated joint space.

        Uses the rigid-body dynamics equation::

            qfrc_constraint = RNEA(q, q̇, q̈) - τ_actuator

        where RNEA(q, q̇, q̈) = M(q)·q̈ + C(q,q̇)·q̇ + g(q).

        This gives the total constraint force projected to actuated joints,
        including ground reaction forces and joint limit forces.  It is the
        *privileged* quantity that is only available in simulation (from the
        dynamics equation) and must be estimated for sim-to-real deployment.

        Args:
            qpos_batch: MuJoCo qpos, shape ``(num_envs, nq_mj)``.
            qvel_batch: MuJoCo qvel, shape ``(num_envs, nv_mj)``.
            qacc_batch: Generalized accelerations, shape ``(num_envs, nv_mj)``.
                Typically estimated from the qvel change between substeps.
            actuator_force_batch: Actuator forces in actuated joint space,
                shape ``(num_envs, nv_actuated)``.  For motor actuators with
                unit gain this equals the ctrl signal from the previous step.

        Returns:
            Constraint forces for actuated joints, shape ``(num_envs, nv_actuated)``.
        """
        rnea_result = self.rnea(qpos_batch, qvel_batch, qacc_batch)
        return rnea_result - np.asarray(actuator_force_batch, dtype=np.float64)
