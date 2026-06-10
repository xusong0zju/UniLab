"""Diagnostic: quantify foot contact force contribution to joint torques.

Uses the MuJoCo batch backend's get_site_jacobian_w and foot force sensors
to compute J_c^T · F_contact and compare against g(q) and C(q,q̇)q̇.

Usage:
    uv run python scripts/diagnose_contact_force.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    print("=" * 70)
    print("Contact Force Diagnostic — G1 Dynamics Compensation")
    print("=" * 70)

    # ── 1. Load config via Hydra ───────────────────────────────────
    config_dir = str(PROJECT_ROOT / "conf" / "offpolicy")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algo=flashsac",
                "task=flashsac/g1_walk_flat/mujoco_cc",
            ],
        )

    # ── 2. Create env ─────────────────────────────────────────────
    import unilab.envs.locomotion.g1.joystick  # noqa: F401
    from unilab.training.common import create_env
    from unilab.training.reward import extract_reward_config

    env_cfg_override = extract_reward_config(cfg)
    env_section = OmegaConf.to_container(getattr(cfg, "env", None), resolve=True) if getattr(cfg, "env", None) else {}
    env_cfg_override.update(env_section or {})

    env = create_env(cfg, num_envs=16, env_cfg_override=env_cfg_override)
    print(f"Env created: {env.num_envs} envs, _num_action={env._num_action}")

    # ── 3. Access internals ───────────────────────────────────────
    dynamics = env._dynamics_model
    backend = env._backend
    mj_model = backend._model

    # Find foot force sensor IDs
    foot_force_ids = {}
    for i in range(mj_model.nsensor):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
        if name in ("left_foot_force", "right_foot_force"):
            foot_force_ids[name] = i
    print(f"Force sensors: {foot_force_ids}")

    if not foot_force_ids:
        print("ERROR: No foot force sensors found!")
        return

    # Site IDs for Jacobian computation
    left_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    print(f"Site IDs: left_foot={left_site_id}, right_foot={right_site_id}")

    # DoF indices for Jacobian (all actuated DoFs = indices 6..nv-1)
    nv = mj_model.nv
    actuated_dof_indices = np.arange(6, nv, dtype=np.int32)

    # ── 4. Run env and collect data ───────────────────────────────
    num_envs = env.num_envs
    nv_act = dynamics.nv_actuated

    state = env.reset(env_indices=np.arange(num_envs))

    num_steps = 500
    record_interval = 5
    rng = np.random.default_rng(42)

    gravity_norms = []
    coriolis_norms = []
    contact_norms = []
    gravity_per_joint = []
    coriolis_per_joint = []
    contact_per_joint = []
    foot_force_norms = []

    for step in range(num_steps):
        actions = rng.standard_normal((num_envs, env._num_action)).astype(np.float32) * 0.1
        state = env.step(actions)

        if step % record_interval != 0:
            continue

        # ── Read Pinocchio dynamics ────────────────────────────────
        full_qpos = backend.get_full_qpos()
        full_qvel = backend.get_full_qvel()

        grav = dynamics.gravity(full_qpos, full_qvel)    # (num_envs, nv_act)
        cor = dynamics.coriolis(full_qpos, full_qvel)    # (num_envs, nv_act)

        # ── Read foot forces from sensors ──────────────────────────
        left_force = backend.get_sensor_data("left_foot_force")    # (num_envs, 3)
        right_force = backend.get_sensor_data("right_foot_force")  # (num_envs, 3)

        # ── Compute J_c^T · F using batch Jacobian ─────────────────
        # get_site_jacobian_w returns (num_envs, 3, len(dof_indices))
        Jp_left, Jr_left = backend.get_site_jacobian_w(left_site_id, actuated_dof_indices)
        Jp_right, Jr_right = backend.get_site_jacobian_w(right_site_id, actuated_dof_indices)

        # τ_contact[env_i] = J_p_left[env_i]^T @ F_left[env_i] + J_p_right[env_i]^T @ F_right[env_i]
        # Jp is (num_envs, 3, nv_act), F is (num_envs, 3)
        # Use einsum: J^T @ F = sum over the 3-dim: J[..., j, k] * F[..., j] for each k
        tau_contact_left = np.einsum("ejk,ej->ek", Jp_left, left_force)    # (num_envs, nv_act)
        tau_contact_right = np.einsum("ejk,ej->ek", Jp_right, right_force)  # (num_envs, nv_act)
        tau_contact = tau_contact_left + tau_contact_right

        # ── Store ──────────────────────────────────────────────────
        gravity_norms.append(np.linalg.norm(grav, axis=1).mean())
        coriolis_norms.append(np.linalg.norm(cor, axis=1).mean())
        contact_norms.append(np.linalg.norm(tau_contact, axis=1).mean())
        foot_force_norms.append(np.linalg.norm(left_force, axis=1).mean())
        foot_force_norms.append(np.linalg.norm(right_force, axis=1).mean())
        gravity_per_joint.append(np.abs(grav).mean(axis=0))
        coriolis_per_joint.append(np.abs(cor).mean(axis=0))
        contact_per_joint.append(np.abs(tau_contact).mean(axis=0))

    # ── 5. Analyze ────────────────────────────────────────────────
    gravity_norms = np.array(gravity_norms)
    coriolis_norms = np.array(coriolis_norms)
    contact_norms = np.array(contact_norms)
    foot_force_norms = np.array(foot_force_norms)
    gravity_per_joint = np.array(gravity_per_joint)
    coriolis_per_joint = np.array(coriolis_per_joint)
    contact_per_joint = np.array(contact_per_joint)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)

    print(f"\n--- Overall Magnitude (L2 norm, avg across envs) ---")
    print(f"  {'Quantity':<25s} {'mean':>8s} {'std':>8s} {'max':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for name, arr in [
        ("Gravity  g(q)", gravity_norms),
        ("Coriolis  C(q,q̇)q̇", coriolis_norms),
        ("Contact  J_c^T·F", contact_norms),
        ("Foot force (per foot)", foot_force_norms),
    ]:
        print(f"  {name:<25s} {arr.mean():8.2f} {arr.std():8.2f} {arr.max():8.2f}")

    print(f"\n--- Ratios ---")
    print(f"  Coriolis / Gravity:  {coriolis_norms.mean()/gravity_norms.mean():.3f}")
    print(f"  Contact / Gravity:   {contact_norms.mean()/gravity_norms.mean():.3f}")

    print(f"\n--- Per-Joint (mean |τ|, leg + waist) ---")
    joint_names = [
        "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
        "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
        "waist_yaw", "waist_pitch", "waist_roll",
    ]
    print(f"  {'Joint':<20s} {'|g(q)|':>8s} {'|C·q̇|':>8s} {'|JcF|':>8s} {'C/g':>6s} {'JcF/g':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for j in range(15):
        g_j = gravity_per_joint[:, j].mean()
        c_j = coriolis_per_joint[:, j].mean()
        ct_j = contact_per_joint[:, j].mean()
        name = joint_names[j]
        c_g = c_j / g_j if g_j > 0.01 else float('inf')
        ct_g = ct_j / g_j if g_j > 0.01 else float('inf')
        print(f"  {name:<20s} {g_j:8.2f} {c_j:8.2f} {ct_j:8.2f} {c_g:6.2f} {ct_g:6.2f}")

    # ── Conclusion ─────────────────────────────────────────────────
    ctg = contact_norms.mean() / gravity_norms.mean()
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")
    print(f"  Contact/Gravity ratio = {ctg:.2f}")
    if ctg > 0.3:
        print(f"  → Contact is a SIGNIFICANT residual ({ctg:.0%} of gravity)")
        print(f"  → Contact-aware compensation is STRONGLY recommended")
    elif ctg > 0.1:
        print(f"  → Contact is a MODERATE residual ({ctg:.0%} of gravity)")
        print(f"  → Contact-aware compensation may help, especially for ankle/knee")
    else:
        print(f"  → Contact is MINOR ({ctg:.0%} of gravity)")


if __name__ == "__main__":
    main()
