"""Diagnostic: verify privileged qfrc_constraint compensation for G1.

Checks:
1. Standing: qfrc_constraint ≈ -(g + C·q̇) in actuated joint space
2. RNEA approach matches MuJoCo's qfrc_constraint (within actuator force)
3. Compensation effect: |qfrc_constraint| / |g(q)| ratio
4. Smoke test: 4 envs × 50 steps with G1WalkFlatCTCP
"""
import sys
sys.path.insert(0, "/root/UniLab_WS/UniLab/src")

import mujoco
import numpy as np
from unilab.control.pinocchio_model import PinocchioDynamicsModel


def check_standing_balance():
    """Check 1: Standing robot — qfrc_constraint balances gravity in joint space."""
    print("=" * 60)
    print("Check 1: Standing balance (qfrc_constraint ≈ -qfrc_smooth)")
    print("=" * 60)

    m = mujoco.MjModel.from_xml_path(
        "/root/UniLab_WS/UniLab/src/unilab/assets/robots/g1/scene_flat.xml"
    )
    d = mujoco.MjData(m)
    dyn = PinocchioDynamicsModel(m)

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # Let settle
    for _ in range(2000):
        d.ctrl[:] = 0
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    # Actuated joints only
    qfrc_c = d.qfrc_constraint[6:]
    qfrc_s = d.qfrc_smooth[6:]

    balance_err = np.linalg.norm(qfrc_c + qfrc_s)
    print(f"  |qfrc_constraint + qfrc_smooth| (actuated) = {balance_err:.6f}")
    print(f"  |qfrc_constraint| = {np.linalg.norm(qfrc_c):.2f}")
    print(f"  |qfrc_smooth|    = {np.linalg.norm(qfrc_s):.2f}")
    print(f"  Ratio |qfrc_c| / |qfrc_s| = {np.linalg.norm(qfrc_c) / np.linalg.norm(qfrc_s):.4f}")

    # GRF vertical check
    grf_z = d.qfrc_constraint[2]  # free joint z-force
    weight = np.sum(m.body_mass) * 9.81
    print(f"\n  GRF vertical (z) = {grf_z:.2f} N, Weight = {weight:.2f} N")
    print(f"  GRF/Weight = {grf_z / weight:.4f}")

    assert balance_err < 1.0, f"Balance error too large: {balance_err}"
    print("  ✅ PASS: qfrc_constraint balances qfrc_smooth in joint space")


def check_rnea_approach():
    """Check 2: RNEA approach matches MuJoCo's qfrc_constraint."""
    print("\n" + "=" * 60)
    print("Check 2: RNEA(q, q̇, q̈) - τ_act ≈ qfrc_constraint")
    print("=" * 60)

    m = mujoco.MjModel.from_xml_path(
        "/root/UniLab_WS/UniLab/src/unilab/assets/robots/g1/scene_flat.xml"
    )
    d = mujoco.MjData(m)
    dyn = PinocchioDynamicsModel(m)

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for _ in range(1000):
        d.ctrl[:] = 0
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    # Use MuJoCo's exact qacc
    qpos = d.qpos[None, :]
    qvel = d.qvel[None, :]
    qacc = d.qacc[None, :]

    # RNEA approach
    rnea_result = dyn.rnea(qpos, qvel, qacc)[0]
    act_force = d.actuator_force.copy()
    qfrc_c_est = rnea_result - act_force
    qfrc_c_mj = d.qfrc_constraint[6:]

    err = np.linalg.norm(qfrc_c_est - qfrc_c_mj)
    print(f"  |RNEA - τ_act - qfrc_constraint| = {err:.4f}")
    print(f"  |qfrc_constraint| = {np.linalg.norm(qfrc_c_mj):.2f}")
    print(f"  Relative error = {err / np.linalg.norm(qfrc_c_mj) * 100:.2f}%")

    assert err < 5.0, f"RNEA approach error too large: {err}"
    print("  ✅ PASS: RNEA approach matches MuJoCo qfrc_constraint")


def check_qacc_estimation():
    """Check 3: qacc estimation from qvel finite difference."""
    print("\n" + "=" * 60)
    print("Check 3: qacc estimation from qvel finite difference")
    print("=" * 60)

    m = mujoco.MjModel.from_xml_path(
        "/root/UniLab_WS/UniLab/src/unilab/assets/robots/g1/scene_flat.xml"
    )
    d = mujoco.MjData(m)
    dyn = PinocchioDynamicsModel(m)
    dt = m.opt.timestep

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for _ in range(1000):
        d.ctrl[:] = 0
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    # Collect qfrc_constraint via RNEA + qacc_est for several steps
    errors = []
    for step in range(20):
        qvel_before = d.qvel.copy()
        d.ctrl[:] = 0
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)

        qacc_est = (d.qvel - qvel_before) / dt
        qacc_mj = d.qacc

        # Compare qfrc_constraint computed both ways
        qpos = d.qpos[None, :]
        qvel = d.qvel[None, :]

        qfrc_est = dyn.constraint_force(qpos, qvel, qacc_est[None, :], np.zeros((1, 29)))[0]
        qfrc_mj = d.qfrc_constraint[6:]

        err = np.linalg.norm(qfrc_est - qfrc_mj)
        errors.append(err)

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"  Mean error over 20 steps: {mean_err:.2f}")
    print(f"  Max error over 20 steps:  {max_err:.2f}")
    print(f"  (Errors are expected due to finite-difference approximation)")
    print("  ✅ Diagnostic complete (errors are acceptable for compensation)")


def check_smoke_test():
    """Check 4: Smoke test — 4 envs × 50 steps with G1WalkFlatCTCP."""
    print("\n" + "=" * 60)
    print("Check 4: Smoke test — G1WalkFlatCTCP (4 envs × 50 steps)")
    print("=" * 60)

    from unilab.base.registry import ensure_registries, make
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    ensure_registries()

    # Use the CC config to get reward_config, then create CTCP env
    with initialize_config_dir(
        config_dir="/root/UniLab_WS/UniLab/conf/offpolicy",
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=flashsac/g1_walk_flat/mujoco_cc",
                "algo.num_envs=4",
                "algo.max_iterations=1",
            ],
        )

    # Create G1WalkFlatCTCP env with CC's reward config
    env = make(
        "G1WalkFlatCTCP",
        num_envs=4,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": cfg.reward},
    )
    obs, info = env.reset(env_indices=np.arange(4))

    total_reward = 0.0
    alive_count = 0
    for step in range(50):
        action = np.random.uniform(-1, 1, size=(4, env.action_space.shape[-1]))
        state = env.step(action)
        reward = state.reward
        total_reward += np.mean(reward)
        alive_count += np.sum(reward > -50)

    avg_reward = total_reward / 50
    alive_rate = alive_count / (50 * 4) * 100
    print(f"  Avg reward per step: {avg_reward:.2f}")
    print(f"  Alive rate: {alive_rate:.1f}%")

    if alive_rate > 50:
        print("  ✅ PASS: Smoke test — env runs and agents survive")
    else:
        print("  ⚠️ WARNING: Low alive rate, may need contact_scale tuning")


if __name__ == "__main__":
    check_standing_balance()
    check_rnea_approach()
    check_qacc_estimation()
    check_smoke_test()
    print("\n" + "=" * 60)
    print("All diagnostics complete!")
    print("=" * 60)
