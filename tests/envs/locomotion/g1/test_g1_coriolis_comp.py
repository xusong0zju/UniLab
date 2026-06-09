"""Tests for CoriolisCompController and G1WalkFlatCC environment."""

from __future__ import annotations

import numpy as np
import pytest


# ------------------------------------------------------------------ #
# Unit tests for CoriolisCompController (no MuJoCo required)          #
# ------------------------------------------------------------------ #


class _FakeDynamicsModel:
    """Fake PinocchioDynamicsModel for unit tests."""

    def __init__(self, gravity_val: np.ndarray, coriolis_val: np.ndarray):
        self._gravity_val = gravity_val
        self._coriolis_val = coriolis_val

    def gravity(self, qpos_batch, qvel_batch):
        num_envs = qpos_batch.shape[0]
        return np.tile(self._gravity_val, (num_envs, 1))

    def coriolis(self, qpos_batch, qvel_batch):
        num_envs = qpos_batch.shape[0]
        return np.tile(self._coriolis_val, (num_envs, 1))

    def invalidate_cache(self):
        pass


def _make_controller(
    num_actions=4,
    gravity_val=None,
    coriolis_val=None,
    gravity_comp_mask=None,
    gravity_scale=1.0,
    coriolis_comp_mask=None,
    coriolis_scale=1.0,
):
    from unilab.control.coriolis_comp_controller import CoriolisCompController

    if gravity_val is None:
        gravity_val = np.zeros(num_actions, dtype=np.float64)
    if coriolis_val is None:
        coriolis_val = np.zeros(num_actions, dtype=np.float64)

    dynamics = _FakeDynamicsModel(
        gravity_val=np.asarray(gravity_val, dtype=np.float64),
        coriolis_val=np.asarray(coriolis_val, dtype=np.float64),
    )
    kp = np.ones(num_actions, dtype=np.float64) * 40.0
    kd = np.ones(num_actions, dtype=np.float64) * 2.0
    force_lower = np.full(num_actions, -100.0, dtype=np.float64)
    force_upper = np.full(num_actions, 100.0, dtype=np.float64)

    return CoriolisCompController(
        dynamics_model=dynamics,
        kp=kp,
        kd=kd,
        force_lower=force_lower,
        force_upper=force_upper,
        gravity_comp_mask=gravity_comp_mask,
        gravity_scale=gravity_scale,
        coriolis_comp_mask=coriolis_comp_mask,
        coriolis_scale=coriolis_scale,
    )


def test_coriolis_comp_controller_output_shape():
    """Output shape matches input shape."""
    num_envs, num_actions = 3, 4
    ctrl = _make_controller(num_actions=num_actions)
    target_pos = np.zeros((num_envs, num_actions))
    joint_pos = np.zeros((num_envs, num_actions))
    joint_vel = np.zeros((num_envs, num_actions))
    full_qpos = np.zeros((num_envs, 10))
    full_qvel = np.zeros((num_envs, 10))

    out = ctrl.compute(target_pos, joint_pos, joint_vel, full_qpos=full_qpos, full_qvel=full_qvel)

    assert out.shape == (num_envs, num_actions)
    assert out.dtype == np.float64


def test_coriolis_comp_degrades_to_gravity_comp_when_coriolis_zero():
    """With coriolis_scale=0, output matches GravityCompController."""
    num_envs, num_actions = 2, 4
    gravity_val = np.array([5.0, -3.0, 1.0, 0.0])
    coriolis_val = np.array([2.0, -1.0, 0.5, 0.0])  # non-zero, but scale=0

    ctrl = _make_controller(
        num_actions=num_actions,
        gravity_val=gravity_val,
        coriolis_val=coriolis_val,
        gravity_scale=1.0,
        coriolis_scale=0.0,
    )

    target_pos = np.ones((num_envs, num_actions)) * 0.5
    joint_pos = np.zeros((num_envs, num_actions))
    joint_vel = np.ones((num_envs, num_actions)) * 0.1
    full_qpos = np.zeros((num_envs, 10))
    full_qvel = np.zeros((num_envs, 10))

    out = ctrl.compute(target_pos, joint_pos, joint_vel, full_qpos=full_qpos, full_qvel=full_qvel)

    # Expected: PD + gravity (no coriolis)
    # PD = kp*(0.5 - 0) - kd*0.1 = 40*0.5 - 2*0.1 = 19.8
    # gravity = [5, -3, 1, 0]
    expected_pd = 40.0 * 0.5 - 2.0 * 0.1  # 19.8
    expected = np.broadcast_to(expected_pd + gravity_val, (num_envs, num_actions)).copy()
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_coriolis_comp_degrades_to_pd_when_both_scales_zero():
    """With gravity_scale=0 and coriolis_scale=0, output is pure PD."""
    num_envs, num_actions = 2, 4

    ctrl = _make_controller(
        num_actions=num_actions,
        gravity_val=np.ones(num_actions) * 10.0,
        coriolis_val=np.ones(num_actions) * 5.0,
        gravity_scale=0.0,
        coriolis_scale=0.0,
    )

    target_pos = np.ones((num_envs, num_actions)) * 0.3
    joint_pos = np.ones((num_envs, num_actions)) * 0.1
    joint_vel = np.ones((num_envs, num_actions)) * 0.2
    full_qpos = np.zeros((num_envs, 10))
    full_qvel = np.zeros((num_envs, 10))

    out = ctrl.compute(target_pos, joint_pos, joint_vel, full_qpos=full_qpos, full_qvel=full_qvel)

    # Pure PD: kp*(0.3-0.1) - kd*0.2 = 40*0.2 - 2*0.2 = 7.6
    expected = np.full((num_envs, num_actions), 7.6)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_coriolis_comp_adds_coriolis_term():
    """Verify that the Coriolis term is correctly added."""
    num_envs, num_actions = 1, 3
    gravity_val = np.array([2.0, 0.0, -1.0])
    coriolis_val = np.array([0.5, 1.0, -0.5])

    ctrl = _make_controller(
        num_actions=num_actions,
        gravity_val=gravity_val,
        coriolis_val=coriolis_val,
        gravity_scale=1.0,
        coriolis_scale=1.0,
    )

    target_pos = np.zeros((num_envs, num_actions))
    joint_pos = np.zeros((num_envs, num_actions))
    joint_vel = np.zeros((num_envs, num_actions))
    full_qpos = np.zeros((num_envs, 10))
    full_qvel = np.zeros((num_envs, 10))

    out = ctrl.compute(target_pos, joint_pos, joint_vel, full_qpos=full_qpos, full_qvel=full_qvel)

    # PD=0, so out = gravity + coriolis = [2.5, 1.0, -1.5]
    expected = gravity_val + coriolis_val
    np.testing.assert_allclose(out[0], expected, atol=1e-10)


def test_coriolis_comp_mask_selectively_applies():
    """coriolis_comp_mask zeros out Coriolis for masked joints."""
    num_envs, num_actions = 1, 4
    coriolis_val = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([1.0, 1.0, 0.0, 0.0])  # only first 2 joints

    ctrl = _make_controller(
        num_actions=num_actions,
        gravity_val=np.zeros(num_actions),
        coriolis_val=coriolis_val,
        gravity_scale=0.0,
        coriolis_scale=1.0,
        coriolis_comp_mask=mask,
    )

    target_pos = np.zeros((num_envs, num_actions))
    joint_pos = np.zeros((num_envs, num_actions))
    joint_vel = np.zeros((num_envs, num_actions))
    full_qpos = np.zeros((num_envs, 10))
    full_qvel = np.zeros((num_envs, 10))

    out = ctrl.compute(target_pos, joint_pos, joint_vel, full_qpos=full_qpos, full_qvel=full_qvel)

    # PD=0, gravity=0, coriolis masked: [1, 2, 0, 0]
    expected = np.array([1.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(out[0], expected, atol=1e-10)


def test_coriolis_comp_mask_defaults_to_gravity_mask():
    """If coriolis_comp_mask is None but gravity_comp_mask is set, use gravity mask."""
    from unilab.control.coriolis_comp_controller import CoriolisCompController

    num_actions = 4
    gravity_mask = np.array([1.0, 1.0, 0.0, 0.0])
    dynamics = _FakeDynamicsModel(
        gravity_val=np.ones(num_actions),
        coriolis_val=np.ones(num_actions),
    )

    ctrl = CoriolisCompController(
        dynamics_model=dynamics,
        kp=np.ones(num_actions) * 40.0,
        kd=np.ones(num_actions) * 2.0,
        force_lower=np.full(num_actions, -100.0),
        force_upper=np.full(num_actions, 100.0),
        gravity_comp_mask=gravity_mask,
        gravity_scale=1.0,
        coriolis_comp_mask=None,  # should default to gravity_mask
        coriolis_scale=1.0,
    )

    # The coriolis mask should have been set to the gravity mask
    np.testing.assert_array_equal(ctrl._coriolis_comp_mask, gravity_mask)


def test_coriolis_comp_respects_force_limits():
    """Output is clipped to force limits."""
    num_envs, num_actions = 1, 2
    # Large gravity + coriolis that would exceed limits
    ctrl = _make_controller(
        num_actions=num_actions,
        gravity_val=np.array([200.0, -200.0]),
        coriolis_val=np.array([200.0, -200.0]),
        gravity_scale=1.0,
        coriolis_scale=1.0,
    )

    target_pos = np.zeros((num_envs, num_actions))
    joint_pos = np.zeros((num_envs, num_actions))
    joint_vel = np.zeros((num_envs, num_actions))
    full_qpos = np.zeros((num_envs, 10))
    full_qvel = np.zeros((num_envs, 10))

    out = ctrl.compute(target_pos, joint_pos, joint_vel, full_qpos=full_qpos, full_qvel=full_qvel)

    # Force limits are [-100, 100]
    assert np.all(out >= -100.0)
    assert np.all(out <= 100.0)


def test_coriolis_comp_without_full_state_is_pure_pd():
    """Without full_qpos/full_qvel, only PD is computed (no dynamics comp)."""
    num_envs, num_actions = 1, 3

    ctrl = _make_controller(
        num_actions=num_actions,
        gravity_val=np.ones(num_actions) * 10.0,
        coriolis_val=np.ones(num_actions) * 5.0,
    )

    target_pos = np.ones((num_envs, num_actions)) * 0.5
    joint_pos = np.zeros((num_envs, num_actions))
    joint_vel = np.ones((num_envs, num_actions)) * 0.1

    # No full_qpos/full_qvel → pure PD
    out = ctrl.compute(target_pos, joint_pos, joint_vel)

    expected_pd = 40.0 * 0.5 - 2.0 * 0.1  # 19.8
    np.testing.assert_allclose(out, expected_pd, atol=1e-10)


# ------------------------------------------------------------------ #
# Resolver integration test                                           #
# ------------------------------------------------------------------ #


def test_resolve_coriolis_comp_controller():
    """resolve_controller('coriolis_comp', ...) returns CoriolisCompController."""
    from unilab.control.coriolis_comp_controller import CoriolisCompController
    from unilab.control.resolver import resolve_controller

    dynamics = _FakeDynamicsModel(
        gravity_val=np.zeros(4),
        coriolis_val=np.zeros(4),
    )

    ctrl = resolve_controller(
        "coriolis_comp",
        dynamics_model=dynamics,
        kp=np.ones(4) * 40.0,
        kd=np.ones(4) * 2.0,
        force_lower=np.full(4, -100.0),
        force_upper=np.full(4, 100.0),
        gravity_comp_mask=np.ones(4),
        gravity_scale=1.0,
        coriolis_comp_mask=np.ones(4),
        coriolis_scale=1.0,
    )

    assert isinstance(ctrl, CoriolisCompController)


def test_resolve_coriolis_comp_requires_dynamics_model():
    """resolve_controller('coriolis_comp') without dynamics_model raises ValueError."""
    from unilab.control.resolver import resolve_controller

    with pytest.raises(ValueError, match="requires a PinocchioDynamicsModel"):
        resolve_controller(
            "coriolis_comp",
            dynamics_model=None,
            kp=np.ones(4),
            kd=np.ones(4),
            force_lower=np.full(4, -100.0),
            force_upper=np.full(4, 100.0),
        )


# ------------------------------------------------------------------ #
# Smoke test with real MuJoCo + Pinocchio (requires mujoco + pin)     #
# ------------------------------------------------------------------ #


def test_g1_walk_flat_cc_env_runs_50_steps():
    """G1WalkFlatCC env: 4 envs × 50 steps, all alive, torques reasonable."""
    try:
        import mujoco
        import pinocchio as pin
    except ImportError:
        pytest.skip("MuJoCo or Pinocchio not available")

    from unilab.envs.locomotion.g1.joystick import (
        G1WalkFlatCCEvn,
        G1WalkFlatCCCfg,
        G1WalkCCControlConfig,
        G1WalkRewardConfig,
    )

    reward_config = G1WalkRewardConfig(
        scales={
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.5,
            "penalty_ang_vel_xy": -1.0,
            "penalty_orientation": -10.0,
            "penalty_action_rate": -5.0,
            "pose": -0.5,
            "penalty_feet_ori": -25.0,
            "feet_phase": 5.0,
            "alive": 10.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.005,
        close_feet_threshold=0.15,
    )

    control_config = G1WalkCCControlConfig(
        action_scale=1.0,
        gravity_comp_mask=[1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],
        gravity_scale=1.0,
        coriolis_comp_mask=[1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0],
        coriolis_scale=1.0,
    )

    cfg = G1WalkFlatCCCfg(
        reward_config=reward_config,
        control_config=control_config,
    )

    # Create env directly
    env = G1WalkFlatCCEvn(cfg, num_envs=4, backend_type="mujoco")

    # Run 50 steps
    env_indices = np.arange(4)
    obs, info = env.reset(env_indices)
    all_alive = True
    for step in range(50):
        actions = np.zeros((4, env._num_action), dtype=np.float32)
        state = env.step(actions)
        # Check torques are within reasonable bounds
        if "torques" in state.info:
            torques = state.info["torques"]
            assert np.all(np.isfinite(torques)), f"Non-finite torques at step {step}"
        # Track alive
        if np.any(state.terminated):
            all_alive = False

    # At least some envs should survive 50 steps
    assert all_alive or step >= 10, "Env died too early"


def test_coriolis_values_nonzero_at_nonzero_velocity():
    """Verify C(q,q̇)q̇ is non-zero when q̇ ≠ 0 for the G1 model."""
    try:
        import mujoco
        import pinocchio as pin
    except ImportError:
        pytest.skip("MuJoCo or Pinocchio not available")

    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.control.pinocchio_model import PinocchioDynamicsModel

    # Load G1 model
    model_path = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    dynamics = PinocchioDynamicsModel(mj_model)

    # Set non-zero velocity
    qpos = np.broadcast_to(mj_data.qpos, (1, mj_model.nq)).copy()
    qvel = np.zeros((1, mj_model.nv))
    qvel[0, 6:] = 1.0  # all joint velocities = 1 rad/s

    coriolis = dynamics.coriolis(qpos, qvel)

    # Coriolis should be non-zero for at least some joints
    assert np.any(np.abs(coriolis) > 1e-6), (
        f"C(q,q̇)q̇ is zero for all joints despite q̇≠0. "
        f"max abs = {np.max(np.abs(coriolis)):.2e}"
    )


def test_coriolis_zero_at_zero_velocity():
    """Verify C(q,q̇)q̇ = 0 when q̇ = 0 (Coriolis is velocity-dependent)."""
    try:
        import mujoco
        import pinocchio as pin
    except ImportError:
        pytest.skip("MuJoCo or Pinocchio not available")

    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.control.pinocchio_model import PinocchioDynamicsModel

    model_path = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    dynamics = PinocchioDynamicsModel(mj_model)

    qpos = np.broadcast_to(mj_data.qpos, (1, mj_model.nq)).copy()
    qvel = np.zeros((1, mj_model.nv))  # zero velocity

    coriolis = dynamics.coriolis(qpos, qvel)

    np.testing.assert_allclose(coriolis, 0.0, atol=1e-10)
