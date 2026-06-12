"""Tests for IMUGravityCompController and G1WalkFlatIMUGC environment."""

from __future__ import annotations

import numpy as np
import pytest


# ------------------------------------------------------------------ #
# Unit tests for IMUGravityCompController (no MuJoCo required)        #
# ------------------------------------------------------------------ #


class _FakeDynamicsModel:
    """Fake PinocchioDynamicsModel for unit tests."""

    def __init__(self, gravity_val: np.ndarray):
        self._gravity_val = gravity_val

    def gravity(self, qpos_batch, qvel_batch):
        num_envs = qpos_batch.shape[0]
        return np.tile(self._gravity_val, (num_envs, 1))

    def invalidate_cache(self):
        pass


def _make_imu_gc_controller(
    num_actions=4,
    gravity_val=None,
    gravity_comp_mask=None,
    gravity_scale=1.0,
    swing_boost=0.0,
    disturbance_scale=0.0,
):
    from unilab.control.imu_gc_controller import IMUGravityCompController

    if gravity_val is None:
        gravity_val = np.zeros(num_actions, dtype=np.float64)

    dynamics = _FakeDynamicsModel(
        gravity_val=np.asarray(gravity_val, dtype=np.float64),
    )

    kp = np.ones(num_actions, dtype=np.float64) * 40.0
    kd = np.ones(num_actions, dtype=np.float64) * 2.0
    force_lower = np.full(num_actions, -200.0, dtype=np.float64)
    force_upper = np.full(num_actions, 200.0, dtype=np.float64)

    return IMUGravityCompController(
        dynamics_model=dynamics,
        kp=kp,
        kd=kd,
        force_lower=force_lower,
        force_upper=force_upper,
        gravity_comp_mask=gravity_comp_mask,
        gravity_scale=gravity_scale,
        swing_boost=swing_boost,
        disturbance_scale=disturbance_scale,
    )


class TestIMUGravityCompController:
    def test_output_shape(self):
        num_envs, num_actions = 8, 4
        ctrl = _make_imu_gc_controller(num_actions=num_actions)
        target = np.zeros((num_envs, num_actions))
        pos = np.zeros((num_envs, num_actions))
        vel = np.zeros((num_envs, num_actions))
        result = ctrl.compute(target, pos, vel)
        assert result.shape == (num_envs, num_actions)

    def test_no_boost_no_disturbance_matches_gc(self):
        """With swing_boost=0 and disturbance_scale=0, output matches pure GC."""
        num_envs, num_actions = 4, 6
        gravity_val = np.array([10.0, -5.0, 3.0, 0.0, -8.0, 2.0])
        gravity_comp_mask = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])

        imu_gc = _make_imu_gc_controller(
            num_actions=num_actions,
            gravity_val=gravity_val,
            gravity_comp_mask=gravity_comp_mask,
            gravity_scale=1.0,
            swing_boost=0.0,
            disturbance_scale=0.0,
        )

        from unilab.control.gravity_comp_controller import GravityCompController

        dynamics = _FakeDynamicsModel(gravity_val=gravity_val)
        gc = GravityCompController(
            dynamics_model=dynamics,
            kp=imu_gc.kp,
            kd=imu_gc.kd,
            force_lower=np.full(num_actions, -200.0),
            force_upper=np.full(num_actions, 200.0),
            gravity_comp_mask=gravity_comp_mask,
            gravity_scale=1.0,
        )

        target = np.random.randn(num_envs, num_actions)
        pos = np.random.randn(num_envs, num_actions)
        vel = np.random.randn(num_envs, num_actions)
        full_qpos = np.zeros((num_envs, 7))  # dummy
        full_qvel = np.zeros((num_envs, 6))  # dummy

        result_imu_gc = imu_gc.compute(
            target, pos, vel, full_qpos=full_qpos, full_qvel=full_qvel
        )
        result_gc = gc.compute(target, pos, vel, full_qpos=full_qpos, full_qvel=full_qvel)
        np.testing.assert_allclose(result_imu_gc, result_gc, atol=1e-10)

    def test_swing_boost_applied(self):
        """swing_boost > 0 should add extra gravity compensation on swing joints."""
        num_envs, num_actions = 2, 4
        gravity_val = np.array([10.0, 5.0, -3.0, 8.0])
        gravity_comp_mask = np.array([1.0, 1.0, 1.0, 1.0])

        ctrl = _make_imu_gc_controller(
            num_actions=num_actions,
            gravity_val=gravity_val,
            gravity_comp_mask=gravity_comp_mask,
            gravity_scale=1.0,
            swing_boost=0.5,
        )

        target = np.zeros((num_envs, num_actions))
        pos = np.zeros((num_envs, num_actions))
        vel = np.zeros((num_envs, num_actions))
        full_qpos = np.zeros((num_envs, 7))
        full_qvel = np.zeros((num_envs, 4))

        # Without swing mask
        result_no_swing = ctrl.compute(target, pos, vel, full_qpos=full_qpos, full_qvel=full_qvel).copy()

        # With swing mask: only joint 0 and 2 are swing
        swing_mask = np.zeros((num_envs, num_actions))
        swing_mask[:, 0] = 1.0
        swing_mask[:, 2] = 1.0
        result_with_swing = ctrl.compute(
            target, pos, vel,
            full_qpos=full_qpos, full_qvel=full_qvel,
            swing_mask_per_env=swing_mask,
        ).copy()

        # Swing joints should have extra 0.5 * g(q)
        diff = result_with_swing - result_no_swing
        # Joint 0: extra 0.5 * 10.0 = 5.0
        np.testing.assert_allclose(diff[:, 0], 0.5 * 10.0, atol=1e-10)
        # Joint 1: no swing, no extra
        np.testing.assert_allclose(diff[:, 1], 0.0, atol=1e-10)
        # Joint 2: extra 0.5 * (-3.0) = -1.5
        np.testing.assert_allclose(diff[:, 2], 0.5 * (-3.0), atol=1e-10)
        # Joint 3: no swing, no extra
        np.testing.assert_allclose(diff[:, 3], 0.0, atol=1e-10)

    def test_disturbance_correction_applied(self):
        """disturbance_scale > 0 should add tau_disturbance to output."""
        num_envs, num_actions = 2, 4
        ctrl = _make_imu_gc_controller(
            num_actions=num_actions,
            gravity_scale=0.0,  # no gravity for cleaner test
            disturbance_scale=0.3,
        )

        target = np.zeros((num_envs, num_actions))
        pos = np.zeros((num_envs, num_actions))
        vel = np.zeros((num_envs, num_actions))

        tau_disturbance = np.array([[1.0, -2.0, 3.0, -4.0], [5.0, 6.0, -7.0, 8.0]])

        result = ctrl.compute(target, pos, vel, tau_disturbance=tau_disturbance)
        # Expected: PD + 0.3 * tau_disturbance, with PD = 0 (target=0, pos=0, vel=0)
        expected = 0.3 * tau_disturbance
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_force_clipping(self):
        """Output should be clipped to force limits."""
        num_envs, num_actions = 1, 2
        gravity_val = np.array([500.0, -500.0])  # will exceed limits
        ctrl = _make_imu_gc_controller(
            num_actions=num_actions,
            gravity_val=gravity_val,
            gravity_scale=1.0,
        )

        target = np.zeros((num_envs, num_actions))
        pos = np.zeros((num_envs, num_actions))
        vel = np.zeros((num_envs, num_actions))
        full_qpos = np.zeros((num_envs, 7))
        full_qvel = np.zeros((num_envs, 2))

        result = ctrl.compute(target, pos, vel, full_qpos=full_qpos, full_qvel=full_qvel)
        assert np.all(result >= -200.0)
        assert np.all(result <= 200.0)


class TestSwingMaskConstruction:
    """Test swing mask construction logic (unit level)."""

    def test_gait_phase_swing_detection(self):
        """Phase > π should indicate swing, ≤ π stance."""
        num_envs = 4
        gait_phase = np.array([
            [0.0, np.pi],        # left stance, right stance boundary
            [np.pi + 0.1, 0.5],  # left swing, right stance
            [np.pi / 2, 3 * np.pi / 2],  # left stance, right swing
            [2 * np.pi - 0.1, np.pi - 0.1],  # left swing, right stance
        ])

        left_swing = gait_phase[:, 0] > np.pi
        right_swing = gait_phase[:, 1] > np.pi

        assert left_swing[0] == False
        assert left_swing[1] == True
        assert left_swing[2] == False
        assert left_swing[3] == True

        assert right_swing[0] == False  # π is not > π
        assert right_swing[1] == False
        assert right_swing[2] == True
        assert right_swing[3] == False
