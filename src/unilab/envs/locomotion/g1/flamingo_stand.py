"""G1 Flamingo Stand (金鸡独立) environment.

Single-leg stance task for Unitree G1 humanoid robot.
Uses arm + waist compensation as the primary disturbance-rejection strategy.
Designed for FlashSAC training with penalty_-prefixed reward keys and
PenaltyCurriculum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.dr_utils import build_interval_push_plan
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.locomotion.g1.joystick import (
    G1DomainRandConfig,
    CurriculumConfig,
    InitState,
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    compute_aggregated_foot_contact,
)


# ═══════════════════════════════════════════════════════════════════════
# Domain Randomization Config
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class G1FlamingoStandDomainRandConfig(G1DomainRandConfig):
    """DR config for flamingo stand with multi-body push support."""

    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    # Multi-body push: randomly select one per push interval.
    # Empty = push pelvis only (default behaviour).
    push_body_names: list[str] = field(default_factory=list)
    # Push (disabled by default, enable in Phase 3)
    push_robots: bool = False
    push_interval: int = 150
    max_force: list[float] = field(default_factory=lambda: [3.0, 3.0, 1.5])


# ═══════════════════════════════════════════════════════════════════════
# Reward Config
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FlamingoStandRewardConfig:
    """Reward configuration for G1 Flamingo Stand task.

    Uses penalty_ prefix convention for FlashSAC training.
    Pose weights are zoned: legs HIGH, arms VERY LOW, waist LOW.
    """

    scales: dict[str, float] = field(default_factory=dict)
    base_height_target: float = 0.754
    min_base_height: float = 0.35
    max_tilt_deg: float = 60.0
    close_feet_threshold: float = 0.15
    # Passthrough fields from G1RewardConfig (unused but needed for Hydra compat)
    tracking_sigma: float = 0.25
    gait_frequency: float = 1.5
    feet_phase_swing_height: float = 0.09
    feet_phase_tracking_sigma: float = 0.005
    min_forward_speed_for_gait_reward: float = 0.0

    # ── zoned pose weights (29-dim, matches actuator order) ──────
    # Indices:  0-5=left_leg  6-11=right_leg  12-14=waist
    #          15-21=left_arm  22-28=right_arm
    pose_weights: list[float] = field(
        default_factory=lambda: [
            # Left leg (lifted): HIGH — maintain flamingo pose
            30.0, 20.0, 20.0, 30.0, 25.0, 25.0,
            # Right leg (support): HIGH — stable pillar
            20.0, 15.0, 15.0, 25.0, 30.0, 30.0,
            # Waist: LOW — freedom for balance rotation
            1.0, 1.0, 1.0,
            # Left arm: VERY LOW — free for counter-balance swinging
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            # Right arm: VERY LOW — free for counter-balance swinging
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
        ]
    )

    # ── target pose offsets from default_angles (29-dim) ─────────
    # Right leg = support (at default), Left leg = lifted.
    target_pose_offsets: list[float] = field(
        default_factory=lambda: [
            # All zero — the "flamingo" keyframe IS the target pose.
            # Policy tracks deviation from keyframe default_angles.
            0.0,
        ]
        * 29
    )


# ═══════════════════════════════════════════════════════════════════════
# Main Config
# ═══════════════════════════════════════════════════════════════════════


@registry.envcfg("G1FlamingoStand")
@dataclass
class G1FlamingoStandCfg(G1BaseCfg):
    """Configuration for G1 Flamingo Stand task."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    reward_config: FlamingoStandRewardConfig | None = None
    domain_rand: G1FlamingoStandDomainRandConfig = field(
        default_factory=G1FlamingoStandDomainRandConfig
    )
    reset_base_qvel_limit: float = 0.05
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


# ═══════════════════════════════════════════════════════════════════════
# DR Provider
# ═══════════════════════════════════════════════════════════════════════


class G1FlamingoStandDRProvider(LocomotionDRProvider):
    """DR provider for flamingo stand — zero commands, minimal initial velocity."""

    def __init__(self, *, base_kp=None, base_kd=None,
                 base_body_mass=None, base_geom_friction=None,
                 ground_geom_id=None, base_dof_armature=None):
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id
        self._base_dof_armature = base_dof_armature

    def _get_base_actuator_gains(self, env):
        return self._base_kp, self._base_kd

    def _get_reset_randomization_baselines(self, env):
        return (self._base_body_mass, self._base_geom_friction,
                self._ground_geom_id, self._base_dof_armature)

    def _get_qvel_limit(self, env):
        return float(env.cfg.reset_base_qvel_limit)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        """Override to support multi-body push: randomly pick a body part each push."""
        from unilab.dr import IntervalRandomizationPlan

        domain_rand = getattr(env.cfg, "domain_rand", None)
        if domain_rand is None or not getattr(domain_rand, "push_robots", False):
            return None
        if step_counter % domain_rand.push_interval != 0:
            return None

        push_names = getattr(domain_rand, "push_body_names", None)
        if push_names:
            # Multi-body: randomly pick ONE body for all envs each push
            import mujoco as _mj
            model = env._backend.model
            body_ids = []
            for name in push_names:
                try:
                    bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, name)
                    body_ids.append(bid)
                except Exception:
                    pass
            if body_ids:
                chosen = np.random.choice(body_ids)
                num_envs = env.num_envs
                force = env._backend._sample_push_force(domain_rand.max_force)
                # reshape to (num_envs, 1, 3)
                return IntervalRandomizationPlan(
                    body_ids=np.array([chosen]),
                    body_force=force.reshape(num_envs, 1, 3),
                )
            # Fall through to default pelvis push
            return build_interval_push_plan(env, step_counter)

        # Default: single-body pelvis push
        return build_interval_push_plan(env, step_counter)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        # Standing task: all velocity commands are zero
        return np.zeros((num_reset, 3), dtype=get_global_dtype())

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        return {}  # No gait_phase for standing task

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)


# ═══════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════


class G1FlamingoStandEnv(G1BaseEnv):
    """G1 Flamingo Stand (金鸡独立) environment.

    Right-leg support, left-leg lifted. Uses zoned pose weights to give
    arms and waist freedom for active counter-balance against disturbances.

    Starts from the "flamingo" keyframe — left leg already lifted.
    Policy only needs to learn to maintain the pose, not lift the leg.
    """

    _cfg: G1FlamingoStandCfg
    _reward_cfg: FlamingoStandRewardConfig
    _keyframe_name = "flamingo"

    def __init__(self, cfg: G1FlamingoStandCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        # ── pose config ────────────────────────────────────────────
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError(
                f"pose_weights length {self._pose_weights.shape[0]} != num_action {self._num_action}"
            )
        target_offsets = np.array(self._reward_cfg.target_pose_offsets, dtype=get_global_dtype())
        self._target_pose = self.default_angles + target_offsets

        # ── curriculum ─────────────────────────────────────────────
        self._episode_tracker: EpisodeLengthTracker | None = None
        self._penalty_curriculum: PenaltyCurriculum | None = None
        if cfg.curriculum.enabled:
            self._episode_tracker = EpisodeLengthTracker(num_envs)
            self._penalty_curriculum = PenaltyCurriculum(
                self,
                enabled=True,
                initial_scale=cfg.curriculum.initial_scale,
                min_scale=cfg.curriculum.min_scale,
                max_scale=cfg.curriculum.max_scale,
                level_down_threshold=cfg.curriculum.level_down_threshold,
                level_up_threshold=cfg.curriculum.level_up_threshold,
                degree=cfg.curriculum.degree,
            )

        # ── reward / DR ────────────────────────────────────────────
        self._init_reward_functions()

        # Collect DR baselines from backend
        base_kp, base_kd = None, None
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()

        base_body_mass = None
        if cfg.domain_rand.randomize_body_mass:
            base_body_mass = backend.get_body_mass()

        base_geom_friction, ground_geom_id = None, None
        if cfg.domain_rand.randomize_ground_friction:
            base_geom_friction = backend.get_geom_friction()
            ground_geom_id = getattr(backend, "_ground_geom_id", 0)

        dr_provider = G1FlamingoStandDRProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_body_mass=base_body_mass,
            base_geom_friction=base_geom_friction,
            ground_geom_id=ground_geom_id,
        )
        self._init_domain_randomization(dr_provider)

        # For external force visualization
        self._last_push_force: np.ndarray = np.zeros(3, dtype=np.float64)

    def step(self, actions: np.ndarray):
        """Override step to expose last_push_force for visualization."""
        from unilab.base.np_env import NpEnvState
        import time
        if self._state is None:
            self.init_state()
        assert self._state is not None
        state = self._state

        ctrl = self.apply_action(actions, state)
        if self._dr_manager is not None:
            self._dr_manager.apply_interval_randomization_if_due(self.step_counter)
        self._state = state.replace(truncated=np.zeros_like(state.truncated))
        self._clear_step_final_observation()

        # ── capture push force before physics clears it ──
        pid = getattr(self._backend, "_push_body_id", 1)
        self._last_push_force = self._backend._pending_xfrc_applied[0, 6*pid:6*pid+3].copy()

        self._backend.step(ctrl, self._cfg.sim_substeps)
        self._state = self.update_state(self._state)
        self._state.info["steps"] += 1
        self.step_counter += 1
        t = self._compute_truncated(self._state)
        np.logical_or(self._state.truncated, t, out=self._state.truncated)
        if self._autoreset and np.any(self._state.terminated | self._state.truncated):
            self._reset_done_envs()
        np.nan_to_num(self._state.reward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return self._state

    # ── observation spec ───────────────────────────────────────────────

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3)+gravity(3)+diff(29)+dof_vel(29)+last_actions(29)+contact(2)=95
        # critic = actor_base(95) + linvel(3) = 98
        return {"obs": 95, "critic": 98}

    # ── reward dispatch ────────────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "penalty_orientation": rewards.orientation,
            "penalty_base_height": self._reward_base_height_deviation,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "pose": self._reward_flamingo_pose,
            "penalty_support_foot_contact": self._reward_support_foot_contact,
            "penalty_lifted_foot_contact": self._reward_lifted_foot_contact,
            "com_over_support": self._reward_com_over_support,
            "penalty_action_rate": rewards.action_rate,
            "alive": rewards.alive,
            "penalty_feet_ori": self._reward_feet_ori,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
        }

    def _build_reward_context(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> RewardContext:
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=0.25,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    # ── observation building ───────────────────────────────────────────

    def _uses_walk_observation_profile(self) -> bool:
        """Check whether to use walk observation profile.

        FlashSAC with penalty_ keys triggers walk profile (reduced gyro/dof_vel scaling).
        """
        scales = getattr(self._reward_cfg, "scales", None)
        if scales is not None:
            if any(
                key in scales
                for key in (
                    "penalty_orientation",
                    "penalty_ang_vel_xy",
                    "penalty_action_rate",
                    "alive",
                )
            ):
                return True
            if any(key in scales for key in ("orientation", "ang_vel_xy", "action_rate")):
                return False

        curriculum = getattr(self._cfg, "curriculum", None)
        return bool(curriculum is not None and curriculum.enabled)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        last_actions = info.get("current_actions", np.zeros_like(diff))
        walk_profile = self._uses_walk_observation_profile()

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)

        actor_gyro_scale = 0.25 if walk_profile else 1.0
        actor_dof_vel_scale = 0.05 if walk_profile else 1.0

        # Contact states (aggregated bool → float)
        # Slice to match input size (may be subset during reset)
        num = dof_pos.shape[0]
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)[:num]
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)[:num]
        contact_states = np.column_stack(
            [
                left_contact.astype(get_global_dtype()),
                right_contact.astype(get_global_dtype()),
            ]
        )

        actor = np.concatenate(
            [
                noisy_gyro * actor_gyro_scale,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * actor_dof_vel_scale,
                last_actions,
                contact_states,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        critic_gyro_scale = 0.25 if walk_profile else 1.0
        critic_dof_vel_scale = 0.05 if walk_profile else 1.0
        critic_linvel_scale = 2.0 if walk_profile else 1.0

        critic_base = np.concatenate(
            [
                gyro * critic_gyro_scale,
                -gravity,
                diff,
                dof_vel * critic_dof_vel_scale,
                last_actions,
                contact_states,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [
                critic_base,
                np.asarray(linvel * critic_linvel_scale, dtype=get_global_dtype()),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        return {"obs": actor, "critic": critic}

    # ── state update (step callback) ───────────────────────────────────

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # Termination: tilt or base too low
        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._terrain_relative_base_height() < self._reward_cfg.min_base_height,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        # Curriculum update
        done = state.terminated | state.truncated
        if (
            self._episode_tracker is not None
            and self._penalty_curriculum is not None
            and np.any(done)
        ):
            done_indices = np.where(done)[0]
            episode_lengths = state.info["steps"][done_indices] + 1
            self._episode_tracker.update(episode_lengths)
            self._penalty_curriculum.update(self._episode_tracker.average_length)

            if "log" not in state.info:
                state.info["log"] = {}
            state.info["log"]["curriculum/average_episode_length"] = float(
                self._episode_tracker.average_length
            )
            state.info["log"]["curriculum/penalty_scale"] = float(
                self._penalty_curriculum.current_scale
            )

        return state

    # ═══════════════════════════════════════════════════════════════
    # Custom Reward Functions
    # ═══════════════════════════════════════════════════════════════

    def _reward_base_height_deviation(self, ctx: RewardContext) -> np.ndarray:
        """Penalty for base height deviation from target."""
        return np.square(ctx.base_height - ctx.base_height_target)

    def _reward_flamingo_pose(self, ctx: RewardContext) -> np.ndarray:
        """Weighted L2 penalty for joint deviation from flamingo target pose.

        Uses zoned pose_weights: legs HIGH, arms VERY LOW, waist LOW.
        """
        diff = ctx.dof_pos - self._target_pose  # deviation from target (not default_angles)
        return np.asarray(
            np.sum(self._pose_weights * np.square(diff), axis=1), dtype=get_global_dtype()
        )

    def _reward_support_foot_contact(self, ctx: RewardContext) -> np.ndarray:
        """Penalty when the support (right) foot loses ground contact."""
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        return np.asarray(~right_contact, dtype=get_global_dtype())

    def _reward_lifted_foot_contact(self, ctx: RewardContext) -> np.ndarray:
        """Penalty when the lifted (left) foot touches the ground (prevents cheating)."""
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        return np.asarray(left_contact, dtype=get_global_dtype())

    def _reward_com_over_support(self, ctx: RewardContext) -> np.ndarray:
        """Reward for keeping COM horizontally over the support foot.

        Uses exponential decay with sigma ≈ 0.14m (foot radius).
        """
        com_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        right_foot_pos = np.asarray(
            self._backend.get_sensor_data("right_foot_pos"), dtype=get_global_dtype()
        )
        dist = np.linalg.norm(com_pos[:, :2] - right_foot_pos[:, :2], axis=1)
        return np.asarray(np.exp(-(dist**2) / 0.02), dtype=get_global_dtype())

    def _reward_feet_ori(self, ctx: RewardContext) -> np.ndarray:
        """Penalty for feet orientation deviation from flat (reused from walk)."""
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext) -> np.ndarray:
        """Penalty when feet are too close together in xy-plane."""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        threshold = self._reward_cfg.close_feet_threshold
        return np.where(
            feet_dist < threshold,
            np.square(feet_dist - threshold),
            0.0,
        )


# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

registry.register_env("G1FlamingoStand", G1FlamingoStandEnv, sim_backend="mujoco")
