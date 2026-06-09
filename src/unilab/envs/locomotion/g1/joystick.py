"""G1 joystick locomotion environments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.augmentation import SymmetryObsLayout
from unilab.base.backend import create_backend
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    Commands,
    sample_heading_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv


@dataclass
class G1DomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


def sample_gait_phase_pairs(rng, num_samples: int, mode: str) -> np.ndarray:
    if mode == "independent":
        return np.asarray(
            np.column_stack(
                [
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                ]
            ),
            dtype=get_global_dtype(),
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,))
    return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())


def sample_reset_base_qvel(rng, num_samples: int, limit: float) -> np.ndarray:
    return np.asarray(rng.uniform(-limit, limit, size=(num_samples, 6)), dtype=get_global_dtype())


def build_upper_body_pose_weights(pose_weights: list[float]) -> np.ndarray:
    weights = np.asarray(pose_weights, dtype=get_global_dtype()).copy()
    weights[:12] = 0.0
    return np.asarray(weights, dtype=get_global_dtype())


def compute_feet_phase_height_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    def cubic_bezier_height(phi: np.ndarray, swing_height: float) -> np.ndarray:
        phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def cubic_bezier_interpolation(
            y_start: np.ndarray, y_end: np.ndarray, t: np.ndarray
        ) -> np.ndarray:
            y_diff = y_end - y_start
            bezier = t**3 + 3 * (t**2 * (1 - t))
            return np.asarray(y_start + y_diff * bezier, dtype=get_global_dtype())

        stance = cubic_bezier_interpolation(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
        swing = cubic_bezier_interpolation(
            np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1
        )
        return np.where(x <= 0.5, stance, swing)

    left_target = cubic_bezier_height(gait_phase[:, 0], swing_height)
    right_target = cubic_bezier_height(gait_phase[:, 1], swing_height)
    return left_target, right_target


LEFT_FOOT_CONTACT_SENSORS = [f"left_foot_contact_{i}" for i in range(4)]
RIGHT_FOOT_CONTACT_SENSORS = [f"right_foot_contact_{i}" for i in range(4)]


def _scalarize_sensor_values(sensor_values: np.ndarray) -> np.ndarray:
    sensor_array = np.asarray(sensor_values, dtype=get_global_dtype())
    if sensor_array.ndim == 1:
        return sensor_array
    if sensor_array.ndim == 2 and sensor_array.shape[1] == 1:
        return sensor_array[:, 0]
    raise ValueError(f"Expected scalar sensor values, got shape {sensor_array.shape}")


def compute_aggregated_foot_contact(backend: Any, sensor_names: list[str]) -> np.ndarray:
    contacts = [_scalarize_sensor_values(backend.get_sensor_data(name)) for name in sensor_names]
    return np.asarray(np.any(np.stack(contacts, axis=1) > 0.5, axis=1), dtype=np.bool_)


def compute_feet_phase_contact_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    contact_height_threshold = swing_height * 0.5
    return left_target <= contact_height_threshold, right_target <= contact_height_threshold


def compute_forward_speed_gate(linvel: np.ndarray, min_forward_speed: float) -> np.ndarray:
    forward_speed = np.maximum(linvel[:, 0], 0.0)
    return np.asarray(forward_speed >= min_forward_speed, dtype=get_global_dtype())


def compute_forward_command_mask(commands: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(commands[:, 0], 0.0) > 1.0e-6, dtype=get_global_dtype())


@dataclass
class G1RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    min_forward_speed_for_gait_reward: float = 0.0
    close_feet_threshold: float = 0.15
    pose_weights: list[float] = field(
        default_factory=lambda: [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ]
    )


@dataclass
class G1WalkLegacyRewardConfig(G1RewardConfig):
    pass


@dataclass
class CurriculumConfig:
    enabled: bool = False
    initial_scale: float = 0.5
    min_scale: float = 0.5
    max_scale: float = 1.0
    level_down_threshold: float = 150.0
    level_up_threshold: float = 750.0
    degree: float = 0.001


@dataclass
class G1WalkEnvCfg(G1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: G1RewardConfig | None = None
    domain_rand: G1DomainRandConfig = field(default_factory=G1DomainRandConfig)
    gait_phase_init_mode: str = "offset_phase"
    reset_base_qvel_limit: float = 0.5
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


class G1WalkDomainRandomizationProvider(LocomotionDRProvider):
    def __init__(self, *, base_kp: np.ndarray | None = None, base_kd: np.ndarray | None = None):
        self._base_kp = base_kp
        self._base_kd = base_kd

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._base_kp, self._base_kd

    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates = {"gait_phase": self._sample_gait_phase(env, num_reset)}
        if getattr(env.cfg.commands, "heading_command", False):
            updates["heading_commands"] = sample_heading_commands(env, num_reset)
        return updates

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())

        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

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
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)  # type: ignore[no-any-return]


class G1WalkEnv(G1BaseEnv):
    _cfg: G1WalkEnvCfg
    _reward_cfg: Any

    def __init__(self, cfg: G1WalkEnvCfg, num_envs=1, backend_type="mujoco"):
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

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)
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

        self._init_reward_functions()
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()
            dr_provider = G1WalkDomainRandomizationProvider(base_kp=base_kp, base_kd=base_kd)
        else:
            dr_provider = G1WalkDomainRandomizationProvider()
        self._init_domain_randomization(dr_provider)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(29) + dof_vel(29) + action(29) + cmd(3) + phase(2) = 98
        return {"obs": 98, "critic": 101}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "penalty_orientation": rewards.orientation,
            "ang_vel_xy": rewards.ang_vel_xy,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "penalty_action_rate": rewards.action_rate,
            "base_height": rewards.base_height,
            "pose": rewards.weighted_pose,
            "upper_body_pose": self._reward_upper_body_pose,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "feet_double_stance": self._reward_feet_double_stance,
            "feet_air_time": self._reward_feet_air_time,
            "alive": rewards.alive,
        }

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._terrain_relative_base_height() < self._reward_cfg.min_base_height,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

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

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        walk_profile = self._uses_walk_observation_profile()

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        actor_gyro_scale = 0.25 if walk_profile else 1.0
        actor_dof_vel_scale = 0.05 if walk_profile else 1.0

        actor = np.concatenate(
            [
                noisy_gyro * actor_gyro_scale,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * actor_dof_vel_scale,
                last_actions,
                command,
                gait_phase,
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
                command,
                gait_phase,
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

    def _uses_walk_observation_profile(self) -> bool:
        scales = getattr(getattr(self, "_reward_cfg", None), "scales", None)
        if scales is None:
            reward_cfg = getattr(self._cfg, "reward_config", None)
            scales = getattr(reward_cfg, "scales", None)

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

    def _actor_symmetry_obs_layout(self) -> SymmetryObsLayout:
        return (
            ("gyro", 3),
            ("gravity", 3),
            ("dof_pos", self._num_action),
            ("dof_vel", self._num_action),
            ("actions", self._num_action),
            ("command", 3),
            ("gait_phase", 2),
        )

    def get_symmetry_obs_layouts(self) -> dict[str, SymmetryObsLayout]:
        actor_layout = self._actor_symmetry_obs_layout()
        return {
            "obs": actor_layout,
            "critic": (*actor_layout, ("linvel", 3)),
        }

    def build_symmetry_augmentation(self, *, device: str):
        if self._backend.backend_type != "mujoco":
            return None
        from unilab.envs.locomotion.g1.symmetry import G1SymmetryAugmentation

        return G1SymmetryAugmentation(
            self._backend.model,
            self.get_symmetry_obs_layouts(),
            device=device,
        )

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
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_feet_phase(self, ctx: RewardContext):
        """Reward gait phase tracking by encouraging the expected swing-foot height."""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        left_error = np.square(left_foot[:, 2] - left_target)
        right_error = np.square(right_foot[:, 2] - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        return compute_forward_speed_gate(linvel, min_forward_speed)

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        actual_delta = left_foot[:, 2] - right_foot[:, 2]
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_feet_ori(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1)

    def _reward_upper_body_pose(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(self._upper_body_pose_weights * np.square(diff), axis=1),
            dtype=get_global_dtype(),
        )


def _walk_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        initial_scale=0.5,
        min_scale=0.5,
        max_scale=1.0,
        level_down_threshold=150.0,
        level_up_threshold=750.0,
        degree=0.001,
    )


@dataclass
class G1WalkControlConfig:
    action_scale: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class G1WalkRewardConfig(G1RewardConfig):
    """Align reward weights with holosoma G1 walking."""


@registry.envcfg("G1WalkFlat")
@dataclass
class G1WalkFlatCfg(G1WalkEnvCfg):
    reward_config: G1WalkRewardConfig | None = None
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    control_config: G1WalkControlConfig = field(default_factory=G1WalkControlConfig)  # type: ignore[assignment]
    curriculum: CurriculumConfig = field(default_factory=_walk_curriculum)


@registry.envcfg("G1WalkRough")
@dataclass
class G1WalkRoughCfg(G1WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_rough.xml")
        )
    )


registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="motrix")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="motrix")


# ======================================================================== #
# G1 Walk with Gravity Compensation (motor actuator + Pinocchio)            #
# ======================================================================== #


@dataclass
class G1WalkGCControlConfig:
    """Control config for G1 with motor actuators and gravity compensation."""

    action_scale: float = 1.0
    simulate_action_latency: bool = False
    gravity_comp_mask: list[float] | None = None
    gravity_scale: float = 1.0


def _build_g1_gc_backend_reset_randomization(
    env: Any, num_reset: int
) -> ResetRandomizationPayload | None:
    """Build reset DR payloads for G1 motor-actuator mode.

    kp/kd are intentionally excluded here — they are handled by the env
    in ``sample_reset_motor_gains()`` / ``set_motor_gains()``.
    """
    from unilab.dr import ResetRandomizationPayload

    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()

    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))

    if getattr(domain_rand, "randomize_com", False):
        low, high = domain_rand.com_offset_x
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset

    if getattr(domain_rand, "randomize_gravity", False):
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        if gravity_range.shape != (2, 3):
            raise ValueError(
                f"domain_rand.gravity_range must have shape (2, 3), got {gravity_range.shape}"
            )
        low = np.minimum(gravity_range[0], gravity_range[1])
        high = np.maximum(gravity_range[0], gravity_range[1])
        payload.gravity = np.random.uniform(low=low, high=high, size=(num_reset, 3))

    return None if payload.is_empty() else payload


class G1WalkGCDomainRandomizationProvider(LocomotionDRProvider):
    """DR provider for G1 motor-actuator environments.

    kp/kd randomization is handled by the env (not the backend), since
    motor actuators don't have gains in the MuJoCo model. Other DR terms
    (base mass, gravity, ground friction, etc.) are delegated to the backend
    as usual.
    """

    def __init__(self, *, base_kp: np.ndarray, base_kd: np.ndarray):
        self._base_kp = base_kp
        self._base_kd = base_kd

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        """Validate DR using our own builder that excludes kp/kd from backend payload."""
        from unilab.dr.dr_utils import validate_interval_push_support

        payload = _build_g1_gc_backend_reset_randomization(env, num_reset=1)
        if payload is not None:
            unsupported = capabilities.get_unsupported_reset_terms(payload.requested_terms())
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support G1 GC reset "
                    f"randomization terms: {names}"
                )
        validate_interval_push_support(env, capabilities)

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return None — kp/kd DR is handled by the env, not the backend."""
        return None, None

    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates: dict[str, np.ndarray] = {
            "gait_phase": self._sample_gait_phase(env, num_reset),
        }
        if getattr(env.cfg.commands, "heading_command", False):
            updates["heading_commands"] = sample_heading_commands(env, num_reset)
        return updates

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())
        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

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
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)  # type: ignore[no-any-return]

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        """Build reset plan with kp/kd randomized in the env, not the backend."""
        from unilab.dr import ResetPlan
        from unilab.dr.dr_utils import zero_actions

        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        limit = self._get_qvel_limit(env)
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-limit, limit, size=(num_reset, 6)), dtype=get_global_dtype()
        )

        # Motor gain randomization — handled by the env, not the backend
        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)

        info_updates: dict[str, Any] = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
        }
        info_updates.update(self._build_extra_info_updates(env, num_reset))
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=_build_g1_gc_backend_reset_randomization(env, num_reset),
        )


@registry.envcfg("G1WalkFlatGC")
@dataclass
class G1WalkFlatGCCfg(G1WalkEnvCfg):
    """Config for G1 walk with gravity compensation (motor actuators)."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    control_config: G1WalkGCControlConfig = field(  # type: ignore[assignment]
        default_factory=G1WalkGCControlConfig
    )
    curriculum: CurriculumConfig = field(default_factory=_walk_curriculum)


@registry.env("G1WalkFlatGC", sim_backend="mujoco")
class G1WalkFlatGCEvn(G1BaseEnv):
    """G1 walk with motor actuators and Pinocchio gravity compensation.

    Key differences from G1WalkEnv (position actuators):
    1. MuJoCo position actuators → motor (torque) actuators
    2. PD control computed in ``pre_step_control`` (not by MuJoCo)
    3. Gravity compensation added: τ = kp(q_d-q) - kd·q̇ + g(q)
    4. kp/kd domain randomization handled in env (not backend)

    The policy's obs/action space is identical to G1WalkFlat for fair A/B
    comparison. The only difference is the dynamics seen by the policy:
    gravity is compensated, so the policy only needs to track positions
    without fighting gravity.
    """

    _cfg: G1WalkFlatGCCfg

    def __init__(self, cfg: G1WalkFlatGCCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        # Create backend
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

        # Switch MuJoCo position actuators → motor actuators before materialization.
        # This modifies the model in-place so that subsequent pool compilation
        # uses motor (torque) actuator mode.
        from unilab.control.actuator_switch import switch_to_motor_actuators
        from unilab.control.pinocchio_model import PinocchioDynamicsModel

        actuator_info = switch_to_motor_actuators(backend._model)

        # Build Pinocchio dynamics model from MuJoCo model (cold path)
        dynamics_model = PinocchioDynamicsModel(backend._model)

        # Initialize base env (G1BaseEnv, which passes backend to LocomotionBaseEnv)
        # Note: We must switch actuators before super().__init__ materializes the pool,
        # but the pool is lazily created so modifying the model in-place works.
        super().__init__(cfg, backend, num_envs)

        # --- G1WalkEnv setup (replicated, since we inherit from G1BaseEnv) ---
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)

        # Episode length tracking / curriculum
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

        self._init_reward_functions()

        # --- Motor actuator specific setup ---
        num_actions = self._num_action
        self._base_motor_kp = actuator_info.kp.copy()
        self._base_motor_kd = actuator_info.kd.copy()
        self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, num_actions)).copy()
        self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, num_actions)).copy()
        self._force_lower = actuator_info.force_lower.copy()
        self._force_upper = actuator_info.force_upper.copy()

        # Gravity compensation setup
        gc_config = cfg.control_config
        gravity_comp_mask = None
        if gc_config.gravity_comp_mask is not None:
            gravity_comp_mask = np.asarray(gc_config.gravity_comp_mask, dtype=np.float64)
            if gravity_comp_mask.shape[0] != num_actions:
                raise ValueError(
                    f"gravity_comp_mask length ({gravity_comp_mask.shape[0]}) "
                    f"must match num_actions ({num_actions})"
                )

        # Build the controller
        from unilab.control.gravity_comp_controller import GravityCompController

        self._controller = GravityCompController(
            dynamics_model=dynamics_model,
            kp=self._base_motor_kp,
            kd=self._base_motor_kd,
            force_lower=self._force_lower,
            force_upper=self._force_upper,
            gravity_comp_mask=gravity_comp_mask,
            gravity_scale=gc_config.gravity_scale,
        )
        self._dynamics_model = dynamics_model
        self._last_motor_ctrl = np.zeros((num_envs, num_actions), dtype=get_global_dtype())

        # Register pre_step_control callback
        self._backend.set_pre_step_control(self._pre_step_motor_control)

        # DR: kp/kd handled by env, not backend
        dr_provider = G1WalkGCDomainRandomizationProvider(
            base_kp=self._base_motor_kp, base_kd=self._base_motor_kd
        )
        self._init_domain_randomization(dr_provider)

    def _init_action_space(self) -> None:
        """Override: motor actuators use forcerange as ctrlrange, but the policy
        outputs target positions in [-1, 1], not raw torques.

        Use a [-1, 1] action space matching G1WalkFlat for fair comparison.
        """
        import gymnasium as gym

        nu = self._backend.num_actuators
        self._action_space = gym.spaces.Box(-1.0, 1.0, (nu,), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Same as G1WalkFlat for fair comparison
        return {"obs": 98, "critic": 101}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "penalty_orientation": rewards.orientation,
            "ang_vel_xy": rewards.ang_vel_xy,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "penalty_action_rate": rewards.action_rate,
            "base_height": rewards.base_height,
            "pose": rewards.weighted_pose,
            "upper_body_pose": self._reward_upper_body_pose,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "feet_double_stance": self._reward_feet_double_stance,
            "feet_air_time": self._reward_feet_air_time,
            "alive": rewards.alive,
        }

    def sample_reset_motor_gains(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample randomized motor gains for reset (like Go2W pattern)."""
        kp = np.broadcast_to(self._base_motor_kp, (num_reset, self._num_action)).copy()
        kd = np.broadcast_to(self._base_motor_kd, (num_reset, self._num_action)).copy()
        domain_rand = self._cfg.domain_rand
        if domain_rand.randomize_kp:
            low, high = domain_rand.kp_multiplier_range
            kp *= np.random.uniform(low, high, size=(num_reset, 1))
        if domain_rand.randomize_kd:
            low, high = domain_rand.kd_multiplier_range
            kd *= np.random.uniform(low, high, size=(num_reset, 1))
        return kp, kd

    def set_motor_gains(self, env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        """Update motor gains for specified environments."""
        self._motor_kp[env_ids] = np.asarray(kp, dtype=np.float64)
        self._motor_kd[env_ids] = np.asarray(kd, dtype=np.float64)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Same as G1WalkEnv.apply_action — returns target joint positions."""
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        ctrl: np.ndarray = actions * self._cfg.control_config.action_scale + self.default_angles
        return ctrl

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        """Pre-step callback: convert target positions → motor torques with gravity comp.

        This is called by the backend before each physics substep.
        Gravity is computed once per ctrl_dt and cached across substeps.
        """
        # Invalidate gravity cache at each substep boundary
        # (The cache will be valid for the first substep call; subsequent
        # substeps within the same ctrl_dt will reuse the cached gravity.)
        self._dynamics_model.invalidate_cache()

        joint_pos = backend.get_dof_pos()
        joint_vel = backend.get_dof_vel()
        full_qpos = backend.get_full_qpos()
        full_qvel = backend.get_full_qvel()

        # Update controller gains (may have been randomized by DR at reset)
        self._controller.kp = self._motor_kp
        self._controller.kd = self._motor_kd

        # Compute motor torques: τ = kp(q_d - q) - kd·q̇ + g(q)
        motor_ctrl = self._controller.compute(
            policy_ctrl,
            joint_pos,
            joint_vel,
            full_qpos=full_qpos,
            full_qvel=full_qvel,
        )
        self._last_motor_ctrl = motor_ctrl
        return motor_ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Override to include gravity-compensation-specific state updates."""
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._terrain_relative_base_height() < self._reward_cfg.min_base_height,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        # Store torques in info
        state.info["torques"] = self._last_motor_ctrl.copy()

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

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

    # --- G1WalkEnv reward methods (delegated) ---

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def _uses_walk_observation_profile(self) -> bool:
        scales = getattr(getattr(self, "_reward_cfg", None), "scales", None)
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

    def _actor_symmetry_obs_layout(self) -> SymmetryObsLayout:
        return (
            ("gyro", 3),
            ("gravity", 3),
            ("dof_pos", self._num_action),
            ("dof_vel", self._num_action),
            ("actions", self._num_action),
            ("command", 3),
            ("gait_phase", 2),
        )

    def get_symmetry_obs_layouts(self) -> dict[str, SymmetryObsLayout]:
        actor_layout = self._actor_symmetry_obs_layout()
        return {
            "obs": actor_layout,
            "critic": (*actor_layout, ("linvel", 3)),
        }

    def build_symmetry_augmentation(self, *, device: str):
        if self._backend.backend_type != "mujoco":
            return None
        from unilab.envs.locomotion.g1.symmetry import G1SymmetryAugmentation

        return G1SymmetryAugmentation(
            self._backend.model,
            self.get_symmetry_obs_layouts(),
            device=device,
        )

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        walk_profile = self._uses_walk_observation_profile()

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        actor_gyro_scale = 0.25 if walk_profile else 1.0
        actor_dof_vel_scale = 0.05 if walk_profile else 1.0

        actor = np.concatenate(
            [
                noisy_gyro * actor_gyro_scale,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * actor_dof_vel_scale,
                last_actions,
                command,
                gait_phase,
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
                command,
                gait_phase,
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
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        return compute_forward_speed_gate(linvel, min_forward_speed)

    def _reward_feet_phase(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        left_error = np.square(left_foot[:, 2] - left_target)
        right_error = np.square(right_foot[:, 2] - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        actual_delta = left_foot[:, 2] - right_foot[:, 2]
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_feet_ori(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1)

    def _reward_upper_body_pose(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(self._upper_body_pose_weights * np.square(diff), axis=1),
            dtype=get_global_dtype(),
        )


# ======================================================================== #
# G1 Walk with Coriolis + Centrifugal Compensation (motor actuator + Pinocchio) #
# ======================================================================== #


@dataclass
class G1WalkCCControlConfig:
    """Control config for G1 with motor actuators, gravity + Coriolis compensation."""

    action_scale: float = 1.0
    simulate_action_latency: bool = False
    gravity_comp_mask: list[float] | None = None
    gravity_scale: float = 1.0
    coriolis_comp_mask: list[float] | None = None
    coriolis_scale: float = 1.0


@registry.envcfg("G1WalkFlatCC")
@dataclass
class G1WalkFlatCCCfg(G1WalkEnvCfg):
    """Config for G1 walk with gravity + Coriolis compensation (motor actuators)."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    control_config: G1WalkCCControlConfig = field(  # type: ignore[assignment]
        default_factory=G1WalkCCControlConfig
    )
    curriculum: CurriculumConfig = field(default_factory=_walk_curriculum)


@registry.env("G1WalkFlatCC", sim_backend="mujoco")
class G1WalkFlatCCEvn(G1BaseEnv):
    """G1 walk with motor actuators and Pinocchio gravity + Coriolis compensation.

    Key differences from G1WalkFlatGCEvn (gravity compensation only):
    1. Additionally compensates Coriolis + centrifugal forces: C(q,q̇)q̇
    2. τ = kp(q_d-q) - kd·q̇ + g(q) + C(q,q̇)q̇

    The Coriolis term becomes significant at higher locomotion speeds where
    joint velocities are non-trivial (fast walking, torso rotation, swing legs).
    At low speeds it degrades gracefully to gravity-only compensation.

    The policy's obs/action space is identical to G1WalkFlat for fair A/B
    comparison.
    """

    _cfg: G1WalkFlatCCCfg

    def __init__(self, cfg: G1WalkFlatCCCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        # Create backend
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

        # Switch MuJoCo position actuators → motor actuators before materialization.
        from unilab.control.actuator_switch import switch_to_motor_actuators
        from unilab.control.pinocchio_model import PinocchioDynamicsModel

        actuator_info = switch_to_motor_actuators(backend._model)

        # Build Pinocchio dynamics model from MuJoCo model (cold path)
        dynamics_model = PinocchioDynamicsModel(backend._model)

        # Initialize base env
        super().__init__(cfg, backend, num_envs)

        # --- G1WalkEnv setup (replicated, since we inherit from G1BaseEnv) ---
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)

        # Episode length tracking / curriculum
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

        self._init_reward_functions()

        # --- Motor actuator specific setup ---
        num_actions = self._num_action
        self._base_motor_kp = actuator_info.kp.copy()
        self._base_motor_kd = actuator_info.kd.copy()
        self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, num_actions)).copy()
        self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, num_actions)).copy()
        self._force_lower = actuator_info.force_lower.copy()
        self._force_upper = actuator_info.force_upper.copy()

        # Gravity + Coriolis compensation setup
        cc_config = cfg.control_config
        gravity_comp_mask = None
        if cc_config.gravity_comp_mask is not None:
            gravity_comp_mask = np.asarray(cc_config.gravity_comp_mask, dtype=np.float64)
            if gravity_comp_mask.shape[0] != num_actions:
                raise ValueError(
                    f"gravity_comp_mask length ({gravity_comp_mask.shape[0]}) "
                    f"must match num_actions ({num_actions})"
                )

        coriolis_comp_mask = None
        if cc_config.coriolis_comp_mask is not None:
            coriolis_comp_mask = np.asarray(cc_config.coriolis_comp_mask, dtype=np.float64)
            if coriolis_comp_mask.shape[0] != num_actions:
                raise ValueError(
                    f"coriolis_comp_mask length ({coriolis_comp_mask.shape[0]}) "
                    f"must match num_actions ({num_actions})"
                )

        # Build the controller
        from unilab.control.coriolis_comp_controller import CoriolisCompController

        self._controller = CoriolisCompController(
            dynamics_model=dynamics_model,
            kp=self._base_motor_kp,
            kd=self._base_motor_kd,
            force_lower=self._force_lower,
            force_upper=self._force_upper,
            gravity_comp_mask=gravity_comp_mask,
            gravity_scale=cc_config.gravity_scale,
            coriolis_comp_mask=coriolis_comp_mask,
            coriolis_scale=cc_config.coriolis_scale,
        )
        self._dynamics_model = dynamics_model
        self._last_motor_ctrl = np.zeros((num_envs, num_actions), dtype=get_global_dtype())

        # Register pre_step_control callback
        self._backend.set_pre_step_control(self._pre_step_motor_control)

        # DR: kp/kd handled by env, not backend (reuse G1WalkGCDomainRandomizationProvider)
        dr_provider = G1WalkGCDomainRandomizationProvider(
            base_kp=self._base_motor_kp, base_kd=self._base_motor_kd
        )
        self._init_domain_randomization(dr_provider)

    def _init_action_space(self) -> None:
        """Override: motor actuators use forcerange as ctrlrange, but the policy
        outputs target positions in [-1, 1], not raw torques.

        Use a [-1, 1] action space matching G1WalkFlat for fair comparison.
        """
        import gymnasium as gym

        nu = self._backend.num_actuators
        self._action_space = gym.spaces.Box(-1.0, 1.0, (nu,), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # Same as G1WalkFlat for fair comparison
        return {"obs": 98, "critic": 101}

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "penalty_orientation": rewards.orientation,
            "ang_vel_xy": rewards.ang_vel_xy,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "penalty_action_rate": rewards.action_rate,
            "base_height": rewards.base_height,
            "pose": rewards.weighted_pose,
            "upper_body_pose": self._reward_upper_body_pose,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "feet_double_stance": self._reward_feet_double_stance,
            "feet_air_time": self._reward_feet_air_time,
            "alive": rewards.alive,
        }

    def sample_reset_motor_gains(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample randomized motor gains for reset (like Go2W pattern)."""
        kp = np.broadcast_to(self._base_motor_kp, (num_reset, self._num_action)).copy()
        kd = np.broadcast_to(self._base_motor_kd, (num_reset, self._num_action)).copy()
        domain_rand = self._cfg.domain_rand
        if domain_rand.randomize_kp:
            low, high = domain_rand.kp_multiplier_range
            kp *= np.random.uniform(low, high, size=(num_reset, 1))
        if domain_rand.randomize_kd:
            low, high = domain_rand.kd_multiplier_range
            kd *= np.random.uniform(low, high, size=(num_reset, 1))
        return kp, kd

    def set_motor_gains(self, env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        """Update motor gains for specified environments."""
        self._motor_kp[env_ids] = np.asarray(kp, dtype=np.float64)
        self._motor_kd[env_ids] = np.asarray(kd, dtype=np.float64)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Same as G1WalkEnv.apply_action — returns target joint positions."""
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        ctrl: np.ndarray = actions * self._cfg.control_config.action_scale + self.default_angles
        return ctrl

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        """Pre-step callback: convert target positions → motor torques with gravity + Coriolis comp.

        This is called by the backend before each physics substep.
        Gravity is computed once per ctrl_dt and cached across substeps.
        Coriolis depends on q̇ and is recomputed each substep.
        """
        self._dynamics_model.invalidate_cache()

        joint_pos = backend.get_dof_pos()
        joint_vel = backend.get_dof_vel()
        full_qpos = backend.get_full_qpos()
        full_qvel = backend.get_full_qvel()

        # Update controller gains (may have been randomized by DR at reset)
        self._controller.kp = self._motor_kp
        self._controller.kd = self._motor_kd

        # Compute motor torques: τ = kp(q_d - q) - kd·q̇ + g(q) + C(q,q̇)q̇
        motor_ctrl = self._controller.compute(
            policy_ctrl,
            joint_pos,
            joint_vel,
            full_qpos=full_qpos,
            full_qvel=full_qvel,
        )
        self._last_motor_ctrl = motor_ctrl
        return motor_ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Override to include compensation-specific state updates."""
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._terrain_relative_base_height() < self._reward_cfg.min_base_height,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        # Store torques in info
        state.info["torques"] = self._last_motor_ctrl.copy()

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

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

    # --- G1WalkEnv reward methods (delegated) ---

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def _uses_walk_observation_profile(self) -> bool:
        scales = getattr(getattr(self, "_reward_cfg", None), "scales", None)
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

    def _actor_symmetry_obs_layout(self) -> SymmetryObsLayout:
        return (
            ("gyro", 3),
            ("gravity", 3),
            ("dof_pos", self._num_action),
            ("dof_vel", self._num_action),
            ("actions", self._num_action),
            ("command", 3),
            ("gait_phase", 2),
        )

    def get_symmetry_obs_layouts(self) -> dict[str, SymmetryObsLayout]:
        actor_layout = self._actor_symmetry_obs_layout()
        return {
            "obs": actor_layout,
            "critic": (*actor_layout, ("linvel", 3)),
        }

    def build_symmetry_augmentation(self, *, device: str):
        if self._backend.backend_type != "mujoco":
            return None
        from unilab.envs.locomotion.g1.symmetry import G1SymmetryAugmentation

        return G1SymmetryAugmentation(
            self._backend.model,
            self.get_symmetry_obs_layouts(),
            device=device,
        )

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        walk_profile = self._uses_walk_observation_profile()

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        actor_gyro_scale = 0.25 if walk_profile else 1.0
        actor_dof_vel_scale = 0.05 if walk_profile else 1.0

        actor = np.concatenate(
            [
                noisy_gyro * actor_gyro_scale,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * actor_dof_vel_scale,
                last_actions,
                command,
                gait_phase,
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
                command,
                gait_phase,
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
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        return compute_forward_speed_gate(linvel, min_forward_speed)

    def _reward_feet_phase(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        left_error = np.square(left_foot[:, 2] - left_target)
        right_error = np.square(right_foot[:, 2] - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        actual_delta = left_foot[:, 2] - right_foot[:, 2]
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_feet_ori(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1)

    def _reward_upper_body_pose(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(self._upper_body_pose_weights * np.square(diff), axis=1),
            dtype=get_global_dtype(),
        )
