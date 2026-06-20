"""G1 Multi-Skill Locomotion: walk, stand, flamingo — with multi-body push.

Extends G1WalkEnv with:
- Flamingo keyframe support (single-leg stance episodes)
- Multi-body persistent push (random body parts, 2s duration, overlapping)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import IntervalRandomizationPlan, ResetPlan
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.g1.joystick import (
    Commands,
    CurriculumConfig,
    G1DomainRandConfig,
    G1WalkDomainRandomizationProvider,
    G1WalkEnv,
    G1WalkEnvCfg,
    G1WalkFlatCfg,
    G1WalkRewardConfig,
    InitState,
    compute_aggregated_foot_contact,
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch


# ═══════════════════════════════════════════════════════════════════════
# Multi-body persistent push config
# ═══════════════════════════════════════════════════════════════════════

G1_PUSH_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
]


@dataclass
class MultiBodyPushConfig:
    """Configuration for multi-body persistent push."""

    push_body_names: list[str] = field(default_factory=lambda: list(G1_PUSH_BODY_NAMES))
    push_duration_steps: int = 100  # ~2 seconds at 50 Hz
    push_interval: int = 150  # steps between new push initiations


@dataclass
class MultiSkillDomainRandConfig(G1DomainRandConfig):
    """DR config with multi-body persistent push."""

    multi_body_push: MultiBodyPushConfig = field(default_factory=MultiBodyPushConfig)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Skill Env Config
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MultiSkillCommands(Commands):
    """Commands with flamingo mode support."""

    rel_flamingo_envs: float = 0.0  # fraction of envs in flamingo (single-leg) mode
    rel_fallen_envs: float = 0.0    # fraction of envs starting from fallen postures


@registry.envcfg("G1MultiSkill")
@dataclass
class G1MultiSkillCfg(G1WalkEnvCfg):
    """Config for G1 multi-skill locomotion: walk + stand + flamingo + multi-push."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    commands: MultiSkillCommands = field(default_factory=MultiSkillCommands)
    domain_rand: MultiSkillDomainRandConfig = field(default_factory=MultiSkillDomainRandConfig)
    reward_config: G1WalkRewardConfig | None = None
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Skill DR Provider
# ═══════════════════════════════════════════════════════════════════════


class MultiSkillDRProvider(G1WalkDomainRandomizationProvider):
    """DR provider with multi-keyframe support and multi-body persistent push."""

    def __init__(
        self,
        *,
        base_kp=None,
        base_kd=None,
        base_body_mass=None,
        base_geom_friction=None,
        ground_geom_id=None,
        base_dof_armature=None,
        stand_qpos=None,
        stand_ctrl=None,
        flamingo_qpos=None,
        flamingo_ctrl=None,
        kneeling_qpos=None,
        kneeling_ctrl=None,
    ):
        super().__init__(base_kp=base_kp, base_kd=base_kd)
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id
        self._base_dof_armature = base_dof_armature
        self._stand_qpos = stand_qpos
        self._stand_ctrl = stand_ctrl
        self._flamingo_qpos = flamingo_qpos
        self._flamingo_ctrl = flamingo_ctrl
        self._kneeling_qpos = kneeling_qpos
        self._kneeling_ctrl = kneeling_ctrl
        # Persistent push state: (num_envs, push_duration_steps, 3) per body
        self._active_pushes: dict[int, np.ndarray] = {}  # body_id -> (num_envs, 3) force
        self._push_remaining: dict[int, np.ndarray] = {}  # body_id -> (num_envs,) steps left

    def _get_reset_randomization_baselines(self, env):
        return (self._base_body_mass, self._base_geom_friction,
                self._ground_geom_id, self._base_dof_armature)

    def _get_qvel_limit(self, env):
        return float(getattr(env.cfg, "reset_base_qvel_limit", 0.05))

    def _sample_commands(self, env, num_reset):
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)

        # Standing envs
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0

        # Flamingo envs (subset of standing)
        flamingo_prob = float(getattr(env.cfg.commands, "rel_flamingo_envs", 0.0))
        is_flamingo = np.random.uniform(size=(num_reset,)) < flamingo_prob if flamingo_prob > 0 else np.zeros(num_reset, dtype=bool)
        if np.any(is_flamingo):
            commands[is_flamingo] = 0.0

        # Store flamingo flag in env for keyframe selection during build_reset_plan
        env._is_flamingo = is_flamingo

        # Fallen envs — start from a fallen posture (Phase D: fall recovery)
        fallen_prob = float(getattr(env.cfg.commands, "rel_fallen_envs", 0.0))
        is_fallen = np.random.uniform(size=(num_reset,)) < fallen_prob if fallen_prob > 0 else np.zeros(num_reset, dtype=bool)
        if np.any(is_fallen):
            commands[is_fallen] = 0.0  # stay in place while getting up
        env._is_fallen = is_fallen

        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands

    def build_reset_plan(self, env, env_ids):
        """Build reset plan with per-env keyframe selection (stand vs flamingo)."""
        num_reset = len(env_ids)
        flamingo_arr = getattr(env, "_is_flamingo", None)
        if flamingo_arr is None or len(flamingo_arr) != env.num_envs:
            flamingo_arr = np.zeros(env.num_envs, dtype=bool)
        is_flamingo = flamingo_arr[env_ids]

        # Start from stand keyframe by default
        qpos_all = np.tile(self._stand_qpos, (num_reset, 1))
        # Override flamingo envs with flamingo keyframe
        if np.any(is_flamingo):
            qpos_all[is_flamingo] = self._flamingo_qpos

        # ---- Fallen postures (Phase D/G) with kneeling curriculum ----
        fallen_arr = getattr(env, "_is_fallen", None)
        if fallen_arr is None or len(fallen_arr) != env.num_envs:
            fallen_arr = np.zeros(env.num_envs, dtype=bool)
        is_fallen = fallen_arr[env_ids]
        if np.any(is_fallen):
            n_fallen = int(np.sum(is_fallen))
            # Half kneeling (easier), half full-fallen (harder)
            if self._kneeling_qpos is not None:
                half = n_fallen // 2
                kneeling_idx = np.zeros(n_fallen, dtype=bool)
                kneeling_idx[:half] = True
                np.random.shuffle(kneeling_idx)
                # Build a (n_fallen, 36) array: kneeling rows take kneel posture,
                # the rest keep qpos_all[is_fallen] (stand). All three operands
                # of np.where must broadcast to (n_fallen, 36); tiling kneel to
                # only (n_kneel, 36) previously crashed with shape mismatch.
                fallen_block = np.where(
                    kneeling_idx[:, None],
                    np.tile(self._kneeling_qpos, (n_fallen, 1)),
                    qpos_all[is_fallen],
                )
                qpos_all[is_fallen] = fallen_block
                # Full fallen for the rest
                n_fallen = n_fallen - half
                is_fallen[is_fallen] = ~kneeling_idx  # only the non-kneeling ones stay as fallen
                # Re-count after split
                if n_fallen == 0:
                    is_fallen[:] = False
            # Random fallen posture per env: 0=supine, 1=prone, 2=side
            fallen_type = np.random.randint(0, 3, size=(int(np.sum(is_fallen)),))
            # Base lowered to just above termination (0.35m)
            fallen_qpos = np.tile(self._stand_qpos, (n_fallen, 1))
            fallen_qpos[:, 2] = 0.35  # base z near ground
            # Tilt quaternion based on type
            for i, ft in enumerate(fallen_type):
                if ft == 0:  # supine: roll back ~80deg
                    half_angle = np.deg2rad(80) / 2
                    q = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])
                elif ft == 1:  # prone: pitch forward ~-70deg
                    half_angle = np.deg2rad(-70) / 2
                    q = np.array([np.cos(half_angle), 0, np.sin(half_angle), 0])
                else:  # side: roll 45 + pitch 30
                    q_roll = np.array([np.cos(np.deg2rad(45)/2), np.sin(np.deg2rad(45)/2), 0, 0])
                    q_pitch = np.array([np.cos(np.deg2rad(30)/2), 0, np.sin(np.deg2rad(30)/2), 0])
                    q = np_quat_mul(q_roll.reshape(1,4), q_pitch.reshape(1,4))[0]
                fallen_qpos[i, 3:7] = np_quat_mul(
                    fallen_qpos[i, 3:7].reshape(1,4), q.reshape(1,4)
                )[0]
            qpos_all[is_fallen] = fallen_qpos

        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos_all[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos_all[:, 3:7] = np_quat_mul(qpos_all[:, 3:7], np_yaw_to_quat(yaw))
        qpos_all[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos_all[:, 0:3], yaw=yaw)

        limit = self._get_qvel_limit(env)
        qvel[:, 0:6] = np.random.uniform(-limit, limit, size=(num_reset, 6)).astype(get_global_dtype())

        # Use flamingo ctrl for flamingo envs
        current_actions = zero_actions(num_reset, env._num_action)
        if np.any(is_flamingo):
            current_actions[is_flamingo] = np.asarray(self._flamingo_ctrl, dtype=get_global_dtype())

        info_updates = {
            "commands": self._sample_commands(env, num_reset),
            "current_actions": current_actions,
            "last_actions": zero_actions(num_reset, env._num_action),
            "gait_phase": self._sample_gait_phase(env, num_reset),
        }

        env._spawn.record_episode_start(env_ids, qpos_all[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos_all,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(
                env, num_reset,
                base_kp=self._base_kp, base_kd=self._base_kd,
                base_body_mass=self._base_body_mass,
                base_geom_friction=self._base_geom_friction,
                ground_geom_id=self._ground_geom_id,
                base_dof_armature=self._base_dof_armature,
            ),
        )

    def _sample_gait_phase(self, env, num_reset):
        mode = getattr(env.cfg, "gait_phase_init_mode", "offset_phase")
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())
        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

    def _compute_reset_obs(self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel):
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)

    def build_interval_randomization_plan(self, env, step_counter):
        """Multi-body persistent push: random body, force lasts push_duration steps."""
        domain_rand = getattr(env.cfg, "domain_rand", None)
        if domain_rand is None or not getattr(domain_rand, "push_robots", False):
            return None

        mb = getattr(domain_rand, "multi_body_push", None)
        if mb is None:
            return build_interval_push_plan(env, step_counter)

        num_envs = env.num_envs
        push_names = mb.push_body_names
        duration = mb.push_duration_steps
        interval = mb.push_interval

        # Initiate new push at interval
        if step_counter % interval == 0:
            import mujoco as _mj
            model = env._backend.model
            body_ids = []
            for name in push_names:
                try:
                    body_ids.append(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, name))
                except Exception:
                    pass
            if body_ids:
                chosen = np.random.choice(body_ids)
                force = env._backend._sample_push_force(domain_rand.max_force)
                self._active_pushes[chosen] = force.copy()
                # Random duration 0.5~2.0s (25~100 steps at 50Hz) per env
                dur = np.random.randint(25, 101, size=(num_envs,))
                self._push_remaining[chosen] = dur

        # Decrement all active pushes
        expired = []
        for bid in list(self._active_pushes.keys()):
            self._push_remaining[bid] -= 1
            done = self._push_remaining[bid] <= 0
            if np.any(done):
                self._push_remaining[bid] = np.where(done, 0, self._push_remaining[bid])
            if np.all(done):
                expired.append(bid)

        # Clean expired
        for bid in expired:
            del self._active_pushes[bid]
            del self._push_remaining[bid]

        # Build plan with all active pushes (can be multiple overlapping)
        if self._active_pushes:
            all_body_ids = []
            all_forces = []
            for bid, force in self._active_pushes.items():
                all_body_ids.append(bid)
                all_forces.append(force.reshape(num_envs, 1, 3))
            return IntervalRandomizationPlan(
                body_ids=np.array(all_body_ids),
                body_force=np.concatenate(all_forces, axis=1),
            )

        return None


# ═══════════════════════════════════════════════════════════════════════
# Multi-Skill Environment
# ═══════════════════════════════════════════════════════════════════════


class G1MultiSkillEnv(G1WalkEnv):
    """G1 Multi-Skill locomotion: walk, stand, flamingo, with multi-body push."""

    _cfg: G1MultiSkillCfg
    _keyframe_name = "stand"  # default, flamingo envs override at reset

    def __init__(self, cfg: G1MultiSkillCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        from unilab.base.backend import create_backend
        backend = create_backend(
            backend_type, cfg.scene, num_envs, cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=getattr(cfg.domain_rand, "push_body_name", None),
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        # Call G1BaseEnv.__init__ directly (skip G1WalkEnv init to set up our own DR)
        from unilab.envs.locomotion.g1.base import G1BaseEnv
        G1BaseEnv.__init__(self, cfg, backend, num_envs)

        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        import math as _math
        self._gait_phase_delta = float(
            2.0 * _math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.asarray(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        from unilab.envs.locomotion.g1.joystick import build_upper_body_pose_weights
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)

        from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
        self._episode_tracker = None
        self._penalty_curriculum = None
        if cfg.curriculum.enabled:
            self._episode_tracker = EpisodeLengthTracker(num_envs)
            self._penalty_curriculum = PenaltyCurriculum(
                self, enabled=True,
                initial_scale=cfg.curriculum.initial_scale,
                min_scale=cfg.curriculum.min_scale,
                max_scale=cfg.curriculum.max_scale,
                level_down_threshold=cfg.curriculum.level_down_threshold,
                level_up_threshold=cfg.curriculum.level_up_threshold,
                degree=cfg.curriculum.degree,
            )

        self._init_reward_functions()
        # Add flamingo-specific rewards (not in G1WalkRewardConfig)
        self._reward_fns["penalty_lifted_foot_contact"] = self._reward_lifted_foot_contact
        self._reward_fns["penalty_support_foot_contact"] = self._reward_support_foot_contact
        self._reward_fns["com_over_support"] = self._reward_com_over_support
        # Height-adaptive orientation: reduced penalty near ground (HoST-inspired get-up guidance)
        self._reward_fns["penalty_orientation_adaptive"] = self._reward_orientation_adaptive
        # HumanUP-inspired get-up rewards (Phase G v2)
        self._reward_fns["height_exp"] = self._reward_height_exp
        self._reward_fns["delta_height"] = self._reward_delta_height
        self._reward_fns["stand_feet"] = self._reward_stand_feet
        self._reward_fns["soft_symmetry"] = self._reward_soft_symmetry
        self._reward_fns["uprightness_exp"] = self._reward_uprightness_exp

        # Pre-compute both keyframes
        stand_qpos = backend.get_keyframe_qpos("stand")
        stand_ctrl = stand_qpos[-self._num_action:].copy() if len(stand_qpos) > self._num_action else None
        flamingo_qpos = backend.get_keyframe_qpos("flamingo")
        flamingo_ctrl = flamingo_qpos[-self._num_action:].copy() if len(flamingo_qpos) > self._num_action else None
        kneeling_qpos = backend.get_keyframe_qpos("kneeling")
        kneeling_ctrl = kneeling_qpos[-self._num_action:].copy() if len(kneeling_qpos) > self._num_action else None

        base_kp, base_kd = None, None
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()

        base_body_mass = None
        if getattr(cfg.domain_rand, "randomize_body_mass", False):
            base_body_mass = backend.get_body_mass()
        base_geom_friction, ground_geom_id = None, None
        if getattr(cfg.domain_rand, "randomize_ground_friction", False):
            base_geom_friction = backend.get_geom_friction()
            ground_geom_id = getattr(backend, "_ground_geom_id", 0)

        dr_provider = MultiSkillDRProvider(
            base_kp=base_kp, base_kd=base_kd,
            base_body_mass=base_body_mass,
            base_geom_friction=base_geom_friction,
            ground_geom_id=ground_geom_id,
            stand_qpos=stand_qpos,
            stand_ctrl=stand_ctrl,
            flamingo_qpos=flamingo_qpos,
            flamingo_ctrl=flamingo_ctrl,
            kneeling_qpos=kneeling_qpos,
            kneeling_ctrl=kneeling_ctrl,
        )
        # Flamingo flag storage for reset (resized dynamically)
        self._is_flamingo: np.ndarray = np.zeros(0, dtype=bool)
        self._is_fallen: np.ndarray = np.zeros(0, dtype=bool)
        self._init_domain_randomization(dr_provider)

        self._last_push_force: np.ndarray = np.zeros(3, dtype=np.float64)

    # ── Flamingo-specific reward functions ───────────────────────────
    def _reward_lifted_foot_contact(self, ctx):
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        return np.asarray(left_contact, dtype=get_global_dtype())

    def _reward_support_foot_contact(self, ctx):
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        return np.asarray(~right_contact, dtype=get_global_dtype())

    def _reward_com_over_support(self, ctx):
        com = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        rf = np.asarray(self._backend.get_sensor_data("right_foot_pos"), dtype=get_global_dtype())
        d = np.linalg.norm(com[:, :2] - rf[:, :2], axis=1)
        return np.asarray(np.exp(-(d**2) / 0.02), dtype=get_global_dtype())

    # ── HumanUP-inspired GetUp rewards ──────────────────────────────

    def _reward_height_exp(self, ctx: RewardContext) -> np.ndarray:
        """Exponential height reward: exp(h_base) - 1. Every cm counts."""
        h = self._backend.get_base_pos()[:, 2]
        return np.asarray(np.exp(np.clip(h, 0.05, 0.8)) - 1.0, dtype=get_global_dtype())

    def _reward_delta_height(self, ctx: RewardContext) -> np.ndarray:
        """Reward ANY upward movement: +1 when height increases between steps."""
        h = self._backend.get_base_pos()[:, 2]
        prev_h = ctx.info.get("_prev_base_z", h)
        ctx.info["_prev_base_z"] = h
        return np.asarray((h > prev_h).astype(get_global_dtype()))

    def _reward_stand_feet(self, ctx: RewardContext) -> np.ndarray:
        """Reward standing on feet: both feet in contact AND feet near ground."""
        h_feet_l = np.asarray(self._backend.get_sensor_data("left_foot_pos")[:, 2], dtype=get_global_dtype())
        h_feet_r = np.asarray(self._backend.get_sensor_data("right_foot_pos")[:, 2], dtype=get_global_dtype())
        lc = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        rc = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        on_feet = (lc | rc) & (np.abs(h_feet_l) < 0.2) & (np.abs(h_feet_r) < 0.2)
        return np.asarray(on_feet, dtype=get_global_dtype())

    def _reward_soft_symmetry(self, ctx: RewardContext) -> np.ndarray:
        """Soft symmetry: penalize left-right action asymmetry."""
        actions = ctx.info.get("current_actions", np.zeros((ctx.num_envs, self._num_action)))
        # Indices 0-5 = left leg, 6-11 = right leg, 12-14 = waist
        left_leg = actions[:, 0:6]; right_leg = actions[:, 6:12]
        leg_asym = np.sum(np.abs(left_leg - right_leg), axis=1)
        # Waist should be near zero
        waist_dev = np.abs(actions[:, 12]) + np.abs(actions[:, 13])  # roll+yaw
        return np.asarray(leg_asym + waist_dev, dtype=get_global_dtype())

    def _reward_uprightness_exp(self, ctx: RewardContext) -> np.ndarray:
        """Exponential uprightness: exp(-g_xy^2) rewards being upright."""
        g = ctx.gravity
        assert g is not None
        g_xy_sq = np.square(g[:, 0]) + np.square(g[:, 1])
        return np.asarray(np.exp(-g_xy_sq), dtype=get_global_dtype())

    def _reward_orientation_adaptive(self, ctx: RewardContext) -> np.ndarray:
        """Orientation penalty scaled by height — gentle when near ground (getting up).

        HoST-inspired: reduces tilt penalty when robot is low, letting it explore
        getting-up motions without being crushed by orientation error.
        Above 0.55m: full penalty. Below: linear ramp from 10% to 100%.
        """
        g = ctx.gravity
        assert g is not None
        orientation_error = np.square(g[:, 0]) + np.square(g[:, 1])
        base_z = self._backend.get_base_pos()[:, 2]
        # Height gate: 0.35m→0.1, 0.55m→1.0
        height_scale = np.clip((base_z - 0.35) / 0.20, 0.1, 1.0)
        return np.asarray(orientation_error * height_scale, dtype=get_global_dtype())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 98, "critic": 101}

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        obs_dict = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        return obs_dict

    def _geo_features(self, obs_dict: dict) -> None:
        """Append explicit geometric features to observation dict (in-place).

        Adds 4 dims: tilt angle, fall heading sin/cos, COM-support distance.
        Gives Mamba direct geometric understanding beyond raw gravity vector.
        """
        g = self._backend.get_sensor_data(self._cfg.sensor.upvector)  # (B, 3)
        tilt = np.arccos(np.clip(g[:, 2], -1.0, 1.0))  # (B,)
        heading = np.arctan2(g[:, 1], g[:, 0])  # (B,)
        com = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, :2]  # (B, 2)
        rf = np.asarray(self._backend.get_sensor_data("right_foot_pos"), dtype=get_global_dtype())[:, :2]
        com_dist = np.linalg.norm(com - rf, axis=1)  # (B,)
        geo = np.column_stack([
            tilt.astype(get_global_dtype()),
            np.sin(heading).astype(get_global_dtype()),
            np.cos(heading).astype(get_global_dtype()),
            com_dist.astype(get_global_dtype()),
        ])  # (B, 4)
        for key in obs_dict:
            obs_dict[key] = np.concatenate([obs_dict[key], geo], axis=1)

    def step(self, actions: np.ndarray):
        """Override to capture push forces for visualization."""
        if self._state is None:
            self.init_state()
        assert self._state is not None
        state = self._state

        ctrl = self.apply_action(actions, state)
        if self._dr_manager is not None:
            self._dr_manager.apply_interval_randomization_if_due(self.step_counter)

        pid = getattr(self._backend, "_push_body_id", 1)
        xf = self._backend._pending_xfrc_applied[:, 6*pid:6*pid+3].copy()
        if xf.size >= 3:
            self._last_push_force = xf[0]

        self._state = state.replace(truncated=np.zeros_like(state.truncated))
        self._clear_step_final_observation()

        # HoST-inspired assistive upward force: helps robot discover get-up
        aux_scale = getattr(self, "_aux_force_scale", 1.0)
        if aux_scale > 0:
            h = self._backend.get_base_pos()[:, 2]
            gap = np.maximum(0.0, 0.55 - h)
            if np.any(gap > 0):
                pid = getattr(self._backend, "_push_body_id", 1)
                f_up = gap * 400.0 * aux_scale  # N per env
                self._backend._pending_xfrc_applied[:, 6*pid+2] += np.asarray(f_up, dtype=np.float64)

        self._backend.step(ctrl, self._cfg.sim_substeps)
        self._state = self.update_state(self._state)
        self._state.info["steps"] += 1
        self.step_counter += 1
        t = self._compute_truncated(self._state)
        np.logical_or(self._state.truncated, t, out=self._state.truncated)

        # ── Command resampling for mode transitions mid-episode ──
        resample_s = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resample_s > 0:
            resample_interval = int(resample_s / self._cfg.ctrl_dt)
            if resample_interval > 0 and self.step_counter % resample_interval == 0 and self.step_counter > 0:
                # Resample velocity commands for all envs
                new_cmds = self._dr_manager._provider._sample_commands(self, self._num_envs)
                self._state.info["commands"] = new_cmds

        if self._autoreset and np.any(self._state.terminated | self._state.truncated):
            self._reset_done_envs()
        np.nan_to_num(self._state.reward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return self._state


# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

registry.register_env("G1MultiSkill", G1MultiSkillEnv, sim_backend="mujoco")
