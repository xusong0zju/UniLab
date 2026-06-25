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
    compute_forward_command_mask,
    compute_forward_speed_gate,
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


# ═══════════════════════════════════════════════════════════════════════
# Left/right joint pairing for symmetry features (A8.3)
# Indices follow G1 actuator ordering (verified via mujoco mj_id2name):
#   0-5 left leg, 6-11 right leg, 12-14 waist (center), 15-21 left arm, 22-28 right arm
# ═══════════════════════════════════════════════════════════════════════

# 13 left/right joint pairs: (left_idx, right_idx) — same joint name, opposite side.
G1_LEFT_RIGHT_PAIRS: list[tuple[int, int]] = [
    (0, 6),   # hip_pitch
    (1, 7),   # hip_roll
    (2, 8),   # hip_yaw
    (3, 9),   # knee
    (4, 10),  # ankle_pitch
    (5, 11),  # ankle_roll
    (15, 22),  # shoulder_pitch
    (16, 23),  # shoulder_roll
    (17, 24),  # shoulder_yaw
    (18, 25),  # elbow
    (19, 26),  # wrist_roll
    (20, 27),  # wrist_pitch
    (21, 28),  # wrist_yaw
]
# Center (mid-sagittal) joints — inherently symmetric, kept as-is.
G1_CENTER_JOINT_IDX: list[int] = [12, 13, 14]  # waist yaw/roll/pitch


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
    fallen_base_z: float = 0.55     # base height of fallen posture (half-kneel).
    # 0.55 keeps phaseI/H behavior unchanged; phaseJ lowers to 0.40 for a more
    # pronounced get-up motion (still above min_base_height so no instant term).


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
    # HoST-inspired assistive upward force config. Defaults keep phaseI/H
    # behavior (aux on at threshold 0.55, no decay). phaseJ tunes these for
    # get-up curriculum (scale 1.0→0 to withdraw assistance, threshold 0.45).
    aux_force_scale: float = 1.0
    aux_force_threshold: float = 0.55
    # Aux decay window in env-steps: aux_scale linearly decays 1.0→0 over
    # [aux_decay_start_step, aux_decay_end_step] so one training run teaches
    # stand-with-assist → stand-unassisted. Defaults 0 disable decay (phaseI/H).
    aux_decay_start_step: int = 0
    aux_decay_end_step: int = 0
    # Phase K: pushover get-up. When True, fallen envs are pushed over (stand→
    # random-direction force→ settle→ PD to randomized lie joints) during reset
    # so they start episodes from a real grounded pose (not the settle-straight
    # pose). Defaults False keeps phaseI/H/J behavior.
    fallen_pushover: bool = False
    # Max tilt (deg) for fallen envs before termination. phaseK sets 170 to
    # allow true grounded poses (tilt>60) to not instantly terminate. Other envs
    # still use max_tilt_deg. Default 80 keeps old behavior.
    max_tilt_fallen_deg: float = 80.0
    pushover_force: float = 50.0       # push magnitude (N) during pushover
    pushover_push_steps: int = 15      # steps applying push force
    pushover_settle_steps: int = 150   # free-fall settle steps
    pushover_pd_steps: int = 150       # PD-control to lie-joint steps
    # Phase L: assistive force gain (N per meter of gap). Default 400 keeps
    # phaseI/H/J behavior. phaseL lowers to 150 so assist nudges near ground
    # instead of lifting the robot airborne (phaseK lifted to base_z 0.88).
    aux_force_gain: float = 400.0


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
        """Sample velocity commands only (no skill flags).

        Skill identity (flamingo/fallen) is set exclusively by
        ``_sample_skill_flags`` at reset and held for the whole episode; this
        method is reused by mid-episode command resampling to switch vx/vy
        (walk<->run transitions) without touching identity.
        """
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)

        # Standing envs: zero the command so they hold still.
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0

        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands

    def _sample_skill_flags(self, env, env_ids):
        """Mutually-exclusive skill sampling + scatter-write to full-length flags.

        Single uniform draw per env partitions into fallen / flamingo / walk:
        ``u < fallen_prob`` -> fallen, ``fallen_prob <= u < fallen_prob +
        flamingo_prob`` -> flamingo, otherwise walk. Writes
        ``env._is_flamingo[env_ids]`` / ``env._is_fallen[env_ids]`` so the full
        arrays stay aligned with all envs across the episode. Returns the
        local (num_reset,) velocity commands, zeroed for non-walk identities.
        """
        num_reset = len(env_ids)
        commands = self._sample_commands(env, num_reset)

        fallen_prob = float(getattr(env.cfg.commands, "rel_fallen_envs", 0.0))
        flamingo_prob = float(getattr(env.cfg.commands, "rel_flamingo_envs", 0.0))
        u = np.random.uniform(size=(num_reset,))
        is_fallen = u < fallen_prob if fallen_prob > 0 else np.zeros(num_reset, dtype=bool)
        is_flamingo = (
            (~is_fallen)
            & (u < fallen_prob + flamingo_prob)
            if (flamingo_prob > 0)
            else np.zeros(num_reset, dtype=bool)
        )
        # Non-walk identities hold still: zero their velocity command.
        non_walk = is_fallen | is_flamingo
        if np.any(non_walk):
            commands[non_walk] = 0.0

        # Scatter-write into the full-length arrays (overwrite only reset envs).
        env._is_flamingo[env_ids] = is_flamingo
        env._is_fallen[env_ids] = is_fallen
        return commands

    def build_reset_plan(self, env, env_ids):
        """Build reset plan with per-env keyframe selection (stand vs flamingo)."""
        num_reset = len(env_ids)
        # Sample skill identity FIRST: writes full-length env._is_flamingo /
        # env._is_fallen (stable for the episode) and returns local commands.
        commands = self._sample_skill_flags(env, env_ids)
        # Flags are full-length and aligned; index the reset subset directly.
        is_flamingo = env._is_flamingo[env_ids]
        is_fallen = env._is_fallen[env_ids]

        # Start from stand keyframe by default
        qpos_all = np.tile(self._stand_qpos, (num_reset, 1))
        # Override flamingo envs with flamingo keyframe
        if np.any(is_flamingo):
            qpos_all[is_flamingo] = self._flamingo_qpos

        # ---- Fallen postures (Phase D/G) with kneeling curriculum ----
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
            # Base lowered to a half-kneel height: low enough to require
            # recovery, but above min_base_height so it doesn't terminate
            # instantly on reset. Read from cfg so phaseJ can lower it (0.40)
            # without changing phaseI/H behavior (0.55).
            fallen_base_z = float(getattr(env.cfg.commands, "fallen_base_z", 0.55))
            fallen_qpos = np.tile(self._stand_qpos, (n_fallen, 1))
            fallen_qpos[:, 2] = fallen_base_z  # half-kneel base z
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
            "commands": commands,
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
        # Phase L: mid-height reward encourages the half-kneel intermediate state
        # (base_z in [0.30,0.55]) so the policy learns to climb from ground to
        # kneeling before standing — not just lift to standing in one shot.
        self._reward_fns["mid_height"] = self._reward_mid_height
        # Phase N: roll-to-supine reward (encourage rolling to face-up before
        # getting up). State-gated to ground (base_z<0.40) so it only fires
        # during the roll-over phase, not after standing. Masked to fallen envs.
        self._reward_fns["roll_to_supine"] = self._reward_roll_to_supine
        # Running flight phase reward (high-speed airborne, walk/stand envs only)
        self._reward_fns["feet_flight"] = self._reward_feet_flight
        # v3: route alive by command — walk envs (cmd vx != 0) get NO alive bonus,
        # forcing them to earn via tracking instead of standing still. Standing/
        # flamingo/fallen envs (cmd = 0) keep alive to encourage holding still.
        self._reward_fns["alive"] = self._reward_alive

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
        # Skill-identity flags, full num_envs length, held stable across an
        # episode (set at reset, never overwritten by mid-episode command
        # resampling). Kept as bool arrays so reward routing can mask by skill.
        self._is_flamingo: np.ndarray = np.zeros(self._num_envs, dtype=bool)
        self._is_fallen: np.ndarray = np.zeros(self._num_envs, dtype=bool)
        self._init_domain_randomization(dr_provider)

        self._last_push_force: np.ndarray = np.zeros(3, dtype=np.float64)
        # Phase K: cache of pre-generated grounded-lie qpos (generated once at
        # init via pushover, then reset just samples from cache + perturbation —
        # avoids running 315 physics steps per reset which crippled the collector.
        self._fallen_qpos_cache: np.ndarray | None = None
        if getattr(cfg, "fallen_pushover", False):
            self._fallen_qpos_cache = self._pregenerate_fallen_cache(n_cache=24)

    # ── Skill masks for reward routing ──────────────────────────────
    # Identity flags are held per-env for the whole episode (see
    # _sample_skill_flags). Reward functions multiply by these masks so that
    # skill-specific terms (flamingo single-leg, fallen get-up, walk gait)
    # only act on the envs that own that skill — preventing the cross-skill
    # leakage that previously pulled walking/standing/get-up envs into a
    # single-leg posture.
    def _flamingo_mask(self) -> np.ndarray:
        return np.asarray(self._is_flamingo, dtype=get_global_dtype())

    def _fallen_mask(self) -> np.ndarray:
        return np.asarray(self._is_fallen, dtype=get_global_dtype())

    def _walk_mask(self) -> np.ndarray:
        # Walk/stand envs: neither flamingo nor fallen.
        return np.asarray(~self._is_flamingo & ~self._is_fallen, dtype=get_global_dtype())

    # ── Flamingo-specific reward functions (masked to flamingo envs) ──
    def _reward_lifted_foot_contact(self, ctx):
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        return np.asarray(left_contact, dtype=get_global_dtype()) * self._flamingo_mask()

    def _reward_support_foot_contact(self, ctx):
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        return np.asarray(~right_contact, dtype=get_global_dtype()) * self._flamingo_mask()

    def _reward_com_over_support(self, ctx):
        com = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        rf = np.asarray(self._backend.get_sensor_data("right_foot_pos"), dtype=get_global_dtype())
        d = np.linalg.norm(com[:, :2] - rf[:, :2], axis=1)
        return np.asarray(np.exp(-(d**2) / 0.02), dtype=get_global_dtype()) * self._flamingo_mask()

    # ── HumanUP-inspired GetUp rewards ──────────────────────────────

    def _reward_height_exp(self, ctx: RewardContext) -> np.ndarray:
        """Exponential height reward: exp(h_base) - 1. Every cm counts.

        Masked to fallen envs so it only motivates getting up, not inflating
        reward for already-standing envs (which would let tracking be ignored).
        """
        h = self._backend.get_base_pos()[:, 2]
        return np.asarray(np.exp(np.clip(h, 0.05, 0.8)) - 1.0, dtype=get_global_dtype()) * self._fallen_mask()

    def _reward_delta_height(self, ctx: RewardContext) -> np.ndarray:
        """Reward ANY upward movement: +1 when height increases between steps.

        Masked to fallen envs. NOTE: prev_h is updated for ALL envs (the side
        effect must run every step so the next comparison is correct), only
        the returned reward is masked.
        """
        h = self._backend.get_base_pos()[:, 2]
        prev_h = ctx.info.get("_prev_base_z", h)
        ctx.info["_prev_base_z"] = h
        return np.asarray((h > prev_h).astype(get_global_dtype())) * self._fallen_mask()

    def _reward_stand_feet(self, ctx: RewardContext) -> np.ndarray:
        """Reward standing on feet: both feet in contact AND feet near ground.

        Masked to fallen envs — this is the terminal "stood up" bonus for the
        get-up skill, not a general standing reward.
        """
        h_feet_l = np.asarray(self._backend.get_sensor_data("left_foot_pos")[:, 2], dtype=get_global_dtype())
        h_feet_r = np.asarray(self._backend.get_sensor_data("right_foot_pos")[:, 2], dtype=get_global_dtype())
        lc = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        rc = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        on_feet = (lc | rc) & (np.abs(h_feet_l) < 0.2) & (np.abs(h_feet_r) < 0.2)
        return np.asarray(on_feet, dtype=get_global_dtype()) * self._fallen_mask()

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
        """Exponential uprightness: exp(-g_xy^2) rewards being upright.

        Masked to fallen envs — a get-up bonus. Standing/walking envs are kept
        upright by the general penalty_orientation term instead.
        """
        g = ctx.gravity
        assert g is not None
        g_xy_sq = np.square(g[:, 0]) + np.square(g[:, 1])
        return np.asarray(np.exp(-g_xy_sq), dtype=get_global_dtype()) * self._fallen_mask()

    def _reward_mid_height(self, ctx: RewardContext) -> np.ndarray:
        """Phase L: reward the half-kneel intermediate height (base_z ~0.42m).

        Gaussian peak at h=0.42 (half-kneel between ground 0.16 and stand 0.754).
        Encourages climbing from ground to kneeling before standing. Masked to
        fallen envs. Helps the policy discover the get-up motion sequence by
        rewarding intermediate progress (not only the final standing state).
        """
        h = self._backend.get_base_pos()[:, 2]
        r = np.exp(-np.square((h - 0.42) / 0.12))
        return np.asarray(r, dtype=get_global_dtype()) * self._fallen_mask()

    def _reward_roll_to_supine(self, ctx: RewardContext) -> np.ndarray:
        """Phase N: encourage rolling to supine (face-up, back-down) before get-up.

        Uses the FULL gravity vector angle against the canonical supine target.
        G1 body frame (verified 2026-06-24): +x=front(chest), +y=left, +z=up(head).
        Standing g_body=[0,0,-1] (gravity to feet/-z). The BACK faces body -x.
        Supine (back on ground) => back(-x) aligns with world gravity(-z), so
        world gravity [0,0,-1] projects to body [-1,0,0]. => g_target=[-1,0,0].

        Prone (face-down, chest to ground) => g_body=[+1,0,0] (opposite).
        Side-lying (left/right arm down) => g_body≈[0,±1,0].
        Real fallen poses scatter across all these; roll_to_supine guides to [-1,0,0].

        State-gated to ground (base_z<0.40). Masked to fallen envs. sigma=0.40.
        """
        g = ctx.gravity
        assert g is not None
        g_target = np.array([-1.0, 0.0, 0.0], dtype=get_global_dtype())  # back-down
        cos_angle = np.clip(g @ g_target, -1.0, 1.0)  # (N,)
        supine_score = np.exp(-np.square((1.0 - cos_angle) / 0.40))
        h = self._backend.get_base_pos()[:, 2]
        ground_gate = (h < 0.40).astype(get_global_dtype())
        return np.asarray(supine_score * ground_gate, dtype=get_global_dtype()) * self._fallen_mask()

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

    # ── Walking rewards (masked to walk/stand envs only) ─────────────
    # The inherited G1WalkEnv feet_phase rewards gate on linvel, but a pushed
    # flamingo/fallen env can momentarily have forward speed and spuriously
    # earn a gait reward that conflicts with single-leg/get-up behavior. Gate
    # them on the walk identity as well so they only fire for walk/stand envs.
    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        gate = compute_forward_speed_gate(linvel, min_forward_speed)
        return np.asarray(gate, dtype=get_global_dtype()) * self._walk_mask()

    def _reward_feet_double_stance(self, ctx: RewardContext) -> np.ndarray:
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        # Encourage double-stance only at walking speeds (vx < flight_speed_threshold);
        # at running speeds the feet_flight reward owns the airborne phase, so
        # the two never overlap (walk: vx in (0, threshold) rewards double support,
        # run: vx >= threshold rewards flight).
        flight_threshold = float(getattr(self._reward_cfg, "flight_speed_threshold", 1.0))
        walk_speed = np.asarray(
            (commands[:, 0] > 1.0e-6) & (commands[:, 0] < flight_threshold),
            dtype=get_global_dtype(),
        )
        return np.asarray(double_stance * walk_speed, dtype=get_global_dtype()) * self._walk_mask()

    def _reward_feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1) * self._walk_mask()

    def _reward_feet_flight(self, ctx: RewardContext) -> np.ndarray:
        """Reward the airborne (flight) phase of running: both feet off ground.

        Fires only for walk/stand envs commanded above flight_speed_threshold,
        so the policy learns a real running gait (brief double-lift) instead
        of always keeping a foot down. Distinct from double_stance which owns
        the low-speed double-support phase.
        """
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        flight_threshold = float(getattr(self._reward_cfg, "flight_speed_threshold", 1.0))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        airborne = np.asarray(~left_contact & ~right_contact, dtype=get_global_dtype())
        fast = np.asarray(commands[:, 0] >= flight_threshold, dtype=get_global_dtype())
        return np.asarray(airborne * fast, dtype=get_global_dtype()) * self._walk_mask()

    def _reward_alive(self, ctx: RewardContext) -> np.ndarray:
        """Route alive bonus by commanded speed (v3).

        Walk/run envs (commanded speed != 0) get NO alive bonus — they must
        earn via tracking_lin_vel instead of standing still to farm alive.
        Standing / flamingo / fallen envs (command = 0) keep alive=1 to
        encourage holding still. This breaks the "stand-and-farm" local
        optimum that small networks fall into when alive is unconditional.
        """
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        # commanded horizontal speed magnitude (vx, vy)
        cmd_speed = np.sqrt(commands[:, 0] ** 2 + commands[:, 1] ** 2)
        threshold = float(getattr(self._reward_cfg, "alive_cmd_threshold", 0.1))
        # alive only where commanded speed is ~0 (stand / flamingo / fallen)
        return np.asarray(cmd_speed < threshold, dtype=get_global_dtype())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # base 98 (gyro3+gravity3+dof_pos29+dof_vel29+action29+cmd3+phase2)
        # + geo 3 (tilt, heading_sin, heading_cos — pure gravity-derived, batch-safe)
        # + sym 29 (13 pairs (l+r)/2 + 13 pairs (l-r)/2 + 3 center joints)
        # = 130 ; critic = actor + linvel 3 = 133
        return {"obs": 130, "critic": 133}

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        obs_dict = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        # A8.2 geometric invariants (SO3 partial invariants, gravity-only → batch-safe
        # in both reset (num_reset) and step (num_envs) paths).
        geo = self._geo_features(gravity)  # (B, 3)
        # A8.3 left/right symmetry invariants from dof_pos.
        sym = self._sym_features(dof_pos)  # (B, 29)
        extra = np.concatenate([geo, sym], axis=1, dtype=get_global_dtype())  # (B, 32)
        for key in obs_dict:
            obs_dict[key] = np.concatenate([obs_dict[key], extra], axis=1)
        return obs_dict

    def _geo_features(self, gravity: np.ndarray) -> np.ndarray:
        """Explicit geometric invariants (SO3 partial invariants). (B, 3)

        tilt, heading sin/cos. Pure functions of the gravity vector so they
        are batch-safe in both reset (num_reset) and step (num_envs) paths —
        no backend query (which returns full num_envs and would mismatch).
        """
        g = np.asarray(gravity, dtype=get_global_dtype())
        tilt = np.arccos(np.clip(g[:, 2], -1.0, 1.0))  # (B,)
        heading = np.arctan2(g[:, 1], g[:, 0])  # (B,)
        return np.column_stack([
            tilt.astype(get_global_dtype()),
            np.sin(heading).astype(get_global_dtype()),
            np.cos(heading).astype(get_global_dtype()),
        ])  # (B, 3)

    def _sym_features(self, dof_pos: np.ndarray) -> np.ndarray:
        """Left/right symmetry invariants from joint positions. (B, 29)

        For 13 left/right joint pairs: symmetric component (l+r)/2 (13 dims,
        invariant under left-right mirror) + anti-symmetric component (l-r)/2
        (13 dims, flips sign under mirror). Plus 3 center (mid-sagittal)
        joints kept as-is (inherently symmetric).
        Gives Mamba an explicit symmetry decomposition so it can implicitly
        learn left-right equivalence without an equivariant architecture.
        """
        dp = np.asarray(dof_pos, dtype=get_global_dtype())
        left_idx = np.array([p[0] for p in G1_LEFT_RIGHT_PAIRS])
        right_idx = np.array([p[1] for p in G1_LEFT_RIGHT_PAIRS])
        sym = (dp[:, left_idx] + dp[:, right_idx]) * 0.5  # (B, 13)
        asym = (dp[:, left_idx] - dp[:, right_idx]) * 0.5  # (B, 13)
        center = dp[:, np.array(G1_CENTER_JOINT_IDX, dtype=np.int64)]  # (B, 3)
        return np.concatenate([sym, asym, center], axis=1, dtype=get_global_dtype())  # (B, 29)

    def _actor_symmetry_obs_layout(self):
        # Base layout (98) + A8 extras appended in _compute_obs:
        #   geo 3 (SO3 invariants, identity under mirror)
        #   sym 29 (sym/center identity, asym flips — approximated as identity
        #          since FlashSAC double_buffer_runner does not invoke augment;
        #          layout only needs to pass dim validation in mirror_obs).
        from unilab.base.augmentation import SymmetryObsLayout
        base: SymmetryObsLayout = super()._actor_symmetry_obs_layout()
        return (*base, ("geo", 3), ("sym", 29))

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Override base apply_action to advance gait_phase each step.

        The inherited LocomotionBaseEnv.apply_action only stores last/current
        actions and builds ctrl — it never advances gait_phase, so the phase
        sampled at reset stays static for the whole episode. With a static
        phase target, the feet_phase reward encourages a fixed single-foot
        lift instead of an alternating gait. Advance both legs phases here so
        walking envs track a moving phase target (alternating stance/swing).
        """
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase
        ctrl: np.ndarray = exec_actions * self._cfg.control_config.action_scale + self.default_angles
        return ctrl

    # ── Phase K: pushover get-up reset ───────────────────────────────
    def reset(self, env_indices: np.ndarray):
        """Standard reset, then (if fallen_pushover) push fallen envs over and
        PD-control them into a randomized grounded lie pose. Recomputes obs so
        the returned obs matches the real (grounded) qpos, not the stand pose.
        """
        obs, info = super().reset(env_indices)
        if getattr(self._cfg, "fallen_pushover", False):
            self._fallen_pushover_warmup(env_indices)
            obs = self._recompute_reset_obs(env_indices, info)
        return obs, info

    def _pregenerate_fallen_cache(self, n_cache: int = 24) -> np.ndarray:
        """Run pushover+PD once at init to generate a cache of n_cache grounded-lie
        qpos (n_cache, nq). Reset then samples from cache + small perturbation
        (O(1), no physics) instead of running 315 steps per reset (which crippled
        the collector). Uses a temp single-env run to avoid disturbing the pool."""
        cfg = self._cfg
        provider = self._dr_manager._provider
        stand_qpos = provider._stand_qpos
        stand_ctrl = stand_qpos[-self._num_action:].copy()
        nq = self._backend._model.nq
        nv = self._backend._model.nv
        N = self._num_envs
        # Push the SHOULDER (not pelvis) for a large lever arm above the CoM so
        # the push force produces a toppling TORQUE, not just translation.
        # Pushing pelvis (CoM height) only slides the robot without toppling
        # (verified: 500N on pelvis => 0% fall; 100N on shoulder => 100% fall).
        import mujoco as _mj
        _push_body_name = getattr(cfg, "pushover_body_name", "left_shoulder_pitch_link")
        pid = int(_mj.mj_name2id(self._backend._model, _mj.mjtObj.mjOBJ_BODY, _push_body_name))
        if pid <= 0:
            pid = int(getattr(self._backend, "_push_body_id", 1))  # fallback pelvis
        pf = float(getattr(cfg, "pushover_force", 50.0))
        push_steps = int(getattr(cfg, "pushover_push_steps", 15))
        settle_steps = int(getattr(cfg, "pushover_settle_steps", 150))
        pd_steps = int(getattr(cfg, "pushover_pd_steps", 150))
        cache = []
        # Generate n_cache poses by running pushover on the full pool (all envs
        # participate; we collect each env's final qpos as a cache entry, repeat
        # until we have n_cache distinct ones).
        batch = 0
        while len(cache) < n_cache:
            batch += 1
            # reset all envs to stand (upright, random yaw)
            qpos_stand = np.tile(stand_qpos, (N, 1))
            qpos_stand[:, 0:2] += np.random.uniform(-0.3, 0.3, (N, 2))
            yaw = np.random.uniform(-np.pi, np.pi, (N,))
            from unilab.envs.common.rotation import np_yaw_to_quat as _yaw_q
            qpos_stand[:, 3:7] = np_quat_mul(qpos_stand[:, 3:7], _yaw_q(yaw))
            self._backend.set_state(np.arange(N, dtype=np.int32), qpos_stand, np.zeros((N, nv)))
            push_ang = np.random.uniform(0, 2 * np.pi, (N,))
            ctrl_stand = np.tile(stand_ctrl, (N, 1)).astype(get_global_dtype())
            ctrl_zero = np.zeros((N, self._num_action), dtype=get_global_dtype())
            # Randomize push body per batch across high-leverage upper-body
            # sites (shoulder pitch/roll/yaw, left & right) so falls cover
            # diverse directions — not always the same shoulder. All are above
            # the CoM (z>0.9) so the push topples rather than slides.
            import mujoco as _mj2
            _push_bodies = [
                "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                "left_shoulder_roll_link", "right_shoulder_roll_link",
                "left_shoulder_yaw_link", "right_shoulder_yaw_link",
            ]
            _push_ids = [
                _mj2.mj_name2id(self._backend._model, _mj2.mjtObj.mjOBJ_BODY, n)
                for n in _push_bodies
            ]
            _push_ids = [b for b in _push_ids if b > 0]
            batch_pid = int(np.random.choice(_push_ids)) if _push_ids else pid
            # Push phase: ctrl=stand (lock upright, stiff) so the push force
            # topples the robot as a rigid body. (ctrl=0 makes legs buckle and
            # absorbs the push without toppling.)
            for _ in range(push_steps):
                force = np.zeros((N, 1, 3))
                force[:, 0, 0] = pf * np.cos(push_ang)
                force[:, 0, 1] = pf * np.sin(push_ang)
                self._backend.apply_body_force(np.array([batch_pid], dtype=np.int32), force)
                self._backend.step(ctrl_stand, 1)
            for _ in range(settle_steps):
                self._backend.step(ctrl_zero, 1)
            # PD phase: drive joints to an EXTENDED lie target (elbows straight,
            # arms alongside body) so the cached pose is a natural extended lie,
            # not retaining fall-time elbow flex. Previously used cur_dof which
            # kept ~90° elbow bend from the fall.
            lie_target = self._sample_lie_joints(N)
            for _ in range(pd_steps):
                self._backend.step(lie_target.astype(get_global_dtype()), 1)
            # collect final qpos, filter to real grounded poses
            cur = self._backend._physics_state[:, self._backend._idx_qpos : self._backend._idx_qpos + nq].copy()
            bz = cur[:, 2]
            # tilt from gravity
            grav = self._backend.get_sensor_data(self._cfg.sensor.upvector)
            tilt = np.degrees(np.arccos(np.clip(np.abs(grav[:, 2]), 0, 1)))
            grounded = (bz < 0.45) & (tilt > 50)
            for i in np.where(grounded)[0]:
                cache.append(cur[i].copy())
                if len(cache) >= n_cache:
                    break
            if batch > 5:
                break  # safety: don't loop forever if pushover fails
        if len(cache) == 0:
            # fallback: use stand (shouldn't happen, but don't crash init)
            cache = [stand_qpos.copy()]
        return np.stack(cache[:n_cache] if len(cache) >= n_cache else cache)

    def _fallen_pushover_warmup(self, env_ids: np.ndarray) -> None:
        """Sample grounded-lie qpos from the pre-generated cache + small
        perturbation, set_state directly (no physics warmup at reset time).
        O(1) per reset — does not slow the collector."""
        if self._fallen_qpos_cache is None or len(self._fallen_qpos_cache) == 0:
            return
        nq = self._backend._model.nq
        nv = self._backend._model.nv
        is_fallen = np.asarray(self._is_fallen)
        N = self._num_envs
        cache = self._fallen_qpos_cache
        n_cache = len(cache)
        # sample a cache entry per env (cycle if more envs than cache)
        idx = np.random.randint(0, n_cache, size=(N,))
        qpos = np.tile(cache[0], (N, 1))
        qpos[:] = cache[idx]
        # small perturbation: xy position + joint noise
        qpos[:, 0:2] += np.random.uniform(-0.15, 0.15, (N, 2))
        joint_noise = np.random.uniform(-0.1, 0.1, (N, self._num_action))
        qpos[:, 7 : 7 + self._num_action] += joint_noise
        # zero velocities
        self._backend.set_state(np.arange(N, dtype=np.int32), qpos, np.zeros((N, nv)))

    def _sample_lie_joints(self, n: int) -> np.ndarray:
        """Randomized grounded-lie joint targets (n, num_action).

        Body EXTENDED on the ground (real supine/prone lie, not curled):
        - elbows STRAIGHT (0 rad, not the 34° default) — arms extended along body
        - knees near straight (small flex, legs extended)
        - shoulders arms alongside body (not splayed)
        - small perturbations simulate natural ground settling.
        Previously PD used cur_dof (retained fall-time elbow flex ~90°)."""
        da = self.default_angles  # (num_action,) stand/default joints
        j = np.tile(da, (n, 1)).astype(np.float64)
        # G1 joint layout (0-28): 0-5 left leg, 6-11 right leg, 12-14 waist,
        # 15-20 left arm, 21-26 right arm. (hip_pitch=0/6, knee=3/9, elbow=18/24)
        for lr in (0, 6):  # left/right leg base index
            j[:, lr] += np.random.uniform(-0.2, 0.2, n)      # hip_pitch
            j[:, lr + 1] += np.random.uniform(-0.1, 0.1, n)  # hip_roll
            j[:, lr + 3] = np.random.uniform(-0.1, 0.2, n)   # knee (near straight, slight flex)
            j[:, lr + 4] += np.random.uniform(-0.1, 0.1, n)  # ankle_pitch
            j[:, lr + 5] += np.random.uniform(-0.1, 0.1, n)  # ankle_roll
        j[:, 12] += np.random.uniform(-0.1, 0.1, n)  # waist yaw
        j[:, 13] += np.random.uniform(-0.1, 0.1, n)  # waist roll
        j[:, 14] += np.random.uniform(-0.1, 0.1, n)  # waist pitch
        for lr in (15, 21):  # left/right arm
            j[:, lr] = np.random.uniform(-0.1, 0.1, n)      # shoulder_pitch (arms alongside body, not splayed)
            j[:, lr + 1] += np.random.uniform(-0.1, 0.1, n)  # shoulder_roll
            j[:, lr + 2] += np.random.uniform(-0.1, 0.1, n)  # shoulder_yaw
            j[:, lr + 3] = np.random.uniform(-0.1, 0.1, n)   # elbow STRAIGHT (0 rad, not 34° default)
        return j

    def _recompute_reset_obs(self, env_ids: np.ndarray, info: dict) -> dict:
        """Recompute reset obs from current (post-warmup) physics state.
        build_reset_observation fetches linvel/gyro/gravity/dof_pos/dof_vel
        internally from the backend, so obs matches the real grounded qpos."""
        provider = self._dr_manager._provider
        return provider.build_reset_observation(self, env_ids, info)

    def step(self, actions: np.ndarray):
        """Override to capture push forces for visualization."""
        if self._state is None:
            self.init_state()
        assert self._state is not None
        state = self._state

        ctrl = self.apply_action(actions, state)
        if self._dr_manager is not None:
            self._dr_manager.apply_interval_randomization_if_due(self.step_counter)

        # Record last push force for visualization only. There is no declared
        # SimBackend method to read staged xfrc_applied, so this is a known
        # CLAUDE.md red-line kept isolated to the visualization path (does not
        # affect training). Tolerant: skips if the private field is absent.
        try:
            pid_v = int(getattr(self._backend, "_push_body_id", 1))
            xfrc = getattr(self._backend, "_pending_xfrc_applied", None)
            if xfrc is not None and xfrc.shape[-1] >= 6 * pid_v + 3:
                self._last_push_force = xfrc[:, 6 * pid_v : 6 * pid_v + 3][0].copy()
        except (TypeError, IndexError, AttributeError):
            pass

        self._state = state.replace(truncated=np.zeros_like(state.truncated))
        self._clear_step_final_observation()

        # HoST-inspired assistive upward force: helps robot discover get-up.
        # Goes through the SimBackend.apply_body_force interface (world-frame,
        # accumulates into the next step's xfrc_applied) instead of mutating the
        # backend-private _pending_xfrc_applied array directly — CLAUDE.md
        # requires env-layer to call only declared SimBackend methods.
        aux_scale = float(getattr(self._cfg, "aux_force_scale", 1.0))
        # Stage II aux decay: linearly decay aux_scale over a step window so the
        # policy learns to stand without assistance within one training run
        # (instead of training 8000 iters fully-assisted then re-training to
        # withdraw). Defaults disable decay (start>=end) keeping phaseI/H behavior.
        decay_start = int(getattr(self._cfg, "aux_decay_start_step", 0))
        decay_end = int(getattr(self._cfg, "aux_decay_end_step", 0))
        if decay_end > decay_start > 0:
            sc = self.step_counter
            if sc >= decay_end:
                aux_scale = 0.0
            elif sc > decay_start:
                aux_scale = aux_scale * (1.0 - (sc - decay_start) / (decay_end - decay_start))
        if aux_scale > 0:
            h = self._backend.get_base_pos()[:, 2]
            # Threshold from cfg (default 0.55 keeps phaseI/H behavior; phaseJ
            # sets 0.45 to match its lower fallen_base_z=0.40).
            aux_threshold = float(getattr(self._cfg, "aux_force_threshold", 0.55))
            gap = np.maximum(0.0, aux_threshold - h)
            if np.any(gap > 0):
                pid = int(getattr(self._backend, "_push_body_id", 1))
                # Phase L: gain configurable (default 400 keeps phaseI/H/J behavior;
                # phaseL sets 150 so assist only nudges near ground, doesn't lift
                # the robot into the air — phaseK lifted to base_z 0.88, too high).
                aux_gain = float(getattr(self._cfg, "aux_force_gain", 400.0))
                f_up = gap * aux_gain * aux_scale  # N per env
                force = np.zeros((self._num_envs, 1, 3), dtype=np.float64)
                force[:, 0, 2] = f_up  # world-frame z+
                self._backend.apply_body_force(np.array([pid], dtype=np.int32), force)

        self._backend.step(ctrl, self._cfg.sim_substeps)
        self._state = self.update_state(self._state)
        # Phase K: relax termination for fallen envs so true grounded poses
        # (tilt>60, low base_z) don't instantly terminate. Non-fallen envs keep
        # the base max_tilt_deg/min_base_height. Default max_tilt_fallen_deg=80
        # keeps phaseI/H/J behavior unchanged.
        max_tilt_fallen = float(getattr(self._cfg, "max_tilt_fallen_deg", 80.0))
        if max_tilt_fallen > self._reward_cfg.max_tilt_deg and np.any(self._is_fallen):
            grav = self._backend.get_sensor_data(self._cfg.sensor.upvector)
            tilt = np.arccos(np.clip(grav[:, 2], -1, 1))
            fallen_strict = (tilt > np.deg2rad(self._reward_cfg.max_tilt_deg)) | (
                self._terrain_relative_base_height() < self._reward_cfg.min_base_height
            )
            fallen_relaxed = (tilt > np.deg2rad(max_tilt_fallen)) | (
                self._terrain_relative_base_height() < self._reward_cfg.min_base_height
            )
            # fallen envs use relaxed threshold; non-fallen keep strict
            new_term = np.where(self._is_fallen, fallen_relaxed, fallen_strict)
            self._state = self._state.replace(terminated=new_term.astype(self._state.terminated.dtype))
        self._state.info["steps"] += 1
        self.step_counter += 1
        t = self._compute_truncated(self._state)
        np.logical_or(self._state.truncated, t, out=self._state.truncated)

        # ── Command resampling for mode transitions mid-episode ──
        resample_s = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resample_s > 0:
            resample_interval = int(resample_s / self._cfg.ctrl_dt)
            if resample_interval > 0 and self.step_counter % resample_interval == 0 and self.step_counter > 0:
                # Resample velocity commands for all envs (walk<->run/turn
                # transitions). Skill identity (flamingo/fallen) is held for
                # the whole episode, so re-zero their commands after resampling.
                new_cmds = self._dr_manager._provider._sample_commands(self, self._num_envs)
                non_walk = self._is_flamingo | self._is_fallen
                if np.any(non_walk):
                    new_cmds[non_walk] = 0.0
                self._state.info["commands"] = new_cmds

        if self._autoreset and np.any(self._state.terminated | self._state.truncated):
            self._reset_done_envs()
        np.nan_to_num(self._state.reward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return self._state


# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

registry.register_env("G1MultiSkill", G1MultiSkillEnv, sim_backend="mujoco")
