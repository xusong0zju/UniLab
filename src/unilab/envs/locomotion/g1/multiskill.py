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
    # Phase P: HumanUP getup reference trajectory (29-DoF aligned npz). Empty = no tracking.
    getup_traj_file: str = ""
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
    # Phase P (方向B): min base height for FALLEN envs before termination.
    # fallen envs start supine and must be allowed to flounder on the ground
    # (base_z dips below standing min_base_height during get-up attempts).
    # Default = -1.0 keeps behavior identical to min_base_height (no separate
    # fallen floor) — only set lower in phaseP so fallen envs don't terminate
    # the instant base_z < min_base_height. phaseP sets 0.0 (effectively off,
    # tilt>170 still terminates). Non-fallen envs always use min_base_height.
    min_base_height_fallen: float = -1.0
    # Phase P 翘臀修复: enable MuJoCo body tracking sensors so env can read
    # arbitrary body world positions via get_body_pos_w (needed for
    # _reward_head_height on torso_link). Default False keeps phaseI/H/N
    # behavior (no injected sensors). phaseP sets True.
    add_body_sensors: bool = False
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

    def _trajectory_frame_qpos(self, env, frame: int, n: int) -> np.ndarray:
        """Phase P: build qpos (n, nq) from HumanUP trajectory at given frame.

        base_z from head_height (mapped), base quat = tilt around body y (倒地90°→站立0°),
        joints = trajectory dof_pos. Used for phase-aligned getup start."""
        traj_dof = env._getup_traj_dof[frame]  # (29,)
        traj_h = float(env._getup_traj_height[frame])
        stand = self._stand_qpos
        qpos = np.tile(stand, (n, 1))
        # base_z from head_height
        base_z = 0.12 + (0.754 - 0.12) * (traj_h - 0.054) / (1.278 - 0.054)
        qpos[:, 2] = base_z
        # base quat: tilt around body y (倒地90°→站立0°)
        tilt_deg = 90.0 * (1.278 - traj_h) / (1.278 - 0.054)
        tilt_deg = max(0.0, min(90.0, tilt_deg))
        h_half = np.deg2rad(tilt_deg) / 2
        # base quat: tilt around body y axis. NEGATIVE sin (绕y轴-90°) for
        # supine (back-down, g_body=[-1,0,0]). Positive gave prone (g_body=[+1,0,0]).
        q = np.array([np.cos(h_half), 0, -np.sin(h_half), 0])
        qpos[:, 3:7] = q
        # joints (29)
        na = env._num_action
        qpos[:, 7:7 + na] = traj_dof[:na]
        return qpos

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

        # Phase P: if getup_traj_file loaded, fallen envs start from trajectory
        # frame-0 pose (aligned with dof/height tracking from step 0). Overrides
        # pushover/kneeling. Done AFTER legacy fallen logic, BEFORE yaw randomization
        # (yaw would break trajectory orientation alignment).
        # IMPORTANT: use the ORIGINAL env._is_fallen[env_ids] flag (all fallen envs),
        # NOT the local `is_fallen` — the kneeling curriculum above mutates the local
        # is_fallen (removes kneeling envs), so using it here would leave half the
        # fallen envs in a kneeling/standing pose instead of the trajectory supine
        # frame-0. This was the "half envs start standing, half supine" root cause.
        if env._getup_traj_len > 0 and np.any(env._is_fallen[env_ids]) and not getattr(env.cfg, "fallen_pushover", False):
            # Phase P: when NOT using pushover, fallen envs start from trajectory
            # frame-0 (stable supine, quat fixed to back-down). When pushover=true,
            # robot starts from random grounded pose and rolls to supine on its
            # own, then tracking triggers — so skip trajectory override.
            # Re-derive the ORIGINAL fallen mask (kneeling curriculum mutated
            # the local is_fallen; we need all fallen envs incl. kneeling ones).
            orig_fallen = env._is_fallen[env_ids]
            n_f = int(np.sum(orig_fallen))
            traj_qpos = self._trajectory_frame_qpos(env, 0, n_f)  # (n_f, nq)
            qpos_all[orig_fallen] = traj_qpos
            # restore local is_fallen to original so downstream (yaw skip etc.)
            # treats all fallen envs consistently.
            is_fallen = orig_fallen.copy()

        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos_all[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        # Phase P: fallen+traj envs keep trajectory orientation (skip yaw rotation
        # so tracking stays aligned). Only non-traj envs get yaw randomization.
        if env._getup_traj_len > 0:
            yaw_apply = np.where(is_fallen, 0.0, yaw)
        else:
            yaw_apply = yaw
        qpos_all[:, 3:7] = np_quat_mul(qpos_all[:, 3:7], np_yaw_to_quat(yaw_apply))
        qpos_all[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos_all[:, 0:3], yaw=yaw_apply)

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

    def _init_action_space(self) -> None:
        """Phase P 三基准GC: action = [29 joint targets, 3 GC weights in [-1,1]].

        Override locomotion base (which sets action_space to (nu,) = 29 from
        actuator ctrl_range). We append 3 GC-weight dims bounded [-1,1] so the
        policy can output w_foot/w_pelvis/w_hand. _num_action stays 29 (joint
        space) — set in __init__ after super() makes it 32.
        """
        import gymnasium as gym
        ctrl_range = self._backend.get_actuator_ctrl_range()
        nu = self._backend.num_actuators  # 29
        low = np.concatenate([ctrl_range[:, 0], np.full(3, -1.0)])
        high = np.concatenate([ctrl_range[:, 1], np.full(3, 1.0)])
        self._action_space = gym.spaces.Box(low, high, (nu + 3,), dtype=float)
        self._num_policy_action = nu + 3  # 32

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
            add_body_sensors=getattr(cfg, "add_body_sensors", False),
        )
        # Call G1BaseEnv.__init__ directly (skip G1WalkEnv init to set up our own DR)
        # Phase P 重做: switch MuJoCo position actuators → motor (torque) actuators
        # BEFORE pool materialization, so we can use GravityCompController (τ=PD+g(q)).
        # Without gravity comp, pure-PD (kp~28) can't overcome torso gravity in supine
        # get-up — robot physically couldn't bend its waist (only reached 15% of target),
        # causing all the "lying flat / can't get up" failures. See G1WalkEnv init
        # (joystick.py:898-904) for the same pattern.
        from unilab.control.actuator_switch import switch_to_motor_actuators
        from unilab.control.pinocchio_model import PinocchioDynamicsModel
        actuator_info = switch_to_motor_actuators(backend._model)
        dynamics_model = PinocchioDynamicsModel(backend._model)

        from unilab.envs.locomotion.g1.base import G1BaseEnv
        G1BaseEnv.__init__(self, cfg, backend, num_envs)
        # Phase P 三基准GC: _num_action must stay 29 (joint space). locomotion base
        # set it to action_space.shape[0]=32 (incl 3 GC weights); restore to 29 so
        # all joint-space buffers (pose_weights, current_actions, default_angles,
        # controller kp/kd) remain 29-dim. apply_action splits 32→29+3.
        self._num_action = self._backend.num_actuators  # 29
        # default_angles was sliced with the old (32) _num_action — recompute with 29.
        import numpy as _np_da
        _dtype = get_global_dtype() if self._use_global_dtype else np.float64
        self.default_angles = np.asarray(self._init_qpos[-self._num_action:], dtype=_dtype)

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
        self._reward_fns["head_height"] = self._reward_head_height
        self._reward_fns["head_height_target"] = self._reward_head_height_target  # Phase Q UHG式
        self._reward_fns["hand_support"] = self._reward_hand_support  # Phase Q 手撑地引导
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
        # Phase P: HumanUP reference trajectory tracking (dof + height)
        self._reward_fns["dof_tracking"] = self._reward_dof_tracking
        self._reward_fns["height_tracking"] = self._reward_height_tracking
        self._reward_fns["orientation_tracking"] = self._reward_orientation_tracking
        self._reward_fns["gc_weight_align"] = self._reward_gc_weight_align  # Phase P 三基准GC
        # Running flight phase reward (high-speed airborne, walk/stand envs only)
        self._reward_fns["feet_flight"] = self._reward_feet_flight
        # v3: route alive by command — walk envs (cmd vx != 0) get NO alive bonus,
        # forcing them to earn via tracking instead of standing still. Standing/
        # flamingo/fallen envs (cmd = 0) keep alive to encourage holding still.
        self._reward_fns["alive"] = self._reward_alive

        # Phase P 重做: motor actuator + gravity compensation setup.
        # Replicates G1WalkEnv's setup (joystick.py:941-977) which multiskill
        # skipped by not calling G1WalkEnv.__init__. Without this, multiskill used
        # pure position actuators (no g(q) feedforward), and the low PD kp (~28 for
        # waist) couldn't overcome torso gravity in supine get-up — robot physically
        # couldn't bend its waist to follow the trajectory, causing "lying flat".
        # With motor actuators + GravityCompController: τ = kp(q_d-q) - kd·q̇ + g(q),
        # the gravity term cancels the torso weight so PD only needs to drive motion.
        num_actions = self._num_action
        self._base_motor_kp = actuator_info.kp.copy()
        self._base_motor_kd = actuator_info.kd.copy()
        self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, num_actions)).copy()
        self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, num_actions)).copy()
        self._force_lower = actuator_info.force_lower.copy()
        self._force_upper = actuator_info.force_upper.copy()

        gc_config = cfg.control_config
        gravity_comp_mask = None
        if getattr(gc_config, "gravity_comp_mask", None) is not None:
            gravity_comp_mask = np.asarray(gc_config.gravity_comp_mask, dtype=np.float64)
            if gravity_comp_mask.shape[0] != num_actions:
                raise ValueError(
                    f"gravity_comp_mask length ({gravity_comp_mask.shape[0]}) "
                    f"must match num_actions ({num_actions})"
                )
        from unilab.control.gravity_comp_controller import GravityCompController
        # Phase P 三基准GC: foot chain = both legs (0-11), hand chain = both arms (15-28).
        # g_foot adds torso-rest gravity to leg joints (foot-support case);
        # g_hand adds below-hand gravity to arm joints (hand-support case).
        # Waist (12-14) excluded from hand chain to avoid double-count (v1 simplification).
        _chain_mask = np.zeros(num_actions, dtype=bool)
        _foot_mask = _chain_mask.copy(); _foot_mask[0:12] = True   # both legs
        _hand_mask = _chain_mask.copy(); _hand_mask[15:29] = True  # both arms (excl waist)
        self._controller = GravityCompController(
            dynamics_model=dynamics_model,
            kp=self._base_motor_kp,
            kd=self._base_motor_kd,
            force_lower=self._force_lower,
            force_upper=self._force_upper,
            gravity_comp_mask=gravity_comp_mask,
            gravity_scale=getattr(gc_config, "gravity_scale", 1.0),
            foot_chain_mask=_foot_mask,
            hand_chain_mask=_hand_mask,
        )
        self._dynamics_model = dynamics_model
        self._last_motor_ctrl = np.zeros((num_envs, num_actions), dtype=get_global_dtype())
        # Phase P 三基准GC: per-env GC weights [w_foot, w_pelvis, w_hand], default
        # all-pelvis (current behavior until policy learns to use them).
        self._gc_weights = np.tile(np.array([0.0, 1.0, 0.0], dtype=get_global_dtype()), (num_envs, 1))
        # Register pre_step_control callback so backend converts target positions
        # (from apply_action) → motor torques (τ=PD+g(q)) before each physics step.
        self._backend.set_pre_step_control(self._pre_step_motor_control)

        # Pre-compute both keyframes
        stand_qpos = backend.get_keyframe_qpos("stand")
        stand_ctrl = stand_qpos[-self._num_action:].copy() if len(stand_qpos) > self._num_action else None
        flamingo_qpos = backend.get_keyframe_qpos("flamingo")
        flamingo_ctrl = flamingo_qpos[-self._num_action:].copy() if len(flamingo_qpos) > self._num_action else None
        kneeling_qpos = backend.get_keyframe_qpos("kneeling")
        kneeling_ctrl = kneeling_qpos[-self._num_action:].copy() if len(kneeling_qpos) > self._num_action else None

        # Phase P: load HumanUP get-up reference trajectory (29-DoF aligned).
        # Used by dof_tracking/height_tracking rewards. Optional — if file
        # missing, tracking rewards return 0 (phaseI/H/N unaffected).
        self._getup_traj_dof = None
        self._getup_traj_height = None
        self._getup_traj_len = 0
        self._getup_traj_base_z = None  # per-frame base_z (head_height mapped), for adaptive frame advance
        traj_file = str(getattr(cfg, "getup_traj_file", ""))
        if traj_file:
            import numpy as _np_traj
            _td = _np_traj.load(traj_file)
            self._getup_traj_dof = _np_traj.asarray(_td["dof_pos"], dtype=get_global_dtype())  # (T,29)
            self._getup_traj_height = _np_traj.asarray(_td["head_height"], dtype=get_global_dtype()).flatten()  # (T,)
            self._getup_traj_len = self._getup_traj_dof.shape[0]
            # head_height -> base_z (same mapping as _reward_height_tracking /
            # _trajectory_frame_qpos): supine head 0.054 -> base 0.12,
            # standing head 1.278 -> base 0.754. Ascending with frame.
            hh = self._getup_traj_height
            self._getup_traj_base_z = (0.12 + (0.754 - 0.12) * (hh - 0.054) / (1.278 - 0.054)).astype(get_global_dtype())
            # Phase P 半蹲修复: precompute per-frame expected body orientation g_x
            # (for _reward_orientation_tracking). tilt=90*(1.278-h)/(1.278-0.054),
            # g_x = -sin(tilt). frame0 g_x=-1 (supine), frame57 g_x=0 (standing),
            # monotonic in between (trajectory never flips to prone side g_x>0).
            _tilt = np.clip(90.0 * (1.278 - hh) / (1.278 - 0.054), 0.0, 90.0)
            self._getup_traj_ref_gx = (-np.sin(np.deg2rad(_tilt))).astype(get_global_dtype())
            print(f"[Phase P] loaded getup traj: {self._getup_traj_len} frames, dof{self._getup_traj_dof.shape}, base_z[{self._getup_traj_base_z[0]:.3f}->{self._getup_traj_base_z[-1]:.3f}], ref_gx[{self._getup_traj_ref_gx[0]:.3f}->{self._getup_traj_ref_gx[-1]:.3f}]")

        # Phase P 翘臀修复: cache torso_link body id for _reward_head_height
        # (the "head/upper body" reward). g1.xml has no separate head body —
        # torso_link is the topmost torso body (head/cam integrated into it).
        # Breaks the "bridge" local optimum (legs lift pelvis but torso stays on
        # ground) by rewarding torso rise via SimBackend.get_body_pos_w.
        # torso_z: standing ~0.80, supine/bridge ~0.12.
        self._head_body_id = np.array([self._backend.get_body_id("torso_link")], dtype=np.int32)
        # Phase P 三基准GC: wrist body ids for gc_weight_align hand-contact heuristic
        self._left_wrist_id = np.array([self._backend.get_body_id("left_wrist_roll_link")], dtype=np.int32)
        self._right_wrist_id = np.array([self._backend.get_body_id("right_wrist_roll_link")], dtype=np.int32)

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
        # Phase P: per-env tracking frame. -1 = not yet supine (rolling over),
        # >=0 = tracking getup trajectory (incremented each step after supine trigger).
        # Triggered when g_body·[-1,0,0] > 0.5 (reached supine). Reuses phaseN
        # rollover ability — robot rolls to supine on its own, THEN tracking starts.
        self._tracking_frame: np.ndarray = np.full(self._num_envs, -1, dtype=np.int32)
        # Phase P 重做: per-env step counter for TIME-FORCED frame advance.
        # Replaces adaptive (base_z-anchored) advance which formed a self-consistent
        # loop (robot tracks frame-15 well → frame doesn't advance → base doesn't
        # rise → stays at frame-15). Time-forced advance (every FRAME_ADVANCE_PERIOD
        # steps +1 frame) forces the robot to follow the FULL trajectory sequence
        # (curl-up → hip-back → reach-arms → stand), not dwell on a comfortable frame.
        # -1 = not triggered yet; 0+ counts steps since supine trigger.
        self._tracking_step_count: np.ndarray = np.full(self._num_envs, -1, dtype=np.int32)
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

    def _reward_head_height(self, ctx: RewardContext) -> np.ndarray:
        """Phase P 翘臀修复: reward upper-body (torso) rise to break the bridge
        local optimum.

        Without this, the policy learns to lift only the pelvis with the legs
        (satisfying height_exp/base_height/stand_feet) while the torso/head
        stays on the ground — a "bridge"/hip-thrust pose that scores high on
        pelvis-based rewards but never actually stands up.

        Uses torso_link (g1.xml has no separate head body; torso is the topmost
        torso body). torso_z: standing ~0.80, supine/bridge ~0.12. exp(h)-1 form
        (every cm counts, same shape as height_exp but on the torso). Masked to
        fallen envs. Uses SimBackend.get_body_pos_w (declared interface, not a
        backend-private field). Variable named _head_body_id for intent; holds
        torso_link id.
        """
        head_z = self._backend.get_body_pos_w(self._head_body_id)[:, 0, 2]
        return np.asarray(np.exp(np.clip(head_z, 0.1, 0.8)) - 1.0, dtype=get_global_dtype()) * self._fallen_mask()

    def _reward_head_height_target(self, ctx: RewardContext) -> np.ndarray:
        """Phase Q 借鉴UHG: reward head_z approaching a STAGED desired height.

        UHG uses exp(-10*(head_h - desired_height)^2) with desired=standing height.
        But G1 is large (standing head_z~0.73, supine ~0.12); a single target from
        supine is too sparse — the robot can't explore the stand-up motion. So we
        STAGE desired_h to ramp up as training progresses (by global step_counter):
          step < 1.5M (iter~400):  desired_h=0.35 (half-kneel, easy to reach)
          step < 3.0M (iter~800):  desired_h=0.55 (half-stand)
          step >= 3.0M:            desired_h=0.73 (standing)
        Each stage is reachable, pulling the robot up incrementally. Uses torso_link
        z (same as _reward_head_height). Masked to fallen envs.

        NOTE: step_counter is global cumulative (not per-episode), so all envs share
        the same stage at a given training step — correct for a curriculum.
        """
        head_z = self._backend.get_body_pos_w(self._head_body_id)[:, 0, 2]
        sc = self.step_counter
        if sc < 1_500_000:
            desired_h = 0.35
        elif sc < 3_000_000:
            desired_h = 0.55
        else:
            desired_h = 0.73
        err = head_z - desired_h
        # coef 10→50: 之前太宽(躺平head0.07拿4.6/10=46%, critic学到躺平Q高0.86>起身0.838,
        # actor跟critic躺平). coef=50让躺平只拿0.2, 起身拿10, 强逼起身(Q诊断确认).
        return np.asarray(np.exp(-50.0 * np.square(err)), dtype=get_global_dtype()) * self._fallen_mask()

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

        Phase P 半蹲修复: when a getup trajectory is loaded, gate this reward
        to base_z > 0.65 (near-standing). The trajectory's get-up arc requires
        the body to TILT mid-rise (frame 15 wants g_x=-0.67, a 47° lean), so
        rewarding verticality mid-rise conflicts with orientation_tracking and
        lets the policy farm a vertical half-squat. Near the top (base>0.65)
        the trajectory is ~vertical anyway, so uprightness safely takes over
        to pin the final stand. No trajectory → gate=1 (phaseI/H/N unchanged).
        """
        g = ctx.gravity
        assert g is not None
        g_xy_sq = np.square(g[:, 0]) + np.square(g[:, 1])
        upright = np.exp(-g_xy_sq)
        if self._getup_traj_len > 0:
            h = self._backend.get_base_pos()[:, 2]
            # Phase Q: gate 0.65→0.4 (UHG式, 半跪就奖励直立, 配合分阶段高度课程)
            gate = (h > 0.4).astype(get_global_dtype())
            upright = upright * gate
        return np.asarray(upright, dtype=get_global_dtype()) * self._fallen_mask()

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
        # Phase P 方向B: gate off once supine is reached (tracking started).
        # With relaxed termination, robot could otherwise farm roll reward by
        # lying supine indefinitely. Once _tracking_frame>=0 (supine reached),
        # tracking rewards take over; roll reward only pays for the GETTING-
        # supine process (not staying supine). No-op when no getup traj.
        rolling = np.ones_like(supine_score, dtype=get_global_dtype())
        if self._getup_traj_len > 0:
            rolling = (self._tracking_frame < 0).astype(get_global_dtype())
        return np.asarray(supine_score * ground_gate * rolling, dtype=get_global_dtype()) * self._fallen_mask()

    def _reward_dof_tracking(self, ctx: RewardContext) -> np.ndarray:
        """Phase P: track HumanUP get-up trajectory joint angles.

        Frame = per-env _tracking_frame (supine-triggered: -1 while rolling,
        0+ after reaching supine). Returns 0 for envs not yet supine (still
        rolling over via phaseN ability). Masked to fallen envs.

        Uses MEAN per-joint squared error (err / n_joints) so that sigma has a
        per-joint physical meaning (rad). With sum-MSE, sigma had to absorb the
        29-joint scale and early-exploration errors (~11) drove exp to ~0
        regardless of sigma. mean-MSE makes exp(-mse/sigma^2) responsive: sigma=0.5
        ≈ 28° per-joint tolerance, 1.0 ≈ 57°. Early exploration still scores >0
        so the gradient can pull joints toward the reference."""
        if self._getup_traj_len == 0:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        frames = np.clip(self._tracking_frame, 0, self._getup_traj_len - 1)
        active = (self._tracking_frame >= 0).astype(get_global_dtype())  # 0 if rolling
        ref_dof = self._getup_traj_dof[frames]  # (N, 29)
        cur_dof = ctx.dof_pos
        n = min(ref_dof.shape[1], cur_dof.shape[1])
        mse = np.sum(np.square(cur_dof[:, :n] - ref_dof[:, :n]), axis=1) / float(n)
        sigma = float(getattr(self._reward_cfg, "dof_tracking_sigma", 1.5))
        return np.asarray(np.exp(-mse / (sigma * sigma)), dtype=get_global_dtype()) * active * self._fallen_mask()

    def _reward_height_tracking(self, ctx: RewardContext) -> np.ndarray:
        """Phase P: track HumanUP get-up trajectory head/base height.
        Supine-triggered (0 while rolling). Masked to fallen envs."""
        if self._getup_traj_len == 0:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        frames = np.clip(self._tracking_frame, 0, self._getup_traj_len - 1)
        active = (self._tracking_frame >= 0).astype(get_global_dtype())
        ref_h = self._getup_traj_height[frames]
        ref_base_z = 0.12 + (0.754 - 0.12) * (ref_h - 0.054) / (1.278 - 0.054)
        cur_z = self._backend.get_base_pos()[:, 2]
        err = np.square(cur_z - ref_base_z)
        sigma = float(getattr(self._reward_cfg, "height_tracking_sigma", 0.15))
        return np.asarray(np.exp(-err / (sigma * sigma)), dtype=get_global_dtype()) * active * self._fallen_mask()

    def _reward_orientation_tracking(self, ctx: RewardContext) -> np.ndarray:
        """Phase P 半蹲修复: track trajectory frame's expected body orientation (g_x).

        Without this, the policy farms uprightness (keeps body vertical) at a
        half-squat base_z~0.46 instead of tilting per the trajectory's get-up
        arc (frame 15 wants g_x=-0.67, a 47° tilt). The adaptive frame index is
        anchored by base_z, so a robot at the right height but wrong orientation
        gets stuck mid-trajectory with dof_tracking~0.1 (joints don't match).

        Trajectory g_x is precomputed per frame (_getup_traj_ref_gx): frame 0
        g_x=-1 (supine), frame 57 g_x=0 (standing), monotonic, never flips to
        the prone side (g_x>0). Rewarding alignment to this forces the robot to
        follow the trajectory's tilt arc, not just its height.

        Supine-triggered (0 while rolling). sigma=0.3 (~17° tolerance). Masked
        to fallen envs. Uses ctx.gravity (same source as roll_to_supine)."""
        if self._getup_traj_len == 0:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        frames = np.clip(self._tracking_frame, 0, self._getup_traj_len - 1)
        active = (self._tracking_frame >= 0).astype(get_global_dtype())
        ref_gx = self._getup_traj_ref_gx[frames]  # (N,) expected g_x per frame
        cur_gx = ctx.gravity[:, 0]
        err = np.square(cur_gx - ref_gx)
        sigma = 0.3
        return np.asarray(np.exp(-err / (sigma * sigma)), dtype=get_global_dtype()) * active * self._fallen_mask()

    def _reward_gc_weight_align(self, ctx: RewardContext) -> np.ndarray:
        """Phase P 三基准GC: align policy GC weights to actual contact state.

        The 3 GC weights (w_foot, w_pelvis, w_hand) only weakly affect reward
        through dynamics, so pure RL tends to collapse them. This auxiliary
        reward shapes them toward the contact-grounded target:
          c_foot = fraction of feet in contact (0 / 0.5 / 1)
          c_hand = fraction of hands in contact (0 / 0.5 / 1) — v1 uses wrist-z heuristic
        Penalizes |w_foot - c_foot| + |w_hand - c_hand| so the policy learns to
        declare the correct support basis (foot when feet down, hand when hands down).
        Masked to fallen envs (get-up only). Small weight (configurable).
        """
        if not hasattr(self, "_gc_weights") or self._gc_weights is None:
            return np.zeros(ctx.num_envs, dtype=get_global_dtype())
        w_foot = self._gc_weights[:, 0]
        w_hand = self._gc_weights[:, 2]
        # Foot contact fraction (0/0.5/1)
        lc = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        rc = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        c_foot = (lc.astype(get_global_dtype()) + rc.astype(get_global_dtype())) * 0.5
        # Hand contact: v1 heuristic — wrist-link z < 0.10 (near ground)
        # (no hand contact sensors; wrist height is a reasonable proxy during get-up)
        lw = self._backend.get_body_pos_w(self._left_wrist_id)[:, 0, 2]
        rw = self._backend.get_body_pos_w(self._right_wrist_id)[:, 0, 2]
        c_hand = ((lw < 0.10).astype(get_global_dtype()) + (rw < 0.10).astype(get_global_dtype())) * 0.5
        lam = float(getattr(self._reward_cfg, "gc_weight_align_lambda", 0.05))
        penalty = np.abs(w_foot - c_foot) + np.abs(w_hand - c_hand)
        return np.asarray(-lam * penalty, dtype=get_global_dtype()) * self._fallen_mask()

    def _reward_hand_support(self, ctx: RewardContext) -> np.ndarray:
        """Phase Q: hands reach ground during get-up, then lift when standing.

        Two-phase shaping to avoid the "hands stuck on ground" failure:
          - base_z < 0.5 (get-up phase): REWARD wrists low (~0.05m) — the
            "弯腰同时双手撑地" lever that's physically needed to rise from supine.
          - base_z >= 0.5 (near standing): PENALIZE wrists low — once the legs
            carry the body, hands must come UP, not stay planted. Without this,
            the policy can farm hand_support by keeping hands down while standing
            (wrong, unstable stance).

        Reward(low phase) = +exp(-(wrist_z-0.05)^2/σ^2);  Penalty(high phase)
        = -exp(-(wrist_z-0.05)^2/σ^2) * (wrist actually low). Per wrist, mean.
        Masked to fallen envs. σ=0.10.
        """
        lw = self._backend.get_body_pos_w(self._left_wrist_id)[:, 0, 2]
        rw = self._backend.get_body_pos_w(self._right_wrist_id)[:, 0, 2]
        sigma = 0.10
        low_score = (np.exp(-np.square((lw - 0.05) / sigma)) +
                     np.exp(-np.square((rw - 0.05) / sigma))) * 0.5  # 0..1
        h = self._backend.get_base_pos()[:, 2]
        # get-up phase (base<0.5): reward hands down; standing phase (base>=0.5): penalize hands down
        getup = (h < 0.5).astype(get_global_dtype())
        standing = (h >= 0.5).astype(get_global_dtype())
        r = low_score * getup - low_score * standing
        return np.asarray(r, dtype=get_global_dtype()) * self._fallen_mask()

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

        Phase P 三基准GC: action is 32-dim = [29 joint targets, 3 GC weights].
        Split joint targets (→ ctrl) from GC weights (w_foot/w_pelvis/w_hand in
        [-1,1] → [0,1]). current_actions stores only the 29 joint part so the obs
        "actions" group (29, per obs_groups_spec) stays consistent.

        The inherited LocomotionBaseEnv.apply_action only stores last/current
        actions and builds ctrl — it never advances gait_phase, so the phase
        sampled at reset stays static for the whole episode. With a static
        phase target, the feet_phase reward encourages a fixed single-foot
        lift instead of an alternating gait. Advance both legs phases here so
        walking envs track a moving phase target (alternating stance/swing).
        """
        # Phase P 三基准GC: split 32-dim action into 29 joints + 3 weights.
        joint_actions = actions[:, : self._num_action]  # (N, 29)
        weight_actions = actions[:, self._num_action : self._num_action + 3]  # (N, 3) in [-1,1]
        # latency applies to BOTH parts consistently (use joint part's last_actions shape)
        prev_joint = state.info.get("current_actions", np.zeros_like(joint_actions))
        state.info["last_actions"] = prev_joint
        state.info["current_actions"] = joint_actions  # obs "actions" group = 29-dim
        if self._cfg.control_config.simulate_action_latency:
            exec_joint = prev_joint
            # weights: also lag (store separately is complex; use current for simplicity)
            exec_weights = weight_actions
        else:
            exec_joint = joint_actions
            exec_weights = weight_actions
        # GC weights in [0,1]: w_foot, w_pelvis, w_hand
        self._gc_weights = ((exec_weights + 1.0) * 0.5).astype(get_global_dtype())
        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase
        ctrl: np.ndarray = exec_joint * self._cfg.control_config.action_scale + self.default_angles
        return ctrl

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        """Pre-step callback: convert target positions → motor torques with gravity comp.

        Replicates G1WalkEnv._pre_step_motor_control (joystick.py:1060). Called by
        the backend before each physics substep. Gravity computed once per ctrl_dt
        and cached across substeps.
        """
        self._dynamics_model.invalidate_cache()
        joint_pos = backend.get_dof_pos()
        joint_vel = backend.get_dof_vel()
        full_qpos = backend.get_full_qpos()
        full_qvel = backend.get_full_qvel()
        # Update controller gains (may have been randomized by DR at reset)
        self._controller.kp = self._motor_kp
        self._controller.kd = self._motor_kd
        # τ = kp(q_d - q) - kd·q̇ + (w_foot·g_foot + w_pelvis·g_pelvis + w_hand·g_hand)
        motor_ctrl = self._controller.compute(
            policy_ctrl,
            joint_pos,
            joint_vel,
            full_qpos=full_qpos,
            full_qvel=full_qvel,
            gc_weights=self._gc_weights,  # Phase P 三基准GC: per-env [w_foot,w_pelvis,w_hand]
        )
        self._last_motor_ctrl = motor_ctrl
        return motor_ctrl

    # ── Phase K: pushover get-up reset ───────────────────────────────
    def reset(self, env_indices: np.ndarray):
        """Standard reset, then (if fallen_pushover) push fallen envs over and
        PD-control them into a randomized grounded lie pose. Recomputes obs so
        the returned obs matches the real (grounded) qpos, not the stand pose.
        """
        obs, info = super().reset(env_indices)
        # Phase P: reset tracking frame to -1 (not yet supine, must roll over first)
        self._tracking_frame[env_indices] = -1
        self._tracking_step_count[env_indices] = -1  # Phase P 重做: reset step counter too
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

        # Phase P: update per-env tracking frame (supine-triggered).
        # -1 (rolling) -> 0 when robot reaches supine (g_body·[-1,0,0] > 0.5),
        # then increments each step. Reuses phaseN rollover: robot rolls to
        # supine on its own, THEN tracking starts (frame aligned to supine state).
        # IMPORTANT: use the upvector sensor (SAME source as ctx.gravity in
        # _reward_roll_to_supine, see joystick.py _build_reward_context). Do NOT
        # read obs[3:6] — the obs gravity is negated (-gravity, see joystick.py
        # _compute_obs), so reading obs inverts the supine sign and the trigger
        # never fires (this was the dof_tracking==0 root cause before this fix).
        # NOTE: backend.get_gravity() returns the world gravity constant [0,0,-g],
        # NOT per-env body-frame projected gravity — do not use it here.
        if self._getup_traj_len > 0:
            g_body = self._backend.get_sensor_data(self._cfg.sensor.upvector)  # (N,3) body-frame
            # supine (back-down) => g_body·[-1,0,0] = -g_x ≈ +1
            cos_supine = g_body[:, 0] * (-1.0)
            # trigger: not yet tracking AND reached supine
            not_triggered = self._tracking_frame < 0
            reached_supine = cos_supine > 0.5
            newly_triggered = not_triggered & reached_supine
            self._tracking_frame[newly_triggered] = 0
            self._tracking_step_count[newly_triggered] = 0  # start counting on trigger
            # Phase P 重做: TIME-FORCED frame advance (replaces adaptive base_z-anchored).
            # Adaptive advance formed a self-consistent loop: robot tracks frame-15
            # well → base_z stays ~0.46 → frame re-anchors to 15 → robot dwells there
            # forever (never rises). Time-forced advance pushes frame forward every
            # FRAME_ADVANCE_PERIOD steps regardless of robot state, forcing the robot
            # to follow the FULL trajectory sequence (curl-up → hip-back → reach-arms
            # → stand) or fall behind and lose dof_tracking/orientation_tracking reward.
            # Rate: every 5 steps +1 frame → 58*5=290 steps = 5.8s to stand (joint
            # speed demand 6.4 rad/s, within G1's ~10 rad/s limit). At end, hold last frame.
            FRAME_ADVANCE_PERIOD = 5
            tracking = self._tracking_frame >= 0
            if np.any(tracking):
                self._tracking_step_count[tracking] += 1
                new_frame = np.minimum(
                    self._tracking_step_count[tracking] // FRAME_ADVANCE_PERIOD,
                    self._getup_traj_len - 1,
                )
                self._tracking_frame[tracking] = new_frame.astype(np.int32)

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
            # Phase P 方向B: fallen envs use a separate (lower) base-height floor
            # so they can flounder on the ground after falling without instant
            # termination. min_base_height_fallen<0 falls back to min_base_height
            # (preserves phaseK/J behavior).
            min_h_fallen = float(getattr(self._cfg, "min_base_height_fallen", -1.0))
            if min_h_fallen < 0.0:
                min_h_fallen = self._reward_cfg.min_base_height
            fallen_strict = (tilt > np.deg2rad(self._reward_cfg.max_tilt_deg)) | (
                self._terrain_relative_base_height() < self._reward_cfg.min_base_height
            )
            fallen_relaxed = (tilt > np.deg2rad(max_tilt_fallen)) | (
                self._terrain_relative_base_height() < min_h_fallen
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
