"""D v3 play 诊断：录每步运动学数据，看机器人怎么倒的。

复用 train_offpolicy.play_offpolicy 的 env+actor 构建（已验证能跑通视频），
只把后半段的"录视频"换成"记录 base_z/tilt/command/action/reward"。
不影响训练（独立进程，只读 checkpoint）。

用法: uv run python unilab_mamba/diag_play.py
"""
import os, sys, csv
sys.path.insert(0, "src"); sys.path.insert(0, ".")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from typing import Any, cast

# ── 1. Hydra compose (和 train_offpolicy.py 的 @hydra.main 等价) ──
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=os.path.abspath("conf/offpolicy"), version_base="1.3"):
    cfg = compose(config_name="config", overrides=[
        "task=flashsac/g1_multiskill/mamba_phaseD_v3",
        "algo=flashsac",
        "training.play_only=true",
        "training.play_env_num=1",
        "algo.load_run=-1",
    ])

algo_name = "flashsac"

# ── 2. 触发 env/registry 注册 (train_offpolicy main 里会调 ensure_registries) ──
from unilab.base.registry import ensure_registries
ensure_registries()

# ── 3. 复用 play_offpolicy 的 env/actor 构建 ──
# 直接 import train_offpolicy 模块的函数，确保和真实 play 路径完全一致
import scripts.train_offpolicy as tplay

env_cfg_override = tplay.build_offpolicy_env_cfg_override(algo_name, cfg)
device = tplay.default_device(torch, cfg.training.device)
print(f"Using device for play: {device}")

from unilab.training import create_env
env = cast(Any, create_env(cfg, num_envs=cfg.training.play_env_num, env_cfg_override=env_cfg_override))

obs_dim, critic_obs_dim = tplay.resolve_play_obs_dims(env.obs_groups_spec)
action_dim = int(env.action_space.shape[0])
print(f"obs_dim={obs_dim} action_dim={action_dim} play_env_num={cfg.training.play_env_num}")

actor_algo_type, actor_kwargs = tplay.resolve_play_actor_spec(algo_name, cfg, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim)
from unilab.algos.torch.common.actor_factory import build_actor
ap = cfg.algo.algo_params
actor = build_actor(
    "flashsac", obs_dim, action_dim, cfg.algo.actor_hidden_dim, cfg.algo.use_layer_norm, device,
    actor_num_blocks=ap.actor_num_blocks, actor_noise_zeta_mu=ap.actor_noise_zeta_mu,
    actor_noise_zeta_max=ap.actor_noise_zeta_max,
    use_mamba_actor=getattr(ap, "use_mamba_actor", False),
    mamba_d_state=getattr(ap, "mamba_d_state", 16), mamba_n_tokens=getattr(ap, "mamba_n_tokens", 4),
)
actor.eval()

# ── 4. 加载 checkpoint (用 train_offpolicy 的 resolve_checkpoint_path, 和 play 一致) ──
load_path, load_path_dir = tplay.resolve_checkpoint_path(
    os.path.abspath("."), cfg.algo.algo_log_name, cfg.training.task_name, cfg.algo.load_run)
print(f"Loading model: {load_path}")
checkpoint = torch.load(load_path, map_location=device, weights_only=True)
actor.load_state_dict(checkpoint["actor"])
print("actor 加载成功")

# ── 4. 诊断 step 循环 (不录视频,只记录) ──
from unilab.base.observations import split_obs_dict

env.init_state()
obs_dict, info_out = env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
obs_np = np.asarray(tplay.extract_play_obs(obs_dict), dtype=np.float32)

rows = []
with torch.inference_mode():
    for t in range(300):
        ot = torch.from_numpy(obs_np).to(device)
        a = actor.explore(ot, deterministic=True).cpu().numpy()
        state = env.step(a)
        obs_np = np.asarray(tplay.extract_play_obs(state.obs), dtype=np.float32)
        info = state.info if hasattr(state, "info") else {}

        # 抓运动学量 (从 env backend 直接取, 和 multiskill.py:538 一致)
        try:
            base_z = float(np.asarray(env._backend.get_base_pos())[:, 2].flatten()[0])
        except Exception:
            base_z = -1.0
        # command: 在 state.info["commands"] (multiskill.py:615)
        cmd = info.get("commands")
        cmd = np.asarray(cmd).flatten()[:3] if cmd is not None else np.full(3, np.nan)
        # reward / terminated
        r = float(np.asarray(state.reward).flatten()[0]) if state.reward is not None else 0.0
        term = bool(np.asarray(state.terminated).flatten()[0]) if state.terminated is not None else False
        rows.append([t, base_z, cmd[0], cmd[1], cmd[2], float(np.abs(a).max()), float(a.std()), r, int(term)])
        if t % 25 == 0 or term:
            print(f"t={t:3d} base_z={base_z:.3f} cmd={cmd.round(2)} |a|max={np.abs(a).max():.2f} a.std={a.std():.2f} r={r:.2f} term={term}")
        if term and t > 5:
            print(f"  ⚠ t={t} 摔倒终止!"); break

# ── 5. 报告 ──
out = "/tmp/dv3_play_diag.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "base_z", "cmd_x", "cmd_y", "cmd_z", "abs_a_max", "a_std", "reward", "terminated"])
    w.writerows(rows)
z = np.array([r[1] for r in rows])
print(f"\n===== 报告 (共{len(rows)}步, checkpoint={os.path.basename(load_path)}) =====")
print(f"base_z: 初始={z[0]:.3f} 末值={z[-1]:.3f} min={z.min():.3f} max={z.max():.3f}")
print(f"  base_z<0.5(快倒)的步: {int((z<0.5).sum())}/{len(z)}")
print(f"  base_z 首10均={z[:10].mean():.3f} 末10均={z[-10:].mean():.3f}  (末<首=持续下降)")
print(f"CSV: {out}")
