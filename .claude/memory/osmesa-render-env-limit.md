---
name: osmesa-render-env-limit
description: "osmesa软渲染play视频时env数上限——24 envs死锁,16 envs可行"
metadata: 
  node_type: memory
  type: project
  originSessionId: 21771dda-f1da-453c-962e-badcc8c86561
---

这台 autodl 容器 EGL 离屏渲染坏(EGLError),MuJoCo 只能用 osmesa 软渲染录视频。osmesa 对场景复杂度敏感:

- **16 envs × 800/1000 帧**:可行
- **24 envs × 1000 帧**:死锁卡住(主进程 CPU 0%,只起 2 个进程非 8 worker,CPU 总占用 8%),杀掉重来
- 特征:`render_many` 把多机器人塞一个场景,osmesa 大场景易死锁;且 spawn worker 在 osmesa 下 init 失败静默退出,"8 processes" 实际不并行(单进程 CPU 7%)

**两个独立坑**:
1. env 数过多 → osmesa 大场景死锁。固定 play_env_num=16,别超。
2. spawn worker 不起来 → 即使 16 envs 也会卡(CPU 7%,只 1 个 python 进程)。根因是 `render_many.py` 用 `multiprocessing spawn` Pool,worker 在 osmesa 下 init 失败;之前卡死还会在 `/dev/shm/` 残留 `sem.mp-*` 信号量,进一步毒化后续渲染。

**解法(已落地)**:给 `src/unilab/visualization/render_many.py` 加了 `UNILAB_RENDER_PROCESSES` 环境变量(默认8)。设 `=1` 强制串行渲染,绕过 spawn 死锁——单进程在主进程里渲染,CPU 能跑满多核(实测 403-443%),1000 帧约 8-12 分钟稳定出片。卡死过的机器先清 `/dev/shm/sem.mp-*` 再重跑。

**录视频标准命令**:
```bash
MUJOCO_GL=osmesa UNILAB_RENDER_PROCESSES=1 uv run scripts/train_offpolicy.py \
  task=flashsac/g1_multiskill/mamba_phaseD_v3 algo=flashsac \
  training.play_only=true training.play_render_mode=record training.no_play=false \
  training.export_onnx=false \
  algo.load_run=<run目录名> '+algo.checkpoint=<iter>' \
  training.play_env_num=16 training.play_steps=1000
```
注意 `+algo.checkpoint` 要带 `+`(struct 模式追加字段),`algo.load_run` 是 run 目录名(如 `2026-06-20_22-42-08_mujoco`)。视频输出到 `<run目录>/play_video.mp4`,会覆盖该目录旧视频。

若需聚焦单机器人看清步态/起身,要改 train_offpolicy.py:597 的 camera_kwargs 加 `cam_tracking=true`(走 render_states_get_frames_tracking),但需改源码。详见 [[mamba-multiskill-eval-strict]]。
