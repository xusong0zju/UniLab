#!/usr/bin/env bash
# 线索B · 远程盯坎判断脚本（Claude 可在本机直接跑，不依赖 ssh 后台）
# 判断：正常推进 / 慢但活着 / 真卡死，并报告当前 iter 与离 940/980 的距离
# 卡死三件套（来自 memory mamba-locomotion-empirical-lessons）：
#   GPU 连续≥5次采样恒0% + 进程CPU TIME不增 + events停写 + 有僵尸compile_worker
cd /root/UniLab_WS/UniLab

echo "===== 进程 ====="
pgrep -af "train_offpolicy|mamba_phaseD" | grep -v grep || echo "  无训练进程（可能未启动/已退出）"
echo "--- 僵尸 compile_worker ---"
ps -ef | awk '$2~/Z/' | head || true

echo; echo "===== GPU (连续采3次，间隔2s) ====="
for i in 1 2 3; do
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | sed "s/^/  采样$i: /"
  sleep 2
done

echo; echo "===== 日志尾部 /tmp/dv3_train.log ====="
tail -8 /tmp/dv3_train.log 2>/dev/null || echo "  无日志"

echo; echo "===== 最新 run 的 iter 进度 ====="
RUNDIR=$(ls -dt logs/flash_sac/G1MultiSkill/2026-06-*_mujoco 2>/dev/null | head -1)
echo "  RUNDIR=$RUNDIR"
if [ -n "$RUNDIR" ]; then
  echo "  events 最后修改: $(stat -c '%y' "$RUNDIR"/*.tfevents* 2>/dev/null | head -1)"
  export PATH=/root/miniconda3/envs/unilab/bin:$PATH
  .venv/bin/python - "$RUNDIR" <<'EOF' 2>/dev/null
import sys
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(sys.argv[1], size_guidance={'scalars':0})
ea.Reload()
for t in ["axis/iteration","reward/mean","reward/alive","train/critic_loss","timing/learner_train_ms"]:
    try:
        v = ea.Scalars(t); print(f"  {t}: last={v[-1].value:.3f} (n={len(v)})")
    except Exception: pass
EOF
fi
