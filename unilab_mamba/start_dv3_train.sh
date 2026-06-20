#!/usr/bin/env bash
# 线索B · 方案C 训练启动脚本（必须在【本机终端 / 云机 web 终端】里跑，不能 ssh 后台）
# 用法：在本机终端执行  bash unilab_mamba/start_dv3_train.sh
# 原理：use_compile:false + official kernel + collector 上 cuda（方案C，已验证跑到 iter 170 不卡）
set -euo pipefail

cd /root/UniLab_WS/UniLab
export PATH=/root/miniconda3/envs/unilab/bin:$PATH

# 复用 train 会话；已存在则进入，不在则新建
tmux has-session -t train 2>/dev/null && {
  echo "[已有 train 会话，直接 attach]"; tmux attach -t train; exit 0
}

tmux new -s train "cd /root/UniLab_WS/UniLab && \
  export PATH=/root/miniconda3/envs/unilab/bin:\$PATH && \
  uv run scripts/train_offpolicy.py task=flashsac/g1_multiskill/mamba_phaseD_v3 algo=flashsac 2>&1 | tee /tmp/dv3_train.log"
# 进入后 Ctrl+B 然后 D 脱离（训练继续），tmux attach -t train 重新进入看进度
