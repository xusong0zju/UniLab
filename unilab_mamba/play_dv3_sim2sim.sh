#!/usr/bin/env bash
# Sim2Sim 回放脚本：加载 D v3 最新 checkpoint，录 30s mp4（含各技能及过渡）
#
# 不影响训练的保证：
#   1. play_only=true        只回放不训练，不碰训练 buffer/权重
#   2. export_onnx=false    不往训练 run 目录写 onnx（避免和训练存盘 model_N.pt 撞）
#   3. 独立进程             回放与训练进程隔离；GPU 余量 ~20G，回放约用 1-2G
#   4. resampling_time 仅作 play override，不写回配置文件，不影响训练
#
# 30s + 各技能及过渡 的实现：
#   - ctrl_dt=0.02 (50Hz) → play_steps=1500 = 30s
#   - resampling_time=8.0 → 每 8s resample 命令(含技能切换)，30s 内 walk→stand→flamingo→walk 过渡 3-4 次
#     （D v3 训练时 resampling_time=0.0 不切，play 时显式开启以覆盖技能过渡）
#
# 用法（本机终端，建议另开 tmux 窗口，不要占用训练的 tmux）：
#   tmux new -s play
#   bash /root/UniLab_WS/UniLab/unilab_mamba/play_dv3_sim2sim.sh
#
# 输出：logs/flash_sac/G1MultiSkill/<run_dir>/play_video.mp4
# 日志：/tmp/dv3_play.log
set -euo pipefail
cd /root/UniLab_WS/UniLab
export PATH=/root/miniconda3/envs/unilab/bin:$PATH

uv run scripts/train_offpolicy.py \
  task=flashsac/g1_multiskill/mamba_phaseD_v3 \
  algo=flashsac \
  training.play_only=true \
  training.play_render_mode=record \
  training.export_onnx=false \
  training.play_env_num=1 \
  training.play_steps=1500 \
  algo.load_run=-1 \
  env.commands.resampling_time=8.0 \
  2>&1 | tee /tmp/dv3_play.log

echo
echo "===== 完成 ====="
echo "视频应位于："
ls -la logs/flash_sac/G1MultiSkill/2026-06-20_13-02-39_mujoco/play_video.mp4 2>/dev/null || \
  find logs/flash_sac/G1MultiSkill/2026-06-20_13-02-39_mujoco/ -name "*.mp4" 2>/dev/null
