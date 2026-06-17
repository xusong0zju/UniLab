#!/bin/bash
# Run Mamba Phase C1→C2→C3→C4 sequentially, each warm-starting from previous.
# Usage: nohup bash scripts/run_mamba_phaseC.sh > /tmp/phaseC_all.log 2>&1 &

set -e
ROOT=/root/UniLab_WS/UniLab

C1_MODEL=$ROOT/logs/flash_sac/G1MultiSkill/2026-06-16_17-46-20_mujoco/model_4000.pt

echo "=== Phase C1: 50N, 5000 iter ==="
uv run $ROOT/scripts/train_offpolicy.py task=flashsac/g1_multiskill/mamba_phaseC1 algo=flashsac algo.load_run=$C1_MODEL
C1_DIR=$(ls -td $ROOT/logs/flash_sac/G1MultiSkill/*/ | head -1)
C1_BEST=$(ls -t ${C1_DIR}model_*.pt | head -1)
echo "C1 done: $C1_BEST"

echo "=== Phase C2: 100N, 7000 iter ==="
uv run $ROOT/scripts/train_offpolicy.py task=flashsac/g1_multiskill/mamba_phaseC2 algo=flashsac algo.load_run=$C1_BEST
C2_DIR=$(ls -td $ROOT/logs/flash_sac/G1MultiSkill/*/ | head -1)
C2_BEST=$(ls -t ${C2_DIR}model_*.pt | head -1)
echo "C2 done: $C2_BEST"

echo "=== Phase C3: 200N, 10000 iter ==="
uv run $ROOT/scripts/train_offpolicy.py task=flashsac/g1_multiskill/mamba_phaseC3 algo=flashsac algo.load_run=$C2_BEST
C3_DIR=$(ls -td $ROOT/logs/flash_sac/G1MultiSkill/*/ | head -1)
C3_BEST=$(ls -t ${C3_DIR}model_*.pt | head -1)
echo "C3 done: $C3_BEST"

echo "=== Phase C4: 400N, 15000 iter ==="
uv run $ROOT/scripts/train_offpolicy.py task=flashsac/g1_multiskill/mamba_phaseC4 algo=flashsac algo.load_run=$C3_BEST
echo "ALL DONE"
