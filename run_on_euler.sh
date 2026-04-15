#!/bin/bash
# ============================================================
# ME759 Course Participation — Euler run script
# Copy stencil.cu + this file to Euler, then run:
#   chmod +x run_on_euler.sh && sbatch run_on_euler.sh
# ============================================================

#SBATCH --job-name=me759_stencil
#SBATCH --output=stencil_results_%j.txt
#SBATCH --error=stencil_errors_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# ── Load modules ──────────────────────────────────────────────
module load cuda
module list

# ── Print GPU info ────────────────────────────────────────────
nvidia-smi

# ── Compile ───────────────────────────────────────────────────
# Change -arch flag based on GPU:
#   A100 = sm_80,  H100 = sm_90,  V100 = sm_70,  K20 = sm_35
nvcc -O3 -arch=sm_80 stencil.cu -o stencil -lm
echo "Compiled OK"

# ── Run benchmark ────────────────────────────────────────────
echo ""
echo "=========================================="
echo "BENCHMARK: N=1M, R=8"
echo "=========================================="
./stencil 1048576 8 256

echo ""
echo "=========================================="
echo "BENCHMARK: N=4M, R=8"
echo "=========================================="
./stencil 4194304 8 256

echo ""
echo "=========================================="
echo "BENCHMARK: N=1M, R=1"
echo "=========================================="
./stencil 1048576 1 256

echo ""
echo "=========================================="
echo "BENCHMARK: N=1M, R=16"
echo "=========================================="
./stencil 1048576 16 256

# ── ncu profiling (roofline) ─────────────────────────────────
echo ""
echo "=========================================="
echo "NCU PROFILE: naive kernel, N=1M, R=8, block=256"
echo "=========================================="
ncu --set full \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum,\
dram__bytes.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
    -o ncu_naive \
    ./stencil 1048576 8 256
echo "Saved: ncu_naive.ncu-rep"

echo ""
echo "NCU PROFILE: tiled kernel — run stencil_tiled only"
echo "(edit stencil.cu to comment out naive if needed)"
echo ""

# ── ncu roofline export ───────────────────────────────────────
# Export roofline chart data (use ncu-ui on your laptop to view .ncu-rep):
echo "To view roofline on your laptop:"
echo "  scp euler:~/me759/ncu_naive.ncu-rep ."
echo "  ncu-ui ncu_naive.ncu-rep"
echo ""
echo "=== DONE ==="
