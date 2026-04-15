#!/bin/bash
#SBATCH --job-name=me759_stencil
#SBATCH --output=stencil_out_%j.txt
#SBATCH --error=stencil_err_%j.txt
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -e

# ══════════════════════════════════════════════════════════════
# WRITE stencil.cu
# ══════════════════════════════════════════════════════════════
cat > stencil.cu << 'CUDA_EOF'
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARMUP_ITERS  5
#define BENCH_ITERS  50

__global__ void stencil_naive(const float* __restrict__ in,
                               float* __restrict__ out,
                               const float* __restrict__ w,
                               int N, int R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R && i < N - R) {
        float acc = 0.f;
        for (int r = -R; r <= R; r++)
            acc += w[r + R] * in[i + r];
        out[i] = acc;
    }
}

__global__ void stencil_tiled(const float* __restrict__ in,
                               float* __restrict__ out,
                               const float* __restrict__ w,
                               int N, int R)
{
    extern __shared__ float tile[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x + R;

    if (gid < N) tile[lid] = in[gid];
    if (threadIdx.x < R) {
        int g = gid - R;
        tile[lid - R] = (g >= 0) ? in[g] : 0.f;
    }
    if (threadIdx.x >= blockDim.x - R) {
        int g = gid + R;
        tile[lid + R] = (g < N) ? in[g] : 0.f;
    }
    __syncthreads();

    if (gid >= R && gid < N - R) {
        float acc = 0.f;
        for (int r = -R; r <= R; r++)
            acc += w[r + R] * tile[lid + r];
        out[gid] = acc;
    }
}

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1048576;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8);
    printf("GPU: %s\n", prop.name);
    printf("Peak BW: %.1f GB/s\n", peak_bw / 1e9);
    printf("N: %d\n\n", N);

    float* h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int radii[]  = {1, 4, 8, 16};
    int bsizes[] = {64, 128, 256, 512, 1024};

    /* ── Sweep radius (block=256) ── */
    printf("RADIUS_SWEEP block=256\n");
    printf("%-6s %-12s %-12s %-10s %-14s %-14s\n",
           "R", "naive_ms", "tiled_ms", "speedup", "naive_bw_GBs", "tiled_bw_GBs");

    for (int ri = 0; ri < 4; ri++) {
        int R = radii[ri];
        int bs = 256;
        int nBlocks = (N + bs - 1) / bs;
        size_t shmem = (bs + 2*R) * sizeof(float);

        float* h_w = (float*)malloc((2*R+1)*sizeof(float));
        float* d_w;
        for (int i = 0; i < 2*R+1; i++) h_w[i] = 1.f/(2*R+1);
        CHECK(cudaMalloc(&d_w, (2*R+1)*sizeof(float)));
        CHECK(cudaMemcpy(d_w, h_w, (2*R+1)*sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t s, e;
        CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

        for (int i = 0; i < WARMUP_ITERS; i++)
            stencil_naive<<<nBlocks,bs>>>(d_in,d_out,d_w,N,R);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(s));
        for (int i = 0; i < BENCH_ITERS; i++)
            stencil_naive<<<nBlocks,bs>>>(d_in,d_out,d_w,N,R);
        CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
        float t_n; CHECK(cudaEventElapsedTime(&t_n,s,e)); t_n /= BENCH_ITERS;

        for (int i = 0; i < WARMUP_ITERS; i++)
            stencil_tiled<<<nBlocks,bs,shmem>>>(d_in,d_out,d_w,N,R);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(s));
        for (int i = 0; i < BENCH_ITERS; i++)
            stencil_tiled<<<nBlocks,bs,shmem>>>(d_in,d_out,d_w,N,R);
        CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
        float t_t; CHECK(cudaEventElapsedTime(&t_t,s,e)); t_t /= BENCH_ITERS;

        double bn = (double)N*((2*R+1)+1)*4 / (t_n*1e-3) / 1e9;
        double bt = (double)N*2*4           / (t_t*1e-3) / 1e9;

        printf("%-6d %-12.4f %-12.4f %-10.2f %-14.1f %-14.1f\n",
               R, t_n, t_t, t_n/t_t, bn, bt);

        CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
        cudaFree(d_w); free(h_w);
    }

    /* ── Sweep block size (R=8) ── */
    printf("\nBLOCK_SWEEP R=8\n");
    printf("%-10s %-12s %-12s %-10s\n", "blockSize", "naive_ms", "tiled_ms", "speedup");

    int R = 8;
    float* h_w8 = (float*)malloc((2*R+1)*sizeof(float));
    float* d_w8;
    for (int i = 0; i < 2*R+1; i++) h_w8[i] = 1.f/(2*R+1);
    CHECK(cudaMalloc(&d_w8, (2*R+1)*sizeof(float)));
    CHECK(cudaMemcpy(d_w8, h_w8, (2*R+1)*sizeof(float), cudaMemcpyHostToDevice));

    for (int bi = 0; bi < 5; bi++) {
        int bs = bsizes[bi];
        int nBlocks = (N + bs - 1) / bs;
        size_t shmem = (bs + 2*R) * sizeof(float);

        cudaEvent_t s, e;
        CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));

        for (int i = 0; i < WARMUP_ITERS; i++)
            stencil_naive<<<nBlocks,bs>>>(d_in,d_out,d_w8,N,R);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(s));
        for (int i = 0; i < BENCH_ITERS; i++)
            stencil_naive<<<nBlocks,bs>>>(d_in,d_out,d_w8,N,R);
        CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
        float t_n; CHECK(cudaEventElapsedTime(&t_n,s,e)); t_n /= BENCH_ITERS;

        for (int i = 0; i < WARMUP_ITERS; i++)
            stencil_tiled<<<nBlocks,bs,shmem>>>(d_in,d_out,d_w8,N,R);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(s));
        for (int i = 0; i < BENCH_ITERS; i++)
            stencil_tiled<<<nBlocks,bs,shmem>>>(d_in,d_out,d_w8,N,R);
        CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
        float t_t; CHECK(cudaEventElapsedTime(&t_t,s,e)); t_t /= BENCH_ITERS;

        printf("%-10d %-12.4f %-12.4f %-10.2f\n", bs, t_n, t_t, t_n/t_t);

        CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    }

    cudaFree(d_w8); cudaFree(d_in); cudaFree(d_out);
    free(h_w8); free(h_in);
    return 0;
}
CUDA_EOF

# ══════════════════════════════════════════════════════════════
# WRITE roofline.py
# ══════════════════════════════════════════════════════════════
cat > roofline.py << 'PY_EOF'
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Parse benchmark output ────────────────────────────────────
results = {}   # R -> (t_naive_ms, t_tiled_ms)
gpu_name  = "Unknown GPU"
peak_bw_gbs = None
N = 1048576

with open(sys.argv[1]) as f:
    lines = f.readlines()

mode = None
for line in lines:
    line = line.strip()
    if line.startswith("GPU:"): gpu_name  = line.split("GPU:")[1].strip()
    if line.startswith("Peak BW:"): peak_bw_gbs = float(line.split()[2])
    if line.startswith("N:"): N = int(line.split()[1])
    if line.startswith("RADIUS_SWEEP"): mode = "radius"; continue
    if line.startswith("BLOCK_SWEEP"):  mode = "block";  continue
    if mode == "radius":
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit():
            R, t_n, t_t = int(parts[0]), float(parts[1]), float(parts[2])
            results[R] = (t_n, t_t)

print(f"GPU: {gpu_name}  |  Peak BW: {peak_bw_gbs:.0f} GB/s  |  N={N}")
print(f"Parsed radii: {list(results.keys())}")

# ── GPU specs ─────────────────────────────────────────────────
GPUS = {
    "K20 (Kepler 2012)":  {"fp": 3.52e12,  "bw": 208e9,  "col": "#5577AA", "ls": ":"},
    "V100 (Volta 2017)":  {"fp": 14.0e12,  "bw": 900e9,  "col": "#7799CC", "ls": "--"},
    "A100 (Ampere 2020)": {"fp": 19.5e12,  "bw": 2000e9, "col": "#00B4D8", "ls": "-"},
    "H100 (Hopper 2022)": {"fp": 60.0e12,  "bw": 3350e9, "col": "#FFFFFF", "ls": "-"},
}

BG    = "#0D1B2A"
PANEL = "#0F2035"
MUTED = "#7A9BB5"
ORANGE = "#FF9944"
GREEN  = "#44DD88"
WHITE  = "#FFFFFF"

fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xscale("log"); ax.set_yscale("log")
ai_x = np.logspace(-1.5, 2, 500)

for name, g in GPUS.items():
    roof = np.minimum(g["bw"] * ai_x, g["fp"]) / 1e12
    ax.plot(ai_x, roof, color=g["col"], lw=2, ls=g["ls"], alpha=0.9, zorder=2)
    ax.text(88, g["fp"]/1e12*1.07, name, color=g["col"],
            fontsize=8, va="bottom", ha="right")
    ax.axvline(g["fp"]/g["bw"], color=g["col"], lw=0.5, ls=":", alpha=0.2, zorder=1)

for R, (t_n, t_t) in sorted(results.items()):
    flops = 2*(2*R+1)*N
    bn = N*((2*R+1)+1)*4
    bt = N*2*4
    ai_n = flops / bn
    ai_t = flops / bt
    pn = flops / (t_n*1e-3) / 1e12
    pt = flops / (t_t*1e-3) / 1e12

    ax.scatter(ai_n, pn, marker="o", s=90, color=ORANGE,
               edgecolors=WHITE, lw=0.8, zorder=5)
    ax.annotate(f"R={R} naive", (ai_n, pn),
                xytext=(5, -15), textcoords="offset points",
                color=ORANGE, fontsize=8)

    ax.scatter(ai_t, pt, marker="^", s=90, color=GREEN,
               edgecolors=WHITE, lw=0.8, zorder=5)
    ax.annotate(f"R={R} tiled", (ai_t, pt),
                xytext=(5, 5), textcoords="offset points",
                color=GREEN, fontsize=8)

    ax.annotate("", xy=(ai_t, pt), xytext=(ai_n, pn),
                arrowprops=dict(arrowstyle="->", color=WHITE, lw=0.8, alpha=0.4))

ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", color=WHITE, fontsize=12, labelpad=8)
ax.set_ylabel("Performance  (TFLOP/s)",              color=WHITE, fontsize=12, labelpad=8)
ax.set_title(f"Roofline — 1D Stencil  ({gpu_name}  vs GPU Generations)",
             color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax.set_xlim(0.05, 100); ax.set_ylim(0.005, 200)
ax.tick_params(colors=MUTED, labelsize=9)
for sp in ax.spines.values(): sp.set_edgecolor("#1A3858")
ax.grid(True, which="both", color="#1A3858", lw=0.5, alpha=0.7)

leg = [Line2D([0],[0], color=g["col"], lw=2, ls=g["ls"], label=n) for n,g in GPUS.items()]
leg += [
    Line2D([0],[0], marker="o", color=ORANGE, lw=0, ms=8,
           markeredgecolor=WHITE, label="Naive kernel"),
    Line2D([0],[0], marker="^", color=GREEN,  lw=0, ms=8,
           markeredgecolor=WHITE, label="Tiled kernel (shmem)"),
]
ax.legend(handles=leg, loc="upper left", framealpha=0.25,
          facecolor=PANEL, edgecolor="#1A3858", labelcolor=WHITE, fontsize=8.5)
ax.text(0.07, 20,  "memory-bound", color=MUTED, fontsize=9, rotation=36, fontstyle="italic")
ax.text(20,  0.01, "compute-bound", color=MUTED, fontsize=9, fontstyle="italic")

plt.tight_layout(pad=0.8)
out = "roofline.png"
plt.savefig(out, dpi=180, facecolor=BG, bbox_inches="tight")
print(f"Saved: {out}")
PY_EOF

# ══════════════════════════════════════════════════════════════
# COMPILE & RUN
# ══════════════════════════════════════════════════════════════
module load cuda
echo "=== nvidia-smi ===" && nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Detect GPU arch
ARCH=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader']).decode()
name = out.strip().lower()
if 'h100' in name: print('sm_90')
elif 'a100' in name: print('sm_80')
elif 'v100' in name: print('sm_70')
else: print('sm_80')
")
echo "Compiling for arch: $ARCH"
nvcc -O3 -arch=$ARCH stencil.cu -o stencil
echo "Compiled OK"
echo ""

# Run benchmark — results go into output file via SBATCH redirect
echo "=== BENCHMARK N=1048576 ==="
./stencil 1048576

# ══════════════════════════════════════════════════════════════
# GENERATE ROOFLINE PLOT
# ══════════════════════════════════════════════════════════════
module load python  2>/dev/null || true
pip3 install --quiet matplotlib numpy 2>/dev/null || true

# The SLURM output file is stencil_out_<jobid>.txt
# We point roofline.py at it after the benchmark section finishes
OUTFILE="stencil_out_${SLURM_JOB_ID}.txt"

# Wait a moment for the OS to flush the output file buffer
sleep 2

python3 roofline.py "$OUTFILE" && echo "Roofline plot saved: roofline.png" \
    || echo "Plot failed — run manually: python3 roofline.py $OUTFILE"

echo ""
echo "=== DONE ==="
echo "Files produced:"
echo "  stencil_out_${SLURM_JOB_ID}.txt  — benchmark numbers"
echo "  roofline.png                      — plot for slides"
echo ""
echo "Copy to laptop:"
echo "  scp euler:$(pwd)/roofline.png ."
echo "  scp euler:$(pwd)/stencil_out_${SLURM_JOB_ID}.txt ."
