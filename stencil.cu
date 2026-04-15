// ME759 Course Participation — 1D Stencil Benchmark
// Tests naive vs shared-memory-tiled kernel across radius R and block sizes
// Usage: ./stencil [N] [R] [blockSize]
//   default: N=1048576, R=8, blockSize=256

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARMUP_ITERS  5
#define BENCH_ITERS  50

// ── Naive stencil ─────────────────────────────────────────────────────────────
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

// ── Shared-memory tiled stencil ───────────────────────────────────────────────
__global__ void stencil_tiled(const float* __restrict__ in,
                               float* __restrict__ out,
                               const float* __restrict__ w,
                               int N, int R)
{
    extern __shared__ float tile[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x + R;

    // Load interior
    if (gid < N)
        tile[lid] = in[gid];

    // Load left halo
    if (threadIdx.x < R) {
        int g = gid - R;
        tile[lid - R] = (g >= 0) ? in[g] : 0.f;
    }

    // Load right halo
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

// ── CUDA error helper ─────────────────────────────────────────────────────────
#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ── Benchmark one config ──────────────────────────────────────────────────────
float benchmark(void (*kernel)(const float*, float*, const float*, int, int),
                const float* d_in, float* d_out, const float* d_w,
                int N, int R, int blockSize, bool tiled)
{
    int nBlocks = (N + blockSize - 1) / blockSize;
    size_t shmem = tiled ? (blockSize + 2 * R) * sizeof(float) : 0;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (tiled)
            stencil_tiled<<<nBlocks, blockSize, shmem>>>(d_in, d_out, d_w, N, R);
        else
            stencil_naive<<<nBlocks, blockSize>>>(d_in, d_out, d_w, N, R);
    }
    CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));

    for (int i = 0; i < BENCH_ITERS; i++) {
        if (tiled)
            stencil_tiled<<<nBlocks, blockSize, shmem>>>(d_in, d_out, d_w, N, R);
        else
            stencil_naive<<<nBlocks, blockSize>>>(d_in, d_out, d_w, N, R);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return ms / BENCH_ITERS;
}

int main(int argc, char** argv)
{
    int N         = (argc > 1) ? atoi(argv[1]) : 1048576;
    int R         = (argc > 2) ? atoi(argv[2]) : 8;
    int blockSize = (argc > 3) ? atoi(argv[3]) : 256;

    // Print GPU info
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Peak BW: %.1f GB/s\n",
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);
    printf("N=%d  R=%d  blockSize=%d\n\n", N, R, blockSize);

    // Allocate host
    float* h_in  = (float*)malloc(N * sizeof(float));
    float* h_w   = (float*)malloc((2*R+1) * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < 2*R+1; i++) h_w[i] = 1.f / (2*R+1);

    // Allocate device
    float *d_in, *d_out, *d_w;
    CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK(cudaMalloc(&d_w,   (2*R+1) * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w,  h_w,  (2*R+1) * sizeof(float), cudaMemcpyHostToDevice));

    // ── Sweep block sizes ─────────────────────────────────────────────────────
    int bsizes[] = {64, 128, 256, 512, 1024};
    int nb = 5;

    printf("%-12s  %-12s  %-12s  %-10s  %-12s  %-12s\n",
           "BlockSize", "Naive(ms)", "Tiled(ms)", "Speedup",
           "Naive BW(GB/s)", "Tiled BW(GB/s)");
    printf("%s\n", "----------------------------------------------------------------------");

    // Bytes transferred per output element:
    //   Naive: (2R+1)*4 reads + 4 write  (no reuse)
    //   Tiled: 4 read + 4 write           (loaded once per element)
    double bytes_naive = (double)N * ((2*R+1) + 1) * sizeof(float);
    double bytes_tiled = (double)N * 2 * sizeof(float);

    for (int bi = 0; bi < nb; bi++) {
        int bs = bsizes[bi];
        float t_naive = benchmark(NULL, d_in, d_out, d_w, N, R, bs, false);
        float t_tiled = benchmark(NULL, d_in, d_out, d_w, N, R, bs, true);

        double bw_naive = bytes_naive / (t_naive * 1e-3) / 1e9;
        double bw_tiled = bytes_tiled / (t_tiled * 1e-3) / 1e9;

        printf("%-12d  %-12.3f  %-12.3f  %-10.2f  %-12.1f  %-12.1f\n",
               bs, t_naive, t_tiled, t_naive / t_tiled, bw_naive, bw_tiled);
    }

    // ── Sweep radius R ────────────────────────────────────────────────────────
    printf("\n%-10s  %-12s  %-12s  %-14s  %-12s\n",
           "Radius R", "FLOPs/elem", "AI(shmem)", "Naive(ms)", "Tiled(ms)");
    printf("%s\n", "--------------------------------------------------------------");

    int radii[] = {1, 4, 8, 16};
    for (int ri = 0; ri < 4; ri++) {
        int r = radii[ri];
        // Reallocate weights for this R
        float* h_wr = (float*)malloc((2*r+1) * sizeof(float));
        float* d_wr;
        for (int i = 0; i < 2*r+1; i++) h_wr[i] = 1.f / (2*r+1);
        CHECK(cudaMalloc(&d_wr, (2*r+1) * sizeof(float)));
        CHECK(cudaMemcpy(d_wr, h_wr, (2*r+1) * sizeof(float), cudaMemcpyHostToDevice));

        // Need to re-benchmark with correct R — use lambda-style via host function
        // Warmup + time naive
        int bs = 256;
        int nBlocks = (N + bs - 1) / bs;

        // Naive
        for (int i = 0; i < WARMUP_ITERS; i++)
            stencil_naive<<<nBlocks, bs>>>(d_in, d_out, d_wr, N, r);
        cudaEvent_t s1, e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
        cudaEventRecord(s1);
        for (int i = 0; i < BENCH_ITERS; i++)
            stencil_naive<<<nBlocks, bs>>>(d_in, d_out, d_wr, N, r);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t_n; cudaEventElapsedTime(&t_n, s1, e1); t_n /= BENCH_ITERS;

        // Tiled
        size_t shmem = (bs + 2*r) * sizeof(float);
        for (int i = 0; i < WARMUP_ITERS; i++)
            stencil_tiled<<<nBlocks, bs, shmem>>>(d_in, d_out, d_wr, N, r);
        cudaEvent_t s2, e2; cudaEventCreate(&s2); cudaEventCreate(&e2);
        cudaEventRecord(s2);
        for (int i = 0; i < BENCH_ITERS; i++)
            stencil_tiled<<<nBlocks, bs, shmem>>>(d_in, d_out, d_wr, N, r);
        cudaEventRecord(e2); cudaEventSynchronize(e2);
        float t_t; cudaEventElapsedTime(&t_t, s2, e2); t_t /= BENCH_ITERS;

        int flops = 2*(2*r+1);
        double ai = (double)flops / 8.0;  // (2R+1)/4 FLOP/byte with shmem (8 bytes: 1 read + 1 write)

        printf("%-10d  %-12d  %-14.3f  %-12.3f  %-12.3f\n",
               r, flops, ai, t_n, t_t);

        cudaFree(d_wr);
        free(h_wr);
    }

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_w));
    free(h_in);
    free(h_w);

    printf("\nDone. Copy these numbers into your slides.\n");
    return 0;
}
