#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "cuphy_pti.hpp"

__host__ __device__ inline uint64_t bswap64(uint64_t x)
{
    return   ((x & 0xff00000000000000ull) >> 56)
           | ((x & 0x00ff000000000000ull) >> 40)
           | ((x & 0x0000ff0000000000ull) >> 24)
           | ((x & 0x000000ff00000000ull) >> 8)
           | ((x & 0x00000000ff000000ull) << 8)
           | ((x & 0x0000000000ff0000ull) << 24)
           | ((x & 0x000000000000ff00ull) << 40)
           | ((x & 0x00000000000000ffull) << 56);
}

__host__ __device__ uint64_t convert_ptpreg_to_ns(uint64_t ptpreg)
{
   uint64_t be_ptpreg = bswap64(ptpreg);
   uint64_t tv_sec = be_ptpreg >> 32;
   uint64_t tv_nsec = be_ptpreg & 0xFFFFFFFF;
   return tv_nsec + tv_sec * 1000000000ULL;
}

__global__ void snapshot_ptp_clock(const volatile uint64_t *d_ptpreg, uint64_t *d_ptp_time_ns, uint64_t *d_gpu_time_v0_ns, uint64_t *d_gpu_time_v1_ns)
{
    uint64_t gpu_tick_v0 = __globaltimer();
    __syncwarp();
    uint64_t ptp_tick_v0 = *d_ptpreg;
    __syncwarp();
    uint64_t gpu_tick_v1 = __globaltimer();

    // Convert BE to LE
    ptp_tick_v0 = convert_ptpreg_to_ns(ptp_tick_v0);

    // Update outputs
    *d_ptp_time_ns = ptp_tick_v0;
    *d_gpu_time_v0_ns = gpu_tick_v0;
    *d_gpu_time_v1_ns = gpu_tick_v1;
}

void cuphy_pti_launch_snapshot_ptp(const volatile uint64_t *d_ptpreg, uint64_t *d_ptp_time_ns, uint64_t *d_gpu_time_v0_ns, uint64_t *d_gpu_time_v1_ns, cudaStream_t stream)
{
    snapshot_ptp_clock<<<1,1,0,stream>>>(d_ptpreg, d_ptp_time_ns, d_gpu_time_v0_ns, d_gpu_time_v1_ns);
}
