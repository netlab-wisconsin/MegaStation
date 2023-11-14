#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void cuphy_pti_launch_snapshot_ptp(const volatile uint64_t *d_ptpreg, uint64_t *d_ptp_time_ns, uint64_t *d_gpu_time_v0_ns, uint64_t *d_gpu_time_v1_ns, cudaStream_t stream);
