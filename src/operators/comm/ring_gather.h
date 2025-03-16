#include <cuda.h>
#include <cuda_runtime.h>

#include "range.h"

inline void ringGather(char *next, const char *curt, int *nx_flag, int *ct_flag,
                       const int rank_id, const RankRange &ranks,
                       const std::size_t batch_sz, cudaStream_t stream) {
  int cur_rank = rank_id;
  int total_ranks = ranks.size();

  for (int j = 0; j < total_ranks - 1; j++) {
    int curt_len = (ranks[cur_rank].end - ranks[cur_rank].start) * batch_sz;
    int curt_start = ranks[cur_rank].start * batch_sz;

    cudaMemcpyAsync(next + ranks[cur_rank].start, curt + curt_start, curt_len,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync((void *)nx_flag, j + 1, sizeof(char), stream);
    cuStreamWaitValue32(stream, CUdeviceptr(ct_flag), j + 1,
                        CU_STREAM_WAIT_VALUE_GEQ);

    cur_rank = (cur_rank + total_ranks - 1) % total_ranks;
  }
}