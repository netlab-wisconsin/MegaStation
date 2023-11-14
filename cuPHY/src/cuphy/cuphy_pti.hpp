#pragma once
#include "nvlog_fmt.hpp"

#define CHECK_CUDA(result)                        \
    if((cudaError_t)result != cudaSuccess)        \
    {                                             \
        NVLOGF_FMT("CUPHY.PTI",                   \
                AERIAL_INTERNAL_EVENT,            \
                "CUDA Runtime Error: {}:{}:{}",   \
                __FILE__,                         \
                __LINE__,                         \
                cudaGetErrorString(result));      \
    }


constexpr int CUPHY_PTI_INDEX_MAX = 20;

typedef enum
{
    CUPHY_PTI_ACTIVITY_PREPREP = 0,
    CUPHY_PTI_ACTIVITY_PREP = 1,
    CUPHY_PTI_ACTIVITY_TRIGGER = 2,
    CUPHY_PTI_ACTIVITY_COUNT
} cuphy_pti_activity_t;

constexpr int CUPHY_PTI_ACTIVITIES_MAX = CUPHY_PTI_ACTIVITY_COUNT;

struct cuphy_pti_all_stats_t
{
    uint64_t *dh_gpu_start_times;
    uint64_t *dh_gpu_stop_times;
    uint32_t *d_cta_counts;
};

struct cuphy_pti_activity_stats_t
{
    uint64_t *dh_gpu_start_time;
    uint64_t *dh_gpu_stop_time;
    uint32_t *d_cta_count;
};

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

void cuphy_pti_init(const char* nic_pci_addr);
void cuphy_pti_set_record_index(int record_index);
int cuphy_pti_get_record_index(void);
void cuphy_pti_calibrate_gpu_timer(cudaStream_t stream);

#define cuphy_pti_get_record_all_activities(x) do \
{ \
    if (cuphy_pti_get_record_index() < 0) \
    { \
        NVLOGF_FMT("CUPHY.PTI",AERIAL_INTERNAL_EVENT,"Called cuphy_pti_get_record() without previously setting the record index for this thread: {}:{}",__FILE__,__LINE__); \
    } \
    x = _cuphy_pti_get_record_all_activities(); \
} while (0)

#define cuphy_pti_get_record_activity(x,activity) do \
{ \
    if (cuphy_pti_get_record_index() < 0) \
    { \
        NVLOGF_FMT("CUPHY.PTI",AERIAL_INTERNAL_EVENT,"Called cuphy_pti_get_record() without previously setting the record index for this thread: {}:{}",__FILE__,__LINE__); \
    } \
    _cuphy_pti_get_record_activity(x,activity); \
} while (0)

struct cuphy_pti_all_stats_t* _cuphy_pti_get_record_all_activities();
void _cuphy_pti_get_record_activity(struct cuphy_pti_activity_stats_t& activity_stats, cuphy_pti_activity_t activity);

#if defined(__cplusplus)
} /* extern "C" */

class CuphyPtiSetIndexScope
{
public:
   CuphyPtiSetIndexScope(int record_index)
   {
      cuphy_pti_set_record_index(record_index);
   };

   ~CuphyPtiSetIndexScope()
   {
      cuphy_pti_set_record_index(-1);
   };
};

#ifdef __CUDACC__

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    asm volatile ( "mov.u64 %0, %globaltimer;"   : "=l"( globaltimer ) );
    return globaltimer;
}

__device__ __forceinline__ void save_start_time(uint32_t *d_cta_count, uint64_t *d_start_time)
{
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
    {
        int current_cta = atomicAdd(d_cta_count,1);
        if (current_cta == 0)
        {
            *d_start_time = __globaltimer();
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void save_stop_time(uint32_t *d_cta_count, uint64_t *d_stop_time)
{
    __syncthreads();
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
    {
        int current_cta = atomicAdd(d_cta_count,1);
        int max_ctas = gridDim.x * gridDim.y * gridDim.z;
        if (current_cta == (2*max_ctas - 1))
        {
            *d_stop_time = __globaltimer();
	    *d_cta_count = 0;
        }
    }
}

class CuphyPtiRecordStartStopTimeScope
{
private:
    uint64_t *m_dh_gpu_stop_time;
    uint32_t *m_d_cta_count;

public:
    __device__ CuphyPtiRecordStartStopTimeScope(struct cuphy_pti_activity_stats_t& activity_stats) :
        m_dh_gpu_stop_time(activity_stats.dh_gpu_stop_time),
        m_d_cta_count(activity_stats.d_cta_count)
    {
#ifdef CUPHY_PTI_ENABLE_TRACING
        save_start_time(activity_stats.d_cta_count, activity_stats.dh_gpu_start_time);
#endif
    };

    __device__ ~CuphyPtiRecordStartStopTimeScope()
    {
#ifdef CUPHY_PTI_ENABLE_TRACING
        save_stop_time(m_d_cta_count, m_dh_gpu_stop_time);
#endif
    };
};

#endif // __CUDACC__

#endif /* defined(__cplusplus) */
