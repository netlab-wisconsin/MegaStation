#include <stdio.h>
#include <cuda.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>
#include <cstdio>

#include "nvlog.hpp"
#include "cuphy_pti.hpp"
#include "util.hpp"

#define TAG "CUPHY.PTI"

uint64_t get_cpu_ns()
{
    struct timespec t;
    int             ret;
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if(ret != 0)
    {
        printf("clock_gettime fail: %d\n",ret);
        exit(1);
    }
    return static_cast<uint64_t>(t.tv_nsec) + static_cast<uint64_t>(t.tv_sec) * 1000000000ULL;
}

__global__ void test_kernel(uint64_t delay_ns, cuphy_pti_activity_stats_t activity_stats)
{
    CuphyPtiRecordStartStopTimeScope scoped_record_start_stop_time(activity_stats); \
    if (blockIdx.x == 0)
    {
        uint64_t t_now = __globaltimer();
        uint64_t t_stop = t_now + delay_ns;

        while (__globaltimer() < t_stop);
    }
}

void usage(const char* progname)
{
    printf("Usage: %s <NIC PCI Address>\n",progname);
    printf("\n  example: %s 0000:cc:00.1\n",progname);
    exit(1);
}

int main(int argc, const char** argv)
{

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "cuphy_pti_example.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
    nvlog_fmtlog_thread_init();

    cudaSetDevice(0);
    if (argc != 2) usage(argv[0]);

    cuphy_pti_init(argv[1]);

    CuphyPtiSetIndexScope cuphy_pti_index_scope(0);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    struct cuphy_pti_activity_stats_t activity_stats;
    cuphy_pti_get_record_activity(activity_stats,CUPHY_PTI_ACTIVITY_PREPREP);

    test_kernel<<<10,256,0,stream>>>(1000000000,activity_stats);
    cudaDeviceSynchronize();

    struct cuphy_pti_all_stats_t *stats;
    cuphy_pti_get_record_all_activities(stats);
    NVLOGW_FMT(TAG,"GPU Start Time: {}  GPU Stop Time: {}  Delta: {}",stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREPREP],stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREPREP], stats->dh_gpu_stop_times[0]-stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREPREP]);

    for (int k=0; k<10; k++)
    {
        cuphy_pti_calibrate_gpu_timer(stream);
        usleep(1000000);
    }

    stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREPREP] = 0;
    stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREPREP] = 0;
    test_kernel<<<10,256,0,stream>>>(1000000000,activity_stats);
    cudaDeviceSynchronize();
    NVLOGW_FMT(TAG,"GPU Start Time: {}  GPU Stop Time: {}  Delta: {}",stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREPREP],stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREPREP], stats->dh_gpu_stop_times[0]-stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREPREP]);
}
