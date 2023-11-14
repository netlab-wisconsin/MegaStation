/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>  /* For SYS_xxx definitions */
#include <syscall.h> /* For SYS_xxx definitions */
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "util.hpp"

#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

#include <chrono>
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

int writeProbe(char const* pFName, void const* pBuffer, size_t nBytes)
{
    FILE*  pFileHandle;
    size_t nWritten;

    /* Write debug files into out directory, if out does not exist already then create it */
    char const* pDirName = "out";
    DIR*        pDir;
    if((pDir = opendir(pDirName)) == NULL)
    {
        if(ENOENT == errno)
        {
            if(mkdir(pDirName, 0777) != 0)
            {
                printf("writeDebugFile: failed to create directory, error: %s\n", strerror(errno));
                return -2;
            }
        }
        else
        {
            printf("writeDebugFile: failed to open existing directory, error: %s\n", strerror(errno));
            return -3;
        }
    }

    /* Append the directory name, file name, instance index and UE index */
    std::string fName = std::string(pDirName) + "/" + std::string(pFName) + "_l.out";

    if(NULL == (pFileHandle = fopen(fName.c_str(), "ab")))
    {
        closedir(pDir);
        printf("Problem opening file, error: %s\n", strerror(errno));
        return -4;
    }

    nWritten = fwrite(pBuffer, 1, (size_t)nBytes, pFileHandle);
    printf("Wrote %d bytes, asked to write %d bytes, error: %s\n", (int)nWritten, (int)nBytes, strerror(errno));
    fclose(pFileHandle);
    closedir(pDir);

    return 0;
}

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_srs_ch_est [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename     Input HDF5 filename, which must contain the following datasets:\n");
    printf("                           DataRx         : received data (frequency-time) to be equalized\n");
    printf("                           FreqInterpCoefs: frequency interpolation filter coefficients used in SRS channel estimation\n");
    printf("    -r  # of iterations    Number of run iterations to run\n");    
    printf("    -h                     Display usage information\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    --H                    0         : No FP16\n");
    printf("                           1(default): FP16 format used for received data samples only\n");
    printf("                           2         : FP16 format used for all front end params\n");
    printf("    --M                    0(default): Descriptor based SRS channel estimator\n");
    printf("    --I                    0(default): Disable iterative processing in the kernel\n");
    printf("                           1         : Enable iterative processing in the kernel\n");
#if 0
    printf("    --M                    0         : Descriptor based SRS channel estimator\n");
    printf("                           1(default): Descriptor based SRS channel estimator as graph node\n");
#endif
}

void setCpuPrio(std::string &name, int instIdx, int cpuId, int policy)
{
    // schdPolicy = SCHED_FIFO // SCHED_RR
    int prio   = sched_get_priority_max(policy);

    //------------------------------------------------------------------
    // Bump up prio
    pid_t       pid = (pid_t)syscall(SYS_gettid);
    sched_param schdPrm;
    schdPrm.sched_priority = prio;

    // pid_t pid = (pid_t) syscall(SYS_gettid);
    int schdSetRet = sched_setscheduler(pid, policy, &schdPrm);
    if(0 == schdSetRet)
    {
        printf("%s Pipeline[%d]: pid %d policy %d prio %d\n", name.c_str(), instIdx, pid, policy, prio);
    }
    else
    {
        printf("%s Pipeline[%d]: Failed to set scheduling algo pid %d, prio %d, return code %d: err %s\n",
               name.c_str(),
               instIdx,
               pid,
               prio,
               schdSetRet,
               strerror(errno));
    }

    //------------------------------------------------------------------
    // Set thread affinity to specified CPU
    
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    CPU_SET(cpuId, &cpuSet);
    // printf("%s Pipeline[%d]: setting affinity of pipeline %d (pid %d) to CPU Id %d\n", m_name.c_str(), instIdx, pid, cpuId);

    int affinitySetRet = sched_setaffinity(pid, sizeof(cpuSet), &cpuSet);
    if(0 == affinitySetRet)
    {
        printf("%s Pipeline[%d]: pid %d set affinity to CPU %d\n", name.c_str(), instIdx, pid, cpuId);
    }
    else
    {
        printf("%s Pipeline[%d]: failed to set affinity pid %d to CPU %d, return code %d err %s\n",
               name.c_str(),
               instIdx,
               pid,
               cpuId,
               affinitySetRet,
               strerror(errno));
    }    
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename;
        std::string outputFilename;
        uint32_t    fp16Mode     = 0xBAD;
        uint32_t    srsChEstMode = 0xBAD;
        uint32_t    iterMode     = 0xBAD;
        uint32_t    nIterations  = 1000;
        uint32_t    delayMs      = 250;
        int         gpuId        = 0;
        int         debugMode    = 0;

        cudaStream_t cuStream;
        cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking);

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'o':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: No output file name given");
                    }
                    outputFilename.assign(argv[iArg++]);
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIterations)) || ((nIterations <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of run iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'H':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (3 <= fp16Mode))
                        {
                            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid FP16 mode {}", fp16Mode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'M':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &srsChEstMode)) || (1 <= srsChEstMode))
                        {
                            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid Channel estimator mode {}", srsChEstMode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'I':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &iterMode)) || (2 <= iterMode))
                        {
                            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid iteration mode {}", iterMode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'D':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &debugMode)) || (3 <= debugMode))
                        {
                            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid debug mode {}", debugMode);
                            exit(1);
                        }
                        ++iArg;
                        break;                        

                    default:
                        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                    }
                    break;
                default:
                    NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }
        //------------------------------------------------------------------
        // Open the input file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

        if(0xBAD == iterMode) iterMode = 0;
        if(0xBAD == srsChEstMode) srsChEstMode = 0;
        if(0xBAD == fp16Mode) fp16Mode = 1;

        // Check FP16 mode of operation
        bool isDataFp16    = true;
        bool isChannelFp16 = false;
        switch(fp16Mode)
        {
        case 0:
            isDataFp16    = false;
            isChannelFp16 = false;
            break;
        case 1:
            isDataFp16    = true;
            isChannelFp16 = false;
            break;
        case 2:
            isDataFp16    = true;
            isChannelFp16 = true;
            break;
        default:
            isDataFp16    = true;
            isChannelFp16 = false;
            break;
        }
        cuphyDataType_t feDataType     = isDataFp16 ? CUPHY_R_16F : CUPHY_R_32F;
        cuphyDataType_t feCplxDataType = isDataFp16 ? CUPHY_C_16F : CUPHY_C_32F;

        cuphyDataType_t feChannelType     = isChannelFp16 ? CUPHY_R_16F : CUPHY_R_32F;
        cuphyDataType_t feCplxChannelType = isChannelFp16 ? CUPHY_C_16F : CUPHY_C_32F;

        //------------------------------------------------------------------
        // Parameters assumed and derived from the input data
        constexpr uint8_t N_TONES_PER_PRB              = 12;
        constexpr uint8_t N_DMRS_GRIDS_PER_PRB         = 2;
        constexpr uint8_t N_DMRS_GRID_TONES_PER_PRB    = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
        constexpr uint8_t N_INTERP_DMRS_TONES_PER_GRID = N_TONES_PER_PRB;

        cuphySrsChEstDynPrms_t srsChEstDynPrms{0};
        srsChEstDynPrms.enIter          = iterMode;
        srsChEstDynPrms.nBSAnts         = 0;
        srsChEstDynPrms.nLayers         = 0;
        srsChEstDynPrms.nPrb            = 0;
        srsChEstDynPrms.scsKHz          = 0;
        srsChEstDynPrms.nCycShifts      = 0;
        srsChEstDynPrms.nCombs          = 0;
        srsChEstDynPrms.srsSymLocBmsk   = 0;
        srsChEstDynPrms.nZc             = 0;
        srsChEstDynPrms.zcSeqNum        = 0;
        srsChEstDynPrms.delaySpreadSecs = 0.0f;

        cuphy::disable_hdf5_error_print(); // Temporarily disable HDF5 stderr printing

        try
        {
            cuphy::cuphyHDF5_struct srsChEstCfg = cuphy::get_HDF5_struct(fInput, "srsChEstCfg");
            srsChEstDynPrms.nBSAnts             = srsChEstCfg.get_value_as<uint16_t>("nRxAnts");
            srsChEstDynPrms.nLayers             = srsChEstCfg.get_value_as<uint8_t>("nLayers");
            srsChEstDynPrms.nPrb                = srsChEstCfg.get_value_as<uint16_t>("nPrb");
            srsChEstDynPrms.scsKHz              = srsChEstCfg.get_value_as<uint16_t>("scsKHz");
            srsChEstDynPrms.nCycShifts          = srsChEstCfg.get_value_as<uint8_t>("nCyclicShifts");
            srsChEstDynPrms.nCombs              = srsChEstCfg.get_value_as<uint8_t>("nCombs");
            srsChEstDynPrms.nZc                 = srsChEstCfg.get_value_as<uint16_t>("nZc");
            srsChEstDynPrms.zcSeqNum            = srsChEstCfg.get_value_as<uint8_t>("zcSeqNum");
            srsChEstDynPrms.srsSymLocBmsk       = srsChEstCfg.get_value_as<uint16_t>("srsSymPosBmsk");
            srsChEstDynPrms.delaySpreadSecs     = srsChEstCfg.get_value_as<float>("delaySpreadSecs");
        }
        catch(const std::exception& exc)
        {
            printf("%s\n", exc.what());
            throw exc;
            // Continue using command line arguments if the input file does not
            // have a config struct.
        }
        cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

        uint8_t nSrsSyms = __builtin_popcount(srsChEstDynPrms.srsSymLocBmsk);

        printf("Config parameters:\n");
        printf("---------------------------------------------------------------\n");
        printf("enIter          : %i\n", srsChEstDynPrms.enIter);
        printf("nBSAnts         : %i\n", srsChEstDynPrms.nBSAnts);
        printf("nLayers         : %i\n", srsChEstDynPrms.nLayers);
        printf("nPrb            : %i\n", srsChEstDynPrms.nPrb);
        printf("scs (kHz)       : %i\n", srsChEstDynPrms.scsKHz);
        printf("nCyclicShifts   : %i\n", srsChEstDynPrms.nCycShifts);
        printf("nCombs          : %i\n", srsChEstDynPrms.nCombs);
        printf("srsSymLocBmsk   : 0x%x\n", srsChEstDynPrms.srsSymLocBmsk);
        printf("nSrsSyms        : %i\n", nSrsSyms);
        printf("nZc             : %i\n", srsChEstDynPrms.nZc);
        printf("zcSeqNum        : %i\n", srsChEstDynPrms.zcSeqNum);
        printf("delaySpread (s) : %e\n", srsChEstDynPrms.delaySpreadSecs);

        //------------------------------------------------------------------
        // Allocate tensors in device memory

        // clang-format off
        cuphy::tensor_device tDataRx          = cuphy::tensor_from_dataset(fInput.open_dataset("DataRx")         , feCplxDataType, cuphy::tensor_flags::align_tight);
        cuphy::tensor_device tFreqInterpCoefs = cuphy::tensor_from_dataset(fInput.open_dataset("FreqInterpCoefs"), feChannelType , cuphy::tensor_flags::align_tight);
        // clang-format on

        printf("Input tensors:\n");
        printf("---------------------------------------------------------------\n");
        printf("DataRx         : %s\n", tDataRx.desc().get_info().to_string(false).c_str());
        printf("FreqInterpCoefs: %s\n", tFreqInterpCoefs.desc().get_info().to_string(false).c_str());

        //------------------------------------------------------------------
        // Allocate an output tensor in device memory
        uint16_t nSrsChEst = srsChEstDynPrms.nPrb / 2;
#if 1 // higher write efficiency: (N_SRS_CH_EST_IN_FREQ, N_BS_ANTS, N_LAYERS)
        cuphy::tensor_device tHEst(CUPHY_C_32F,
                                   nSrsChEst,
                                   srsChEstDynPrms.nBSAnts,
                                   srsChEstDynPrms.nCycShifts,
                                   nSrsSyms * srsChEstDynPrms.nCombs,
                                   cuphy::tensor_flags::align_tight);
#else // BFW friendly writes: (N_BS_ANTS, N_SRS_CH_EST_IN_FREQ, N_LAYERS)
        cuphy::tensor_device tHEst(CUPHY_C_32F,
                                   srsChEstDynPrms.nBSAnts,
                                   nSrsChEst,
                                   srsChEstDynPrms.nCycShifts,
                                   nSrsSyms * srsChEstDynPrms.nCombs,
                                   cuphy::tensor_flags::align_tight);
#endif

#if 0
        // ShiftSeq or ShiftSeq phase
        cuphy::tensor_device tDbg(CUPHY_C_32F,
                                  srsChEstDynPrms.nPrb * CUPHY_N_TONES_PER_PRB / srsChEstDynPrms.nCombs,
                                  cuphy::tensor_flags::align_tight);
#elif 0
        // Interpolator input (post shiftSeq application)
        cuphy::tensor_device tDbg(CUPHY_C_32F,
                                  4 * CUPHY_N_TONES_PER_PRB / srsChEstDynPrms.nCombs, // # of COMB tones per PRB group, PRB group size 4
                                  srsChEstDynPrms.nPrb / 4,                           // # of PRB groups
                                  srsChEstDynPrms.nCycShifts,
                                  srsChEstDynPrms.nCombs,
                                  cuphy::tensor_flags::align_tight);
#elif 1
        // Interpolator output
        cuphy::tensor_device tDbg(CUPHY_C_32F,
                                  (4 / 2),                  // # of COMB tones per PRB group, PRB group size 4, 2 SRS chest per PRB group
                                  srsChEstDynPrms.nPrb / 4, // # of PRB groups
                                  srsChEstDynPrms.nCycShifts,
                                  srsChEstDynPrms.nCombs,
                                  cuphy::tensor_flags::align_tight);
#elif 0
        // UnShiftSeq
        cuphy::tensor_device tDbg(CUPHY_C_32F,
                                  srsChEstDynPrms.nPrb / 2, // 2 SRS chest per PRB
                                  srsChEstDynPrms.nCombs,
                                  cuphy::tensor_flags::align_tight);
#else
        // Interpolator output
        cuphy::tensor_device tDbg(CUPHY_C_32F,
                                  (2),                      // 2 SRS chest per PRB group
                                  srsChEstDynPrms.nPrb / 4, // # of PRB groups
                                  srsChEstDynPrms.nCycShifts,
                                  srsChEstDynPrms.nCombs,
                                  cuphy::tensor_flags::align_tight);
#endif

        printf("Tensor layout:\n");
        printf("---------------------------------------------------------------\n");
        printf("tDataRx         : addr: %p, %s, size: %.1f kB\n",
               tDataRx.addr(),
               tDataRx.desc().get_info().to_string().c_str(),
               tDataRx.desc().get_size_in_bytes() / 1024.0);
        printf("tFreqInterpCoefs: addr: %p, %s, size: %.1f kB\n",
               tFreqInterpCoefs.addr(),
               tFreqInterpCoefs.desc().get_info().to_string().c_str(),
               tFreqInterpCoefs.desc().get_size_in_bytes() / 1024.0);
        printf("tHEst           : addr: %p, %s, size: %.1f kB\n",
               tHEst.addr(),
               tHEst.desc().get_info().to_string().c_str(),
               tHEst.desc().get_size_in_bytes() / 1024.0);
        printf("tDbg            : addr: %p, %s, size: %.1f kB\n",
               tDbg.addr(),
               tDbg.desc().get_info().to_string().c_str(),
               tDbg.desc().get_size_in_bytes() / 1024.0);

        //------------------------------------------------------------------
        // Ensure number of buffers exceeds L2cache size
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));
        uint32_t targetBufLen = 2*deviceProp.l2CacheSize;
        uint32_t bufLen = tDataRx.desc().get_size_in_bytes() + tHEst.desc().get_size_in_bytes();        
        auto div_round_up = [] (uint32_t val, uint32_t divide_by) { return ((val + (divide_by - 1)) / divide_by); };
        uint32_t nBufs = (bufLen > targetBufLen) ? 1 : div_round_up(targetBufLen, bufLen);

        std::vector<cuphy::tensor_device> tDataRxVec(nBufs, tDataRx);
        std::vector<cuphy::tensor_device> tHEstVec(nBufs, tHEst);
        // vectors above invoke tensor copy constructor, tensor copy constructor's invocation of convert function uses default stream
        cudaDeviceSynchronize();

        if((0 == srsChEstMode) || (1 == srsChEstMode))
        {
            std::array<cudaStream_t, CUPHY_SRS_CH_EST_N_HET_CFG> strmVec{{cuStream}};

            //------------------------------------------------------------------
            // Descriptor allocation
            size_t        statDescrSizeBytes, statDescrAlignBytes, dynDescrSizeBytes, dynDescrAlignBytes;
            cuphyStatus_t statusGetWorkspaceSize = cuphySrsChEstGetDescrInfo(&statDescrSizeBytes,
                                                                             &statDescrAlignBytes,
                                                                             &dynDescrSizeBytes,
                                                                             &dynDescrAlignBytes);
            if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

            printf("DescriptorInfo: statDescrSizeBytes %zu statDescrAlignBytes %zu dynDescrSizeBytes %zu dynDescrAlignBytes %zu\n", statDescrSizeBytes, statDescrAlignBytes, dynDescrSizeBytes, dynDescrAlignBytes);

            cuphy::buffer<uint8_t, cuphy::pinned_alloc> statDescrBufCpu(statDescrSizeBytes);
            cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);

            cuphy::buffer<uint8_t, cuphy::device_alloc> statDescrBufGpu(statDescrSizeBytes);
            cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

            //------------------------------------------------------------------
            bool                enableCpuToGpuDescrAsyncCpy = true; // false;
            cuphySrsChEstHndl_t srsChEstHndl;

            uint32_t hetCfgIdx = 0;

            cuphyTensorPrm_t tPrmInterpCoef{.desc = tFreqInterpCoefs.desc().handle(), .pAddr = tFreqInterpCoefs.addr()};
            cuphyStatus_t statusCreate = cuphyCreateSrsChEst(&srsChEstHndl,
                                                             &tPrmInterpCoef,
                                                             enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                             static_cast<void*>(statDescrBufCpu.addr()),
                                                             static_cast<void*>(statDescrBufGpu.addr()),
                                                             strmVec[hetCfgIdx]);

            if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

            // Copy descriptor
            if(!enableCpuToGpuDescrAsyncCpy) cudaMemcpyAsync(statDescrBufGpu.addr(), statDescrBufCpu.addr(), statDescrSizeBytes, cudaMemcpyHostToDevice, strmVec[hetCfgIdx]);
            cudaStreamSynchronize(strmVec[hetCfgIdx]);
                        
            std::vector<cuphy::event_timer> cuEvtTimer1Vec(nIterations);
            std::vector<cuphy::event_timer> cuEvtTimer2Vec(nIterations);
            std::vector<duration<float, std::micro>> durationSetupVec(nIterations);
            
            //------------------------------------------------------------------
            // Ensure all  prior work on the GPU is completed before launching delay kernel (free up the internal FIFOs to accomodate as much
            // of the workload burst that follows the delay kernel)
            cudaDeviceSynchronize();
            
            //------------------------------------------------------------------
            // Bump up CPU thread prio
            if(debugMode > 1)
            {
                std::string pipelineName("SRS");
                int cpuId = 0;
                setCpuPrio(pipelineName, 0, cpuId, SCHED_FIFO);
            }

            //------------------------------------------------------------------
            // Insert delay kernel
            gpu_ms_delay(delayMs, gpuId, strmVec[hetCfgIdx]);

            TimePoint startWallClock = Clock::now();
            
            //------------------------------------------------------------------
            // Burst out all the iterations
            for(uint32_t i = 0; i < nIterations; ++i)
            {
                uint32_t bufIdx = i % nBufs;
#if 1              
                // clang-format off
                cuphyTensorPrm_t tPrmDataRx{.desc = tDataRxVec[bufIdx].desc().handle(), .pAddr = tDataRxVec[bufIdx].addr()};
                cuphyTensorPrm_t tPrmHEst  {.desc = tHEstVec[bufIdx].desc().handle()  , .pAddr = tHEstVec[bufIdx].addr()};
                cuphyTensorPrm_t tPrmDbg   {.desc = tDbg.desc().handle()              , .pAddr = tDbg.addr()};
                // clang-format on
#else
                // clang-format off
                cuphyTensorPrm_t tPrmDataRx{.desc = tDataRx.desc().handle(), .pAddr = tDataRx.addr()};
                cuphyTensorPrm_t tPrmHEst  {.desc = tHEst.desc().handle()  , .pAddr = tHEst.addr()};
                cuphyTensorPrm_t tPrmDbg   {.desc = tDbg.desc().handle()   , .pAddr = tDbg.addr()};
                // clang-format on
#endif

#if 0            
                cudaGraph_t                    graph;
                cudaGraphExec_t                graphExec;
                std::array<cudaGraphNode_t, 1> srsChEstGraphNode; // only 1 graph node for channel est unit test
                std::vector<cudaGraphNode_t>   nodeDependencies;
                uint8_t                        graphNodeUpdateStatus = 0; // false;
                cuphyPuschRxFeGraphNodePrms_t  graphNodePrms         = {.pCreatePrms = nullptr, .pUpdatePrms = nullptr, .pSuccess = &graphNodeUpdateStatus};
#endif
                //------------------------------------------------------------------
                // Non-graph approach
                if(0 == srsChEstMode)
                {
                    if(enableCpuToGpuDescrAsyncCpy) cuEvtTimer1Vec[i].record_begin(strmVec[hetCfgIdx]);
                    TimePoint startWallClkSetup = Clock::now();
                    cuphyStatus_t statusSetup = cuphySetupSrsChEst(srsChEstHndl,
                                                                   &srsChEstDynPrms,
                                                                   &tPrmDataRx,
									   &tPrmHEst,
									   &tPrmDbg,
									   enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
									   static_cast<void*>(dynDescrBufCpu.addr()),
									   static_cast<void*>(dynDescrBufGpu.addr()),
									   strmVec[hetCfgIdx]);
			    TimePoint stopWallClkSetup = Clock::now();
			    if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
			    if(enableCpuToGpuDescrAsyncCpy)  cuEvtTimer1Vec[i].record_end(strmVec[hetCfgIdx]);
			    
			    durationSetupVec[i] = (stopWallClkSetup - startWallClkSetup);
			}
#if 0            
			//------------------------------------------------------------------
			// Graph node creation setup
			else if(1 == srsChEstMode)
			{
			    nodeDependencies.clear();
	    
			    CUDA_CHECK(cudaGraphCreate(&graph, 0));
			    cuphyPuschRxFeCreateGraphNodePrms_t createGraphNodePrms{
				.pGraph        = &graph,
				.pNode         = srsChEstGraphNode.data(),
				.pDependencies = nodeDependencies.data(),
				.nDependencies = nodeDependencies.size()};
	    
			    graphNodePrms = cuphyPuschRxFeGraphNodePrms_t{
				.pCreatePrms = &createGraphNodePrms,
				.pUpdatePrms = nullptr,
				.pSuccess    = &graphNodeUpdateStatus};
	    
			    cuphyStatus_t statusSetup = cuphySetupSrsChEst(srsChEstHndl,
									   cellId,
									   slotNumber,
									   nBSAnts,
									   nLayers,
									   nDMRSSyms,
									   nDMRSGridsPerPRB,
									   activeDMRSGridBmsk,
									   nTotalDMRSPRB,
									   nTotalDataPRB,
									   Nh,
									   &tPrmDataRx,
									   &tPrmHEst,
                                                                   &tPrmDbg,
                                                                   enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                   static_cast<void*>(dynDescrBufCpu.addr()),
                                                                   static_cast<void*>(dynDescrBufGpu.addr()),
                                                                   strmVec[hetCfgIdx],
                                                                   &graphNodePrms);
    
                    if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
                }
#endif
                //------------------------------------------------------------------
                // Copy descriptor if not done within the GPU
                if(!enableCpuToGpuDescrAsyncCpy) 
                {
                    cuEvtTimer1Vec[i].record_begin(strmVec[hetCfgIdx]);
                    cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, strmVec[hetCfgIdx]);
                    cuEvtTimer1Vec[i].record_end(strmVec[hetCfgIdx]);
                }

#if 0            
                //------------------------------------------------------------------
                // Graph node update setup
                if(1 == srsChEstMode)
                {
                    // CUDA_CHECK(cudaGraphAddKernelNode(&srsChEstGraphNode[0], graph, NULL, 0, &srsChEstGraphNodePrms[hetCfgIdx]));
    
                    // cudaGraphExecKernelNodeSetParams step is not needed since there is no update from kernel node addition to node update
                    // CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graphExec, srsChEstGraphNode[0], &srsChEstGraphNodePrms[hetCfgIdx]));
    
                    printf("GraphNodeCreateStatus %d\n", graphNodeUpdateStatus);
    
                    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
                    graphNodeUpdateStatus = 0;// false;
    
                    cuphyPuschRxFeUpdateGraphNodePrms_t updateGraphNodePrms{
                        .pGraphExec = &graphExec,
                        .pNode      = srsChEstGraphNode.data()};
    
                    graphNodePrms = cuphyPuschRxFeGraphNodePrms_t{
                        .pCreatePrms = nullptr,
                        .pUpdatePrms = &updateGraphNodePrms,
                        .pSuccess    = &graphNodeUpdateStatus};
    
                    cuphyStatus_t statusSetup = cuphySetupSrsChEst(srsChEstHndl,
                                                                   cellId,
                                                                   slotNumber,
                                                                   nBSAnts,
                                                                   nLayers,
                                                                   nDMRSSyms,
                                                                   nDMRSGridsPerPRB,
                                                                   activeDMRSGridBmsk,
                                                                   nTotalDMRSPRB,
                                                                   nTotalDataPRB,
                                                                   Nh,
                                                                   &tPrmDataRx,
                                                                   &tPrmHEst,
                                                                   &tPrmDbg,
                                                                   enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                   static_cast<void*>(dynDescrBufCpu.addr()),
                                                                   static_cast<void*>(dynDescrBufGpu.addr()),
                                                                   strmVec[hetCfgIdx],
                                                                   &graphNodePrms);
    
                    if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
    
                    printf("GraphNodeUpdateStatus %d\n", graphNodeUpdateStatus);
    
                    CUDA_CHECK(cudaGraphLaunch(graphExec, strmVec[hetCfgIdx]));
                }
                else
#endif
                //------------------------------------------------------------------                
                // Stream based launch
                {
                    cuEvtTimer2Vec[i].record_begin(strmVec[hetCfgIdx]);

                    cuphyStatus_t statusRun = cuphyRunSrsChEst(srsChEstHndl, strmVec[hetCfgIdx]);
                    if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

                    cuEvtTimer2Vec[i].record_end(strmVec[hetCfgIdx]);
                }

            }
            
            //------------------------------------------------------------------
            // Calculate and display execution time

            float elapsedH2DCpyTimeUs = 0.0f;
            for(auto& cuEvtTimer : cuEvtTimer1Vec)
            {
                cuEvtTimer.synchronize();
                elapsedH2DCpyTimeUs += cuEvtTimer.elapsed_time_ms()*1000;
            }

            float durationSetupUs = 0.0f;
            for(auto& durationSetup : durationSetupVec)
            {
                durationSetupUs += durationSetup.count();
            }

            float elapsedRunTimeUs = 0.0f;
            for(auto& cuEvtTimer : cuEvtTimer2Vec)
            {
                cuEvtTimer.synchronize();
                elapsedRunTimeUs += cuEvtTimer.elapsed_time_ms()*1000;
            }
            
            auto stopWallClock = Clock::now();
            duration<float, std::micro> diff = stopWallClock - startWallClock;

            CUDA_CHECK(cudaStreamSynchronize(cuStream));

            if(debugMode > 0)
            {
                printf("Average (%d runs) elapsed time in usec (w/ %u ms delay kernel): Run = %4.4f, Setup = %4.4f, H2DCpy = %4.4f\n",
                       nIterations,
                       delayMs,
                       elapsedRunTimeUs / nIterations,
                       elapsedH2DCpyTimeUs / nIterations,
                       durationSetupUs / nIterations);
            }
            else
            {
                printf("Average (%d runs) elapsed time in usec (w/ %u ms delay kernel): Run = %4.4f\n",
                       nIterations,
                       delayMs,
                       elapsedRunTimeUs / nIterations);                
            }

#if 0
            printf("Average (%d runs) elapsed time in usec (wall clock w/ %u ms delay kernel) = %4.4f\n",
                   nIterations,
                   delayMs,
                   diff.count() / nIterations);
#endif    

            cuphyStatus_t statusDestroy = cuphyDestroySrsChEst(srsChEstHndl);
            if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);
        }
        else
        {
            printf("Invalid srsChEstMode: %d", srsChEstMode);
            throw cuphy::cuphy_exception(CUPHY_STATUS_INVALID_ARGUMENT);
        }
        cudaStreamSynchronize(cuStream);
        cudaDeviceSynchronize();

        //------------------------------------------------------------------
        // Write outputs

        // Convert to FP32 format for MATLAB readability
        cuphy::tensor_device tOutHEst(CUPHY_C_32F, tHEstVec[0].layout());
        cuphy::tensor_device tOutDbg(CUPHY_C_32F, tDbg.layout());

        cuphyStatus_t tensorConvertStat = cuphyConvertTensor(tOutHEst.desc().handle(),    // dst tensor
                                                             tOutHEst.addr(),             // dst address
                                                             tHEstVec[0].desc().handle(), // src tensor
                                                             tHEstVec[0].addr(),          // src address
                                                             cuStream);                   // CUDA stream
        if(CUPHY_STATUS_SUCCESS != tensorConvertStat) throw cuphy::cuphy_exception(tensorConvertStat);

        tensorConvertStat = cuphyConvertTensor(tOutDbg.desc().handle(), // dst tensor
                                               tOutDbg.addr(),          // dst address
                                               tDbg.desc().handle(),    // src tensor
                                               tDbg.addr(),             // src address
                                               cuStream);               // CUDA stream
        if(CUPHY_STATUS_SUCCESS != tensorConvertStat) throw cuphy::cuphy_exception(tensorConvertStat);

        // Wait for copy to complete
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(cuStream);

        // Write outputs
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutHEst, "HEst");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg, "Dbg");
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
