/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "pusch_rx.hpp"
#include "pusch_utils.hpp"
#include "datasets.hpp"

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_cfo_ta_est [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename     Input HDF5 filename, which must contain the following datasets:\n");
    printf("                           Data_rx      : received data (frequency-time) to be equalized\n");
    printf("                           WFreq        : interpolation filter coefficients used in channel estimation\n");
    printf("                           ShiftSeq     : sequence to be applied to DMRS tones containing descrambling code and delay shift for channel centering\n");
    printf("                           UnShiftSeq   : sequence to remove the delay shift from estimated channel\n");
    printf("    -h                     Display usage information\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    --H                    0         : No FP16\n");
    printf("                           1(default): FP16 format used for received data samples only\n");
    printf("                           2         : FP16 format used for all front end params\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("cfo_ta_est.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int                      iArg = 1;
        std::vector<std::string> inputFilenameVec;
        std::string              outputFilename = std::string();
        uint32_t                 fp16Mode       = 0xBAD;

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
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilenameVec.push_back(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'o':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No output file name given");
                    }
                    outputFilename.assign(argv[iArg++]);
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'H':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (3 <= fp16Mode))
                        {
                            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid FP16 mode {}", fp16Mode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    default:
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                    }
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilenameVec.empty())
        {
            usage();
            exit(1);
        }

        //------------------------------------------------------------------
        // Allocate tensors in device memory

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

        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilenameVec[0].c_str());

        //------------------------------------------------------------------
        // Load API parameters

        cuphy::stream cuStrmMain;

        uint64_t procModeBmsk = 0;
        bool     cpuCopyOn    = false;

        StaticApiDataset staticApiDataset(inputFilenameVec, cuStrmMain.handle(), outputFilename);
        DynApiDataset    dynApiDataset(inputFilenameVec, cuStrmMain.handle(), procModeBmsk, cpuCopyOn, fp16Mode);
        EvalDataset      evalDataset(inputFilenameVec, cuStrmMain.handle());

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //----------------------------------------------------------------
        // Initialize CPU/GPU memory

        uint32_t nUes    = dynApiDataset.cellGrpDynPrm.nUes;
        uint32_t nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;

        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::pinned_alloc> drvdUeGrpPrmsCpuBuffer(nUeGrps);
        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::device_alloc> drvdUeGrpPrmsGpuBuffer(nUeGrps);

        //------------------------------------------------------------------
        // Derive API parameters

        PuschRx puschRx(&staticApiDataset.puschStatPrms, cuStrmMain.handle());

        uint8_t enableRssiMeasurement = 0;
        puschRx.expandFrontEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsCpuBuffer.addr(), enableRssiMeasurement);

        //------------------------------------------------------------------
        // Allocate CfoTaEst output tensor arrays in device memory
        cuphy::tensor_device tCfoPhaseRot(feCplxDataType, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps, cuphy::tensor_flags::align_tight);
        CUDA_CHECK(cudaMemsetAsync(tCfoPhaseRot.addr(), 0, tCfoPhaseRot.desc().get_size_in_bytes(), cuStrmMain.handle()));
        // tCfoPhaseRot.fill(make_cuComplex(0.0f,0.0f), cuStrmMain.handle());
        cuphy::tensor_ref tRefCfoPhaseRot(tCfoPhaseRot.desc(), tCfoPhaseRot.addr());
        //
        cuphy::tensor_device tTaPhaseRot(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps, cuphy::tensor_flags::align_tight);
        CUDA_CHECK(cudaMemsetAsync(tTaPhaseRot.addr(), 0, tTaPhaseRot.desc().get_size_in_bytes(), cuStrmMain.handle()));
        cuphy::tensor_ref tRefTaPhaseRot(tTaPhaseRot.desc(), tTaPhaseRot.addr());
        //
        cuphy::tensor_device tCfoHz(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
        cuphy::tensor_ref    tRefCfoHz(tCfoHz.desc(), tCfoHz.addr());
        //
        cuphy::tensor_device tCfoTaInterCtaSyncCnt(CUPHY_R_32U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
        CUDA_CHECK(cudaMemsetAsync(tCfoTaInterCtaSyncCnt.addr(), 0, tCfoTaInterCtaSyncCnt.desc().get_size_in_bytes(), cuStrmMain.handle()));
        cuphy::tensor_ref tRefCfoTaInterCtaSyncCnt(tCfoTaInterCtaSyncCnt.desc(), tCfoTaInterCtaSyncCnt.addr());
        //
#if CUPHY_ENABLE_SUB_SLOT_PROCESSING
        cuphy::tensor_device tCfoInterCtaSyncCnt(CUPHY_R_32U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
        CUDA_CHECK(cudaMemsetAsync(tCfoInterCtaSyncCnt.addr(), 0, tCfoInterCtaSyncCnt.desc().get_size_in_bytes(), cuStrmMain.handle()));
        cuphy::tensor_ref tRefCfoInterCtaSyncCnt(tCfoInterCtaSyncCnt.desc(), tCfoInterCtaSyncCnt.addr());
        //
        cuphy::tensor_device tTaInterCtaSyncCnt(CUPHY_R_32U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
        CUDA_CHECK(cudaMemsetAsync(tTaInterCtaSyncCnt.addr(), 0, tTaInterCtaSyncCnt.desc().get_size_in_bytes(), cuStrmMain.handle()));
        cuphy::tensor_ref tRefTaInterCtaSyncCnt(tTaInterCtaSyncCnt.desc(), tTaInterCtaSyncCnt.addr());
#endif
        //----------------------------------------------------------------------------------------------------------
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        std::vector<cuphy::tensor_device> tHEstArray;
        std::vector<cuphy::tensor_device> tCfoEstArray;
        std::vector<cuphy::tensor_device> tTaEstArray;
        cuphyTensorPrm_t                  tPrms;

        for(int i = 0; i < nUeGrps; ++i)
        {
            cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrmsCpu = drvdUeGrpPrmsCpuBuffer[i];
            puschRx.copyTensorRef2Info(tRefCfoPhaseRot, drvdUeGrpPrmsCpu.tInfoCfoPhaseRot);
            puschRx.copyTensorRef2Info(tRefTaPhaseRot, drvdUeGrpPrmsCpu.tInfoTaPhaseRot);
            puschRx.copyTensorRef2Info(tRefCfoHz, drvdUeGrpPrmsCpu.tInfoCfoHz);
            puschRx.copyTensorRef2Info(tRefCfoTaInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoCfoTaEstInterCtaSyncCnt);
#if CUPHY_ENABLE_SUB_SLOT_PROCESSING
            puschRx.copyTensorRef2Info(tRefCfoInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoCfoEstInterCtaSyncCnt);
            puschRx.copyTensorRef2Info(tRefTaInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoTaEstInterCtaSyncCnt);
#endif

            // Construct channel estimation tensor
            tHEstArray.push_back(cuphy::tensor_from_dataset(fInput.open_dataset("reference_H_est"), feCplxChannelType, cuphy::tensor_flags::align_default, cuStrmMain.handle()));
            tPrms.desc  = tHEstArray[i].desc().handle();
            tPrms.pAddr = tHEstArray[i].addr();
            copyTensorPrm2Info(tPrms, drvdUeGrpPrmsCpu.tInfoHEst);

            // Construct CfoEst tensor
            tCfoEstArray.push_back(cuphy::tensor_device(CUPHY_C_32F, MAX_ND_SUPPORTED, drvdUeGrpPrmsCpu.nUes, cuphy::tensor_flags::align_tight));
            tPrms.desc  = tCfoEstArray[i].desc().handle();
            tPrms.pAddr = tCfoEstArray[i].addr();
            copyTensorPrm2Info(tPrms, drvdUeGrpPrmsCpu.tInfoCfoEst);

            // Construct TaEst tensor
            tTaEstArray.push_back(cuphy::tensor_device(CUPHY_R_32F, drvdUeGrpPrmsCpu.nUes, cuphy::tensor_flags::align_tight));
            tPrms.desc  = tTaEstArray[i].desc().handle();
            tPrms.pAddr = tTaEstArray[i].addr();
            copyTensorPrm2Info(tPrms, drvdUeGrpPrmsCpu.tInfoTaEst);

#ifdef ENABLE_DEBUG
            printf("Tensor layouts:\n");
            printf("tHEstArray[%d]    : addr: %p, %s, size: %.1f kB\n",
                   i,
                   tHEstArray[i].addr(),
                   tHEstArray[i].desc().get_info().to_string().c_str(),
                   tHEstArray[i].desc().get_size_in_bytes() / 1024.0);
#endif
        }

        //------------------------------------------------------------------
        // CfoTaEst descriptors

        // descriptors hold Kernel parameters in GPU
        size_t        statDescrSizeBytes, statDescrAlignBytes;
        size_t        dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t status = cuphyPuschRxCfoTaEstGetDescrInfo(&statDescrSizeBytes,
                                                                &statDescrAlignBytes,
                                                                &dynDescrSizeBytes,
                                                                &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxCfoTaEstGetDescrInfo()");
        }

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> statDescrBufCpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);

        cuphy::buffer<uint8_t, cuphy::device_alloc> statDescrBufGpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create CfoTaEst object

        bool enableCpuToGpuDescrAsyncCpy = false; // True: static descriptors copied from CPU to GPU at creation.
                                                  // False: static descriptors populated in CPU. Caller needs to copy.
        cuphyPuschRxCfoTaEstHndl_t puschRxCfoTaEstHndl;

        cuphyStatus_t statusCreate = cuphyCreatePuschRxCfoTaEst(&puschRxCfoTaEstHndl,
                                                                static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                static_cast<void*>(statDescrBufCpu.addr()),
                                                                static_cast<void*>(statDescrBufGpu.addr()),
                                                                cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        if(!enableCpuToGpuDescrAsyncCpy)
        {
            cudaMemcpyAsync(statDescrBufGpu.addr(), statDescrBufCpu.addr(), statDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // setup CfoTaEst object

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPuschRxCfoTaEstLaunchCfgs_t cfoEstLaunchCfgs;
        cfoEstLaunchCfgs.nCfgs = CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS;
        uint32_t nMaxPrb       = MAX_N_PRBS_SUPPORTED; //??

        // setup function populates dynamic descriptor and launch config
        cuphyStatus_t cfoEstSetupStatus = cuphySetupPuschRxCfoTaEst(puschRxCfoTaEstHndl,
                                                                    drvdUeGrpPrmsCpuBuffer.addr(),
                                                                    drvdUeGrpPrmsGpuBuffer.addr(),
                                                                    nUeGrps,
                                                                    nMaxPrb,
                                                                    0,
                                                                    enableCpuToGpuDescrAsyncCpy,
                                                                    static_cast<void*>(dynDescrBufCpu.addr()),
                                                                    static_cast<void*>(dynDescrBufGpu.addr()),
                                                                    &cfoEstLaunchCfgs,
                                                                    cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != cfoEstSetupStatus) throw cuphy::cuphy_exception(cfoEstSetupStatus);

        if (cfoEstLaunchCfgs.nCfgs == 0)
        {
            throw cuphy::cuphy_fn_exception(status, "Need at least two time domain channel estimates to compute CFO !! terminating the test ...\n");
        }

        if(!enableCpuToGpuDescrAsyncCpy)
        {
            cudaMemcpyAsync(drvdUeGrpPrmsGpuBuffer.addr(), drvdUeGrpPrmsCpuBuffer.addr(), nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run CfoTaEst

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = cfoEstLaunchCfgs.cfgs[0].kernelNodeParamsDriver;
        CUresult                       cfoEstRunStatus        = cuLaunchKernel(kernelNodeParamsDriver.func,
                                                  kernelNodeParamsDriver.gridDimX,
                                                  kernelNodeParamsDriver.gridDimY,
                                                  kernelNodeParamsDriver.gridDimZ,
                                                  kernelNodeParamsDriver.blockDimX,
                                                  kernelNodeParamsDriver.blockDimY,
                                                  kernelNodeParamsDriver.blockDimZ,
                                                  kernelNodeParamsDriver.sharedMemBytes,
                                                  static_cast<CUstream>(cuStrmMain.handle()),
                                                  kernelNodeParamsDriver.kernelParams,
                                                  kernelNodeParamsDriver.extra);
        if(CUDA_SUCCESS != cfoEstRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);

        //------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPuschRxCfoTaEst(puschRxCfoTaEstHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);
        cudaStreamSynchronize(cuStrmMain.handle());
        cudaDeviceSynchronize();

        //------------------------------------------------------------------
        // save CfoTaEst to h5

        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
            for(uint32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
            {
                // write
                cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tHEstArray[ueGrpIdx], std::string("HEst" + std::to_string(ueGrpIdx)).c_str());
            }
        }

        //------------------------------------------------------------------
        // compare results against reference values
        evalDataset.evalCfoTaEst(tCfoEstArray, tTaEstArray, cuStrmMain.handle());

    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
