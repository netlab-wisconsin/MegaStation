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
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "util.hpp"
#include "pusch_utils.hpp"
#include "datasets.hpp"

#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

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
    printf("cuphy_ex_ch_est [options]\n");
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
    cuphyNvlogFmtHelper nvlog_fmt("ch_est.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::vector<std::string> inputFilenameVec;
        std::string outputFilename =  std::string();
        uint32_t    fp16Mode       = 0xBAD;

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
        // Load API parameters

        cuphy::stream cuStrmMain;

        int descramblingOn    = 1;
        int apiTVflag         = 0; 
        uint64_t procModeBmsk = 0;
        bool cpuCopyOn        = false;

        StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle(), outputFilename, descramblingOn, apiTVflag);
        DynApiDataset     dynApiDataset(inputFilenameVec,   cuStrmMain.handle(), procModeBmsk, cpuCopyOn, fp16Mode, apiTVflag);
        EvalDataset       evalDataset(inputFilenameVec, cuStrmMain.handle(), apiTVflag);

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //------------------------------------------------------------------
        // Derive API parameters

        uint32_t nUes    = dynApiDataset.cellGrpDynPrm.nUes;
        uint32_t nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;

        std::vector<PerTbParams>   tbPrmsVec(nUes);
        cuphyDerivedPuschUeGrpPrms ueGrpPrmsPrime(cuStrmMain.handle());
        cuphyLDPCParams            LDPCprms(&staticApiDataset.puschStatPrms);
        cuphyChEstSettings         chEstSettings(&staticApiDataset.puschStatPrms, cuStrmMain.handle());
        cuphyDerivedPuschCmnPrms   cmnPrms;

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure filters copied
        

        //TODO:expandParameters is deprecated for cell groups. Fix this.
        expandParameters(tbPrmsVec.data(), &staticApiDataset.puschStatPrms, &dynApiDataset.puschDynPrm, staticApiDataset.cellStatPrmVec[0], cmnPrms, ueGrpPrmsPrime, LDPCprms);


        //NOTE: to be removed, these parameters should be included w/h descriptors
        ueGrpPrmsPrime.tDataSymLocGpu.copy(ueGrpPrmsPrime.tDataSymLocCpu, cuStrmMain.handle());
        ueGrpPrmsPrime.tQamInfoGpu.copy(ueGrpPrmsPrime.tQamInfoCpu, cuStrmMain.handle());
        ueGrpPrmsPrime.tStartPrbGpu.copy(ueGrpPrmsPrime.tStartPrbCpu, cuStrmMain.handle());
        ueGrpPrmsPrime.tNumPrbGpu.copy(ueGrpPrmsPrime.tNumPrbGpu, cuStrmMain.handle());
        ueGrpPrmsPrime.tDmrsScIdGpu.copy(ueGrpPrmsPrime.tDmrsScIdCpu, cuStrmMain.handle());

        cudaStreamSynchronize(cuStrmMain.handle());

        //---------------------------------------------------------------------
        // Extract ChEst parameters

        // antennas
        uint16_t nRxAnt  = staticApiDataset.cellStatPrmVec[0].nRxAnt;
        uint32_t nLayers = ueGrpPrmsPrime.nLayers;

        // freq allocation
        uint16_t nPrb                    = ueGrpPrmsPrime.nPrb;
        cuphyTensorPrm_t tPrmStartPrbGpu = ueGrpPrmsPrime.tPrmStartPrbGpu;
        cuphyTensorPrm_t tPrmNumPrbGpu   = ueGrpPrmsPrime.tPrmNumPrbGpu;
        cuphyTensorPrm_t tPrmDmrsScIdGpu = ueGrpPrmsPrime.tPrmDmrsScIdGpu;

        // identification parameters
        uint16_t phyCellId  = staticApiDataset.cellStatPrmVec[0].phyCellId;
        uint16_t slotNum    = dynApiDataset.cellDynPrmVec[0].slotNum;

        // DMRS parameters
        uint32_t activeDMRSGridBmsk   = ueGrpPrmsPrime.activeDMRSGridBmsk;
        uint32_t nDMRSSyms            = ueGrpPrmsPrime.nDMRSSyms;
        uint32_t nDMRSGridsPerPRB     = ueGrpPrmsPrime.nDMRSGridsPerPRB;
        uint16_t chEst0DmrsSymLocBmsk = cmnPrms.chEst0DmrsSymLocBmsk;

        // ChEst receiver settings
        cuphyTensorPrm_t tPrmWFreq        = chEstSettings.tPrmWFreq;
        cuphyTensorPrm_t tPrmWFreq4       = chEstSettings.tPrmWFreq4;
        cuphyTensorPrm_t tPrmWFreqSmall   = chEstSettings.tPrmWFreqSmall;
        cuphyTensorPrm_t tPrmShiftSeq     = chEstSettings.tPrmShiftSeq;
        cuphyTensorPrm_t tPrmShiftSeq4    = chEstSettings.tPrmShiftSeq4;
        cuphyTensorPrm_t tPrmUnShiftSeq   = chEstSettings.tPrmUnShiftSeq;
        cuphyTensorPrm_t tPrmUnShiftSeq4  = chEstSettings.tPrmUnShiftSeq4;
        uint8_t          nTimeChEsts      = chEstSettings.nTimeChEsts;

        // Input
        cuphyTensorPrm_t tPrmDataRx = dynApiDataset.tPrmDataRxVec[0];

        //------------------------------------------------------------------
        // Allocate ChEst output tensor arrays in device memory

        std::vector<cuphy::tensor_device> tHEstArray;
        std::vector<cuphy::tensor_device> tDbgArray;

        cuphyTensorPrm_t* pTPrmHEst = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmDbg  = new cuphyTensorPrm_t[nUeGrps];

        for(int i = 0; i < nUeGrps; ++i)
        {
            tHEstArray.push_back(cuphy::tensor_device(CUPHY_C_32F, 
                                                      nRxAnt, 
                                                      nLayers, 
                                                      CUPHY_N_TONES_PER_PRB * nPrb, 
                                                      nTimeChEsts, 
                                                      cuphy::tensor_flags::align_tight));

            tDbgArray.push_back(cuphy::tensor_device(CUPHY_C_32F,
                                                     CUPHY_N_TONES_PER_PRB * nPrb / 2, 
                                                     nDMRSSyms, 
                                                     1, 
                                                     1, 
                                                     cuphy::tensor_flags::align_tight));


            pTPrmHEst[i].desc  = tHEstArray[i].desc().handle();
            pTPrmHEst[i].pAddr = tHEstArray[i].addr();

            pTPrmDbg[i].desc  = tDbgArray[i].desc().handle();
            pTPrmDbg[i].pAddr = tDbgArray[i].addr();
        }

        //------------------------------------------------------------------
        // ChEst descriptors

        // descriptors hold Kernel parameters in GPU
        size_t        statDescrSizeBytes, statDescrAlignBytes, dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxChEstGetDescrInfo(&statDescrSizeBytes,
                                                                             &statDescrAlignBytes,
                                                                             &dynDescrSizeBytes,
                                                                             &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        std::array<cuphy::buffer<uint8_t, cuphy::pinned_alloc>, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> statDescrBufCpu{statDescrSizeBytes};
        std::array<cuphy::buffer<uint8_t, cuphy::pinned_alloc>, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> dynDescrBufCpu{dynDescrSizeBytes};

        std::array<cuphy::buffer<uint8_t, cuphy::device_alloc>, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> statDescrBufGpu{statDescrSizeBytes};
        std::array<cuphy::buffer<uint8_t, cuphy::device_alloc>, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> dynDescrBufGpu{dynDescrSizeBytes};

        //------------------------------------------------------------------
        // Create ChEst object

        cuphyPuschRxChEstHndl_t puschRxChEstHndl;
        bool                    enableCpuToGpuDescrAsyncCpy = true; // True: static descriptors copied from CPU to GPU at creation. 
                                                                    // False: static descriptors populated in CPU. Caller needs to copy.

        cuphyStatus_t statusCreate = cuphyCreatePuschRxChEst(&puschRxChEstHndl,
                                                             &tPrmWFreq,
                                                             &tPrmWFreq4,
                                                             &tPrmWFreqSmall,
                                                             &tPrmShiftSeq,
                                                             &tPrmShiftSeq4,
                                                             &tPrmUnShiftSeq,
                                                             &tPrmUnShiftSeq4,
                                                             chEstSettings.pSymbolRxStatus,
                                                             enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                             reinterpret_cast<void**>(statDescrBufCpu.data()),
                                                             reinterpret_cast<void**>(statDescrBufGpu.data()),
                                                             cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        if(!enableCpuToGpuDescrAsyncCpy){
            for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
            {
                CUDA_CHECK(cudaMemcpyAsync(statDescrBufGpu[chEstTimeInstIdx].addr(), statDescrBufCpu[chEstTimeInstIdx].addr(), statDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle()));
            }
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // setup ChEst object

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPuschRxChEstLaunchCfgs_t  chEstLaunchCfgs;
        chEstLaunchCfgs.nCfgs = CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS;

        // setup function populates dynamic descriptor and launch config
        uint32_t chEstInstIdx = 0; 
        cuphyStatus_t chEstSetupStatus = CUPHY_STATUS_SUCCESS;

        //TODO: enable this code
#if 0
        cuphyPuschRxEarlyHarqWaitLaunchCfg_t preEarlyHarqWaitCfgs, postEarlyHarqWaitCfgs;

        cuphyStatus_t chEstSetupStatus = cuphySetupPuschRxChEst(puschRxChEstHndl,
                                                                slotNum,
                                                                nRxAnt,
                                                                nLayers,
                                                                nDMRSSyms,
                                                                chEst0DmrsSymLocBmsk,
                                                                nDMRSGridsPerPRB,
                                                                activeDMRSGridBmsk,
                                                                chEstInstIdx,
                                                                nUeGrps,
                                                                ueGrpPrmsPrime.tNumPrbCpu.addr(),
                                                                &tPrmStartPrbGpu,
                                                                &tPrmNumPrbGpu,
                                                                &tPrmDmrsScIdGpu,
                                                                &tPrmDataRx,
                                                                pTPrmHEst,
                                                                pTPrmDbg,
                                                                enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                static_cast<void*>(dynDescrBufCpu.addr()),
                                                                static_cast<void*>(dynDescrBufGpu.addr()),
                                                                &chEstLaunchCfgs,
                                                                0,
                                                                &preEarlyHarqWaitCfgs,
                                                                &postEarlyHarqWaitCfgs,
                                                                cuStrmMain.handle());
#endif

        if(CUPHY_STATUS_SUCCESS != chEstSetupStatus) throw cuphy::cuphy_exception(chEstSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
            {
                CUDA_CHECK(cudaMemcpyAsync(dynDescrBufGpu[chEstTimeInstIdx].addr(), dynDescrBufCpu[chEstTimeInstIdx].addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle()));
            }            
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run ChEst

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = chEstLaunchCfgs.cfgs[0].kernelNodeParamsDriver;
        CUresult chEstRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != chEstRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);

        //------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPuschRxChEst(puschRxChEstHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);
        cudaStreamSynchronize(cuStrmMain.handle());
        cudaDeviceSynchronize();

        //------------------------------------------------------------------
        // save ChEst to h5

        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
            for(uint32_t ueGrpIdx = 0; ueGrpIdx <  nUeGrps; ++ueGrpIdx)
            {
                // write
                cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tHEstArray[ueGrpIdx], std::string("HEst" + std::to_string(ueGrpIdx)).c_str());
            }
        }

        //------------------------------------------------------------------
        // chEst snr

        double chEstSnr = evalDataset.evalChEst(tHEstArray, cuStrmMain.handle());
        NVLOGI_FMT(NVLOG_PUSCH, "snr comparing GPU ChEst output to matlab reference: {} dB", chEstSnr);
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
