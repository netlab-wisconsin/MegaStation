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
#include "util.hpp"
#include "pusch_utils.hpp"
#include "datasets.hpp"

#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

#define BITS_PER_WORD  32

 ////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_pusch_crc [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename     Input API test vector containing intermidate buffers\n");
    printf("    -h                     Display usage information\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
}


////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("pusch_crc.log");
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

        cuphy::stream cuStrmMain(cudaStreamDefault, PUSCH_STREAM_PRIORITY);

        int descramblingOn    = 1;
        int apiTVflag         = 1; 
        uint64_t procModeBmsk = 0;
        bool cpuCopyOn        = false;

        StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle(), outputFilename, descramblingOn, apiTVflag);
        DynApiDataset     dynApiDataset(inputFilenameVec,   cuStrmMain.handle(), procModeBmsk, cpuCopyOn, fp16Mode, apiTVflag);
        EvalDataset       evalDataset(inputFilenameVec, cuStrmMain.handle(), apiTVflag);

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //------------------------------------------------------------------
        // Derive API parameters

        uint32_t nUes = dynApiDataset.cellGrpDynPrm.nUes;

        cuphy::buffer<PerTbParams, cuphy::pinned_alloc>  tbPrmsCpu_buffer(nUes);
        cuphyDerivedPuschCmnPrms                         cmnPrms;
        cuphyDerivedPuschUeGrpPrms                       ueGrpPrmsPrime(cuStrmMain.handle());
        cuphyLDPCParams                                  LDPCprms(&staticApiDataset.puschStatPrms);

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure filters copied
        //TODO:expandParameters is deprecated for cell groups. Fix this.
        expandParameters(tbPrmsCpu_buffer.addr(), &staticApiDataset.puschStatPrms, &dynApiDataset.puschDynPrm, staticApiDataset.cellStatPrmVec[0], cmnPrms, ueGrpPrmsPrime, LDPCprms);

        //------------------------------------------------------------------
        // Extract Pusch CRC parameters

        uint32_t totNumTbs             = dynApiDataset.DataOut.totNumTbs;
        uint32_t totNumCbs             = dynApiDataset.DataOut.totNumCbs;
        uint32_t totNumPayloadBytes    = dynApiDataset.DataOut.totNumPayloadBytes;
        uint32_t nWordsLdpcOut         = totNumCbs * MAX_DECODED_CODE_BLOCK_BIT_SIZE / BITS_PER_WORD;

        uint16_t nSchUes = 0;
        std::vector<uint16_t> schUserIdxsVec(nUes);
        for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            if(tbPrmsCpu_buffer[ueIdx].isDataPresent)
            {
                schUserIdxsVec[nSchUes]  =  ueIdx;
                nSchUes                 += 1;
            }
        }

        //------------------------------------------------------------------
        // GPU input Buffers

        cuphy::buffer<PerTbParams, cuphy::device_alloc> tbPrmsGpu_buffer(nUes);
        cuphy::buffer<uint32_t,    cuphy::device_alloc> ldpcOutGpu_buffer(nWordsLdpcOut);
        
        uint32_t BITS_PER_BYTE     = 8;
        uint32_t ldpcOutBufferSize = totNumCbs * MAX_DECODED_CODE_BLOCK_BIT_SIZE / BITS_PER_BYTE;

        cudaMemcpyAsync(tbPrmsGpu_buffer.addr() , tbPrmsCpu_buffer.addr()      , sizeof(PerTbParams)*nUes       , cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaMemcpyAsync(ldpcOutGpu_buffer.addr(), evalDataset.ldpcOutRef.addr(), sizeof(uint32_t)*nWordsLdpcOut , cudaMemcpyHostToDevice, cuStrmMain.handle());
        cuStrmMain.synchronize();

        //------------------------------------------------------------------
        // GPU output buffers

        cuphy::buffer<uint32_t, cuphy::device_alloc> cbCrcGpu_buffer(totNumCbs);
        cuphy::buffer<uint32_t, cuphy::device_alloc> tbCrcGpu_buffer(totNumTbs);
        cuphy::buffer<uint8_t,  cuphy::device_alloc> tbPayloadGpu_buffer(totNumPayloadBytes);

        //------------------------------------------------------------------
        // Pusch CRC descriptors
        // descriptors hold Kernel parameters in GPU

        size_t  dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxCrcDecodeGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create Pusch Crc object

        cuphyPuschRxCrcDecodeHndl_t  crcDecodeHndl;
        cuphyStatus_t statusCreate = cuphyCreatePuschRxCrcDecode(&crcDecodeHndl, 1);
        if(CUPHY_STATUS_SUCCESS != statusCreate)
        {
            throw cuphy::cuphy_fn_exception(statusCreate, "cuphyCreatePuschRxCrcDecode()");
        }

        //------------------------------------------------------------------
        // setup Pusch Crc object

        bool    enableCpuToGpuDescrAsyncCpy = true; // True: dynamic descriptors copied from CPU to GPU at setup. 
                                                    // False: dynamic descriptors populated in CPU. Caller needs to copy.

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPuschRxCrcDecodeLaunchCfg_t crcLaunchCfgs[2]; // CB CRC + TB CRC

        // setup function populates dynamic descriptor and launch config
        cuphyStatus_t setupCrcDecodeStatus = cuphySetupPuschRxCrcDecode(crcDecodeHndl,
                                                                        nSchUes,
                                                                        schUserIdxsVec.data(),
                                                                        cbCrcGpu_buffer.addr(),
                                                                        tbPayloadGpu_buffer.addr(),
                                                                        ldpcOutGpu_buffer.addr(),
                                                                        tbCrcGpu_buffer.addr(),
                                                                        tbPrmsCpu_buffer.addr(),
                                                                        tbPrmsGpu_buffer.addr(),
                                                                        static_cast<void*>(dynDescrBufCpu.addr()),
                                                                        static_cast<void*>(dynDescrBufGpu.addr()),
                                                                        1,
                                                                        &crcLaunchCfgs[0],
                                                                        &crcLaunchCfgs[1],
                                                                        cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != setupCrcDecodeStatus) throw cuphy::cuphy_exception(setupCrcDecodeStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //----------------------------------------------------------------------------------------------
        // run Pusch Crc

        {
            // code block crc
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = crcLaunchCfgs[0].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                    kernelNodeParamsDriver.gridDimX,
                                    kernelNodeParamsDriver.gridDimY, 
                                    kernelNodeParamsDriver.gridDimZ,
                                    kernelNodeParamsDriver.blockDimX, 
                                    kernelNodeParamsDriver.blockDimY, 
                                    kernelNodeParamsDriver.blockDimZ,
                                    kernelNodeParamsDriver.sharedMemBytes,
                                    static_cast<CUstream>(cuStrmMain.handle()),
                                    kernelNodeParamsDriver.kernelParams,
                                    kernelNodeParamsDriver.extra));
        }
         
        {
            // transport block crc        
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = crcLaunchCfgs[1].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                    kernelNodeParamsDriver.gridDimX,
                                    kernelNodeParamsDriver.gridDimY, 
                                    kernelNodeParamsDriver.gridDimZ,
                                    kernelNodeParamsDriver.blockDimX, 
                                    kernelNodeParamsDriver.blockDimY, 
                                    kernelNodeParamsDriver.blockDimZ,
                                    kernelNodeParamsDriver.sharedMemBytes,
                                    static_cast<CUstream>(cuStrmMain.handle()),
                                    kernelNodeParamsDriver.kernelParams,
                                    kernelNodeParamsDriver.extra));
        }
         

         cuStrmMain.synchronize(); //make sure stream finish works

        //------------------------------------------------------------------
        // save output to h5

        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            // make file
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
            
            // write
            cuphy::tensor_ref tTbCrcOut;
            tTbCrcOut.desc().set(CUPHY_R_32U, static_cast<int>(totNumTbs), cuphy::tensor_flags::align_tight);
            tTbCrcOut.set_addr(static_cast<void*>(tbCrcGpu_buffer.addr()));
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tTbCrcOut, "tbCrcs", cuStrmMain.handle());

            cuphy::tensor_ref tCbCrcOut;
            tCbCrcOut.desc().set(CUPHY_R_32U, static_cast<int>(totNumCbs), cuphy::tensor_flags::align_tight);
            tCbCrcOut.set_addr(static_cast<void*>(cbCrcGpu_buffer.addr()));
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tCbCrcOut,"cbCrcs", cuStrmMain.handle());

            cuphy::tensor_ref tTbPayloadGpu;
            tTbPayloadGpu.desc().set(CUPHY_R_8U, static_cast<int>(totNumPayloadBytes), cuphy::tensor_flags::align_tight);
            tTbPayloadGpu.set_addr(static_cast<void*>(tbPayloadGpu_buffer.addr()));
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tTbPayloadGpu,"tbPayload", cuStrmMain.handle());

            cuStrmMain.synchronize(); //make sure copy done
        }

        //------------------------------------------------------------------
        // evaluate

        evalDataset.evalPuschCrc(tbCrcGpu_buffer.addr(), cbCrcGpu_buffer.addr(), tbPayloadGpu_buffer.addr(), cuStrmMain.handle());

        


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
