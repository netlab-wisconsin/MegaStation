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
#include <numeric>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "util.hpp"

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_channel_eq [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename     Input HDF5 filename, which must contain the following datasets:\n");
    printf("                           Data_sym_loc : locations of data symbols within the subframe\n");
    printf("                           RxxInv   : Symbol energy covariance\n");
    printf("                           Noise_pwr: noise power at frequency-time bins where channel (H) is estimated\n");
    printf("                           H        : Channel coupling matrix in frequency-time \n");
    printf("                           Data_rx  : received data (frequency-time) to be equalized\n");
    printf("                           Data_eq  : equalized output data (in frequency-time)\n");
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
    cuphyNvlogFmtHelper nvlog_fmt("channel_eq.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename;
        std::string outputFilename;
        uint32_t    fp16Mode = 0xBAD;

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
                    inputFilename.assign(argv[iArg++]);
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
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }
        //--------------------------------------------------------------
        // Create a cuPHY context
        cuphy::context ctx;
        
        //------------------------------------------------------------------
        // Open the input file and required datasets
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());


#if 1
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

        // clang-format off
        cuphy::tensor_device tStartPrb      = cuphy::tensor_from_dataset(fInput.open_dataset("StartPrb")     ,                    cuphy::tensor_flags::align_tight, cuStream);
        cuphy::tensor_device tNumPrb        = cuphy::tensor_from_dataset(fInput.open_dataset("NumPrb")       ,                    cuphy::tensor_flags::align_tight, cuStream);
        cuphy::tensor_device tDataSymLoc    = cuphy::tensor_from_dataset(fInput.open_dataset("Data_sym_loc") ,                    cuphy::tensor_flags::align_tight, cuStream);
        cuphy::tensor_device tQamInfo       = cuphy::tensor_from_dataset(fInput.open_dataset("QamInfo")      ,                    cuphy::tensor_flags::align_tight, cuStream);
        cuphy::tensor_device tDataRx        = cuphy::tensor_from_dataset(fInput.open_dataset("DataRx")       , feCplxDataType   , cuphy::tensor_flags::align_tight, cuStream);
        cuphy::tensor_device tRxxInv        = cuphy::tensor_from_dataset(fInput.open_dataset("RxxInv")       , feChannelType    , cuphy::tensor_flags::align_tight, cuStream);
#if 0        
        cuphy::tensor_device tNoisePwr      = cuphy::tensor_from_dataset(fInput.open_dataset("Noise_pwr")    , feChannelType,     cuphy::tensor_flags::align_tight, cuStream);
#else        
        cuphy::tensor_device tNoisePwr      = cuphy::tensor_from_dataset(fInput.open_dataset("Noise_pwr")    , feCplxChannelType, cuphy::tensor_flags::align_tight, cuStream);
#endif        
        cuphy::tensor_device tHEst          = cuphy::tensor_from_dataset(fInput.open_dataset("H")            , feCplxChannelType, cuphy::tensor_flags::align_tight, cuStream);
        // clang-format on

        // Ensure conversion completes
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(cuStream);

        printf("Input tensors:\n");
        printf("---------------------------------------------------------------\n");
        printf("Data_sym_loc   : %s\n", tDataSymLoc.desc().get_info().to_string(false).c_str());
        printf("QamInfo        : %s\n", tQamInfo.desc().get_info().to_string(false).c_str());
        printf("RxxInv         : %s\n", tRxxInv.desc().get_info().to_string(false).c_str());
        printf("Noise_pwr      : %s\n", tNoisePwr.desc().get_info().to_string(false).c_str());
        printf("H              : %s\n", tHEst.desc().get_info().to_string(false).c_str());
        printf("Data_rx        : %s\n", tDataRx.desc().get_info().to_string(false).c_str());

        //------------------------------------------------------------------
        uint16_t nBSAnts = 0;
        uint8_t  nLayers = 0;
        uint16_t nPrb    = 0;
        uint8_t  Nd      = 0;
        uint16_t nUeGrps = 0;

        cuphy::disable_hdf5_error_print(); // Temporarily disable HDF5 stderr printing

        try
        {
            cuphy::cuphyHDF5_struct chEstCfg = cuphy::get_HDF5_struct(fInput, "chEqCfg");
            nBSAnts                          = chEstCfg.get_value_as<uint16_t>("nRxAnts");
            nLayers                          = chEstCfg.get_value_as<uint8_t>("nLayers");
            nPrb                             = chEstCfg.get_value_as<uint16_t>("nPrb");
            Nd                               = chEstCfg.get_value_as<uint8_t>("nDataSyms");
            nUeGrps                          = chEstCfg.get_value_as<uint16_t>("nUeGrps");

            if(nUeGrps != 1)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Error! Equalization test-bench only supports a single user group");
                exit(1);
            }
        }
        catch(const std::exception& exc)
        {
            printf("%s\n", exc.what());
            throw exc;
            // Continue using command line arguments if the input file does not
            // have a config struct.
        }
        cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

        uint16_t Nf = nPrb * CUPHY_N_TONES_PER_PRB;
        uint8_t  Nh = 1;

        printf("Config parameters:\n");
        printf("---------------------------------------------------------------\n");
        printf("nUeGrps            : %i\n", nUeGrps);
        printf("nBSAnts            : %i\n", nBSAnts);
        printf("nLayers            : %i\n", nLayers);
        printf("Nf                 : %i\n", Nf); // # of estimates of H in frequency
        printf("nPrb               : %i\n", nPrb);
        printf("Nh                 : %i\n", Nh); // # of estimates of H in time
        printf("Nd                 : %i\n", Nd); // # of data symbols

        //------------------------------------------------------------------
        // Allocate tensors in device memory
        // clang-format off

        cuphy::tensor_device tCoef(feCplxChannelType,
                                   nBSAnts,
                                   CUPHY_N_TONES_PER_PRB,
                                   nLayers,
                                   nPrb,
                                   cuphy::tensor_flags::align_tight);
        cuphy::tensor_device tReeDiag(feChannelType,
                                      CUPHY_N_TONES_PER_PRB,
                                      nLayers,
                                      nPrb,
                                      cuphy::tensor_flags::align_tight);
        cuphy::tensor_device tDataEq(feCplxDataType,
                                     nLayers,
                                     Nf,
                                     Nd,
                                     cuphy::tensor_flags::align_tight);

       
        cuphy::tensor_device tLLR(feDataType, // feDataType, keeping LLR format to FP32 until backend supports it
                                  CUPHY_QAM_256,
                                  nLayers,
                                  Nf,
                                  Nd,
                                  cuphy::tensor_flags::align_tight);

         cuphy::tensor_device tDbg(feCplxChannelType,
                                   nLayers, // static_cast<int>(nLayers), // static_cast<int>(nBSAnts),
                                   nLayers, // static_cast<int>(nBSAnts), // static_cast<int>(nLayers),
                                   Nf,      // static_cast<int>(Nf),
                                   Nh,
                                   cuphy::tensor_flags::align_tight);

        // clang-format on

        //Input tensor Arrays
        //-------------------------------------------------------------------------------------------

        cuphyTensorPrm_t* pTPrmHEst    = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmRwwInv  = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmQamInfo = new cuphyTensorPrm_t[nUeGrps];

        pTPrmHEst[0].desc  = tHEst.desc().handle();
        pTPrmHEst[0].pAddr = tHEst.addr();

        pTPrmRwwInv[0].desc  = tNoisePwr.desc().handle();
        pTPrmRwwInv[0].pAddr = tNoisePwr.addr();

        pTPrmQamInfo[0].desc  = tQamInfo.desc().handle();
        pTPrmQamInfo[0].pAddr = tQamInfo.addr();


        // //Output tensor Arrays
        // //-------------------------------------------------------------------------------------------

        std::vector<cuphy::tensor_device> tCoefArray;
        std::vector<cuphy::tensor_device> tReeDiagInvArray;
        std::vector<cuphy::tensor_device> tCfoEstArray;
        std::vector<cuphy::tensor_device> tDbgArray;
        std::vector<cuphy::tensor_device> tDataEqArray;
        std::vector<cuphy::tensor_device> tLLRArray;

        cuphyTensorPrm_t* pTPrmCoef       = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmReeDiagInv = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmCfoEst     = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmDbg        = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmDataEq     = new cuphyTensorPrm_t[nUeGrps];
        cuphyTensorPrm_t* pTPrmLLR        = new cuphyTensorPrm_t[nUeGrps];

        for(int i = 0; i < nUeGrps; ++i)
        {
            tCoefArray.push_back(cuphy::tensor_device(feCplxChannelType,
                                                      nBSAnts,
                                                      CUPHY_N_TONES_PER_PRB,
                                                      nLayers,
                                                      nPrb,
                                                      cuphy::tensor_flags::align_tight));

            tReeDiagInvArray.push_back(cuphy::tensor_device(feChannelType,
                                                            CUPHY_N_TONES_PER_PRB,
                                                            nLayers,
                                                            nPrb,
                                                            cuphy::tensor_flags::align_tight));

            tCfoEstArray.push_back(cuphy::tensor_device(CUPHY_C_32F,
                                                        nLayers,
                                                        cuphy::tensor_flags::align_tight));

            tDbgArray.push_back(cuphy::tensor_device(feCplxChannelType,
                                                     nLayers, // static_cast<int>(nLayers), // static_cast<int>(nBSAnts),
                                                     nLayers, // static_cast<int>(nBSAnts), // static_cast<int>(nLayers),
                                                     Nf,      // static_cast<int>(Nf),
                                                     Nh,
                                                     cuphy::tensor_flags::align_tight));

            tDataEqArray.push_back(cuphy::tensor_device(feCplxDataType,
                                                        nLayers,
                                                        Nf,
                                                        Nd,
                                                        cuphy::tensor_flags::align_tight));

            tLLRArray.push_back(cuphy::tensor_device(feDataType, // feDataType, keeping LLR format to FP32 until backend supports it
                                                     CUPHY_QAM_256,
                                                     nLayers,
                                                     Nf,
                                                     Nd,
                                                     cuphy::tensor_flags::align_tight));

            pTPrmCoef[i].desc  = tCoefArray[i].desc().handle();
            pTPrmCoef[i].pAddr = tCoefArray[i].addr();

            pTPrmReeDiagInv[i].desc  = tReeDiagInvArray[i].desc().handle();
            pTPrmReeDiagInv[i].pAddr = tReeDiagInvArray[i].addr();

            pTPrmCfoEst[i].desc  = tCfoEstArray[i].desc().handle();
            pTPrmCfoEst[i].pAddr = tCfoEstArray[i].addr();            

            pTPrmDbg[i].desc  = tDbgArray[i].desc().handle();
            pTPrmDbg[i].pAddr = tDbgArray[i].addr();

            pTPrmDataEq[i].desc  = tDataEqArray[i].desc().handle();
            pTPrmDataEq[i].pAddr = tDataEqArray[i].addr();

            pTPrmLLR[i].desc  = tLLRArray[i].desc().handle();
            pTPrmLLR[i].pAddr = tLLRArray[i].addr();
        }

        //--------------------------------------------------------------------------------------------------------------------------------------

        printf("Tensor layout:\n");
        printf("---------------------------------------------------------------\n");
        printf("tDataSymLoc : addr: %p, %s, size: %.1f kB\n",
               tDataSymLoc.addr(),
               tDataSymLoc.desc().get_info().to_string().c_str(),
               tDataSymLoc.desc().get_size_in_bytes() / 1024.0);
        printf("tQamInfo    : addr: %p, %s, size: %.1f kB\n",
               tQamInfo.addr(),
               tQamInfo.desc().get_info().to_string().c_str(),
               tQamInfo.desc().get_size_in_bytes() / 1024.0);
        printf("tRxxInv     : addr: %p, %s, size: %.1f kB\n",
               tRxxInv.addr(),
               tRxxInv.desc().get_info().to_string().c_str(),
               tRxxInv.desc().get_size_in_bytes() / 1024.0);
        printf("tNoisePwr  : addr: %p, %s, size: %.1f kB\n",
               tNoisePwr.addr(),
               tNoisePwr.desc().get_info().to_string().c_str(),
               tNoisePwr.desc().get_size_in_bytes() / 1024.0);
        printf("tHEst      : addr: %p, %s, size: %.1f kB\n",
               tHEst.addr(),
               tHEst.desc().get_info().to_string().c_str(),
               tHEst.desc().get_size_in_bytes() / 1024.0);
        printf("tCoef      : addr: %p, %s, size: %.1f kB\n",
               tCoef.addr(),
               tCoef.desc().get_info().to_string().c_str(),
               tCoef.desc().get_size_in_bytes() / 1024.0);
        printf("tDataRx    : addr: %p, %s, size: %.1f kB\n",
               tDataRx.addr(),
               tDataRx.desc().get_info().to_string().c_str(),
               tDataRx.desc().get_size_in_bytes() / 1024.0);
        printf("tDataEq    : addr: %p, %s, size: %.1f kB\n",
               tDataEq.addr(),
               tDataEq.desc().get_info().to_string().c_str(),
               tDataEq.desc().get_size_in_bytes() / 1024.0);
        printf("tReeDiag   : addr: %p, %s, size: %.1f kB\n",
               tReeDiag.addr(),
               tReeDiag.desc().get_info().to_string().c_str(),
               tReeDiag.desc().get_size_in_bytes() / 1024.0);
        printf("tCfoEst   : addr: %p, %s, size: %.1f kB\n",
               tReeDiag.addr(),
               tReeDiag.desc().get_info().to_string().c_str(),
               tReeDiag.desc().get_size_in_bytes() / 1024.0);               
        printf("tLLR        : addr: %p, %s, size: %.1f kB\n",
               tLLR.addr(),
               tLLR.desc().get_info().to_string().c_str(),
               tLLR.desc().get_size_in_bytes() / 1024.0);
        printf("tDbg        : addr: %p, %s, size: %.1f kB\n",
               tDbg.addr(),
               tDbg.desc().get_info().to_string().c_str(),
               tDbg.desc().get_size_in_bytes() / 1024.0);


        std::array<cudaStream_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS> strmVec{{cuStream}};

        size_t statDescrSizeBytes, statDescrAlignBytes, coefCompDynDescrSizeBytes, coefCompDynDescrAlignBytes,
            softDemapDynDescrSizeBytes, softDemapDynDescrAlignBytes;

        cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxChEqGetDescrInfo(&statDescrSizeBytes,
                                                                            &statDescrAlignBytes,
                                                                            &coefCompDynDescrSizeBytes,
                                                                            &coefCompDynDescrAlignBytes,
                                                                            &softDemapDynDescrSizeBytes,
                                                                            &softDemapDynDescrAlignBytes);

        printf("DescriptorInfo: statDescrSizeBytes %zu statDescrAlignBytes %zu coefCompDynDescrSizeBytes %zu"
                "coefCompDynDescrAlignBytes %zu softDemapDynDescrSizeBytes %zu softDemapDynDescrAlignBytes %zu\n",
                statDescrSizeBytes,
                statDescrAlignBytes,
                coefCompDynDescrSizeBytes,
                coefCompDynDescrAlignBytes,
                softDemapDynDescrSizeBytes,
                softDemapDynDescrAlignBytes);



        cuphy::buffer<uint8_t, cuphy::pinned_alloc> statDescrBufCpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> coefCompDynDescrBufCpu(coefCompDynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> softDemapDynDescrBufCpu(softDemapDynDescrSizeBytes);

        cuphy::buffer<uint8_t, cuphy::device_alloc> statDescrBufGpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> coefCompDynDescrBufGpu(coefCompDynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> softDemapDynDescrBufGpu(softDemapDynDescrSizeBytes);


        bool                   enableCpuToGpuDescrAsyncCpy = false;
        cuphyPuschRxChEqHndl_t puschRxChEqHndl;

        uint32_t hetCfgIdx = 0;

        cuphyStatus_t statusCreate = cuphyCreatePuschRxChEq(ctx.handle(),
                                                            &puschRxChEqHndl,
                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                            reinterpret_cast<void**>(statDescrBufCpu.addr()),
                                                            reinterpret_cast<void**>(statDescrBufGpu.addr()),
                                                            strmVec[hetCfgIdx]);
        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        if(!enableCpuToGpuDescrAsyncCpy)
        {
            cudaMemcpyAsync(statDescrBufGpu.addr(), statDescrBufCpu.addr(), statDescrSizeBytes, cudaMemcpyHostToDevice, strmVec[hetCfgIdx]);
        }
        cudaStreamSynchronize(strmVec[hetCfgIdx]);

        // clang-format off
        cuphyTensorPrm_t tPrmHEst       {.desc = tHEst.desc().handle()    , .pAddr = tHEst.addr()};
        cuphyTensorPrm_t tPrmNoisePwrInv{.desc = tNoisePwr.desc().handle(), .pAddr = tNoisePwr.addr()};
        cuphyTensorPrm_t tPrmCoef       {.desc = tCoef.desc().handle()    , .pAddr = tCoef.addr()};
        cuphyTensorPrm_t tPrmReeDiagInv {.desc = tReeDiag.desc().handle() , .pAddr = tReeDiag.addr()};
        cuphyTensorPrm_t tPrmDbg        {.desc = tDbg.desc().handle()     , .pAddr = tDbg.addr()};

        cuphyTensorPrm_t tPrmQamInfo   {.desc = tQamInfo.desc().handle()   , .pAddr = tQamInfo.addr()};
        cuphyTensorPrm_t tPrmStartPrb  {.desc = tStartPrb.desc().handle()  , .pAddr = tStartPrb.addr()};
        cuphyTensorPrm_t tPrmNumPrb    {.desc = tNumPrb.desc().handle()    , .pAddr = tNumPrb.addr()};
        cuphyTensorPrm_t tPrmDataSymLoc{.desc = tDataSymLoc.desc().handle(), .pAddr = tDataSymLoc.addr()};
        cuphyTensorPrm_t tPrmDataRx    {.desc = tDataRx.desc().handle()    , .pAddr = tDataRx.addr()};
        cuphyTensorPrm_t tPrmDataEq    {.desc = tDataEq.desc().handle()    , .pAddr = tDataEq.addr()};
        cuphyTensorPrm_t tPrmLLR       {.desc = tLLR.desc().handle()       , .pAddr = tLLR.addr()};
        // clang-format on


        cuphyPuschRxChEqLaunchCfgs_t   chEqCoefCompLaunchCfgs, chEqSoftDemapLaunchCfgs;
        chEqCoefCompLaunchCfgs.nCfgs  = CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS;
        chEqSoftDemapLaunchCfgs.nCfgs = CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS;

        //TODO: enable this code
#if 0
        cuphyStatus_t statusSetupCoefCompute = cuphySetupPuschRxChEqCoefCompute(puschRxChEqHndl,
                                                                                nBSAnts,
                                                                                nLayers,
                                                                                Nh,                                                                            
                                                                                nUeGrps,
                                                                                nPrb,
                                                                                &tPrmNumPrb,
                                                                                pTPrmHEst,
                                                                                pTPrmRwwInv,
                                                                                pTPrmCoef,
                                                                                pTPrmReeDiagInv,
                                                                                pTPrmDbg,
                                                                                enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                                static_cast<void*>(coefCompDynDescrBufCpu.addr()),
                                                                                static_cast<void*>(coefCompDynDescrBufGpu.addr()),
                                                                                &chEqCoefCompLaunchCfgs,
                                                                                strmVec[hetCfgIdx]);
        if(CUPHY_STATUS_SUCCESS != statusSetupCoefCompute) throw cuphy::cuphy_exception(statusSetupCoefCompute);

        cuphyStatus_t statusSetupSoftDemap = cuphySetupPuschRxChEqSoftDemap(puschRxChEqHndl,
                                                                            nBSAnts,
                                                                            nLayers,
                                                                            Nh,
                                                                            Nd,
                                                                            nUeGrps,
                                                                            nPrb,                                                                            
                                                                            0,
                                                                            &tPrmStartPrb,
                                                                            &tPrmNumPrb,
                                                                            &tPrmDataSymLoc,
                                                                            pTPrmQamInfo,
                                                                            pTPrmCoef,
                                                                            pTPrmReeDiagInv,
                                                                            pTPrmCfoEst,
                                                                            &tPrmDataRx,
                                                                            pTPrmDataEq,
                                                                            pTPrmLLR,
                                                                            pTPrmDbg,
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            static_cast<void*>(softDemapDynDescrBufCpu.addr()),
                                                                            static_cast<void*>(softDemapDynDescrBufGpu.addr()),
                                                                            &chEqSoftDemapLaunchCfgs,
                                                                            strmVec[hetCfgIdx]);
        if(CUPHY_STATUS_SUCCESS != statusSetupSoftDemap) throw cuphy::cuphy_exception(statusSetupSoftDemap);
        
        
#endif        

        if(!enableCpuToGpuDescrAsyncCpy)
        {
            cudaMemcpyAsync(coefCompDynDescrBufGpu.addr(), coefCompDynDescrBufCpu.addr(), coefCompDynDescrSizeBytes, cudaMemcpyHostToDevice, strmVec[hetCfgIdx]);
            cudaMemcpyAsync(softDemapDynDescrBufGpu.addr(), softDemapDynDescrBufCpu.addr(), softDemapDynDescrSizeBytes, cudaMemcpyHostToDevice, strmVec[hetCfgIdx]);
        }
        cudaStreamSynchronize(strmVec[hetCfgIdx]);

          
       
        // launch kernels
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < chEqCoefCompLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = chEqCoefCompLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                    kernelNodeParamsDriver.gridDimX,
                                    kernelNodeParamsDriver.gridDimY, 
                                    kernelNodeParamsDriver.gridDimZ,
                                    kernelNodeParamsDriver.blockDimX, 
                                    kernelNodeParamsDriver.blockDimY, 
                                    kernelNodeParamsDriver.blockDimZ,
                                    kernelNodeParamsDriver.sharedMemBytes,
                                    static_cast<CUstream>(cuStream),
                                    kernelNodeParamsDriver.kernelParams,
                                    kernelNodeParamsDriver.extra));
        }
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < chEqSoftDemapLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = chEqSoftDemapLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                    kernelNodeParamsDriver.gridDimX,
                                    kernelNodeParamsDriver.gridDimY, 
                                    kernelNodeParamsDriver.gridDimZ,
                                    kernelNodeParamsDriver.blockDimX, 
                                    kernelNodeParamsDriver.blockDimY, 
                                    kernelNodeParamsDriver.blockDimZ,
                                    kernelNodeParamsDriver.sharedMemBytes,
                                    static_cast<CUstream>(cuStream),
                                    kernelNodeParamsDriver.kernelParams,
                                    kernelNodeParamsDriver.extra));
        }
            

            cuphyStatus_t statusDestroy = cuphyDestroyPuschRxChEq(puschRxChEqHndl);
            if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);


     

        // Wait for kernel to complete execution
        cudaStreamSynchronize(cuStream);

        // Convert to FP32 format for MATLAB readability
        cuphy::tensor_device tOutCoef(CUPHY_C_32F, tCoefArray[0].layout());
        cuphy::tensor_device tOutDataEq(CUPHY_C_32F, tDataEqArray[0].layout());
        cuphy::tensor_device tOutReeDiag(CUPHY_R_32F, tReeDiagInvArray[0].layout());
        cuphy::tensor_device tOutLLR(CUPHY_R_32F, tLLRArray[0].layout());
        cuphy::tensor_device tOutDbg(CUPHY_C_32F, tDbgArray[0].layout());

        cuphyStatus_t tensorConvertStat = cuphyConvertTensor(tOutDataEq.desc().handle(), // dst tensor
                                                             tOutDataEq.addr(),          // dst address
                                                             tDataEqArray[0].desc().handle(),// src tensor
                                                             tDataEqArray[0].addr(),         // src address
                                                             cuStream);                  // CUDA stream
        if(CUPHY_STATUS_SUCCESS != tensorConvertStat) throw cuphy::cuphy_exception(tensorConvertStat);

        tensorConvertStat = cuphyConvertTensor(tOutCoef.desc().handle(),      // dst tensor
                                               tOutCoef.addr(),               // dst address
                                               tCoefArray[0].desc().handle(), // src tensor
                                               tCoefArray[0].addr(),          // src address
                                               cuStream);                     // CUDA stream
        if(CUPHY_STATUS_SUCCESS != tensorConvertStat) throw cuphy::cuphy_exception(tensorConvertStat);

        tensorConvertStat = cuphyConvertTensor(tOutReeDiag.desc().handle(),         // dst tensor
                                               tOutReeDiag.addr(),                  // dst address
                                               tReeDiagInvArray[0].desc().handle(), // src tensor
                                               tReeDiagInvArray[0].addr(),          // src address
                                               cuStream);                           // CUDA stream
        if(CUPHY_STATUS_SUCCESS != tensorConvertStat) throw cuphy::cuphy_exception(tensorConvertStat);

        tensorConvertStat = cuphyConvertTensor(tOutLLR.desc().handle(),      // dst tensor
                                               tOutLLR.addr(),               // dst address
                                               tLLRArray[0].desc().handle(), // src tensor
                                               tLLRArray[0].addr(),          // src address
                                               cuStream);                    // CUDA stream
        if(CUPHY_STATUS_SUCCESS != tensorConvertStat) throw cuphy::cuphy_exception(tensorConvertStat);

        tensorConvertStat = cuphyConvertTensor(tOutDbg.desc().handle(),      // dst tensor
                                               tOutDbg.addr(),               // dst address
                                               tDbgArray[0].desc().handle(), // src tensor
                                               tDbgArray[0].addr(),          // src address
                                               cuStream);                    // CUDA stream
        if(CUPHY_STATUS_SUCCESS != tensorConvertStat) throw cuphy::cuphy_exception(tensorConvertStat);


        // Wait for copy to complete
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(cuStream);

        // Write outputs
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
#if 1
            // Write channel equalizer outputs
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutCoef, "Coef");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDataEq, "DataEq");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutReeDiag, "ReeDiagInv");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutLLR, "LLR");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg, "Dbg");
#else
            // Write channel equalizer outputs
#if 0
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tCoef   , "Coef");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tDataEq , "DataEq");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tReeDiag, "ReeDiag");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tLLR    , "LLR");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tDbg    , "Dbg");
#endif
#endif
        }

        delete[] pTPrmCoef      ;
        delete[] pTPrmReeDiagInv;
        delete[] pTPrmCfoEst    ;
        delete[] pTPrmDbg       ;
        delete[] pTPrmDataEq    ;
        delete[] pTPrmLLR       ;

        // Wait for writes to complete
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(cuStream);

        cudaDeviceSynchronize();
        cudaStreamDestroy(cuStream);
#endif        
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
