/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cuda_runtime.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <bitset>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "cuphy_channels.hpp"
#include "pucch_rx.hpp"
#include "gen_pucch_perf_curve.hpp"
#include <math.h>
#include <fstream>
#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir
#include "fading_chan.cuh"
using namespace std;

void gen_pucch_perf_curve(std::string pucchInputFilename, std::string pucchDbgFilename, std::string perfOutFilename, uint8_t  formatType, std::vector<float> snrVec, uint32_t nItrPerSnr, uint8_t mode, uint8_t seed)
{
    // Initialize main stream
    cuphy::stream cuStrmMain;

    // Input file
    hdf5hpp::hdf5_file inputFile            = hdf5hpp::hdf5_file::open(pucchInputFilename.c_str());
    
    // Setup parameters
    float TTIlen = 0.0005f; // legnth of each TTI, assuming numorology 1
    hdf5hpp::hdf5_dataset dset_carrier              = inputFile.open_dataset("carrier_pars");
    hdf5hpp::hdf5_dataset_elem dset_carrier_elem    = dset_carrier[0];
    uint16_t N_txLayer                              = dset_carrier_elem["N_txAnt"].as<uint16_t>();
    hdf5hpp::hdf5_dataset dset_chan                 = inputFile.open_dataset("chan_pars");
    hdf5hpp::hdf5_dataset_elem dset_chan_elem       = dset_chan[0];
    // chanType: 0 - AWGN, 1 - TDL, 2 - CDL, 3 - P2P 
    uint8_t channType                               = dset_chan_elem["chanType"].as<uint8_t>();
    
    // validate channType
    // currently only AWGN and TDL are implemented
    assert (channType == 0 || channType == 1);

    // Open output files
    std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
    if(!pucchDbgFilename.empty())
    {
        debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(pucchDbgFilename.c_str())));
    }
    
    std::unique_ptr<hdf5hpp::hdf5_file> perfFile;
    bool                     perfOutputFlag = false;
    hdf5hpp::hdf5_file       perfHdf5File;
    if(!perfOutFilename.empty())
    {
        perfOutputFlag = true;
        perfFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(perfOutFilename.c_str())));
        perfHdf5File = hdf5hpp::hdf5_file::open(perfOutFilename.c_str());
    }
    
    // -------------------------------------------------------------------
    // Load PUCCH datasets

    std::vector<std::string> pucchInputFilenameVec(1);
    pucchInputFilenameVec[0] = pucchInputFilename;

    uint64_t procModeBmsk = 0;
    pucchStaticApiDataset  statPucchApiDataset(pucchInputFilenameVec, cuStrmMain.handle(), pucchDbgFilename);
    pucchDynApiDataset     dynPucchApiDataset (pucchInputFilenameVec, cuStrmMain.handle(), procModeBmsk);
    EvalPucchDataset       evalPucchDataset   (pucchInputFilenameVec, cuStrmMain.handle());
    cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied
    // printf("nF0Ucis = %d, nF1Ucis = %d, nF2Ucis = %d, nF3Ucis = %d\n", evalPucchDataset.nF0Ucis ,evalPucchDataset.nF1Ucis ,evalPucchDataset.nF2Ucis ,evalPucchDataset.nF3Ucis);
    cuphyPucchDynPrms_t&  pucchDynPrm   = dynPucchApiDataset.pucchDynPrm;
    cuphyPucchStatPrms_t& pucchStatPrms =  statPucchApiDataset.pucchStatPrms;

    // Recieved signal buffers
    cuphy::rng rng(seed, 0, cuStrmMain.handle());
    uint32_t nSubcarriers  = statPucchApiDataset.cellStatPrm[0].nPrbUlBwp*12;
    uint32_t nGnbAnt       = statPucchApiDataset.cellStatPrm[0].nRxAnt;
    uint32_t slotSizeBytes = nSubcarriers * nGnbAnt * OFDM_SYMBOLS_PER_SLOT * sizeof(__half2);
    
    cuphy::tensor_ref tNoiseFreeTxSlot;
    tNoiseFreeTxSlot.desc().set(CUPHY_C_16F,  nSubcarriers, OFDM_SYMBOLS_PER_SLOT, N_txLayer, cuphy::tensor_flags::align_tight);
    tNoiseFreeTxSlot.set_addr(dynPucchApiDataset.DataIn.pTDataRx->pAddr);
    cuphy::tensor_device tNoisyRxSlot(CUPHY_C_16F, nSubcarriers, OFDM_SYMBOLS_PER_SLOT, nGnbAnt, cuphy::tensor_flags::align_tight);
    dynPucchApiDataset.DataIn.pTDataRx->pAddr = tNoisyRxSlot.addr();

    /*------------------------ Creat fading channel ------------------------*/
    uint8_t phyChannType = 1; // PUCCH
    fadingChan<__half2>* fadeChanPtr = new fadingChan<__half2>(static_cast<__half2*>(tNoiseFreeTxSlot.addr()), static_cast<__half2*>(tNoisyRxSlot.addr()), cuStrmMain.handle(), channType, static_cast<uint16_t>(seed), phyChannType);
    fadeChanPtr->setup(inputFile);

    // Generate perf curves
    int nSnrSteps = snrVec.size();
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tPerfCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tSnrCpu(nSnrSteps);
        
    // Finish setting dynamic parameters
    dynPucchApiDataset.pucchDynPrm.cuStream  = cuStrmMain.handle(); // save stream in dynamic parameters
    dynPucchApiDataset.pucchDynPrm.cpuCopyOn = 1;      // option to copy uci output to CPU immediately after run

    // ---------------------------------------------------------------
    // Create reciever
    cuphyPucchRxHndl_t pucchRxHndl;
        
    cuphyStatus_t statusCreate = cuphyCreatePucchRx(&pucchRxHndl, &pucchStatPrms, cuStrmMain.handle());
    if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);
    cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied
    
    cuphyPucchBatchPrmHndl_t const batchPrmHndl = nullptr;  // batchPrms currently un-used

    // get the number of UCIs per iteration
    uint8_t numUcis;
    cuphyPucchF234OutOffsets_t* pPucchF23OutOffsets = nullptr;
    switch (formatType) {
        case 0:
            numUcis = dynPucchApiDataset.F0UciPrmsVec.size();
            break;
        case 1:
            numUcis = dynPucchApiDataset.F1UciPrmsVec.size();
            break;
        case 2:
            numUcis = dynPucchApiDataset.F2UciPrmsVec.size();
            pPucchF23OutOffsets = pucchDynPrm.pDataOut->pPucchF2OutOffsets;
            break;
        case 3:
            numUcis = dynPucchApiDataset.F3UciPrmsVec.size();
            pPucchF23OutOffsets = pucchDynPrm.pDataOut->pPucchF3OutOffsets;
            break;
        break;
    }
     // test
     //if (formatType == 2) {
    //  printf("formatType = %d\n", formatType);
      //  for (int idx = 0; idx < numUcis; idx++) {
      //      printf("harqPayloadByteOffset = %d\n", pPucchF23OutOffsets[idx].harqPayloadByteOffset);
      //  }
     //}

    for (int snrStepIdx = 0; snrStepIdx < nSnrSteps; snrStepIdx++)
    {
        tPerfCpu(snrStepIdx) = 0;
        float snr            = snrVec[snrStepIdx];
        int numPayloadBits = 0;
        int numUci = 0;

        for (int itrIdx = 0; itrIdx < nItrPerSnr; ++itrIdx)
        {
            // channel fading/noise
            fadeChanPtr -> run(TTIlen * itrIdx, snr);
            cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

            // Setup pucch reciever object
            cuphyStatus_t statusSetup  = cuphySetupPucchRx(pucchRxHndl, &pucchDynPrm, batchPrmHndl);
            if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
            cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

            // Run pucch reciever object
            cuphyStatus_t statusRun = cuphyRunPucchRx(pucchRxHndl, procModeBmsk);
            if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);
            cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

            for (int uciIdx = 0; uciIdx < numUcis; uciIdx++) {
                numUci++;
                switch (formatType) {
                    case 0:
                    {
                        cuphyPucchF0F1UciOut_t& pF0UciOut = pucchDynPrm.pDataOut->pF0UcisOut[uciIdx];
                        numPayloadBits += pF0UciOut.NumHarq;
                        for (int bitIdx = 0; bitIdx < pF0UciOut.NumHarq; bitIdx++) {
                            switch (mode) {
                                case 1:
                                    if (pF0UciOut.HarqValues[bitIdx] == 0) {
                                        tPerfCpu(snrStepIdx) += 1;
                                    }
                                    break;
                                case 3:
                                    if (pF0UciOut.HarqValues[bitIdx] != 0) {
                                        tPerfCpu(snrStepIdx) += 1;
                                    }
                                    break;
                            }
                        }
                        break;
                    }
                    case 1:
                    {
                        cuphyPucchF0F1UciOut_t& pF1UciOut = pucchDynPrm.pDataOut->pF1UcisOut[uciIdx];
                        numPayloadBits += pF1UciOut.NumHarq;
                        for (int bitIdx = 0; bitIdx < pF1UciOut.NumHarq; bitIdx++) {
                             switch (mode) {
                                case 1:                                  
                                case 2:
                                    if (pF1UciOut.HarqValues[bitIdx] == 0) {
                                        tPerfCpu(snrStepIdx) += 1;
                                    }
                                    break;
                                case 3:
                                    if (pF1UciOut.HarqValues[bitIdx] != 0) {
                                        tPerfCpu(snrStepIdx) += 1;
                                    }
                                    break;
                             }
                        }
                        break;
                    }
                    case 2:
                    case 3:
                    { 
                        uint16_t bitLenHarq;
                        uint16_t bitLenCsiPart1;
                        uint16_t bitLenSr;
                        if (formatType == 2) {
                            bitLenHarq     = dynPucchApiDataset.F2UciPrmsVec[uciIdx].bitLenHarq;
                            bitLenCsiPart1 = dynPucchApiDataset.F2UciPrmsVec[uciIdx].bitLenCsiPart1;
                            bitLenSr       = dynPucchApiDataset.F2UciPrmsVec[uciIdx].bitLenSr;
                        } else {
                            bitLenHarq     = dynPucchApiDataset.F3UciPrmsVec[uciIdx].bitLenHarq;
                            bitLenCsiPart1 = dynPucchApiDataset.F3UciPrmsVec[uciIdx].bitLenCsiPart1;
                            bitLenSr       = dynPucchApiDataset.F3UciPrmsVec[uciIdx].bitLenSr;
                        }
                        numPayloadBits += bitLenHarq + bitLenCsiPart1 + bitLenSr;
                        if (mode == 1) { // assumption's that only bitLenHarq > 0
                            uint16_t HarqDetectionStatusOffset = pPucchF23OutOffsets[uciIdx].HarqDetectionStatusOffset;
                            uint8_t HarqDetectionStatus = pucchDynPrm.pDataOut->HarqDetectionStatus[HarqDetectionStatusOffset];
                            if (HarqDetectionStatus == 1 || HarqDetectionStatus == 4) {
                                uint32_t harqPayloadByteOffset = pPucchF23OutOffsets[uciIdx].harqPayloadByteOffset;
                                uint16_t byteLenHarq = ceil(bitLenHarq/8.0);
                                int      numRemBits = bitLenHarq;
                                for (int byteIdx = 0; byteIdx<byteLenHarq; byteIdx++) {
                                    uint8_t     payload = pucchDynPrm.pDataOut->pUciPayloads[harqPayloadByteOffset + byteIdx];
                                    std::string payloadStr = std::bitset<8>(payload).to_string();
                                    if (numRemBits >= 8) {
                                        for (int bitIdx = 0; bitIdx < 8; bitIdx++) {
                                            if (payloadStr[bitIdx] == '0') {
                                                tPerfCpu(snrStepIdx) += 1;
                                            }
                                        }
                                        numRemBits -= 8;
                                    } else {
                                        for (int bitIdx = 0; bitIdx < numRemBits; bitIdx++) {
                                            if (payloadStr[7-bitIdx] == '0') {
                                                tPerfCpu(snrStepIdx) += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        } else if (mode == 3) { // assumption's that only bitLenHarq > 0
                            uint16_t HarqDetectionStatusOffset = pPucchF23OutOffsets[uciIdx].HarqDetectionStatusOffset;
                            uint8_t HarqDetectionStatus = pucchDynPrm.pDataOut->HarqDetectionStatus[HarqDetectionStatusOffset];
                            if (HarqDetectionStatus == 2 || HarqDetectionStatus == 3) {
                                tPerfCpu(snrStepIdx) += bitLenHarq;
                            } else {
                                uint32_t harqPayloadByteOffset = pPucchF23OutOffsets[uciIdx].harqPayloadByteOffset;
                                uint16_t byteLenHarq = ceil(bitLenHarq/8.0);
                                int      numRemBits = bitLenHarq;
                                for (int byteIdx = 0; byteIdx<byteLenHarq; byteIdx++) {
                                    uint8_t     payload = pucchDynPrm.pDataOut->pUciPayloads[harqPayloadByteOffset + byteIdx];
                                    std::string payloadStr = std::bitset<8>(payload).to_string();
                                    if (numRemBits >= 8) {
                                        for (int bitIdx = 0; bitIdx < 8; bitIdx++) {
                                            if (payloadStr[bitIdx] != '0') {
                                                tPerfCpu(snrStepIdx) += 1;
                                            }
                                        }
                                        numRemBits -= 8;
                                    } else {
                                        for (int bitIdx = 0; bitIdx < numRemBits; bitIdx++) {
                                            if (payloadStr[7-bitIdx] != '0') {
                                                tPerfCpu(snrStepIdx) += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        } else if (mode == 4) { // assumption's that bitLenHarq == 0 but bitLenCsiPart1 > 0
                            uint16_t CsiP1DetectionStatusOffset = pPucchF23OutOffsets[uciIdx].CsiP1DetectionStatusOffset;
                            uint8_t CsiP1DetectionStatus = pucchDynPrm.pDataOut->CsiP1DetectionStatus[CsiP1DetectionStatusOffset];
                            if (CsiP1DetectionStatus == 2 || CsiP1DetectionStatus == 3) {
                                tPerfCpu(snrStepIdx) += 1;
                            }
                        }
                        break;
                    }
                }
            }
        }
        if (mode != 4) {
            tPerfCpu(snrStepIdx) = tPerfCpu(snrStepIdx) / static_cast<float>(numPayloadBits);
        } else {
            tPerfCpu(snrStepIdx) = tPerfCpu(snrStepIdx) / static_cast<float>(numUci);
        }
        
        // printf("tPerfCpu = %f\n", tPerfCpu(snrStepIdx));
        tSnrCpu(snrStepIdx) = snr;
    }


    // Save debug output
    if(!pucchDbgFilename.empty())
    {
        cuphyStatus_t statusDebugWrite = cuphyWriteDbgBufSynchPucch(pucchRxHndl, cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != statusDebugWrite) throw cuphy::cuphy_exception(statusDebugWrite);
        cudaStreamSynchronize(cuStrmMain.handle());

    }
    
    // cleanup
    cuphyStatus_t statusDestroy = cuphyDestroyPucchRx(pucchRxHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);
    cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied
    if (fadeChanPtr) {
        delete fadeChanPtr;
    }
    // -----------------------------------------------------------------
    // Save tber to H5

    if(perfOutputFlag)
    {
        cuphy::typed_tensor<CUPHY_R_32F, cuphy::device_alloc> tPerfGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32F, cuphy::device_alloc> tSnrGpu(nSnrSteps);

        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tPerfGpu.addr()), static_cast<void*>(tPerfCpu.addr()), nSnrSteps * sizeof(float), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tSnrGpu.addr()) , static_cast<void*>(tSnrCpu.addr()) , nSnrSteps * sizeof(float), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        cuphy::write_HDF5_dataset(perfHdf5File, tSnrGpu,  "snr", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(perfHdf5File, tPerfGpu, "perf", cuStrmMain.handle());
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        perfHdf5File.close();
    }
}