/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>
#include <queue>
#include <condition_variable>
#include <bitset>
#include <map>
#include <cmath>

#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "pycuphy_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace cuphy;

using namespace std::complex_literals;
namespace py = pybind11;

#ifndef AERIAL_PYTHON_POLAR_DECODER_CPP
#define AERIAL_PYTHON_POLAR_DECODER_CPP

typedef struct
{
    void *                inputDevicePtr;             // Pointer to device memory of input LLR
    void *                outputHostPtr;              // Pointer to host memory of output data
    void *                tempOutputHostPtr;          // Pointer to temporary host memory of output data

    uint32_t              numIterations;              // Number of iterations

    cuphyPolarUciSegPrm_t polarUciSegPrm;             // UCI polar segment parameters
} cuphyPolarDecoderPrms_t;

namespace pycuphy {

class PolarDecoder{

public:
    enum DescriptorTypes {
        POL_COMP_CW_TREE              = 0,
        POL_COMP_CW_TREE_ADDRS        = POL_COMP_CW_TREE + 1,
        POL_SEG_DERM_DEITL            = POL_COMP_CW_TREE_ADDRS + 1,
        POL_SEG_DERM_DEITL_CW_ADDRS   = POL_SEG_DERM_DEITL + 1,
        POL_SEG_DERM_DEITL_UCI_ADDRS  = POL_SEG_DERM_DEITL_CW_ADDRS + 1,
        POL_DECODE                    = POL_SEG_DERM_DEITL_UCI_ADDRS + 1,
        POL_DECODE_LLR_ADDRS          = POL_DECODE + 1,
        POL_DECODE_CB_ADDRS           = POL_DECODE_LLR_ADDRS + 1,
        LIST_POL_DECODE_SCRATCH_ADDRS = POL_DECODE_CB_ADDRS + 1,
        N_POLAR_DECODE_DESCR_TYPES    = LIST_POL_DECODE_SCRATCH_ADDRS + 1
    };

    PolarDecoder(const uint64_t inputDevicePtr,
                 const uint64_t outputHostPtr,
                 const uint64_t tempOutputHostPtr);

    ~PolarDecoder();

    py::array_t<float> decode(
        const py::array& inputLLR,
        const uint8_t numCbs,
        const uint8_t zeroInsertFlag,
        const uint32_t E_seg,
        const uint8_t numCrcBits,
        const uint32_t E_cw,
        const uint16_t K_cw,
        const uint16_t N_cw,
        const uint8_t n_cw,
        uint64_t cuStream
    );

    void setProfilingIterations(const uint16_t numIterations);

private:
    cuphyPolarDecoderPrms_t polarDecoderPrms;
    cuphy::linear_alloc<128, cuphy::device_alloc> m_linearAlloc;

    std::array<size_t, N_POLAR_DECODE_DESCR_TYPES> m_dynDescrSizeBytes{};
    std::array<size_t, N_POLAR_DECODE_DESCR_TYPES> m_dynDescrAlignBytes{};
    cuphy::kernelDescrs<N_POLAR_DECODE_DESCR_TYPES> m_dynDescr;

    cuphyCompCwTreeTypesHndl_t     m_compCwTreeTypesHndl;
    cuphyPolSegDeRmDeItlHndl_t     m_polSegDeRmDeItlHndl;
    cuphyPolarDecoderHndl_t        m_polarDecoderHndl;

    void allocateDescr();
    void createComponents();
    void destroy();

    size_t getMaxMem() const;
};


PolarDecoder::PolarDecoder(const uint64_t inputDevicePtr,
                           const uint64_t outputHostPtr,
                           const uint64_t tempOutputHostPtr) :
    m_linearAlloc(getMaxMem()),
    m_dynDescr("PycuphyPolarDecodeDynDescr") {
    polarDecoderPrms.inputDevicePtr = (void * )inputDevicePtr;
    polarDecoderPrms.outputHostPtr = (void * )outputHostPtr;
    polarDecoderPrms.tempOutputHostPtr = (void * )tempOutputHostPtr;
    polarDecoderPrms.numIterations = 0;

    allocateDescr();
    createComponents();
}


void PolarDecoder::setProfilingIterations(const uint16_t numIterations) {
    polarDecoderPrms.numIterations = numIterations;
}


PolarDecoder::~PolarDecoder() {
    destroy();
}


void PolarDecoder::destroy() {
    cuphyStatus_t statusDestroyCompCw = cuphyDestroyCompCwTreeTypes(m_compCwTreeTypesHndl);
    if(statusDestroyCompCw != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(statusDestroyCompCw);
    }

    cuphyStatus_t statusDestroyDeRm = cuphyDestroyPolSegDeRmDeItl(m_polSegDeRmDeItlHndl);
    if(statusDestroyDeRm != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(statusDestroyDeRm);
    }

    cuphyStatus_t statusDestroyPolDec = cuphyDestroyPolarDecoder(m_polarDecoderHndl);
    if(statusDestroyPolDec != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(statusDestroyPolDec);
    }
}


size_t PolarDecoder::getMaxMem() const {

    int polarListSz = 1; // TODO: Allow polarListSz to be set based on a static parameter
                         // (e.g. defined in cuphyPolarDecoderPrms_t).

    size_t maxCwPrmsMem = sizeof(cuphyPolarCwPrm_t) * CUPHY_MAX_N_POL_CWS;
    size_t maxNumRmBits = 273 * 12 * 14 * 8 * 2; // maxPrb x scPerPrb x symbolPerSlot x maxLayers x maxQamBits
    size_t maxCwTreeLlrsMem = (2 * maxNumRmBits - 1) * polarListSz * sizeof(__half);
    size_t maxCwN = 1024;
    size_t maxCwTreeTypesMem = sizeof(uint8_t) * (2 * maxCwN) * polarListSz * CUPHY_MAX_N_POL_UCI_SEGS;
    size_t maxScratchBuffer  = polarListSz > 1 ? sizeof(bool) * (2 * maxCwN) * polarListSz : 0;
    size_t maxOutMem = (maxCwN / 32) * CUPHY_MAX_N_POL_CWS * sizeof(uint32_t);
    size_t maxMem = maxCwPrmsMem + maxCwTreeLlrsMem + maxCwTreeTypesMem +  maxScratchBuffer + maxOutMem;
    return maxMem;

}


void PolarDecoder::createComponents() {

    cuphyStatus_t statusCreateCompCw = cuphyCreateCompCwTreeTypes(&m_compCwTreeTypesHndl);
    if(statusCreateCompCw != CUPHY_STATUS_SUCCESS) {
         throw cuphy::cuphy_exception(statusCreateCompCw);
    }

    cuphyStatus_t statusCreateDeRm = cuphyCreatePolSegDeRmDeItl(&m_polSegDeRmDeItlHndl);
    if(statusCreateDeRm != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(statusCreateDeRm);
    }

    cuphyStatus_t statusCreate = cuphyCreatePolarDecoder(&m_polarDecoderHndl);
    if(statusCreate != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(statusCreate);
    }
}


void PolarDecoder::allocateDescr() {

    size_t* pDynDescrSizeBytes = m_dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes = m_dynDescrAlignBytes.data();

    cuphyStatus_t status;
    status = cuphyCompCwTreeTypesGetDescrInfo(&pDynDescrSizeBytes[POL_COMP_CW_TREE], &pDynDescrAlignBytes[POL_COMP_CW_TREE]);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCompCwTreeTypesGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_COMP_CW_TREE_ADDRS] = sizeof(uint8_t**) * CUPHY_MAX_N_POL_UCI_SEGS;
    pDynDescrAlignBytes[POL_COMP_CW_TREE_ADDRS] = alignof(uint8_t**);

    status = cuphyPolSegDeRmDeItlGetDescrInfo(&pDynDescrSizeBytes[POL_SEG_DERM_DEITL], &pDynDescrAlignBytes[POL_SEG_DERM_DEITL]);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyPolSegDeRmDeItlGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_SEG_DERM_DEITL_CW_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS;
    pDynDescrAlignBytes[POL_SEG_DERM_DEITL_CW_ADDRS] = alignof(__half*);

    pDynDescrSizeBytes[POL_SEG_DERM_DEITL_UCI_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_UCI_SEGS;
    pDynDescrAlignBytes[POL_SEG_DERM_DEITL_UCI_ADDRS] = alignof(__half*);

    status = cuphyPolarDecoderGetDescrInfo(&pDynDescrSizeBytes[POL_DECODE], &pDynDescrAlignBytes[POL_DECODE]);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyPolarDecoderGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_DECODE_LLR_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS;
    pDynDescrAlignBytes[POL_DECODE_LLR_ADDRS] = alignof(__half*);

    pDynDescrSizeBytes[POL_DECODE_CB_ADDRS]  = sizeof(uint32_t*) * CUPHY_MAX_N_POL_CWS;
    pDynDescrAlignBytes[POL_DECODE_CB_ADDRS] = alignof(uint32_t*);

    pDynDescrSizeBytes[LIST_POL_DECODE_SCRATCH_ADDRS]  = sizeof(bool*) * CUPHY_MAX_N_POL_CWS;
    pDynDescrAlignBytes[LIST_POL_DECODE_SCRATCH_ADDRS] = alignof(bool*);

    m_dynDescr.alloc(m_dynDescrSizeBytes, m_dynDescrAlignBytes);
}


py::array_t<float> PolarDecoder::decode(const py::array& inputLLR,
                                        const uint8_t numCbs,
                                        const uint8_t zeroInsertFlag,
                                        const uint32_t E_seg,
                                        const uint8_t numCrcBits,
                                        const uint32_t E_cw,
                                        const uint16_t K_cw,
                                        const uint16_t N_cw,
                                        const uint8_t n_cw,
                                        uint64_t cuStream) {

    size_t* pDynDescrSizeBytes = m_dynDescrSizeBytes.data();
    auto dynCpuDescrStartAddrs = m_dynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_dynDescr.getGpuStartAddrs();

    // Input tensor.
    cuphy::tensor_device inputLLRTensor = deviceFromNumpy<float>(inputLLR,
                                                                 polarDecoderPrms.inputDevicePtr,
                                                                 CUPHY_R_32F,
                                                                 CUPHY_R_16F,
                                                                 cuphy::tensor_flags::align_tight,
                                                                 (cudaStream_t)cuStream);

    uint16_t nPolUciSegs = 1; // TODO: Remove hard coding later.
    int polarListSz = 1;      // TODO: Allow polarListSz to be set based on a static parameter
                              // (e.g. defined in cuphyPolarDecoderPrms_t).

    // Set polar encoded UCI segment parameters.
    polarDecoderPrms.polarUciSegPrm.nCbs = numCbs;
    polarDecoderPrms.polarUciSegPrm.zeroInsertFlag = zeroInsertFlag;
    polarDecoderPrms.polarUciSegPrm.E_seg = E_seg;
    polarDecoderPrms.polarUciSegPrm.nCrcBits = numCrcBits;
    polarDecoderPrms.polarUciSegPrm.E_cw = E_cw;
    polarDecoderPrms.polarUciSegPrm.K_cw = K_cw;
    polarDecoderPrms.polarUciSegPrm.N_cw = N_cw;
    polarDecoderPrms.polarUciSegPrm.n_cw = n_cw;
    polarDecoderPrms.polarUciSegPrm.childCbIdxs[0] = 0;
    polarDecoderPrms.polarUciSegPrm.childCbIdxs[1] = 0;

    // Setup components.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyCompCwTreeTypesLaunchCfg_t compCwTreeTypesLaunchCfg;
    cuphyPolSegDeRmDeItlLaunchCfg_t polSegDeRmDeItlLaunchCfg;
    cuphyPolarDecoderLaunchCfg_t polarDecoderLaunchCfg;

    std::vector<cuphyPolarUciSegPrm_t> pPolUciSegPrmsCpuVec;
    for(uint16_t segIdx; segIdx < nPolUciSegs; segIdx++) {
        pPolUciSegPrmsCpuVec.push_back(polarDecoderPrms.polarUciSegPrm);
    }
    cuphyPolarUciSegPrm_t* pPolUciSegPrmsCpu = pPolUciSegPrmsCpuVec.data();

    std::vector<__half*>cwLLRsAddrVec(CUPHY_MAX_N_POL_CWS);
    std::vector<cuphyPolarCwPrm_t>cwPrmsVec(CUPHY_MAX_N_POL_CWS);

    uint16_t nPolCws = 0;
    for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx) {
        for(int i = 0; i < pPolUciSegPrmsCpu[segIdx].nCbs; ++i) {
            size_t nBytes = pPolUciSegPrmsCpu[segIdx].N_cw * sizeof(__half);
            cwLLRsAddrVec[nPolCws] = static_cast<__half*>(m_linearAlloc.alloc(nBytes));
            cwPrmsVec[nPolCws].N_cw = pPolUciSegPrmsCpu[segIdx].N_cw;

            pPolUciSegPrmsCpu[segIdx].childCbIdxs[i] = nPolCws;
            nPolCws += 1;
        }
    }

    // Copy polUciSegPrms to GPU.
    cuphyPolarUciSegPrm_t* pPolUciSegPrmsGpu = static_cast<cuphyPolarUciSegPrm_t*>(m_linearAlloc.alloc(sizeof(cuphyPolarUciSegPrm_t) * nPolUciSegs));
    cudaMemcpyAsync(static_cast<void*>(pPolUciSegPrmsGpu),
                    static_cast<void*>(pPolUciSegPrmsCpu),
                    sizeof(cuphyPolarUciSegPrm_t) * nPolUciSegs,
                    cudaMemcpyHostToDevice,
                    (cudaStream_t)cuStream);
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Allocate GPU memory for cwTreeTypes.
    std::vector<uint8_t*> cwTreeTypesAddrVec(nPolUciSegs);
    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx) {
        uint16_t N_cw = pPolUciSegPrmsCpu[segIdx].N_cw;
        cwTreeTypesAddrVec[segIdx] = static_cast<uint8_t*>(m_linearAlloc.alloc(2 * N_cw));
    }

    std::vector<__half*> uciSegLLRsAddrVec(nPolUciSegs);
    for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx) {
        size_t nBytes = polarDecoderPrms.polarUciSegPrm.E_seg * sizeof(__half);
        uciSegLLRsAddrVec[segIdx] = static_cast<__half*>(m_linearAlloc.alloc(nBytes));
        cudaMemcpyAsync(uciSegLLRsAddrVec[segIdx], inputLLRTensor.addr(), nBytes, cudaMemcpyDeviceToDevice, (cudaStream_t)cuStream);
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Codeword parameters
    std::vector<cuphyPolarCwPrm_t> cwPrmsCpuVec(nPolCws);
    uint16_t cwIdx = 0;
    for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx) {
        for(int i = 0; i < polarDecoderPrms.polarUciSegPrm.nCbs; ++i) {
            cwPrmsCpuVec[cwIdx].N_cw         = polarDecoderPrms.polarUciSegPrm.N_cw;
            cwPrmsCpuVec[cwIdx].nCrcBits     = polarDecoderPrms.polarUciSegPrm.nCrcBits;
            cwPrmsCpuVec[cwIdx].A_cw         = polarDecoderPrms.polarUciSegPrm.K_cw - polarDecoderPrms.polarUciSegPrm.nCrcBits;
            cwPrmsCpuVec[cwIdx].pCwTreeTypes = cwTreeTypesAddrVec[segIdx];
            cwIdx += 1;
        }
    }

    cuphyPolarCwPrm_t* pCwPrmsGpu = static_cast<cuphyPolarCwPrm_t*>(m_linearAlloc.alloc(nPolCws * sizeof(cuphyPolarCwPrm_t)));
    cudaMemcpyAsync(pCwPrmsGpu,
                    cwPrmsCpuVec.data(),
                    nPolCws * sizeof(cuphyPolarCwPrm_t),
                    cudaMemcpyHostToDevice,
                    (cudaStream_t)cuStream);
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    cuphyStatus_t compCwTreeTypesSetupStatus = cuphySetupCompCwTreeTypes(m_compCwTreeTypesHndl,
                                                                         nPolUciSegs,
                                                                         pPolUciSegPrmsCpu,
                                                                         pPolUciSegPrmsGpu,
                                                                         cwTreeTypesAddrVec.data(),
                                                                         static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE]),
                                                                         static_cast<void*>(dynGpuDescrStartAddrs[POL_COMP_CW_TREE]),
                                                                         static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE_ADDRS]),
                                                                         enableCpuToGpuDescrAsyncCpy,
                                                                         &compCwTreeTypesLaunchCfg,
                                                                         (cudaStream_t)cuStream);
    if(compCwTreeTypesSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(compCwTreeTypesSetupStatus);
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        cudaMemcpyAsync(static_cast<void*>(dynGpuDescrStartAddrs[POL_COMP_CW_TREE]),
                        static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE]),
                        pDynDescrSizeBytes[POL_COMP_CW_TREE],
                        cudaMemcpyHostToDevice,
                        (cudaStream_t)cuStream);
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Launch kernel using the CUDA driver API.
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriverCompCw = compCwTreeTypesLaunchCfg.kernelNodeParamsDriver;
    CUresult compCwTreeTypesRunStatus = cuLaunchKernel(kernelNodeParamsDriverCompCw.func,
                                                       kernelNodeParamsDriverCompCw.gridDimX,
                                                       kernelNodeParamsDriverCompCw.gridDimY,
                                                       kernelNodeParamsDriverCompCw.gridDimZ,
                                                       kernelNodeParamsDriverCompCw.blockDimX,
                                                       kernelNodeParamsDriverCompCw.blockDimY,
                                                       kernelNodeParamsDriverCompCw.blockDimZ,
                                                       kernelNodeParamsDriverCompCw.sharedMemBytes,
                                                       (cudaStream_t)cuStream,
                                                       kernelNodeParamsDriverCompCw.kernelParams,
                                                       kernelNodeParamsDriverCompCw.extra);
    if(compCwTreeTypesRunStatus != CUDA_SUCCESS) {
        throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    }

    cuphyStatus_t polSegDeRmDeItlSetupStatus = cuphySetupPolSegDeRmDeItl(m_polSegDeRmDeItlHndl,
                                                                         nPolUciSegs,
                                                                         nPolCws,
                                                                         pPolUciSegPrmsCpu,
                                                                         pPolUciSegPrmsGpu,   
                                                                         cwPrmsCpuVec.data(),
                                                                         pCwPrmsGpu,
                                                                         uciSegLLRsAddrVec.data(),
                                                                         cwLLRsAddrVec.data(),
                                                                         static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                                                                         static_cast<void*>(dynGpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                                                                         static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_CW_ADDRS]),
                                                                         static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_UCI_ADDRS]),
                                                                         enableCpuToGpuDescrAsyncCpy,
                                                                         &polSegDeRmDeItlLaunchCfg,
                                                                         (cudaStream_t)cuStream);
    if(polSegDeRmDeItlSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(polSegDeRmDeItlSetupStatus);
    }
    if(!enableCpuToGpuDescrAsyncCpy) {
        cudaMemcpyAsync(static_cast<void*>(dynGpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                        static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                        pDynDescrSizeBytes[POL_SEG_DERM_DEITL],
                        cudaMemcpyHostToDevice,
                        (cudaStream_t)cuStream);
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriverDeRm = polSegDeRmDeItlLaunchCfg.kernelNodeParamsDriver;
    CUresult polSegDeRmDeItlRunStatus = cuLaunchKernel(kernelNodeParamsDriverDeRm.func,
                                                       kernelNodeParamsDriverDeRm.gridDimX,
                                                       kernelNodeParamsDriverDeRm.gridDimY,
                                                       kernelNodeParamsDriverDeRm.gridDimZ,
                                                       kernelNodeParamsDriverDeRm.blockDimX,
                                                       kernelNodeParamsDriverDeRm.blockDimY,
                                                       kernelNodeParamsDriverDeRm.blockDimZ,
                                                       kernelNodeParamsDriverDeRm.sharedMemBytes,
                                                       (cudaStream_t)cuStream,
                                                       kernelNodeParamsDriverDeRm.kernelParams,
                                                       kernelNodeParamsDriverDeRm.extra);
    if(polSegDeRmDeItlRunStatus != CUDA_SUCCESS) {
        throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    uint16_t nPolCwsPolDec = 0;
    std::vector<__half*> cwTreeLLRsGpuAddrVec(CUPHY_MAX_N_POL_CWS);
    for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx) {
        uint16_t N_cw             = pPolUciSegPrmsCpu[segIdx].N_cw;
        uint8_t  nCbs             = pPolUciSegPrmsCpu[segIdx].nCbs;
        //size_t   nBytesCwTree     = sizeof(uint8_t) * (2 * N_cw);
        size_t   nBytesCwLLRs     = sizeof(__half) * N_cw;
        size_t   nBytesCwTreeLLRs = sizeof(__half) * (2 * N_cw) * polarListSz; // More storage is required for list decoder, proportional to the list size.

        for(int i = 0; i < nCbs; ++i) {
            cwTreeLLRsGpuAddrVec[nPolCwsPolDec] = static_cast<__half*>(m_linearAlloc.alloc(nBytesCwTreeLLRs));
            uint16_t offset = N_cw;
            cudaMemcpyAsync(cwTreeLLRsGpuAddrVec[nPolCwsPolDec] + offset,
                            cwLLRsAddrVec.data()[nPolCwsPolDec],
                            nBytesCwLLRs,
                            cudaMemcpyDeviceToDevice,
                            (cudaStream_t)cuStream);
            nPolCwsPolDec += 1;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // GPU scratch buffers used in list decoder.
    std::vector<bool*> listPolScratchGpuAddrVec;
    if (polarListSz > 1) {
        listPolScratchGpuAddrVec.resize(nPolCws);
        for(int cbIdx = 0; cbIdx < nPolCws; ++cbIdx) {
            size_t nBytesScratch = sizeof(bool) * (2 * cwPrmsCpuVec[cbIdx].N_cw) * polarListSz;
            listPolScratchGpuAddrVec[cbIdx] = static_cast<bool*>(m_linearAlloc.alloc(nBytesScratch));
        }
    }

    // GPU output buffers
    uint8_t* pCrcErrorFlags = static_cast<uint8_t*>(m_linearAlloc.alloc(nPolCwsPolDec));
    std::vector<uint32_t*> cbEstsGpuAddrVec(nPolCwsPolDec);

    uint16_t resDim = div_round_up(cwPrmsCpuVec[0].A_cw, (uint16_t)32);

    for(int cbIdx = 0; cbIdx < nPolCwsPolDec; ++cbIdx) {
        uint16_t nWords = div_round_up(cwPrmsCpuVec[cbIdx].A_cw, static_cast<uint16_t>(32));
        cbEstsGpuAddrVec[cbIdx] = static_cast<uint32_t*>(m_linearAlloc.alloc(nWords * sizeof(uint32_t)));
    }

    cuphyStatus_t polarDecoderSetupStatus = cuphySetupPolarDecoder(m_polarDecoderHndl,
                                                                   nPolCwsPolDec,
                                                                   cwTreeLLRsGpuAddrVec.data(),
                                                                   pCwPrmsGpu,
                                                                   cwPrmsCpuVec.data(),
                                                                   cbEstsGpuAddrVec.data(),
                                                                   listPolScratchGpuAddrVec.data(),
                                                                   polarListSz,
                                                                   pCrcErrorFlags,
                                                                   static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                   static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE]),
                                                                   static_cast<void*>(dynGpuDescrStartAddrs[POL_DECODE]),
                                                                   static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_LLR_ADDRS]),
                                                                   static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_CB_ADDRS]),
                                                                   static_cast<void*>(dynCpuDescrStartAddrs[LIST_POL_DECODE_SCRATCH_ADDRS]),
                                                                   &polarDecoderLaunchCfg,
                                                                   (cudaStream_t)cuStream);

    if(polarDecoderSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_exception(polarDecoderSetupStatus);
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        cudaMemcpyAsync(static_cast<void*>(dynGpuDescrStartAddrs[POL_DECODE]),
                        static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE]),
                        pDynDescrSizeBytes[POL_DECODE],
                        cudaMemcpyHostToDevice,
                        (cudaStream_t)cuStream);
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    CUresult polarDecoderRunStatus;
    if(!polarDecoderPrms.numIterations) {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = polarDecoderLaunchCfg.kernelNodeParamsDriver;
        polarDecoderRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
                                               kernelNodeParamsDriver.gridDimX,
                                               kernelNodeParamsDriver.gridDimY,
                                               kernelNodeParamsDriver.gridDimZ,
                                               kernelNodeParamsDriver.blockDimX,
                                               kernelNodeParamsDriver.blockDimY,
                                               kernelNodeParamsDriver.blockDimZ,
                                               kernelNodeParamsDriver.sharedMemBytes,
                                               (cudaStream_t)cuStream,
                                               kernelNodeParamsDriver.kernelParams,
                                               kernelNodeParamsDriver.extra);
        if(polarDecoderRunStatus != CUDA_SUCCESS) {
            throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        }
    }
    else {
        cudaEvent_t start, stop;
        float time = 0.0f;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for(int i = 0; i < polarDecoderPrms.numIterations; i++) {

            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = polarDecoderLaunchCfg.kernelNodeParamsDriver;
            polarDecoderRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
                                                   kernelNodeParamsDriver.gridDimX,
                                                   kernelNodeParamsDriver.gridDimY,
                                                   kernelNodeParamsDriver.gridDimZ,
                                                   kernelNodeParamsDriver.blockDimX,
                                                   kernelNodeParamsDriver.blockDimY,
                                                   kernelNodeParamsDriver.blockDimZ,
                                                   kernelNodeParamsDriver.sharedMemBytes,
                                                   (cudaStream_t)cuStream,
                                                   kernelNodeParamsDriver.kernelParams,
                                                   kernelNodeParamsDriver.extra);
            if(polarDecoderRunStatus != CUDA_SUCCESS) {
                throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        double tput = ((double)polarDecoderPrms.polarUciSegPrm.K_cw * (double)polarDecoderPrms.numIterations * (double)nPolUciSegs) / (1000.0f * time);
        std::cout << "Total time from C++ is " << time * 1000 << " us." << std::endl;
        std::cout << "Internal throughput is " << tput << " Mbps." << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Device to host memory transfer
    uint32_t dim0 = resDim;
    uint32_t dim1 = 1;
    cuphy::tensor_device dOutputTensor = tensor_device(cbEstsGpuAddrVec.data()[0], CUPHY_R_32U, dim0, dim1, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned hOutputTensor = tensor_pinned(polarDecoderPrms.tempOutputHostPtr, CUPHY_R_32U, dim0, dim1, cuphy::tensor_flags::align_tight);

    hOutputTensor.convert(dOutputTensor, (cudaStream_t)cuStream);
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Unpack the bits to float for Numpy.
    toNumpyBitArray<uint32_t>((uint32_t*)polarDecoderPrms.tempOutputHostPtr,
                              (float*)polarDecoderPrms.outputHostPtr,
                              dim0 * 32,
                              dim1);

    // Return the Numpy array.
    return hostToNumpy<float>((float*)polarDecoderPrms.outputHostPtr, cwPrmsCpuVec[0].A_cw, dim1);
}


}

#endif