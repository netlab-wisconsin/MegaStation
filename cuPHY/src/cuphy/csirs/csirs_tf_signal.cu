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
#include "cuphy.hpp"
#include "cuphy_internal.h"

#include "tensor_desc.hpp"
#include "csirs.hpp"
#include "csirs.cuh"
#include "descrambling.hpp"
#include "descrambling.cuh"

using namespace descrambling;

namespace csirsTf
{

//static constexpr int Ng = 273 * 2 * 3;
static constexpr int Ng_elements = 1664 / 8; //Ng=273*2*3 rounded up to be divible by 32 divided by 8

template <typename SignalType, int SEQ_INDEX, int SHIFT>
__device__ void csirsTfSignalGenHelper(uint lenLKBarPrime, uint lenKPrime, CsirsParams& params, uint idxRB, int8_t* seqTable, uint8_t* goldSeq, CsirsSymbLocRow& rowData, uint16_t numReInFreqDim, SignalType* tfSignal, cuphyCsirsPmWOneLayer_t* d_pmw_params)
{
    uint idxKBarLBar = lenLKBarPrime >> SHIFT;
    uint lenLKPrime  = lenLKBarPrime & (SEQ_INDEX - 1);

    //Reminder: based on the table lenKPrime can be either 1 or 2
    const uint lenKPrime_minus_one = ((lenKPrime - 1) & 0x1);
    const uint lPrime              = lenLKPrime >> lenKPrime_minus_one;
    const uint kPrime              = lenLKPrime & lenKPrime_minus_one;

    // Notes: kOffsets is always 0 expect for row0 {0, 4, 8} and row 3 {0, 2}
    // lIndices and lOffsets arrays each have at most 16 elements with values 0 or 1; could use an uint16_t bit element for each and shift and mask instead
    // cdm group index is 0 for row 0 and sequential starting from 0 for the rest.
    uint kBar = params.ki[rowData.kIndices[idxKBarLBar]] + rowData.kOffsets[idxKBarLBar];
    uint lBar = params.li[rowData.lIndices[idxKBarLBar]] + rowData.lOffsets[idxKBarLBar];

    uint k = kBar + kPrime + idxRB * CUPHY_N_TONES_PER_PRB;
    uint l = lBar + lPrime;

    //alpha is rho if numPorts ==1 otherwise 2*rho
    //rho is 0.5f,1 or 3
    uint mPrime = floorf(idxRB * params.alpha) + kPrime + floorf(kBar * params.rho / 12.0f);

    uint8_t gold_seq_value = goldSeq[l * Ng_elements + ((2 * mPrime) >> 3)] >> ((2 * mPrime) & 7);

    const uint8_t enablePrcdBf = params.enablePrcdBf;
    uint16_t pmwPrmIdx         = 0xFFFF;
    uint8_t nPorts = 0;
    if(enablePrcdBf)
    {
        pmwPrmIdx = params.pmwPrmIdx;
        nPorts    = d_pmw_params[pmwPrmIdx].nPorts;
    }

    // Non-ZP computation
    // Note: seqIndexCount is 1, 2, 4 or 8
    // uint jj = (row == 0) ? 0 : idxKBarLBar; //instead of rowData.cdmGroupIndex[idxKBarLBar]; could also remove from table; pass row as argument
    for(int s = 0; s < SEQ_INDEX; ++s)
    {
        int wf = seqTable[s * 2 * 4 + kPrime];
        int wt = seqTable[s * 2 * 4 + 1 * 4 + lPrime];

        SignalType a;
        a.x = params.beta * wf * wt * sqrt(0.5f) * (1.0f - 2.0f * (gold_seq_value & 0x1));
        a.y = params.beta * wf * wt * sqrt(0.5f) * (1.0f - 2.0f * ((gold_seq_value >> 1) & 0x1));

        uint jj = rowData.cdmGroupIndex[idxKBarLBar];
        uint p  = jj * SEQ_INDEX + s;

        if(enablePrcdBf)
        {
            const SignalType zeroValue = make_complex<SignalType>::create(0,0);
            for(int idx = 0; idx < nPorts; idx++)
            {
                 //tfSignal[k + l *  numReInFreqDim + p * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim + idx * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim]  = __hcmadd(a, d_pmw_params[pmwPrmIdx].matrix[idx], zeroValue); // uncoalesced writes
                 tfSignal[k + l *  numReInFreqDim + idx * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim]  = __hcmadd(a, d_pmw_params[pmwPrmIdx].matrix[idx], zeroValue); // uncoalesced writes
            }
        }
        else
        {
            tfSignal[k + l *  numReInFreqDim + p * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim] = a;
        }
    }
}

__global__ void genScramblingKernel(CsirsParams* csirs_params, int num_params, uint8_t* d_scrambling_seq)
{
    CsirsParams& params = csirs_params[blockIdx.x];
    const int    tid    = threadIdx.x;

    const int symbol            = blockIdx.y;
    uint32_t  c_init_scrambling = ((1 << 10) * (OFDM_SYMBOLS_PER_SLOT * params.idxSlotInFrame + symbol + 1) * (2 * params.scrambId + 1) + params.scrambId) & 0x7FFFFFFF;

    constexpr int max_scrambling_elements = 52; // 272*2*3/32
    __builtin_assume_aligned(d_scrambling_seq, sizeof(uint32_t));
    uint32_t* scrambling_seq = reinterpret_cast<uint32_t*>(d_scrambling_seq + blockIdx.x * OFDM_SYMBOLS_PER_SLOT * Ng_elements + symbol * Ng_elements); // it is 32-bit aligned
    if(tid < max_scrambling_elements)
    {
        uint32_t val        = gold32(c_init_scrambling, tid * 32);
        scrambling_seq[tid] = val;
    }
}

// generate tfSignal
template <typename SignalType>
__global__ void genCsirsTfSignalKernel(SignalType** tfSignalArray,
                                 CsirsParams* csirsParams,
                                 int          numParams,
                                 uint32_t*    offsets,
                                 uint32_t     total_offsets,
                                 uint8_t*     goldSeq,
                                 uint16_t*    numReInFreqDimArray,
                                 cuphyCsirsPmWOneLayer_t* d_pmw_params)
{
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int paramNum    = 0;

    // Offset array is of size CUPHY_CSIRS_MAX_NUM_PARAMS(32) + 1
    if(globalIndex >= total_offsets)
    {
        return;
    }

    // find the CSI-RS parameter number for this thread using offset array
    // ToDo: Consider removing dependency on offset array by repeating offset calculation
    // in kernel code.
    while(globalIndex >= offsets[paramNum + 1])
    {
        ++paramNum;
    }

    // local index for this thread within threads spawned for a CSI-RS parameter set
    int localIndex = globalIndex - offsets[paramNum];

    // get the parameter from array
    CsirsParams& params = csirsParams[paramNum];
    int          rowNum = params.row;

    int         cell_index     = params.cell_index;
    SignalType* tfSignal       = tfSignalArray[cell_index];
    uint16_t    numReInFreqDim = numReInFreqDimArray[cell_index];

    // rowNum is 1 index, the table is 0-indexed
    CsirsSymbLocRow& rowData = constRowDataCsirs[rowNum - 1];

    // get CDM table for Wf/wt value
    uint    cdmTableLoc = (uint8_t)(params.cdmType);
    int8_t* seqTable    = &constSeqTableCsirs[cdmTableLoc][0][0][0];

    uint nRB         = params.nRb;
    uint lenKBarLBar = rowData.lenKBarLBar;
    uint lenKPrime   = rowData.lenKPrime;
    uint lenLPrime   = rowData.lenLPrime;

    // needs to be done as number of threads for a parameter are warp size aligned
    // to avoid divergence
    if(localIndex >= nRB * lenKBarLBar * lenKPrime * lenLPrime)
    {
        return;
    }

    // get index values for kBar-LBar array, lPrime and kPrime
    uint idxRB         = localIndex / (lenKBarLBar * lenKPrime * lenLPrime);
    uint lenLKBarPrime = localIndex - idxRB * lenKBarLBar * lenKPrime * lenLPrime;

#if 0
    uint idxKBarLBar = lenLKBarPrime / (lenKPrime * lenLPrime);
    uint lenLKPrime = lenLKBarPrime - idxKBarLBar * lenKPrime * lenLPrime;

    //uint lPrime = lenLKPrime / lenKPrime;
    //uint kPrime = lenLKPrime - lenKPrime * lPrime;
    //Reminder: based on the table lenKPrime can be either 1 or 2
    //Reminder: based on the table lenKPrime can be either 1 or 2
    const uint lenKPrime_minus_one = ((lenKPrime - 1) & 0x1);
    const uint lPrime = lenLKPrime >> lenKPrime_minus_one;
    const uint kPrime = lenLKPrime & lenKPrime_minus_one;

    idxRB += params.startRb;

    bool isEvenRB = (idxRB & 1) == 0;
    if(params.rho == 0.5f) // need to write alternative RB
    {
        if ((params.genEvenRB && !isEvenRB) || (!params.genEvenRB && isEvenRB))
            return;
    }

    uint kBar = params.ki[rowData.kIndices[idxKBarLBar]] + rowData.kOffsets[idxKBarLBar];
    uint lBar = params.li[rowData.lIndices[idxKBarLBar]] + rowData.lOffsets[idxKBarLBar];

    uint k = kBar + kPrime + idxRB * CUPHY_N_TONES_PER_PRB;
    uint l = lBar + lPrime;

    //FIXME alpha is rho if numPorts ==1 otherwise 2*rho
    //rho is 0.5f,1 or 3 -> 1, 2, or 6
    uint mPrime = floorf(idxRB * params.alpha) + kPrime + floorf(kBar * params.rho/12.0f);

    // For ZP CSI-RS use following code and return
    // tfSignal[k + l *  273 * CUPHY_N_TONES_PER_PRB] = 1;

    // Non-ZP computation
    // Note: seqIndexCount is 1, 2, 4 or 8
    for(int s = 0; s < params.seqIndexCount; ++s)
    {
        int wf = seqTable[s*2*4 + kPrime];
        int wt = seqTable[s*2*4 + 1*4 + lPrime];

        SignalType a;
        a.x = params.beta * wf * wt * sqrt(0.5f)*(1.0f-2.0f*goldSeq[paramNum * OFDM_SYMBOLS_PER_SLOT * Ng + l * Ng + 2*mPrime]);
        a.y = params.beta * wf * wt * sqrt(0.5f)*(1.0f-2.0f*goldSeq[paramNum * OFDM_SYMBOLS_PER_SLOT * Ng + l * Ng + 2*mPrime+1]);;

        uint jj = rowData.cdmGroupIndex[idxKBarLBar];
        uint p = jj * params.seqIndexCount + s;

        tfSignal[k + l *  numReInFreqDim + p * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim] = a;
    }
#else
    idxRB += params.startRb;
    bool isEvenRB = (idxRB & 1) == 0;
    if(params.rho == 0.5f) // need to write alternative RB
    {
        if((params.genEvenRB && !isEvenRB) || (!params.genEvenRB && isEvenRB))
            return;
    }
    
    if (params.seqIndexCount == 1) {
        csirsTfSignalGenHelper<SignalType, 1, 0>(lenLKBarPrime, lenKPrime, params, idxRB, seqTable, goldSeq + paramNum * OFDM_SYMBOLS_PER_SLOT * Ng_elements, rowData, numReInFreqDim, tfSignal, d_pmw_params);
    } else if (params.seqIndexCount == 2) {
        csirsTfSignalGenHelper<SignalType, 2, 1>(lenLKBarPrime, lenKPrime, params, idxRB, seqTable, goldSeq +  paramNum * OFDM_SYMBOLS_PER_SLOT * Ng_elements, rowData, numReInFreqDim, tfSignal, d_pmw_params);
    } else if (params.seqIndexCount == 4) {
        csirsTfSignalGenHelper<SignalType, 4, 2>(lenLKBarPrime, lenKPrime, params, idxRB, seqTable, goldSeq +  paramNum * OFDM_SYMBOLS_PER_SLOT * Ng_elements, rowData, numReInFreqDim, tfSignal, d_pmw_params);
    } else { //  (params.seqIndexCount == 8)
        csirsTfSignalGenHelper<SignalType, 8, 3>(lenLKBarPrime, lenKPrime, params, idxRB, seqTable, goldSeq +  paramNum * OFDM_SYMBOLS_PER_SLOT * Ng_elements, rowData, numReInFreqDim, tfSignal, d_pmw_params);
    }
#endif
}

} // namespace csirsTf

using namespace csirsTf;

cuphyStatus_t cuphyCsirsKernelSelect(cuphyGenScramblingLaunchCfg_t*    pGenCsirsScramblingLaunchCfg,
                                     cuphyGenCsirsTfSignalLaunchCfg_t* pGenCsirsTfSignalLaunchCfg,
                                     uint32_t                          numParams)
{
    if(!pGenCsirsScramblingLaunchCfg || !pGenCsirsTfSignalLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    kernelSelectGenScrambling(pGenCsirsScramblingLaunchCfg, numParams);
    kernelSelectGenCsirsTfSignal(pGenCsirsTfSignalLaunchCfg);
    return CUPHY_STATUS_SUCCESS;
}

void kernelSelectGenScrambling(cuphyGenScramblingLaunchCfg_t* pLaunchCfg,
                               uint32_t                       numParams)
{
    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(genScramblingKernel);
    CUDA_CHECK_EXCEPTION_TRY_RESET_ERROR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));

    // launch geometry (can change!)
    dim3 gridDim(numParams, OFDM_SYMBOLS_PER_SLOT);
    dim3 blockDim(64);

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra    = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

void kernelSelectGenCsirsTfSignal(cuphyGenCsirsTfSignalLaunchCfg_t* pLaunchCfg)
{
    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(genCsirsTfSignalKernel<__half2>);
    CUDA_CHECK_EXCEPTION_TRY_RESET_ERROR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));

    // launch geometry
    uint32_t total_offsets = pLaunchCfg->totalNumThreadsLB;
    dim3 blockDim(128);
    dim3 gridDim(div_round_up(total_offsets, blockDim.x));

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}
