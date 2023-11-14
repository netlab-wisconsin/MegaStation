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
#include "cuphy.hpp"
#include "cuphy_api.h" // to be removed
#include <algorithm>
#include <new>
#include "bfc.hpp"
#include "ch_est.hpp"
#include "pusch_noise_intf_est.hpp"
#include "cfo_ta_est.hpp"
#include "channel_eq.hpp"
#include "channel_est.hpp"
#include "pusch_rssi.hpp"
#include "rate_matching.hpp"
#include "crc_decode.hpp"
#include "crc_encode.hpp"
#include "convert_tensor.cuh"
#include "ldpc.hpp"
#include "dl_rate_matching.hpp"
#include "modulation_mapper.hpp"
#include "pdsch_dmrs.hpp"
#include "polar_encoder.hpp"
#include "srs_ch_est.hpp"
#include "tensor_desc.hpp"
#include "cuphy_context.hpp"
#include "rm_decoder.hpp"
#include "simplex_decoder.hpp"
#include "pucch_F0_receiver.hpp"
#include "pucch_F1_receiver.hpp"
#include "pucch_F2_front_end.hpp"
#include "pucch_F3_front_end.hpp"
#include "pucch_F3_csi2Ctrl.hpp"
#include "pucch_F3_segLLRs.hpp"
#include "pucch_F234_uci_seg.hpp"
#include "soft_demapper.hpp"
#include "rng.hpp"
#include "variant.hpp"
#include "tensor_fill.hpp"
#include "tensor_tile.hpp"
#include "tensor_elementwise.hpp"
#include "tensor_reduction.hpp"
#include "comp_cwTreeTypes.hpp"
#include "polar_seg_deRm_deItl.hpp"
#include "uciOnPusch_segLLRs1.hpp"
#include "uciOnPusch_segLLRs2.hpp"
#include "uciOnPusch_segLLRs0.hpp"
#include "uciOnPusch_csi2Ctrl.hpp"
#include "polar_decoder.hpp"
#include "srs_chEst0.hpp"
#include "empty_kernels.cuh"

#include <vector>

////////////////////////////////////////////////////////////////////////
// cuphyGetErrorString()
const char* cuphyGetErrorString(cuphyStatus_t status)
{ // clang-format off
    switch (status)
    {
    case CUPHY_STATUS_SUCCESS:               return "Success";
    case CUPHY_STATUS_INTERNAL_ERROR:        return "Internal error";
    case CUPHY_STATUS_NOT_SUPPORTED:         return "An operation was requested that is not currently supported";
    case CUPHY_STATUS_INVALID_ARGUMENT:      return "An invalid argument was provided";
    case CUPHY_STATUS_ARCH_MISMATCH:         return "Requested computation not supported on current architecture";
    case CUPHY_STATUS_ALLOC_FAILED:          return "Memory allocation failed";
    case CUPHY_STATUS_SIZE_MISMATCH:         return "Operand size mismatch";
    case CUPHY_STATUS_MEMCPY_ERROR:          return "Error performing memory copy";
    case CUPHY_STATUS_INVALID_CONVERSION:    return "Invalid data conversion requested";
    case CUPHY_STATUS_UNSUPPORTED_TYPE:      return "Operation requested on unsupported type";
    case CUPHY_STATUS_UNSUPPORTED_LAYOUT:    return "Operation requested on unsupported tensor layout";
    case CUPHY_STATUS_UNSUPPORTED_RANK:      return "Operation requested on unsupported rank";
    case CUPHY_STATUS_UNSUPPORTED_CONFIG:    return "Operation requested using an unsupported configuration";
    case CUPHY_STATUS_UNSUPPORTED_ALIGNMENT: return "One or more API arguments don't have the required alignment";
    case CUPHY_STATUS_VALUE_OUT_OF_RANGE:    return "Data conversion could not occur because an input value was out of range";
    case CUPHY_STATUS_REF_MISMATCH:          return "Mismatch found when comparing to TV";
    default:                                 return "Unknown status value";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// cuphyGetErrorName()
const char* cuphyGetErrorName(cuphyStatus_t status)
{ // clang-format off
    switch (status)
    {
    case CUPHY_STATUS_SUCCESS:               return "CUPHY_STATUS_SUCCESS";
    case CUPHY_STATUS_INTERNAL_ERROR:        return "CUPHY_STATUS_INTERNAL_ERROR";
    case CUPHY_STATUS_NOT_SUPPORTED:         return "CUPHY_STATUS_NOT_SUPPORTED";
    case CUPHY_STATUS_INVALID_ARGUMENT:      return "CUPHY_STATUS_INVALID_ARGUMENT";
    case CUPHY_STATUS_ARCH_MISMATCH:         return "CUPHY_STATUS_ARCH_MISMATCH";
    case CUPHY_STATUS_ALLOC_FAILED:          return "CUPHY_STATUS_ALLOC_FAILED";
    case CUPHY_STATUS_SIZE_MISMATCH:         return "CUPHY_STATUS_SIZE_MISMATCH";
    case CUPHY_STATUS_MEMCPY_ERROR:          return "CUPHY_STATUS_MEMCPY_ERROR";
    case CUPHY_STATUS_INVALID_CONVERSION:    return "CUPHY_STATUS_INVALID_CONVERSION";
    case CUPHY_STATUS_UNSUPPORTED_TYPE:      return "CUPHY_STATUS_UNSUPPORTED_TYPE";
    case CUPHY_STATUS_UNSUPPORTED_LAYOUT:    return "CUPHY_STATUS_UNSUPPORTED_LAYOUT";
    case CUPHY_STATUS_UNSUPPORTED_RANK:      return "CUPHY_STATUS_UNSUPPORTED_RANK";
    case CUPHY_STATUS_UNSUPPORTED_CONFIG:    return "CUPHY_STATUS_UNSUPPORTED_CONFIG";
    case CUPHY_STATUS_UNSUPPORTED_ALIGNMENT: return "CUPHY_STATUS_UNSUPPORTED_ALIGNMENT";
    case CUPHY_STATUS_VALUE_OUT_OF_RANGE:    return "CUPHY_STATUS_VALUE_OUT_OF_RANGE";
    default:                                 return "CUPHY_UNKNOWN_STATUS";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// cuphyGetDataTypeString()
const char* CUPHYWINAPI cuphyGetDataTypeString(cuphyDataType_t t)
{ // clang-format off
    switch(t)
    {
    case CUPHY_VOID:  return "CUPHY_VOID";
    case CUPHY_BIT:   return "CUPHY_BIT";
    case CUPHY_R_16F: return "CUPHY_R_16F";
    case CUPHY_C_16F: return "CUPHY_C_16F";
    case CUPHY_R_32F: return "CUPHY_R_32F";
    case CUPHY_C_32F: return "CUPHY_C_32F";
    case CUPHY_R_8I:  return "CUPHY_R_8I";
    case CUPHY_C_8I:  return "CUPHY_C_8I";
    case CUPHY_R_8U:  return "CUPHY_R_8U";
    case CUPHY_C_8U:  return "CUPHY_C_8U";
    case CUPHY_R_16I: return "CUPHY_R_16I";
    case CUPHY_C_16I: return "CUPHY_C_16I";
    case CUPHY_R_16U: return "CUPHY_R_16U";
    case CUPHY_C_16U: return "CUPHY_C_16U";
    case CUPHY_R_32I: return "CUPHY_R_32I";
    case CUPHY_C_32I: return "CUPHY_C_32I";
    case CUPHY_R_32U: return "CUPHY_R_32U";
    case CUPHY_C_32U: return "CUPHY_C_32U";
    case CUPHY_R_64F: return "CUPHY_R_64F";
    case CUPHY_C_64F: return "CUPHY_C_64F";
    default:          return "UNKNOWN_TYPE";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// cuphyCreateContext()
cuphyStatus_t CUPHYWINAPI cuphyCreateContext(cuphyContext_t* pcontext,
                                             unsigned int    flags)
{
    if(!pcontext)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pcontext = nullptr;
    try
    {
        cuphy_i::context* c = new cuphy_i::context;
        *pcontext           = static_cast<cuphyContext*>(c);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyContext()
cuphyStatus_t CUPHYWINAPI cuphyDestroyContext(cuphyContext_t ctx)
{
    if(!ctx)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cuphy_i::context* c = static_cast<cuphy_i::context*>(ctx);
    delete c;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateTensorDescriptor()
cuphyStatus_t cuphyCreateTensorDescriptor(cuphyTensorDescriptor_t* tensorDesc)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == tensorDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Allocate the descriptor structure
    tensor_desc* tdesc = new(std::nothrow) tensor_desc;
    if(nullptr == tdesc)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    //------------------------------------------------------------------
    // Populate the return address
    *tensorDesc = tdesc;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyTensorDescriptor()
cuphyStatus_t CUPHYWINAPI cuphyDestroyTensorDescriptor(cuphyTensorDescriptor_t tensorDesc)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == tensorDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Free the structure previously allocated by cuphyCreateTensorDescriptor()
    tensor_desc* tdesc = static_cast<tensor_desc*>(tensorDesc);
    delete tdesc;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyGetTensorDescriptor()
cuphyStatus_t CUPHYWINAPI cuphyGetTensorDescriptor(cuphyTensorDescriptor_t tensorDesc,
                                                   int                     numDimsRequested,
                                                   cuphyDataType_t*        dataType,
                                                   int*                    numDims,
                                                   int                     dimensions[],
                                                   int                     strides[])
{
    //------------------------------------------------------------------
    // Validate arguments
    if((nullptr == tensorDesc) ||
       ((numDimsRequested > 0) && (nullptr == dimensions)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    const tensor_desc* tdesc = static_cast<const tensor_desc*>(tensorDesc);
    if(dataType)
    {
        *dataType = tdesc->type();
    }
    if(numDims)
    {
        *numDims = tdesc->layout().rank();
    }
    if((numDimsRequested > 0) && dimensions)
    {
        std::copy(tdesc->layout().dimensions.begin(),
                  tdesc->layout().dimensions.begin() + numDimsRequested,
                  dimensions);
    }
    if((numDimsRequested > 0) && strides)
    {
        std::copy(tdesc->layout().strides.begin(),
                  tdesc->layout().strides.begin() + numDimsRequested,
                  strides);
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetTensorDescriptor()
cuphyStatus_t CUPHYWINAPI cuphySetTensorDescriptor(cuphyTensorDescriptor_t tensorDesc,
                                                   cuphyDataType_t         type,
                                                   int                     numDim,
                                                   const int               dim[],
                                                   const int               str[],
                                                   unsigned int            flags)
{
    //-----------------------------------------------------------------
    // Validate arguments. Validation of dimension/stride values will
    // occur in the call below.
    // Tensor descriptor must be non-NULL.
    // Dimensions array must be non-NULL.
    if(!tensorDesc || !dim)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //-----------------------------------------------------------------
    // If the user passed TIGHT, we use nullptr as an argument to the
    // internal function.
    const int* strArg = is_set(flags, CUPHY_TENSOR_ALIGN_TIGHT) ? nullptr : str;
    //-----------------------------------------------------------------
    // Adjust the strides array using any optional flags. Adjusting for
    // alignment only makes sense when the number of dimensions is
    // greater than 1.
    std::array<int, CUPHY_DIM_MAX> userStrides;

    if(is_set(flags, CUPHY_TENSOR_ALIGN_COALESCE) &&
       (!is_set(flags, CUPHY_TENSOR_ALIGN_TIGHT)) &&
       (numDim > 1))
    {
        // Use the given array of strides as indices into the dimension
        // vector to determine the actual strides.
        userStrides[0] = 1;
        for(int i = 1; i < numDim; ++i)
        {
            userStrides[i] = dim[i - 1] * userStrides[i - 1];
        }
        // Adjust the alignment if necessary
        if(is_set(flags, CUPHY_TENSOR_ALIGN_COALESCE))
        {
            const int COALESCE_BYTES   = 128;
            const int num_elem_aligned = round_up_to_next(dim[0],
                                                          get_element_multiple_for_alignment(COALESCE_BYTES, type));
            userStrides[1]             = num_elem_aligned;
            for(int i = 2; i < numDim; ++i)
            {
                userStrides[i] = dim[i - 1] * userStrides[i - 1];
            }
        }
        // Use the populated array to set the tensor descriptor below
        strArg = userStrides.data();
    }
    //-----------------------------------------------------------------
    // Modify the tensor descriptor using the given arguments
    tensor_desc& tdesc = static_cast<tensor_desc&>(*tensorDesc);
    return tdesc.set(type, numDim, dim, strArg) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INVALID_ARGUMENT;
}

////////////////////////////////////////////////////////////////////////
// cuphyGetTensorSizeInBytes()
cuphyStatus_t CUPHYWINAPI cuphyGetTensorSizeInBytes(cuphyTensorDescriptor_t tensorDesc,
                                                    size_t*                 psz)
{
    //-----------------------------------------------------------------
    // Validate arguments
    if(!tensorDesc || !psz)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc& tdesc = static_cast<tensor_desc&>(*tensorDesc);
    *psz               = tdesc.get_size_in_bytes();
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyConvertTensor()
cuphyStatus_t CUPHYWINAPI cuphyConvertTensor(cuphyTensorDescriptor_t tensorDescDst,
                                             void*                   dstAddr,
                                             cuphyTensorDescriptor_t tensorDescSrc,
                                             const void*             srcAddr,
                                             cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(!tensorDescDst || !tensorDescSrc || !dstAddr || !srcAddr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc&       tdDst = static_cast<tensor_desc&>(*tensorDescDst);
    const tensor_desc& tdSrc = static_cast<const tensor_desc&>(*tensorDescSrc);
    // Types don't need to match, but they can't be VOID
    if((tdDst.type() == CUPHY_VOID) || tdSrc.type() == CUPHY_VOID)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const tensor_layout_any& layoutDst = tdDst.layout();
    const tensor_layout_any& layoutSrc = tdSrc.layout();
    if(!layoutDst.has_same_size(layoutSrc))
    {
        return CUPHY_STATUS_SIZE_MISMATCH;
    }
    //------------------------------------------------------------------
    // Handle "memcpy" case (same type and strides)
    // We also exclude CUPHY_BIT from memcpy cases, since we may need
    // to mask off "extra" bits from the source tensor.
    if((tdDst.type() == tdSrc.type()) &&
       (tdDst.type() != CUPHY_BIT) &&
       layoutDst.has_same_strides(layoutSrc))
    {
        // Assuming availability of cudaMemcpyDefault (unified virtual
        // addressing), unifiedAddressing property in cudaDeviceProperties
        cudaError_t e = cudaMemcpyAsync(dstAddr,
                                        srcAddr,
                                        tdDst.get_size_in_bytes(),
                                        cudaMemcpyDefault,
                                        strm);
        return (cudaSuccess != e) ? CUPHY_STATUS_MEMCPY_ERROR : CUPHY_STATUS_SUCCESS;
    }
    //------------------------------------------------------------------
    // Handle more complex cases here (different types and/or different
    // layouts).
    // printf("tdDstType %s tdSrcType %s dstAddr %p srcAddr %p\n", cuphyGetDataTypeString(tdDst.type()), cuphyGetDataTypeString(tdSrc.type()), dstAddr, srcAddr);
    return convert_tensor_layout(tdDst, dstAddr, tdSrc, srcAddr, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyBfcCoefCompute()
cuphyStatus_t CUPHYWINAPI cuphyBfcCoefCompute(unsigned int            nBSAnts,
                                              unsigned int            nLayers,
                                              unsigned int            Nprb,
                                              cuphyTensorDescriptor_t tDescH,
                                              const void*             HAddr,
                                              cuphyTensorDescriptor_t tDescLambda,
                                              const void*             lambdaAddr,
                                              cuphyTensorDescriptor_t tDescCoef,
                                              void*                   coefAddr,
                                              cuphyTensorDescriptor_t tDescDbg,
                                              void*                   dbgAddr,
                                              cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tDescH ||
       !HAddr ||
       !tDescLambda ||
       !lambdaAddr ||
       !tDescCoef ||
       !coefAddr ||
       !tDescDbg ||
       !dbgAddr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    //------------------------------------------------------------------
    // clang-format off
    const_tensor_pair tPairH       (static_cast<const tensor_desc&>(*tDescH)     ,  HAddr);
    const_tensor_pair tPairLambda  (static_cast<const tensor_desc&>(*tDescLambda),  lambdaAddr);
    tensor_pair       tPairCoef    (static_cast<const tensor_desc&>(*tDescCoef)  ,  coefAddr);
    tensor_pair       tPairDbg     (static_cast<const tensor_desc&>(*tDescDbg)   ,  dbgAddr);
    // clang-format on

    bfw_coefComp::bfcCoefCompute(static_cast<uint32_t>(nBSAnts),
                        static_cast<uint32_t>(nLayers),
                        static_cast<uint32_t>(Nprb),
                        tPairH,
                        tPairLambda,
                        tPairCoef,
                        tPairDbg,
                        strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphyCreateBfwCoefComp(cuphyBfwCoefCompHndl_t* pBfwCoefCompHndl,
                                                 uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                                 uint8_t                 compressBitwidth,
                                                 uint16_t                nMaxUeGrps,
                                                 uint16_t                nMaxTotalLayers,
                                                 float                   lambda,
                                                 void*                   pStatDescrCpu,
                                                 void*                   pStatDescrGpu,
                                                 void*                   pDynDescrsCpu,
                                                 void*                   pDynDescrsGpu,
                                                 void*                   pHetCfgUeGrpMapCpu,
                                                 void*                   pHetCfgUeGrpMapGpu,
                                                 void*                   pUeGrpPrmsCpu,
                                                 void*                   pUeGrpPrmsGpu,
                                                 void*                   pBfLayerPrmsCpu,
                                                 void*                   pBfLayerPrmsGpu,                                              
                                                 cudaStream_t            strm)
{
    if(!pBfwCoefCompHndl || !pStatDescrCpu || !pStatDescrGpu || !pDynDescrsCpu || !pDynDescrsGpu || !pHetCfgUeGrpMapCpu || !pHetCfgUeGrpMapGpu ||
       !pUeGrpPrmsCpu || !pUeGrpPrmsGpu || !pBfLayerPrmsCpu || !pBfLayerPrmsGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pBfwCoefCompHndl = nullptr;
    try
    {
        bfw_coefComp::bfwCoefComp* pBfwCoefComp = new bfw_coefComp::bfwCoefComp(nMaxUeGrps, nMaxTotalLayers);
        *pBfwCoefCompHndl                       = static_cast<cuphyBfwCoefCompHndl_t>(pBfwCoefComp);

        pBfwCoefComp->init((0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                           compressBitwidth,
                           lambda,
                           pStatDescrCpu,
                           pStatDescrGpu,
                           pDynDescrsCpu,
                           pDynDescrsGpu,
                           pHetCfgUeGrpMapCpu,
                           pHetCfgUeGrpMapGpu,
                           pUeGrpPrmsCpu,
                           pUeGrpPrmsGpu,
                           pBfLayerPrmsCpu,
                           pBfLayerPrmsGpu,
                           strm);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{} EXCEPTION: {}", __FUNCTION__, e.what());
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{} UNKNOWN EXCEPTION", __FUNCTION__);
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphyDestroyBfwCoefComp(cuphyBfwCoefCompHndl_t bfwCoefCompHndl)
{
    if(!bfwCoefCompHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    bfw_coefComp::bfwCoefComp* pBfwCoefComp = static_cast<bfw_coefComp::bfwCoefComp*>(bfwCoefCompHndl);
    delete pBfwCoefComp;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyGetDescrInfoBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphyGetDescrInfoBfwCoefComp(uint16_t               nMaxUeGrps,
                                                       uint16_t               nMaxTotalLayers,
                                                       size_t*                pStatDescrSizeBytes, 
                                                       size_t*                pStatDescrAlignBytes,
                                                       size_t*                pDynDescrSizeBytes, 
                                                       size_t*                pDynDescrAlignBytes,
                                                       size_t*                pHetCfgUeGrpMapSizeBytes, 
                                                       size_t*                pHetCfgUeGrpMapAlignBytes,
                                                       size_t*                pUeGrpPrmsSizeBytes, 
                                                       size_t*                pUeGrpPrmsAlignBytes,
                                                       size_t*                pBfLayerPrmsSizeBytes, 
                                                       size_t*                pBfLayerPrmsAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes || !pDynDescrSizeBytes || !pDynDescrAlignBytes || 
       !pHetCfgUeGrpMapSizeBytes || !pHetCfgUeGrpMapAlignBytes || !pUeGrpPrmsSizeBytes || !pUeGrpPrmsAlignBytes ||
       !pBfLayerPrmsSizeBytes || !pBfLayerPrmsAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    bfw_coefComp::bfwCoefComp::getDescrInfo(nMaxUeGrps,
                                            nMaxTotalLayers,
                                            *pStatDescrSizeBytes,
                                            *pStatDescrAlignBytes,
                                            *pDynDescrSizeBytes, 
                                            *pDynDescrAlignBytes,
                                            *pHetCfgUeGrpMapSizeBytes,
                                            *pHetCfgUeGrpMapAlignBytes,
                                            *pUeGrpPrmsSizeBytes,
                                            *pUeGrpPrmsAlignBytes,
                                            *pBfLayerPrmsSizeBytes,
                                            *pBfLayerPrmsAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphySetupBfwCoefComp(cuphyBfwCoefCompHndl_t        bfwCoefCompHndl,
                                                uint16_t                      nUeGrps,
                                                cuphyBfwUeGrpPrm_t const*     pUeGrpPrms,
                                                uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                cuphySrsChEstBuffInfo_t*      pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                                                cuphyTensorPrm_t*             pTBfwCoef,
#endif
                                                uint8_t**                     pBfwCompCoef,
                                                cuphyBfwCoefCompLaunchCfgs_t* pLaunchCfgs,
                                                cudaStream_t                  strm)
{
    if(!bfwCoefCompHndl || !pUeGrpPrms || !pChEstInfo || !pBfwCompCoef || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    bfw_coefComp::bfwCoefComp* pBfwCoefComp = static_cast<bfw_coefComp::bfwCoefComp*>(bfwCoefCompHndl);

    return pBfwCoefComp->setupCoefComp(nUeGrps,
                                       pUeGrpPrms,
                                       (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                       pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                                       pTBfwCoef,
#endif
                                       pBfwCompCoef,
                                       pLaunchCfgs,
                                       strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyChannelEst1DTimeFrequency()
cuphyStatus_t CUPHYWINAPI cuphyChannelEst1DTimeFrequency(cuphyTensorDescriptor_t tensorDescDst,
                                                         void*                   dstAddr,
                                                         cuphyTensorDescriptor_t tensorDescSymbols,
                                                         const void*             symbolsAddr,
                                                         cuphyTensorDescriptor_t tensorDescFreqFilters,
                                                         const void*             freqFiltersAddr,
                                                         cuphyTensorDescriptor_t tensorDescTimeFilters,
                                                         const void*             timeFiltersAddr,
                                                         cuphyTensorDescriptor_t tensorDescFreqIndices,
                                                         const void*             freqIndicesAddr,
                                                         cuphyTensorDescriptor_t tensorDescTimeIndices,
                                                         const void*             timeIndicesAddr,
                                                         cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tensorDescDst ||
       !dstAddr ||
       !tensorDescSymbols ||
       !symbolsAddr ||
       !tensorDescFreqFilters ||
       !freqFiltersAddr ||
       !tensorDescTimeFilters ||
       !timeFiltersAddr ||
       !tensorDescFreqIndices ||
       !freqIndicesAddr ||
       !tensorDescTimeIndices ||
       !timeIndicesAddr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // clang-format off
    tensor_pair       tDst        (static_cast<const tensor_desc&>(*tensorDescDst),         dstAddr);
    const_tensor_pair tSymbols    (static_cast<const tensor_desc&>(*tensorDescSymbols),     symbolsAddr);
    const_tensor_pair tFreqFilters(static_cast<const tensor_desc&>(*tensorDescFreqFilters), freqFiltersAddr);
    const_tensor_pair tTimeFilters(static_cast<const tensor_desc&>(*tensorDescTimeFilters), timeFiltersAddr);
    const_tensor_pair tFreqIndices(static_cast<const tensor_desc&>(*tensorDescFreqIndices), freqIndicesAddr);
    const_tensor_pair tTimeIndices(static_cast<const tensor_desc&>(*tensorDescTimeIndices), timeIndicesAddr);
    // clang-format on
    channel_est::mmse_1D_time_frequency(tDst,
                                        tSymbols,
                                        tFreqFilters,
                                        tTimeFilters,
                                        tFreqIndices,
                                        tTimeIndices,
                                        strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxChEstGetDescrSizes()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxChEstGetDescrInfo(size_t* pStatDescrSizeBytes, size_t* pStatDescrAlignBytes, size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes || !pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ch_est::puschRxChEst::getDescrInfo(*pStatDescrSizeBytes, *pStatDescrAlignBytes, *pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxChEst()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxChEst(cuphyPuschRxChEstHndl_t* pPuschRxChEstHndl,
                                                  cuphyTensorPrm_t const*  pFreqInterpCoefs,
                                                  cuphyTensorPrm_t const*  pFreqInterpCoefs4,
                                                  cuphyTensorPrm_t const*  pFreqInterpCoefsSmall,
                                                  cuphyTensorPrm_t const*  pShiftSeq,
                                                  cuphyTensorPrm_t const*  pShiftSeq4,
                                                  cuphyTensorPrm_t const*  pUnShiftSeq,
                                                  cuphyTensorPrm_t const*  pUnShiftSeq4,
                                                  const uint32_t*          pSymStats,
                                                  uint8_t                  enableCpuToGpuDescrAsyncCpy,
                                                  void**                   ppStatDescrsCpu,
                                                  void**                   ppStatDescrsGpu,
                                                  cudaStream_t             strm)
{
    if(!pPuschRxChEstHndl || !pFreqInterpCoefs || !pShiftSeq || !pUnShiftSeq || !ppStatDescrsCpu || !ppStatDescrsGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxChEstHndl = nullptr;
    try
    {
        ch_est::puschRxChEst* pChEst = new ch_est::puschRxChEst;
        *pPuschRxChEstHndl           = static_cast<cuphyPuschRxChEstHndl_t>(pChEst);

        //------------------------------------------------------------------

        // clang-format off
        tensor_pair tPairFreqInterpCoefs(static_cast<const tensor_desc&>(*(pFreqInterpCoefs->desc)), pFreqInterpCoefs->pAddr);
        tensor_pair tPairFreqInterpCoefs4(static_cast<const tensor_desc&>(*(pFreqInterpCoefs4->desc)), pFreqInterpCoefs4->pAddr);
        tensor_pair tPairFreqInterpCoefsSmall(static_cast<const tensor_desc&>(*(pFreqInterpCoefsSmall->desc)), pFreqInterpCoefsSmall->pAddr);
        tensor_pair tPairShiftSeq       (static_cast<const tensor_desc&>(*(pShiftSeq->desc))       , pShiftSeq->pAddr);
        tensor_pair tPairShiftSeq4       (static_cast<const tensor_desc&>(*(pShiftSeq4->desc))       , pShiftSeq4->pAddr);
        tensor_pair tPairUnShiftSeq     (static_cast<const tensor_desc&>(*(pUnShiftSeq->desc))     , pUnShiftSeq->pAddr);
        tensor_pair tPairUnShiftSeq4     (static_cast<const tensor_desc&>(*(pUnShiftSeq4->desc))     , pUnShiftSeq4->pAddr);

        // clang-format on

        pChEst->init(tPairFreqInterpCoefs,
                     tPairFreqInterpCoefs4,
                     tPairFreqInterpCoefsSmall,
                     tPairShiftSeq,
                     tPairShiftSeq4,
                     tPairUnShiftSeq,
                     tPairUnShiftSeq4,
                     pSymStats,
                     (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                     ppStatDescrsCpu,
                     ppStatDescrsGpu,
                     strm);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxChEst()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxChEst(cuphyPuschRxChEstHndl_t puschRxChEstHndl)
{
    if(!puschRxChEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ch_est::puschRxChEst* pChEst = static_cast<ch_est::puschRxChEst*>(puschRxChEstHndl);
    delete pChEst;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEst()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEst(cuphyPuschRxChEstHndl_t               puschRxChEstHndl,
                                                 cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsCpu,
                                                 cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsGpu,
                                                 uint16_t                              nUeGrps,
                                                 uint8_t                               enableDftSOfdm,
                                                 uint8_t*                              pPreEarlyHarqWaitKernelStatus_d,
                                                 uint8_t*                              pPostEarlyHarqWaitKernelStatus_d,
                                                 const uint16_t                        waitTimeOutPreEarlyHarqUs,
                                                 const uint16_t                        waitTimeOutPostEarlyHarqUs,
                                                 uint8_t                               enableCpuToGpuDescrAsyncCpy,
                                                 void**                                ppDynDescrsCpu,
                                                 void**                                ppDynDescrsGpu,
                                                 cuphyPuschRxChEstLaunchCfgs_t*        pLaunchCfgs,
                                                 uint8_t                               enableEarlyHarqProc,
                                                 cuphyPuschRxEarlyHarqWaitLaunchCfg_t* pLaunchCfgsPreEHQ,
                                                 cuphyPuschRxEarlyHarqWaitLaunchCfg_t* pLaunchCfgsPostEHQ,
                                                 cudaStream_t                          strm)
{
    ch_est::puschRxChEst* pChEst = static_cast<ch_est::puschRxChEst*>(puschRxChEstHndl);

    return pChEst->setup(pDrvdUeGrpPrmsCpu,
                         pDrvdUeGrpPrmsGpu,
                         nUeGrps,
                         enableDftSOfdm,
                         pPreEarlyHarqWaitKernelStatus_d,
                         pPostEarlyHarqWaitKernelStatus_d,
                         waitTimeOutPreEarlyHarqUs,
                         waitTimeOutPostEarlyHarqUs,
                         (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                         ppDynDescrsCpu,
                         ppDynDescrsGpu,
                         pLaunchCfgs,
                         enableEarlyHarqProc,
                         pLaunchCfgsPreEHQ,
                         pLaunchCfgsPostEHQ,
                         strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxNoiseIntfEstGetDescrSizes()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxNoiseIntfEstGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pusch_noise_intf_est::puschRxNoiseIntfEst::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxNoiseIntfEst()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t* pPuschRxNoiseIntfEstHndl)
{
    if(!pPuschRxNoiseIntfEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxNoiseIntfEstHndl = nullptr;
    try
    {
        pusch_noise_intf_est::puschRxNoiseIntfEst* pNoiseIntfEst = new pusch_noise_intf_est::puschRxNoiseIntfEst;
        *pPuschRxNoiseIntfEstHndl                                = static_cast<cuphyPuschRxNoiseIntfEstHndl_t>(pNoiseIntfEst);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxNoiseIntfEst()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t puschRxNoiseIntfEstHndl)
{
    if(!puschRxNoiseIntfEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pusch_noise_intf_est::puschRxNoiseIntfEst* pNoiseIntfEst = static_cast<pusch_noise_intf_est::puschRxNoiseIntfEst*>(puschRxNoiseIntfEstHndl);
    delete pNoiseIntfEst;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxNoiseIntfEst()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t puschRxNoiseIntfEstHndl,
                                                        cuphyPuschRxUeGrpPrms_t*       pDrvdUeGrpPrmsCpu,
                                                        cuphyPuschRxUeGrpPrms_t*       pDrvdUeGrpPrmsGpu,
                                                        uint16_t                       nUeGrps,
                                                        uint16_t                       nMaxPrb,
                                                        uint8_t                        enableDftSOfdm,
                                                        uint8_t                        dmrsSymbolIdx,
                                                        uint8_t                        enableCpuToGpuDescrAsyncCpy,
                                                        void*                          pDynDescrsCpu,
                                                        void*                          pDynDescrsGpu,
                                                        cuphyPuschRxNoiseIntfEstLaunchCfgs_t* pLaunchCfgs,
                                                        cudaStream_t                   strm,
                                                        uint8_t                        subSlotStageIdx)
{
    pusch_noise_intf_est::puschRxNoiseIntfEst* pNoiseIntfEst = static_cast<pusch_noise_intf_est::puschRxNoiseIntfEst*>(puschRxNoiseIntfEstHndl);
    return pNoiseIntfEst->setup(pDrvdUeGrpPrmsCpu,
                                pDrvdUeGrpPrmsGpu,
                                nUeGrps,
                                nMaxPrb,
                                enableDftSOfdm,
                                dmrsSymbolIdx,
                                (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                pDynDescrsCpu,
                                pDynDescrsGpu,
                                &pLaunchCfgs[subSlotStageIdx * CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS],
                                strm,
                                subSlotStageIdx);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxCfoTaEstGetDescrInfo()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxCfoTaEstGetDescrInfo(size_t* pStatDescrSizeBytes, size_t* pStatDescrAlignBytes, size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes || !pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cfo_ta_est::puschRxCfoTaEst::getDescrInfo(*pStatDescrSizeBytes, *pStatDescrAlignBytes, *pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxCfoTaEst()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t* pPuschRxCfoTaEstHndl,
                                                     uint8_t                     enableCpuToGpuDescrAsyncCpy,
                                                     void*                       pStatDescrCpu,
                                                     void*                       pStatDescrGpu,
                                                     cudaStream_t                strm)
{
    if(!pPuschRxCfoTaEstHndl || !pStatDescrCpu || !pStatDescrGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxCfoTaEstHndl = nullptr;
    try
    {
        cfo_ta_est::puschRxCfoTaEst* pCfoTaEst               = new cfo_ta_est::puschRxCfoTaEst;
        *pPuschRxCfoTaEstHndl                                = static_cast<cuphyPuschRxCfoTaEstHndl_t>(pCfoTaEst);
        cfo_ta_est::puschRxCfoTaEstStatDescr_t& statDescrCpu = *(static_cast<cfo_ta_est::puschRxCfoTaEstStatDescr_t*>(pStatDescrCpu));

        //------------------------------------------------------------------
        pCfoTaEst->init((0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                        statDescrCpu,
                        pStatDescrGpu,
                        strm);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxCfoTaEst()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t puschRxCfoTaEstHndl)
{
    if(!puschRxCfoTaEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cfo_ta_est::puschRxCfoTaEst* pCfoTaEst = static_cast<cfo_ta_est::puschRxCfoTaEst*>(puschRxCfoTaEstHndl);
    delete pCfoTaEst;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxCfoTaEst()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t        puschRxCfoTaEstHndl,
                                                    cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsCpu,
                                                    cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsGpu,
                                                    uint16_t                          nUeGrps,
                                                    uint32_t                          nMaxPrb,
                                                    cuphyTensorPrm_t*                 pDbg,
                                                    uint8_t                           enableCpuToGpuDescrAsyncCpy,
                                                    void*                             pDynDescrsCpu,
                                                    void*                             pDynDescrsGpu,
                                                    cuphyPuschRxCfoTaEstLaunchCfgs_t* pLaunchCfgs,
                                                    cudaStream_t                      strm)
{
    if(!puschRxCfoTaEstHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs || (pLaunchCfgs->nCfgs > CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cfo_ta_est::puschRxCfoTaEst*              pPuschRxCfoTaEst = static_cast<cfo_ta_est::puschRxCfoTaEst*>(puschRxCfoTaEstHndl);
    cfo_ta_est::puschRxCfoTaEstDynDescrVec_t& dynDescrVecCpu   = *(static_cast<cfo_ta_est::puschRxCfoTaEstDynDescrVec_t*>(pDynDescrsCpu));
    return pPuschRxCfoTaEst->setup(pDrvdUeGrpPrmsCpu,
                                   pDrvdUeGrpPrmsGpu,
                                   nUeGrps,
                                   nMaxPrb,
                                   (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                   dynDescrVecCpu,
                                   pDynDescrsGpu,
                                   pLaunchCfgs,
                                   strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxChEqGetDescrInfo()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxChEqGetDescrInfo(size_t* pStatDescrSizeBytes,
                                                       size_t* pStatDescrAlignBytes,
                                                       size_t* pCoefCompDynDescrSizeBytes,
                                                       size_t* pCoefCompDynDescrAlignBytes,
                                                       size_t* pSoftDemapDynDescrSizeBytes,
                                                       size_t* pSoftDemapDynDescrAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes ||
       !pCoefCompDynDescrSizeBytes || !pCoefCompDynDescrAlignBytes ||
       !pSoftDemapDynDescrSizeBytes || !pSoftDemapDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq::getDescrInfo(*pStatDescrSizeBytes,
                                          *pStatDescrAlignBytes,
                                          *pCoefCompDynDescrSizeBytes,
                                          *pCoefCompDynDescrAlignBytes,
                                          *pSoftDemapDynDescrSizeBytes,
                                          *pSoftDemapDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxChEq()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxChEq(cuphyContext_t          ctx,
                                                 cuphyPuschRxChEqHndl_t* pPuschRxChEqHndl,
                                                 uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                                 void**                  ppStatDescrCpu,
                                                 void**                  ppStatDescrGpu,
                                                 cudaStream_t            strm)
{
    if(!ctx || !pPuschRxChEqHndl || !ppStatDescrCpu || !ppStatDescrGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxChEqHndl = nullptr;
    try
    {
        channel_eq::puschRxChEq* pChEq = new channel_eq::puschRxChEq;
        *pPuschRxChEqHndl              = static_cast<cuphyPuschRxChEqHndl_t>(pChEq);

        //------------------------------------------------------------------
        pChEq->init(ctx,
                    (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                    ppStatDescrCpu,
                    ppStatDescrGpu,
                    strm);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxChEq()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxChEq(cuphyPuschRxChEqHndl_t puschRxChEqHndl)
{
    if(!puschRxChEqHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq* pChEq = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    delete pChEq;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqCoefCompute()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqCoefCompute(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                           cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                           cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                           uint16_t                      nUeGrps,
                                                           uint16_t                      nMaxPrb,
                                                           uint8_t                       enableCfoCorrection,
                                                           uint8_t                       enablePuschTdi,
                                                           uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                           void**                        ppDynDescrsCpu,
                                                           void**                        ppDynDescrsGpu,
                                                           cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,

                                                           cudaStream_t strm)
{
    if(!puschRxChEqHndl || !ppDynDescrsCpu || !ppDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq* pChEq = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);

    return pChEq->setupCoefCompute(pDrvdUeGrpPrmsCpu,
                                   pDrvdUeGrpPrmsGpu,
                                   nUeGrps,
                                   nMaxPrb,
                                   enableCfoCorrection,
                                   enablePuschTdi,
                                   (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                   ppDynDescrsCpu,
                                   ppDynDescrsGpu,
                                   pLaunchCfgs,
                                   strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqSoftDemap()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqSoftDemap(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                         uint16_t                      nUeGrps,
                                                         uint16_t                      nMaxPrb,
                                                         uint8_t                       enableCfoCorrection,
                                                         uint8_t                       enablePuschTdi,
                                                         uint16_t                      symbolBitmask,
                                                         uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                         void*                         pDynDescrsCpu,
                                                         void*                         pDynDescrsGpu,
                                                         cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                                         cudaStream_t                  strm)
{
    if(!puschRxChEqHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq*                       pChEq          = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    channel_eq::puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu = *(static_cast<channel_eq::puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsCpu));

    return pChEq->setupSoftDemap(pDrvdUeGrpPrmsCpu,
                                 pDrvdUeGrpPrmsGpu,
                                 nUeGrps,
                                 nMaxPrb,
                                 enableCfoCorrection,
                                 enablePuschTdi,
                                 symbolBitmask,
                                 (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                 dynDescrVecCpu,
                                 pDynDescrsGpu,
                                 pLaunchCfgs,
                                 strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqSoftDemapBluesteinWorkspace
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqSoftDemapBluesteinWorkspace(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                         uint16_t                      nUeGrps,
                                                         uint16_t                      nMaxPrb,
                                                         uint8_t                       enableCfoCorrection,
                                                         uint8_t                       enablePuschTdi,
                                                         uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                         void*                         pDynDescrsCpu,
                                                         void*                         pDynDescrsGpu,
                                                         cuphyPuschRxChEqLaunchCfgs_t*  pLaunchCfgs,
                                                         cudaStream_t                  strm)
{
    if(!puschRxChEqHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq*                       pChEq          = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    channel_eq::puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu = *(static_cast<channel_eq::puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsCpu));

    return pChEq->setupSoftDemapBluesteinWorkspace(pDrvdUeGrpPrmsCpu,
                                 pDrvdUeGrpPrmsGpu,
                                 nUeGrps,
                                 nMaxPrb,
                                 enableCfoCorrection,
                                 enablePuschTdi,
                                 (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                 dynDescrVecCpu,
                                 pDynDescrsGpu,
                                 pLaunchCfgs,
                                 strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqSoftDemapIdft()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqSoftDemapIdft(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                         uint16_t                      nUeGrps,
                                                         uint16_t                      nMaxPrb,
                                                         uint8_t                       enableCfoCorrection,
                                                         uint8_t                       enablePuschTdi,
                                                         uint16_t                      symbolBitmask,
                                                         uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                         void*                         pDynDescrsCpu,
                                                         void*                         pDynDescrsGpu,
                                                         cuphyPuschRxChEqLaunchCfgs_t*  pLaunchCfgs,
                                                         cudaStream_t                  strm)
{
    if(!puschRxChEqHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq*                       pChEq          = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    channel_eq::puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu = *(static_cast<channel_eq::puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsCpu));

    return pChEq->setupSoftDemapIdft(pDrvdUeGrpPrmsCpu,
                                 pDrvdUeGrpPrmsGpu,
                                 nUeGrps,
                                 nMaxPrb,
                                 enableCfoCorrection,
                                 enablePuschTdi,
                                 symbolBitmask,
                                 (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                 dynDescrVecCpu,
                                 pDynDescrsGpu,
                                 pLaunchCfgs,
                                 strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqSoftDemapAfterDft()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqSoftDemapAfterDft(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                         uint16_t                      nUeGrps,
                                                         uint16_t                      nMaxPrb,
                                                         uint8_t                       enableCfoCorrection,
                                                         uint8_t                       enablePuschTdi,
                                                         uint16_t                      symbolBitmask,
                                                         uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                         void*                         pDynDescrsCpu,
                                                         void*                         pDynDescrsGpu,
                                                         cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                                         cudaStream_t                  strm)
{
    if(!puschRxChEqHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq*                       pChEq          = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    channel_eq::puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu = *(static_cast<channel_eq::puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsCpu));

    return pChEq->setupSoftDemapAfterDft(pDrvdUeGrpPrmsCpu,
                                 pDrvdUeGrpPrmsGpu,
                                 nUeGrps,
                                 nMaxPrb,
                                 enableCfoCorrection,
                                 enablePuschTdi,
                                 symbolBitmask,
                                 (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                 dynDescrVecCpu,
                                 pDynDescrsGpu,
                                 pLaunchCfgs,
                                 strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxRateMatchGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPuschRxRateMatchGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxRateMatch::getDescrInfo(*pDescrSizeBytes, *pDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxRateMatch()

cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxRateMatch(cuphyPuschRxRateMatchHndl_t* pPuschRxRateMatchHndl,
                                                      int                          FPconfig, // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
                                                      int                          descramblingOn)                    // enable/disable descrambling
{
    if(!pPuschRxRateMatchHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxRateMatchHndl = nullptr;
    try
    {
        puschRxRateMatch* pRateMatch = new puschRxRateMatch;
        *pPuschRxRateMatchHndl       = static_cast<cuphyPuschRxRateMatchHndl_t>(pRateMatch);

        pRateMatch->init(FPconfig, descramblingOn);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxRateMatch()

cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxRateMatch(cuphyPuschRxRateMatchHndl_t       puschRxRateMatchHndl,
                                                     uint16_t                          nSchUes,
                                                     uint16_t*                         pSchUserIdxsCpu,
                                                     const PerTbParams*                pTbPrmsCpu,
                                                     const PerTbParams*                pTbPrmsGpu,
                                                     cuphyTensorPrm_t*                 pTPrmRmIn,
                                                     cuphyTensorPrm_t*                 pTPrmCdm1RmIn,
                                                     void**                            ppRmOut,
                                                     void*                             pCpuDesc,
                                                     void*                             pGpuDesc,
                                                     uint8_t                           enableCpuToGpuDescrAsyncCpy,
                                                     cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,
                                                     cudaStream_t                      strm)
{
    if(!puschRxRateMatchHndl || !pTbPrmsCpu || !pTbPrmsGpu || !pTPrmRmIn || !pTPrmCdm1RmIn || !ppRmOut || !pCpuDesc || !pGpuDesc || !pLaunchCfg || !pSchUserIdxsCpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRxRateMatch* pRateMatch = static_cast<puschRxRateMatch*>(puschRxRateMatchHndl);
    pRateMatch->setup(nSchUes,  pSchUserIdxsCpu, pTbPrmsCpu, pTbPrmsGpu, pTPrmRmIn, pTPrmCdm1RmIn, ppRmOut, pCpuDesc, pGpuDesc, enableCpuToGpuDescrAsyncCpy, pLaunchCfg, strm);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxRateMatch()

cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxRateMatch(cuphyPuschRxRateMatchHndl_t puschRxRateMatchHndl)
{
    if(!puschRxRateMatchHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxRateMatch* pRateMatch = static_cast<puschRxRateMatch*>(puschRxRateMatchHndl);
    delete pRateMatch;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxCrcDecodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPuschRxCrcDecodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxCrcDecode::getDescrInfo(*pDescrSizeBytes, *pDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

// ////////////////////////////////////////////////////////////////////////
// // cuphyCreatePuschRxCrcDecode()

cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t* puschRxCrcDecodeHndl,
                                                      int                          reverseBytes)
{
    if(!puschRxCrcDecodeHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *puschRxCrcDecodeHndl = nullptr;
    try
    {
        puschRxCrcDecode* pCrcDecode = new puschRxCrcDecode;
        *puschRxCrcDecodeHndl        = static_cast<cuphyPuschRxCrcDecodeHndl_t>(pCrcDecode);

        pCrcDecode->init(reverseBytes);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

// ////////////////////////////////////////////////////////////////////////
// // cuphySetupPuschRxCrcDecode()

cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t       puschRxCrcDecodeHndl,
                                                     uint16_t                          nSchUes,
                                                     uint16_t*                         pSchUserIdxsCpu,
                                                     uint32_t*                         pOutputCBCRCs,
                                                     uint8_t*                          pOutputTBs,
                                                     const uint32_t*                   pInputCodeBlocks,
                                                     uint32_t*                         pOutputTBCRCs,
                                                     const PerTbParams*                pTbPrmsCpu,
                                                     const PerTbParams*                pTbPrmsGpu,
                                                     void*                             pCpuDesc,
                                                     void*                             pGpuDesc,
                                                     uint8_t                           enableCpuToGpuDescrAsyncCpy,
                                                     cuphyPuschRxCrcDecodeLaunchCfg_t* pCbCrcLaunchCfg,
                                                     cuphyPuschRxCrcDecodeLaunchCfg_t* pTbCrcLaunchCfg,
                                                     cudaStream_t                      strm)
{
    if(!puschRxCrcDecodeHndl || !pOutputCBCRCs || !pOutputTBs || !pInputCodeBlocks || !pOutputTBCRCs || !pTbPrmsCpu || !pTbPrmsGpu || !pCpuDesc || !pGpuDesc || !pCbCrcLaunchCfg || !pTbCrcLaunchCfg || !pSchUserIdxsCpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRxCrcDecode* pCrcDecode = static_cast<puschRxCrcDecode*>(puschRxCrcDecodeHndl);
    pCrcDecode->setup(nSchUes, pSchUserIdxsCpu, pOutputCBCRCs, pOutputTBs, pInputCodeBlocks, pOutputTBCRCs, pTbPrmsCpu, pTbPrmsGpu, pCpuDesc, pGpuDesc, enableCpuToGpuDescrAsyncCpy, pCbCrcLaunchCfg, pTbCrcLaunchCfg, strm);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxCrcDecode()

cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t puschRxCrcDecodeHndl)
{
    if(!puschRxCrcDecodeHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxCrcDecode* pCrcDecode = static_cast<puschRxCrcDecode*>(puschRxCrcDecodeHndl);
    delete pCrcDecode;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecode()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecode(cuphyLDPCDecoder_t                 decoder,
                                                         cuphyTensorDescriptor_t            tensorDescDst,
                                                         void*                              dstAddr,
                                                         cuphyTensorDescriptor_t            tensorDescLLR,
                                                         const void*                        LLRAddr,
                                                         const cuphyLDPCDecodeConfigDesc_t* config,
                                                         cudaStream_t                       strm)
{
    std::array<int, 4> BG2_Kb = {6, 8, 9, 10};
    if(!decoder ||
       !tensorDescDst ||
       !dstAddr ||
       !tensorDescLLR ||
       !LLRAddr ||
       !config ||
       (config->max_iterations < 0) ||
       (config->BG < 1) ||
       (config->BG > 2) ||
       ((1 == config->BG) ? (config->Kb != 22) :
                            (BG2_Kb.end() == std::find(BG2_Kb.begin(), BG2_Kb.end(), config->Kb))) ||
       (config->Z < 2) ||
       (config->Z > 384) ||
       (config->num_parity_nodes < 4) ||
       ((1 == config->BG) ? (config->num_parity_nodes > 46) : (config->num_parity_nodes > 42)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    // clang-format off
    ldpc::decoder&    d = static_cast<ldpc::decoder&>(*decoder);
    tensor_pair       tDst(static_cast<const tensor_desc&>(*tensorDescDst), dstAddr);
    const_tensor_pair tLLR(static_cast<const tensor_desc&>(*tensorDescLLR), LLRAddr);
    // clang-format on
    //------------------------------------------------------------------
    // Check for LLR type mismatch
    if(config->llr_type != tLLR.first.get().type())
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    return d.decode(tDst,
                    tLLR,
                    *config,
                    strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCTransportBlockDecode()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCTransportBlockDecode(cuphyLDPCDecoder_t           decoder,
                                                                       const cuphyLDPCDecodeDesc_t* decodeDesc,
                                                                       cudaStream_t                 strm)
{
    if(!decoder ||
       !decodeDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder& d = static_cast<ldpc::decoder&>(*decoder);
    return d.decode_tb(*decodeDesc, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize(cuphyLDPCDecoder_t                 decoder,
                                                                         const cuphyLDPCDecodeConfigDesc_t* config,
                                                                         int                                numCodeWords,
                                                                         size_t*                            sizeInBytes)
{
    static const std::array<int, 2> BG_valid = {1, 2};
    static const std::array<int, 5> Kb_valid = {22, 10, 9, 8, 6};
    // clang-format off
    static const std::array<int, 51> Z_valid =
    {
        2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
       15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
       48,  52,  56,  60,  64,  72,  80,  88,  96, 104, 112, 120, 128,
      144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384
    };
    // clang-format on
    if(!decoder ||
       !config ||
       (std::find(BG_valid.begin(), BG_valid.end(), config->BG) == BG_valid.end()) ||
       (std::find(Kb_valid.begin(), Kb_valid.end(), config->Kb) == Kb_valid.end()) ||
       (std::find(Z_valid.begin(), Z_valid.end(), config->Z) == Z_valid.end()) ||
       (numCodeWords <= 0) ||
       !sizeInBytes ||
       (config->num_parity_nodes < 4) ||
       ((1 == config->BG) ? (config->num_parity_nodes > 46) : (config->num_parity_nodes > 42)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder&          d             = static_cast<ldpc::decoder&>(*decoder);
    std::pair<bool, size_t> workspaceSize = d.workspace_size(*config, numCodeWords);
    if(workspaceSize.first)
    {
        *sizeInBytes = workspaceSize.second;
        return CUPHY_STATUS_SUCCESS;
    }
    else
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateLDPCDecoder()
cuphyStatus_t CUPHYWINAPI cuphyCreateLDPCDecoder(cuphyContext_t      context,
                                                 cuphyLDPCDecoder_t* pdecoder,
                                                 unsigned int        flags)
{
    if(!pdecoder || !context)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pdecoder             = nullptr;
    cuphy_i::context& ctx = static_cast<cuphy_i::context&>(*context);
    try
    {
        ldpc::decoder* d = new ldpc::decoder(ctx);
        *pdecoder        = static_cast<cuphyLDPCDecoder_t>(d);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyLDPCDecoder()
cuphyStatus_t CUPHYWINAPI cuphyDestroyLDPCDecoder(cuphyLDPCDecoder_t decoder)
{
    if(!decoder)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder* d = static_cast<ldpc::decoder*>(decoder);
    delete d;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecodeSetNormalization()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeSetNormalization(cuphyLDPCDecoder_t           decoder,
                                                                         cuphyLDPCDecodeConfigDesc_t* decodeDesc)
{
    if(!decoder || !decodeDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder& d = static_cast<ldpc::decoder&>(*decoder);
    return d.set_normalization(*decodeDesc);
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(cuphyLDPCDecoder_t             decoder,
                                                                            cuphyLDPCDecodeLaunchConfig_t* launchConfig)
{
    if(!decoder || !launchConfig)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder& d = static_cast<ldpc::decoder&>(*decoder);
    return d.get_launch_config(*launchConfig);
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxRssi()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxRssi(cuphyPuschRxRssiHndl_t* pPuschRxRssiHndl)
{
    if(!pPuschRxRssiHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxRssiHndl = nullptr;
    try
    {
        puschRx_rssi::puschRxRssi* pPuschRxRssi = new puschRx_rssi::puschRxRssi;
        *pPuschRxRssiHndl                       = static_cast<cuphyPuschRxRssiHndl_t>(pPuschRxRssi);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxRssi()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxRssi(cuphyPuschRxRssiHndl_t puschRxRssiHndl)
{
    if(!puschRxRssiHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRx_rssi::puschRxRssi* pPuschRxRssi = static_cast<puschRx_rssi::puschRxRssi*>(puschRxRssiHndl);
    delete pPuschRxRssi;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxRssi()
cuphyStatus_t cuphySetupPuschRxRssi(cuphyPuschRxRssiHndl_t        puschRxRssiHndl,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                    uint16_t                      nUeGrps,
                                    uint32_t                      nMaxPrb,
                                    uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                    void*                         pDynDescrsCpu,
                                    void*                         pDynDescrsGpu,
                                    cuphyPuschRxRssiLaunchCfgs_t* pLaunchCfgs,
                                    cudaStream_t                  strm)
{
    if(!puschRxRssiHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRx_rssi::puschRxRssi*              pPuschRssi     = static_cast<puschRx_rssi::puschRxRssi*>(puschRxRssiHndl);
    puschRx_rssi::puschRxRssiDynDescrVec_t& dynDescrVecCpu = *(static_cast<puschRx_rssi::puschRxRssiDynDescrVec_t*>(pDynDescrsCpu));

    return pPuschRssi->setupRssiMeas(pDrvdUeGrpPrmsCpu,
                                     pDrvdUeGrpPrmsGpu,
                                     nUeGrps,
                                     nMaxPrb,
                                     (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                     dynDescrVecCpu,
                                     pDynDescrsGpu,
                                     pLaunchCfgs,
                                     strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxRsrp()
cuphyStatus_t cuphySetupPuschRxRsrp(cuphyPuschRxRssiHndl_t        puschRxRssiHndl,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                    uint16_t                      nUeGrps,
                                    uint32_t                      nMaxPrb,
                                    uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                    void*                         pDynDescrsCpu,
                                    void*                         pDynDescrsGpu,
                                    cuphyPuschRxRsrpLaunchCfgs_t* pLaunchCfgs,
                                    cudaStream_t                  strm)
{
    if(!puschRxRssiHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRx_rssi::puschRxRssi*              pPuschRssi     = static_cast<puschRx_rssi::puschRxRssi*>(puschRxRssiHndl);
    puschRx_rssi::puschRxRsrpDynDescrVec_t& dynDescrVecCpu = *(static_cast<puschRx_rssi::puschRxRsrpDynDescrVec_t*>(pDynDescrsCpu));

    return pPuschRssi->setupRsrpMeas(pDrvdUeGrpPrmsCpu,
                                     pDrvdUeGrpPrmsGpu,
                                     nUeGrps,
                                     nMaxPrb,
                                     (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                     dynDescrVecCpu,
                                     pDynDescrsGpu,
                                     pLaunchCfgs,
                                     strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxRssiGetDescrInfo()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxRssiGetDescrInfo(size_t* pRssiDynDescrSizeBytes, size_t* pRssiDynDescrAlignBytes, size_t* pRsrpDynDescrSizeBytes, size_t* pRsrpDynDescrAlignBytes)
{
    if(!pRssiDynDescrSizeBytes || !pRssiDynDescrAlignBytes || !pRsrpDynDescrSizeBytes || !pRsrpDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRx_rssi::puschRxRssi::getDescrInfo(*pRssiDynDescrSizeBytes, *pRssiDynDescrAlignBytes, *pRsrpDynDescrSizeBytes, *pRsrpDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPolarEncRateMatch()
cuphyStatus_t CUPHYWINAPI cuphyPolarEncRateMatch(unsigned int   nInfoBits,
                                                 unsigned int   nTxBits,
                                                 uint8_t const* pInfoBits,
                                                 uint32_t*      pNCodedBits,
                                                 uint8_t*       pCodedBits,
                                                 uint8_t*       pTxBits,
                                                 uint32_t       procModeBmsk,
                                                 cudaStream_t   strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if((!pInfoBits) || (!pNCodedBits) || (!pCodedBits) || (!pTxBits) ||
       (nInfoBits < 1) || (nInfoBits > CUPHY_POLAR_ENC_MAX_INFO_BITS) ||
       (nTxBits < 1) || (nTxBits > CUPHY_POLAR_ENC_MAX_TX_BITS))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // Ensure 4byte (32b) alignment on all input buffers
    if(((reinterpret_cast<uintptr_t>(pInfoBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pCodedBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pTxBits) & 0x3) != 0))
    {
        return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
    }

    //------------------------------------------------------------------
    polar_encoder::encodeRateMatch(static_cast<uint32_t>(nInfoBits),
                                   static_cast<uint32_t>(nTxBits),
                                   pInfoBits,
                                   pNCodedBits,
                                   pCodedBits,
                                   pTxBits,
                                   procModeBmsk,
                                   strm);


    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;

    //return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyRunPolarEncRateMatchSSBs(
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRmSSBCfg,
    uint8_t const*                            pInfoBits,
    uint8_t*                                  pCodedBits,
    uint8_t*                                  pTxBits,
    uint16_t                                  nSSBs,
    cudaStream_t                              strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if((!pInfoBits) || (!pCodedBits) || (!pTxBits) || (nSSBs == 0))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // Ensure 4byte (32b) alignment on all input buffers
    if(((reinterpret_cast<uintptr_t>(pInfoBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pCodedBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pTxBits) & 0x3) != 0))
    {
        return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
    }

    CUresult e = launch_kernel(pEncdRmSSBCfg->kernelNodeParamsDriver, strm);
    return (e == CUDA_SUCCESS) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;

}


////////////////////////////////////////////////////////////////////////
// cuphySrsChEstGetDescrSizes()
cuphyStatus_t CUPHYWINAPI cuphySrsChEstGetDescrInfo(size_t* pStatDescrSizeBytes, size_t* pStatDescrAlignBytes, size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes || !pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    srs_ch_est::srsChEst::getDescrInfo(*pStatDescrSizeBytes, *pStatDescrAlignBytes, *pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateSrsChEst()
cuphyStatus_t CUPHYWINAPI cuphyCreateSrsChEst(cuphySrsChEstHndl_t*    pSrsChEstHndl,
                                              cuphyTensorPrm_t const* pFreqInterpCoefs,
                                              uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                              void*                   pStatDescrCpu,
                                              void*                   pStatDescrGpu,
                                              cudaStream_t            strm)
{
    if(!pSrsChEstHndl || !pFreqInterpCoefs || !pStatDescrCpu || !pStatDescrGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pSrsChEstHndl = nullptr;
    try
    {
        srs_ch_est::srsChEst* pChEst                  = new srs_ch_est::srsChEst;
        *pSrsChEstHndl                                = static_cast<cuphySrsChEstHndl_t>(pChEst);
        srs_ch_est::srsChEstStatDescr_t& statDescrCpu = *(static_cast<srs_ch_est::srsChEstStatDescr_t*>(pStatDescrCpu));

        //------------------------------------------------------------------
        tensor_pair tPairFreqInterpCoefs(static_cast<const tensor_desc&>(*(pFreqInterpCoefs->desc)), pFreqInterpCoefs->pAddr);

        pChEst->init(tPairFreqInterpCoefs,
                     (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                     statDescrCpu,
                     pStatDescrGpu,
                     strm);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroySrsChEst()
cuphyStatus_t CUPHYWINAPI cuphyDestroySrsChEst(cuphySrsChEstHndl_t srsChEstHndl)
{
    if(!srsChEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    srs_ch_est::srsChEst* pChEst = static_cast<srs_ch_est::srsChEst*>(srsChEstHndl);
    delete pChEst;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupSrsChEst()
cuphyStatus_t CUPHYWINAPI cuphySetupSrsChEst(cuphySrsChEstHndl_t           srsChEstHndl,
                                             cuphySrsChEstDynPrms_t const* pDynPrms,
                                             cuphyTensorPrm_t*             pDataRx,
                                             cuphyTensorPrm_t*             pHEst,
                                             cuphyTensorPrm_t*             pDbg,
                                             uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                             void*                         pDynDescrsCpu,
                                             void*                         pDynDescrsGpu,
                                             cudaStream_t                  strm)
{
    if(!srsChEstHndl || !pDynPrms || !pDataRx || !pHEst || !pDbg || !pDynDescrsCpu || !pDynDescrsGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    srs_ch_est::srsChEst*              pChEst         = static_cast<srs_ch_est::srsChEst*>(srsChEstHndl);
    srs_ch_est::srsChEstDynDescrVec_t& dynDescrVecCpu = *(static_cast<srs_ch_est::srsChEstDynDescrVec_t*>(pDynDescrsCpu));

    //------------------------------------------------------------------
    // clang-format off
    tensor_pair tPairDataRx(static_cast<const tensor_desc&>(*(pDataRx->desc)), pDataRx->pAddr);
    tensor_pair tPairHEst  (static_cast<const tensor_desc&>(*(pHEst->desc))  , pHEst->pAddr  );
    tensor_pair tPairDbg   (static_cast<const tensor_desc&>(*(pDbg->desc))   , pDbg->pAddr   );
    // clang-format on

    pChEst->setup(pDynPrms,
                  tPairDataRx,
                  tPairHEst,
                  tPairDbg,
                  (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                  dynDescrVecCpu,
                  pDynDescrsGpu,
                  strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyRunSrsChEst()
cuphyStatus_t CUPHYWINAPI cuphyRunSrsChEst(cuphySrsChEstHndl_t srsChEstHndl, cudaStream_t strm)
{
    if(!srsChEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    srs_ch_est::srsChEst* pChEst = static_cast<srs_ch_est::srsChEst*>(srsChEstHndl);

    // @todo: one CUDA stream per heterogenous config for stream (i.e. non-graph) based kernel launch
    srs_ch_est::srsChEstStrmVec_t strmVec{strm};
    pChEst->run(strmVec);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCrcEncodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyCrcEncodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(crcEncodeDescr_t);
    *pDescrAlignBytes = alignof(crcEncodeDescr_t);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPrepareCrcEncodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPrepareCrcEncodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(prepareCrcEncodeDescr_t);
    *pDescrAlignBytes = alignof(prepareCrcEncodeDescr_t);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyLDPCEncodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyLDPCEncodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes, uint16_t maxUEs, size_t* pWorkspaceBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes || !pWorkspaceBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //*pDescrSizeBytes = sizeof(ldpcEncodeDescr_t);
    //*pDescrAlignBytes = alignof(ldpcEncodeDescr_t);
    *pDescrSizeBytes  = sizeof(ldpcEncodeDescr_t_array);
    *pDescrAlignBytes = alignof(ldpcEncodeDescr_t_array);
    *pWorkspaceBytes  = 2 * maxUEs * sizeof(LDPC_output_t); // 2x because it includes output and input
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDlRateMatchingGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyDlRateMatchingGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(dlRateMatchingDescr_t);
    *pDescrAlignBytes = alignof(dlRateMatchingDescr_t);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyModulationGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyModulationGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(modulationDescr_t);
    *pDescrAlignBytes = alignof(modulationDescr_t);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPdschDmrsGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPdschDmrsGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(pdschDmrsDescr_t);
    *pDescrAlignBytes = alignof(pdschDmrsDescr_t);
    return CUPHY_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////
// cuphyPdschCsirsPrepGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPdschCsirsPrepGetDescrInfo(size_t* pDescrSizeBytes,
                                                          size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(pdschCsirsPrepDescr_t);
    *pDescrAlignBytes = alignof(pdschCsirsPrepDescr_t);
    return CUPHY_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////
// RM decoder

cuphyStatus_t CUPHYWINAPI cuphyRmDecoderGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    rmDecoder::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyCreateRmDecoder(cuphyContext_t        context,
                                               cuphyRmDecoderHndl_t* pHndl,
                                               unsigned int          flags,
                                               void*                 pMemoryFootprint)
{
    if(!pHndl || !context)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pHndl                = nullptr;
    cuphy_i::context& ctx = static_cast<cuphy_i::context&>(*context);
    try
    {
        rmDecoder* d = new rmDecoder(ctx);
        *pHndl       = static_cast<cuphyRmDecoderHndl_t>(d);
        d->init(pMemoryFootprint);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphySetupRmDecoder(cuphyRmDecoderHndl_t       hndl,
                                              uint16_t                   nCws,
                                              cuphyRmCwPrm_t*            pCwPrmsGpu,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                                              void*                      pCpuDynDesc,                 // pointer to descriptor in cpu
                                              void*                      pGpuDynDesc,                 // pointer to descriptor in gpu
                                              cuphyRmDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                                              cudaStream_t               strm)                                      // stream to perform copy
{
    if(!hndl || !pCwPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    rmDecoder* pDecoder = static_cast<rmDecoder*>(hndl);

    pDecoder->setup(nCws,
                    pCwPrmsGpu,
                    static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                    static_cast<rmDecoderDynDescr_t*>(pCpuDynDesc),
                    pGpuDynDesc,
                    pLaunchCfg,
                    strm);

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyRmDecoder(cuphyRmDecoderHndl_t hndl)
{
    if(!hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    rmDecoder* pRmDecoder = static_cast<rmDecoder*>(hndl);
    delete pRmDecoder;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// Simplex decoder

cuphyStatus_t CUPHYWINAPI cuphySimplexDecoderGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    SimplexDecoder::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyCreateSimplexDecoder(cuphySimplexDecoderHndl_t* pHndl)
{
    if(!pHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pHndl = nullptr;
    try
    {
        SimplexDecoder* d = new SimplexDecoder();
        *pHndl            = static_cast<cuphySimplexDecoderHndl_t>(d);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphySetupSimplexDecoder(cuphySimplexDecoderHndl_t       simplexDecoderHndl,
                                                   uint16_t                        nCws,
                                                   cuphySimplexCwPrm_t*            pCwPrmsCpu,
                                                   cuphySimplexCwPrm_t*            pCwPrmsGpu,
                                                   uint8_t                         enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                                                   void*                           pCpuDynDesc,                 // pointer to descriptor in cpu
                                                   void*                           pGpuDynDesc,                 // pointer to descriptor in gpu
                                                   cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                                                   cudaStream_t                    strm)                                           // stream to perform copy
{
    if(!pCwPrmsCpu || !pCwPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    SimplexDecoder* pDecoder = static_cast<SimplexDecoder*>(simplexDecoderHndl);

    pDecoder->setup(nCws,
                    pCwPrmsCpu,
                    pCwPrmsGpu,
                    static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                    static_cast<simplexDecoderDynDescr_t*>(pCpuDynDesc),
                    pGpuDynDesc,
                    pLaunchCfg,
                    strm);

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyDestroySimplexDecoder(cuphySimplexDecoderHndl_t simplexDecoderHndl)
{
    if(!simplexDecoderHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    SimplexDecoder* pSimplexDecoder = static_cast<SimplexDecoder*>(simplexDecoderHndl);
    delete pSimplexDecoder;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPucchF0RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF0RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF0Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF0Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF0Rx(cuphyPucchF0RxHndl_t* pPucchF0RxHndl, cudaStream_t strm)
{
    if(!pPucchF0RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF0RxHndl = nullptr;
    try
    {
        pucchF0Rx* pPucchF0Rx = new pucchF0Rx(strm);
        *pPucchF0RxHndl       = static_cast<cuphyPucchF0RxHndl_t>(pPucchF0Rx);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPucchF0Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF0Rx(cuphyPucchF0RxHndl_t       pucchF0RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              cuphyPucchF0F1UciOut_t*    pF0UcisOut,
                                              uint16_t                   nCells,
                                              uint16_t                   nF0Ucis,
                                              cuphyPucchUciPrm_t*        pF0UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF0RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF0RxHndl || !pDataRx || !pF0UcisOut || !pF0UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF0Rx* pPucchF0Rx = static_cast<pucchF0Rx*>(pucchF0RxHndl);

    pPucchF0Rx->setup(pDataRx,
                      pF0UcisOut,
                      nCells,
                      nF0Ucis,
                      pF0UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF0RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF0Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF0Rx(cuphyPucchF0RxHndl_t pucchF0RxHndl)
{
    if(!pucchF0RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF0Rx* pPucchF0Rx = static_cast<pucchF0Rx*>(pucchF0RxHndl);
    delete pPucchF0Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// cuphyPucchF1RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF1RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF1Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF1Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF1Rx(cuphyPucchF1RxHndl_t* pPucchF1RxHndl, cudaStream_t strm)
{
    if(!pPucchF1RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF1RxHndl = nullptr;
    try
    {
        pucchF1Rx* pPucchF1Rx = new pucchF1Rx(strm);
        *pPucchF1RxHndl       = static_cast<cuphyPucchF1RxHndl_t>(pPucchF1Rx);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPucchF1Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF1Rx(cuphyPucchF1RxHndl_t       pucchF1RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              cuphyPucchF0F1UciOut_t*    pF1UcisOut,
                                              uint16_t                   nCells,
                                              uint16_t                   nF1Ucis,
                                              cuphyPucchUciPrm_t*        pF1UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF1RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF1RxHndl || !pDataRx || !pF1UcisOut || !pF1UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF1Rx* pPucchF1Rx = static_cast<pucchF1Rx*>(pucchF1RxHndl);

    pPucchF1Rx->setup(pDataRx,
                      pF1UcisOut,
                      nCells,
                      nF1Ucis,
                      pF1UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF1RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF1Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF1Rx(cuphyPucchF1RxHndl_t pucchF1RxHndl)
{
    if(!pucchF1RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF1Rx* pPucchF1Rx = static_cast<pucchF1Rx*>(pucchF1RxHndl);
    delete pPucchF1Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF2RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF2RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF2Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}
////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF2Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF2Rx(cuphyPucchF2RxHndl_t* pPucchF2RxHndl, cudaStream_t strm)
{
    if(!pPucchF2RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF2RxHndl = nullptr;
    try
    {
        pucchF2Rx* pPucchF2Rx = new pucchF2Rx(strm);
        *pPucchF2RxHndl       = static_cast<cuphyPucchF2RxHndl_t>(pPucchF2Rx);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}
////////////////////////////////////////////////////////////////////////
// cuphySetupPucchF2Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF2Rx(cuphyPucchF2RxHndl_t       pucchF2RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              __half**                   pDescramLLRaddrs,
                                              uint8_t*                   pDTXflags,
                                              float*                     pSinr,
                                              float*                     pRssi,
                                              float*                     pRsrp,
                                              float*                     pInterf,
                                              float*                     pNoiseVar,
                                              float*                     pTaEst,
                                              uint16_t                   nCells,
                                              uint16_t                   nF2Ucis,
                                              cuphyPucchUciPrm_t*        pF2UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF2RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF2RxHndl || !pDataRx || !pDescramLLRaddrs || !pDTXflags || !pSinr || !pRssi || !pRsrp || !pInterf || !pNoiseVar || !pF2UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF2Rx* pPucchF2Rx = static_cast<pucchF2Rx*>(pucchF2RxHndl);

    pPucchF2Rx->setup(pDataRx,
                      pDescramLLRaddrs,
                      pDTXflags,
                      pSinr,
                      pRssi,
                      pRsrp,
                      pInterf,
                      pNoiseVar,
                      pTaEst,
                      nCells,
                      nF2Ucis,
                      pF2UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF2RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF2Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF2Rx(cuphyPucchF2RxHndl_t pucchF2RxHndl)
{
    if(!pucchF2RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF2Rx* pPucchF2Rx = static_cast<pucchF2Rx*>(pucchF2RxHndl);
    delete pPucchF2Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF3RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF3RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF3Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF3Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3Rx(cuphyPucchF3RxHndl_t* pPucchF3RxHndl, cudaStream_t strm)
{
    if(!pPucchF3RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF3RxHndl = nullptr;
    try
    {
        pucchF3Rx* pPucchF3Rx = new pucchF3Rx(strm);
        *pPucchF3RxHndl       = static_cast<cuphyPucchF3RxHndl_t>(pPucchF3Rx);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPucchF3Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3Rx(cuphyPucchF3RxHndl_t       pucchF3RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              __half**                   pDescramLLRaddrs,
                                              uint8_t*                   pDTXflags,
                                              float*                     pSinr,
                                              float*                     pRssi,
                                              float*                     pRsrp,
                                              float*                     pInterf,
                                              float*                     pNoiseVar,
                                              float*                     pTaEst,
                                              uint16_t                   nCells,
                                              uint16_t                   nF3Ucis,
                                              cuphyPucchUciPrm_t*        pF3UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF3RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF3RxHndl || !pDataRx || !pDescramLLRaddrs || !pDTXflags || !pSinr || !pRssi || !pRsrp || !pInterf || !pNoiseVar || !pF3UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3Rx* pPucchF3Rx = static_cast<pucchF3Rx*>(pucchF3RxHndl);

    pPucchF3Rx->setup(pDataRx,
                      pDescramLLRaddrs,
                      pDTXflags,
                      pSinr,
                      pRssi,
                      pRsrp,
                      pInterf,
                      pNoiseVar,
                      pTaEst,
                      nCells,
                      nF3Ucis,
                      pF3UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF3RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF3Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3Rx(cuphyPucchF3RxHndl_t pucchF3RxHndl)
{
    if(!pucchF3RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3Rx* pPucchF3Rx = static_cast<pucchF3Rx*>(pucchF3RxHndl);
    delete pPucchF3Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPucchF3Csi2CtrlGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF3Csi2CtrlGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF3Csi2Ctrl::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF3Csi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t* pPucchF3Csi2CtrlHndl)
{
    if(!pPucchF3Csi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF3Csi2CtrlHndl = nullptr;
    try
    {
        pucchF3Csi2Ctrl* pPucchF3Csi2Ctrl = new pucchF3Csi2Ctrl;
        *pPucchF3Csi2CtrlHndl             = static_cast<cuphyPucchF3Csi2CtrlHndl_t>(pPucchF3Csi2Ctrl);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPucchF3Csi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t           pucchF3Csi2CtrlHndl,
                                                    uint16_t                             nCsi2Ucis,                 
                                                    uint16_t*                            pCsi2UciIdxsCpu,
                                                    cuphyPucchUciPrm_t*                  pUciPrmsCpu,                   
                                                    cuphyPucchUciPrm_t*                  pUciPrmsGpu,
                                                    cuphyPucchCellStatPrm_t*             pCellStatPrmsGpu,
                                                    cuphyPucchF234OutOffsets_t*          pPucchF3OutOffsetsCpu,    
                                                    uint8_t*                             pUciPayloadsGpu,              
                                                    uint16_t*                            pNumCsi2BitsGpu,               
                                                    cuphyPolarUciSegPrm_t*               pCsi2PolarSegPrmsGpu,          
                                                    cuphyPolarCwPrm_t*                   pCsi2PolarCwPrmsGpu,          
                                                    cuphyRmCwPrm_t*                      pCsi2RmCwPrmsGpu,            
                                                    cuphySimplexCwPrm_t*                 pCsi2SpxCwPrmsGpu,                  
                                                    void*                                pCpuDynDesc,
                                                    void*                                pGpuDynDesc,
                                                    bool                                 enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPucchF3Csi2CtrlLaunchCfg_t*     pLaunchCfg,
                                                    cudaStream_t                         strm)
{
    if(!pucchF3Csi2CtrlHndl || !pCsi2UciIdxsCpu || !pUciPrmsCpu || !pUciPrmsGpu || !pCellStatPrmsGpu || !pPucchF3OutOffsetsCpu || !pUciPayloadsGpu || !pNumCsi2BitsGpu || !pCsi2PolarSegPrmsGpu || !pCsi2PolarCwPrmsGpu || !pCsi2RmCwPrmsGpu || !pCsi2SpxCwPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    pucchF3Csi2Ctrl* pPucchF3Csi2Ctrl = static_cast<pucchF3Csi2Ctrl*>(pucchF3Csi2CtrlHndl);

    pPucchF3Csi2Ctrl->setup(nCsi2Ucis,
                               pCsi2UciIdxsCpu,
                               pUciPrmsCpu,
                               pUciPrmsGpu,
                               pCellStatPrmsGpu,
                               pPucchF3OutOffsetsCpu,
                               pUciPayloadsGpu,
                               pNumCsi2BitsGpu,
                               pCsi2PolarSegPrmsGpu,
                               pCsi2PolarCwPrmsGpu,
                               pCsi2RmCwPrmsGpu,
                               pCsi2SpxCwPrmsGpu,
                               static_cast<pucchF3Csi2CtrlDynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t pucchF3Csi2CtrlHndl)
{
    if(!pucchF3Csi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3Csi2Ctrl* pPucchF3Csi2Ctrl = static_cast<pucchF3Csi2Ctrl*>(pucchF3Csi2CtrlHndl);
    delete pPucchF3Csi2Ctrl;
    return CUPHY_STATUS_SUCCESS;
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF3SegLLRsGetDescrInfo
cuphyStatus_t CUPHYWINAPI cuphyPucchF3SegLLRsGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                          size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF3SegLLRs::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF3SegLLRs
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t* pPucchF3SegLLRsHndl)
{
    if(!pPucchF3SegLLRsHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF3SegLLRsHndl = nullptr;
    try
    {
        pucchF3SegLLRs* pPucchF3SegLLRs = new pucchF3SegLLRs;
        *pPucchF3SegLLRsHndl             = static_cast<cuphyPucchF3SegLLRsHndl_t>(pPucchF3SegLLRs);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPucchF3SegLLRs
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t            pucchF3SegLLRsHndl,
                                                   uint16_t                             nF3Ucis,
                                                   cuphyPucchUciPrm_t*                  pF3UciPrms,
                                                   __half**                             pDescramLLRaddrs,
                                                   void*                                pCpuDynDesc,
                                                   void*                                pGpuDynDesc,
                                                   bool                                 enableCpuToGpuDescrAsyncCpy,
                                                   cuphyPucchF3SegLLRsLaunchCfg_t*      pLaunchCfg,
                                                   cudaStream_t                         strm)
{
    if(!pucchF3SegLLRsHndl || !pF3UciPrms || !pDescramLLRaddrs || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    pucchF3SegLLRs* pPucchF3SegLLRs = static_cast<pucchF3SegLLRs*>(pucchF3SegLLRsHndl);

    pPucchF3SegLLRs->setup(nF3Ucis,
                           pF3UciPrms,
                           pDescramLLRaddrs,
                           static_cast<pucchF3SegLLRsDynDescr_t*>(pCpuDynDesc),
                           pGpuDynDesc,
                           enableCpuToGpuDescrAsyncCpy,
                           pLaunchCfg,
                           strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF3SegLLRs
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t pucchF3SegLLRsHndl)
{
    if(!pucchF3SegLLRsHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3SegLLRs* pPucchF3SegLLRs = static_cast<pucchF3SegLLRs*>(pucchF3SegLLRsHndl);
    delete pPucchF3SegLLRs;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF234UciSegGetDescrInfo
cuphyStatus_t CUPHYWINAPI cuphyPucchF234UciSegGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                           size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF234UciSeg::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF234UciSeg
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF234UciSeg(cuphyPucchF234UciSegHndl_t* pPucchF234UciSegHndl)
{
    if(!pPucchF234UciSegHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF234UciSegHndl = nullptr;
    try
    {
        pucchF234UciSeg* pPucchF234UciSeg = new pucchF234UciSeg;
        *pPucchF234UciSegHndl             = static_cast<cuphyPucchF234UciSegHndl_t>(pPucchF234UciSeg);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPucchF234UciSeg
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF234UciSeg(cuphyPucchF234UciSegHndl_t       pucchF234UciSegHndl,
                                                    uint16_t                         nF2Ucis,
                                                    uint16_t                         nF3Ucis,
                                                    cuphyPucchUciPrm_t*              pF2UciPrms,
                                                    cuphyPucchUciPrm_t*              pF3UciPrms,
                                                    cuphyPucchF234OutOffsets_t*&     pF2OutOffsetsCpu,
                                                    cuphyPucchF234OutOffsets_t*&     pF3OutOffsetsCpu,
                                                    uint8_t*                         uciPayloadsGpu,
                                                    void*                            pCpuDynDesc,
                                                    void*                            pGpuDynDesc,
                                                    bool                             enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPucchF234UciSegLaunchCfg_t* pLaunchCfg,
                                                    cudaStream_t                     strm)
{
    if(!pucchF234UciSegHndl || (!pF2UciPrms && !pF3UciPrms) || (!pF2OutOffsetsCpu && !pF3OutOffsetsCpu) || !uciPayloadsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    pucchF234UciSeg* pPucchF234UciSeg = static_cast<pucchF234UciSeg*>(pucchF234UciSegHndl);

    pPucchF234UciSeg->setup(nF2Ucis,
                            nF3Ucis,
                            pF2UciPrms,
                            pF3UciPrms,
                            pF2OutOffsetsCpu,
                            pF3OutOffsetsCpu,
                            uciPayloadsGpu,
                            static_cast<pucchF234UciSegDynDescr_t*>(pCpuDynDesc),
                            pGpuDynDesc,
                            enableCpuToGpuDescrAsyncCpy,
                            pLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF234UciSeg
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF234UciSeg(cuphyPucchF234UciSegHndl_t pPucchF234UciSegHndl)
{
    if(!pPucchF234UciSegHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF234UciSeg* pPucchF234UciSeg = static_cast<pucchF234UciSeg*>(pPucchF234UciSegHndl);
    delete pPucchF234UciSeg;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyModulateSymbol()
cuphyStatus_t cuphyModulateSymbol(cuphyTensorDescriptor_t tSym,
                                  void*                   pSym,
                                  cuphyTensorDescriptor_t tBits,
                                  const void*             pBits,
                                  int                     log2_QAM,
                                  cudaStream_t            strm)
{
    std::array<int, 5> valid_log_mod = {CUPHY_QAM_2,
                                        CUPHY_QAM_4,
                                        CUPHY_QAM_16,
                                        CUPHY_QAM_64,
                                        CUPHY_QAM_256};
    //------------------------------------------------------------------
    // Validate inputs
    if(!tSym ||
       !pSym ||
       !tBits ||
       !pBits ||
       valid_log_mod.end() == std::find(valid_log_mod.begin(),
                                        valid_log_mod.end(),
                                        log2_QAM))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    const tensor_desc& tSymDesc = static_cast<const tensor_desc&>(*tSym);
    const tensor_desc& tBitDesc = static_cast<const tensor_desc&>(*tBits);
    //------------------------------------------------------------------
    // Validate tensor types
    if(CUPHY_BIT != tBitDesc.type() ||
       (CUPHY_C_32F != tSymDesc.type() && (CUPHY_C_16F != tSymDesc.type())))
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    // Validate tensor sizes
    if((0 != (tBitDesc.layout().dimensions[0] % log2_QAM)) ||
       (tBitDesc.layout().dimensions[0] / log2_QAM != tSymDesc.layout().dimensions[0]))
    {
        return CUPHY_STATUS_SIZE_MISMATCH;
    }
    return cuphy_i::symbol_modulate(tSymDesc,
                                    pSym,
                                    tBitDesc,
                                    pBits,
                                    log2_QAM,
                                    strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyDemodulateSymbol()
cuphyStatus_t cuphyDemodulateSymbol(cuphyContext_t          context,
                                    cuphyTensorDescriptor_t tLLR,
                                    void*                   pLLR,
                                    cuphyTensorDescriptor_t tSym,
                                    const void*             pSym,
                                    int                     log2_QAM,
                                    float                   noiseVariance,
                                    cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!context ||
       !tLLR ||
       !pLLR ||
       !tSym ||
       !pSym ||
       (log2_QAM < 1) ||
       (log2_QAM > 8) ||
       (noiseVariance <= 0.0f))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc& tLLRDesc = static_cast<tensor_desc&>(*tLLR);
    tensor_desc& tSymDesc = static_cast<tensor_desc&>(*tSym);
    if((tLLRDesc.type() != CUPHY_R_32F) &&
       (tLLRDesc.type() != CUPHY_R_16F))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    if((tSymDesc.type() != CUPHY_C_32F) &&
       (tSymDesc.type() != CUPHY_C_16F))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    cuphy_i::context& ctx = static_cast<cuphy_i::context&>(*context);
    return cuphy_i::soft_demap(ctx,
                               tLLRDesc,
                               pLLR,
                               tSymDesc,
                               pSym,
                               log2_QAM,
                               noiseVariance,
                               strm);
}

cuphyStatus_t CUPHYWINAPI cuphySetGenericEmptyKernelNodeGridConstantParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, uint16_t descr_size)
{
    if(pNodeParams == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if((descr_size != 32) && (descr_size != 48))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, (descr_size == 32) ?  reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_32B)  : reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_48B)); // TODO expand as needed
    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 32;
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = pKernelParams;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphySetGenericEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, int ptrArgsCnt, void** pKernelParams)
{
    if((pNodeParams == nullptr) || (ptrArgsCnt < 0) || (ptrArgsCnt > 9))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    void* kernel_function_ptr[10] = {reinterpret_cast<void*>(graphs_empty_kernel),
                                     reinterpret_cast<void*>(graphs_empty_kernel_1_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_2_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_3_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_4_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_5_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_6_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_7_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_8_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_9_ptr_arg)};

    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, kernel_function_ptr[ptrArgsCnt]);
    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 32;
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = pKernelParams;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;
    pNodeParams->kern           = nullptr;
    pNodeParams->ctx            = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphySetEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams)
{
#if 0
    return cuphySetGenericEmptyKernelNodeParams(pNodeParams, 0, nullptr);
#else
    if(pNodeParams == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, reinterpret_cast<void*>(graphs_empty_kernel));
    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 32;
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = nullptr;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;
    pNodeParams->kern           = nullptr;
    pNodeParams->ctx            = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
#endif
}

void CUPHYWINAPI cuphySetD2HMemcpyNodeParams(CUDA_MEMCPY3D *memcpyParams, void* src_d, void* dst_h, size_t size_in_bytes) {
    *memcpyParams = {0};
    memcpyParams->WidthInBytes = size_in_bytes;
    memcpyParams->Height = 1;
    memcpyParams->Depth = 1;
    memcpyParams->dstHost = dst_h;
    memcpyParams->dstMemoryType = CU_MEMORYTYPE_HOST;
    memcpyParams->srcDevice = reinterpret_cast<CUdeviceptr>(src_d);
    memcpyParams->srcMemoryType = CU_MEMORYTYPE_DEVICE;
}


////////////////////////////////////////////////////////////////////////
// cuphyCreateRandomNumberGenerator()
cuphyStatus_t CUPHYWINAPI cuphyCreateRandomNumberGenerator(cuphyRNG_t*        pRNG,
                                                           unsigned long long seed,
                                                           unsigned int       flags,
                                                           cudaStream_t       strm)
{
    if(!pRNG)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pRNG = nullptr;
    try
    {
        cuphy_i::rng* r = new cuphy_i::rng(seed, strm);
        *pRNG           = static_cast<cuphyRNG_t>(r);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyRandomNumberGenerator()
cuphyStatus_t CUPHYWINAPI cuphyDestroyRandomNumberGenerator(cuphyRNG_t rng)
{
    if(!rng)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cuphy_i::rng* r = static_cast<cuphy_i::rng*>(rng);
    delete r;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyRandomUniform()
cuphyStatus_t CUPHYWINAPI cuphyRandomUniform(cuphyRNG_t              rng,
                                             cuphyTensorDescriptor_t tDst,
                                             void*                   pDst,
                                             const cuphyVariant_t*   minValue,
                                             const cuphyVariant_t*   maxValue,
                                             cudaStream_t            strm)
{
    if(!rng ||
       !tDst ||
       !pDst ||
       !minValue ||
       !maxValue)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc&  tDesc = static_cast<tensor_desc&>(*tDst);
    cuphy_i::rng& r     = static_cast<cuphy_i::rng&>(*rng);
    return r.uniform(tDesc, pDst, *minValue, *maxValue, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyRandomNormal()
cuphyStatus_t CUPHYWINAPI cuphyRandomNormal(cuphyRNG_t              rng,
                                            cuphyTensorDescriptor_t tDst,
                                            void*                   pDst,
                                            const cuphyVariant_t*   mean,
                                            const cuphyVariant_t*   stddev,
                                            cudaStream_t            strm)
{
    if(!rng ||
       !tDst ||
       !pDst ||
       !mean ||
       !stddev)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc&  tDesc = static_cast<tensor_desc&>(*tDst);
    cuphy_i::rng& r     = static_cast<cuphy_i::rng&>(*rng);
    return r.normal(tDesc, pDst, *mean, *stddev, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyConvertVariant()
cuphyStatus_t CUPHYWINAPI cuphyConvertVariant(cuphyVariant_t* v,
                                              cuphyDataType_t t)
{
    if(!v ||
       (CUPHY_VOID == t))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy_i::convert_variant(*v, t);
}

////////////////////////////////////////////////////////////////////////
// cuphyFillTensor()
cuphyStatus_t CUPHYWINAPI cuphyFillTensor(cuphyTensorDescriptor_t tDst,
                                          void*                   pDst,
                                          const cuphyVariant_t*   v,
                                          cudaStream_t            strm)
{
    if(!tDst || !pDst || !v || (CUPHY_VOID == v->type))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy_i::tensor_fill(static_cast<tensor_desc&>(*tDst),
                                pDst,
                                *v,
                                strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyTileTensor()
cuphyStatus_t CUPHYWINAPI cuphyTileTensor(cuphyTensorDescriptor_t tDst,
                                          void*                   pDst,
                                          cuphyTensorDescriptor_t tSrc,
                                          const void*             pSrc,
                                          int                     tileRank,
                                          const int*              tileExtents,
                                          cudaStream_t            strm)
{
    //------------------------------------------------------------------
    if(!tDst || !pDst || !tSrc || !pSrc || (0 == tileRank) || !tileExtents || (tileRank > CUPHY_DIM_MAX))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Types must match (to avoid having an exponential number of
    // conversion tiling kernels).
    const tensor_desc& descDst = static_cast<tensor_desc&>(*tDst);
    const tensor_desc& descSrc = static_cast<tensor_desc&>(*tSrc);
    if(descDst.type() != descSrc.type())
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    return cuphy_i::tensor_tile(descDst,
                                pDst,
                                descSrc,
                                pSrc,
                                tileRank,
                                tileExtents,
                                strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyTensorElementWiseOperation()
cuphyStatus_t CUPHYWINAPI cuphyTensorElementWiseOperation(cuphyTensorDescriptor_t tDst,
                                                          void*                   pDst,
                                                          cuphyTensorDescriptor_t tSrcA,
                                                          const void*             pSrcA,
                                                          const cuphyVariant_t*   alpha,
                                                          cuphyTensorDescriptor_t tSrcB,
                                                          const void*             pSrcB,
                                                          const cuphyVariant_t*   beta,
                                                          cuphyElementWiseOp_t    elemOp,
                                                          cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Note that input B is optional for some operations, but the
    // destination and input A are required.
    if(!tDst || !pDst || !tSrcA || !pSrcA)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    // If there is a descriptor for B, there must also be an address
    if((nullptr == tSrcB) != (nullptr == pSrcB))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const tensor_desc& descDst  = static_cast<const tensor_desc&>(*tDst);
    const tensor_desc& descSrcA = static_cast<const tensor_desc&>(*tSrcA);
    return cuphy_i::tensor_elementwise(descDst,
                                       pDst,
                                       descSrcA,
                                       pSrcA,
                                       alpha,
                                       tSrcB,
                                       pSrcB,
                                       beta,
                                       elemOp,
                                       strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyTensorReduction()
cuphyStatus_t CUPHYWINAPI cuphyTensorReduction(cuphyTensorDescriptor_t tDst,
                                               void*                   pDst,
                                               cuphyTensorDescriptor_t tSrc,
                                               const void*             pSrc,
                                               cuphyReductionOp_t      redOp,
                                               int                     dim,
                                               size_t                  workspaceSize,
                                               void*                   workspace,
                                               cudaStream_t            strm)
{
    if(!tDst || !pDst || !tSrc || !pSrc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if((dim < 0) || (dim >= CUPHY_DIM_MAX))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const tensor_desc& descDst = static_cast<const tensor_desc&>(*tDst);
    const tensor_desc& descSrc = static_cast<const tensor_desc&>(*tSrc);
    return cuphy_i::tensor_reduction(descDst,
                                     pDst,
                                     descSrc,
                                     pSrc,
                                     redOp,
                                     dim,
                                     workspaceSize,
                                     workspace,
                                     strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyCompCwTreeTypesGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyCompCwTreeTypesGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    compCwTreeTypes::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateCompCwTreeTypes()

cuphyStatus_t CUPHYWINAPI cuphyCreateCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t* pCompCwTreeTypesHndl)
{
    if(!pCompCwTreeTypesHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pCompCwTreeTypesHndl = nullptr;
    try
    {
        compCwTreeTypes* pCompCwTreeTypes = new compCwTreeTypes;
        *pCompCwTreeTypesHndl             = static_cast<cuphyCompCwTreeTypesHndl_t>(pCompCwTreeTypes);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupCompCwTreeTypes()
cuphyStatus_t CUPHYWINAPI cuphySetupCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t       compCwTreeTypesHndl,
                                                    uint16_t                         nPolUciSegs,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,
                                                    uint8_t**                        pCwTreeTypesAddrs,
                                                    void*                            pCpuDynDescCompTree,
                                                    void*                            pGpuDynDescCompTree,
                                                    void*                            pCpuDynDescCompTreeAddrs,
                                                    uint8_t                          enableCpuToGpuDescrAsyncCpy,
                                                    cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg,
                                                    cudaStream_t                     strm)
{
    if(!compCwTreeTypesHndl || !pPolUciSegPrmsCpu || !pPolUciSegPrmsGpu || !pCpuDynDescCompTree || !pGpuDynDescCompTree || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    compCwTreeTypes* pCompCwTreeTypes = static_cast<compCwTreeTypes*>(compCwTreeTypesHndl);

    auto pCpuDynDesc = static_cast<compCwTreeTypesDynDescr_t*>(pCpuDynDescCompTree);
    pCpuDynDesc->pCwTreeTypesAddrs = static_cast<uint8_t**> (pCpuDynDescCompTreeAddrs);

    pCompCwTreeTypes->setup(nPolUciSegs,
                            pPolUciSegPrmsCpu,
                            pPolUciSegPrmsGpu,
                            pCwTreeTypesAddrs,
                            pCpuDynDesc,
                            pGpuDynDescCompTree,
                            enableCpuToGpuDescrAsyncCpy,
                            pLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyCompCwTreeTypes()
cuphyStatus_t CUPHYWINAPI cuphyDestroyCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t compCwTreeTypesHndl)
{
    if(!compCwTreeTypesHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    compCwTreeTypes* pCompCwTreeTypes = static_cast<compCwTreeTypes*>(compCwTreeTypesHndl);
    delete pCompCwTreeTypes;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPolSegDeRmDeItlDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPolSegDeRmDeItlGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    polSegDeRmDeItl::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePolSegDeRmDeItl()

cuphyStatus_t CUPHYWINAPI cuphyCreatePolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t* pPolSegDeRmDeItlHndl)
{
    if(!pPolSegDeRmDeItlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPolSegDeRmDeItlHndl = nullptr;
    try
    {
        polSegDeRmDeItl* pPolSegDeRmDeItl = new polSegDeRmDeItl;
        *pPolSegDeRmDeItlHndl             = static_cast<cuphyPolSegDeRmDeItlHndl_t>(pPolSegDeRmDeItl);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPolSegDeRmDeItl()
cuphyStatus_t CUPHYWINAPI cuphySetupPolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t       polSegDeRmDeItlHndl,
                                                    uint16_t                         nPolUciSegs,
                                                    uint16_t                         nPolCws,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,
                                                    const cuphyPolarCwPrm_t*         pPolCwPrmsCpu,
                                                    const cuphyPolarCwPrm_t*         pPolCwPrmsGpu,
                                                    __half**                         pUciSegLLRsAddrs,
                                                    __half**                         pCwLLRsAddrs,
                                                    void*                            pCpuDynDescDrDi,
                                                    void*                            pGpuDynDescDrDi,
                                                    void*                            pCpuDynDescDrDiCwAddrs,
                                                    void*                            pCpuDynDescDrDiUciAddrs,
                                                    uint8_t                          enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPolSegDeRmDeItlLaunchCfg_t* pLaunchCfg,
                                                    cudaStream_t                     strm)
{
    if(!polSegDeRmDeItlHndl || !pPolUciSegPrmsCpu || !pPolCwPrmsCpu || !pPolUciSegPrmsGpu || !pPolUciSegPrmsGpu || !pUciSegLLRsAddrs || !pCwLLRsAddrs || !pCpuDynDescDrDi || !pGpuDynDescDrDi || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    polSegDeRmDeItl* pPolSegDeRmDeItl = static_cast<polSegDeRmDeItl*>(polSegDeRmDeItlHndl);

    auto pCpuDynDesc              = static_cast<polSegDeRmDeItlDynDescr_t*>(pCpuDynDescDrDi);
    pCpuDynDesc->pCwLLRsAddrs     = static_cast<__half**>(pCpuDynDescDrDiCwAddrs);
    pCpuDynDesc->pUciSegLLRsAddrs = static_cast<__half**>(pCpuDynDescDrDiUciAddrs);

    pPolSegDeRmDeItl->setup(nPolUciSegs,
                            nPolCws,
                            pPolUciSegPrmsCpu,
                            pPolUciSegPrmsGpu,
                            pPolCwPrmsCpu,
                            pPolCwPrmsGpu,
                            pUciSegLLRsAddrs,
                            pCwLLRsAddrs,
                            pCpuDynDesc,
                            pGpuDynDescDrDi,
                            enableCpuToGpuDescrAsyncCpy,
                            pLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPolSegDeRmDeItl()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t polSegDeRmDeItlHndl)
{
    if(!polSegDeRmDeItlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    polSegDeRmDeItl* pPolSegDeRmDeItl = static_cast<polSegDeRmDeItl*>(polSegDeRmDeItlHndl);
    delete pPolSegDeRmDeItl;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschSegLLrs1DescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs1GetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschSegLLRs1::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschSegLLRs1()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t* pUciOnPuschSegLLRs1Hndl)
{
    if(!pUciOnPuschSegLLRs1Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschSegLLRs1Hndl = nullptr;
    try
    {
        uciOnPuschSegLLRs1* pUciOnPuschSegLLRs1 = new uciOnPuschSegLLRs1;
        *pUciOnPuschSegLLRs1Hndl                = static_cast<cuphyUciOnPuschSegLLRs1Hndl_t>(pUciOnPuschSegLLRs1);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupUciOnPuschSegLLRs1()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t       uciOnPuschSegLLRs1Hndl,
                                                       uint16_t                            nUciUes,
                                                       uint16_t*                           pUciUserIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       uint16_t*                           pNumPrbs,
                                                       uint8_t                             startSym,
                                                       uint8_t                             nPuschSym,
                                                       uint8_t                             nPuschDataSym,
                                                       uint8_t*                            pDataSymIdxs,
                                                       uint8_t                             nPuschDmrsSym,
                                                       uint8_t*                            pDmrsSymIdxs,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschSegLLRs1Hndl || !pUciUserIdxs || !pTbPrmsCpu || !pTbPrmsGpu || !pTensorPrmsEqOutLLRs || !pNumPrbs || !pDataSymIdxs || !pDmrsSymIdxs || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschSegLLRs1* pUciOnPuschSegLLRs1 = static_cast<uciOnPuschSegLLRs1*>(uciOnPuschSegLLRs1Hndl);

    pUciOnPuschSegLLRs1->setup(uciOnPuschSegLLRs1Hndl,
                               nUciUes,
                               pUciUserIdxs,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               nUeGrps,
                               pTensorPrmsEqOutLLRs,
                               pNumPrbs,
                               startSym,
                               nPuschSym,
                               nPuschDataSym,
                               pDataSymIdxs,
                               nPuschDmrsSym,
                               pDmrsSymIdxs,
                               static_cast<uciOnPuschSegLLRs1DynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               enableCpuToGpuDescrAsyncCpy,
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschSegLLRs1()
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t uciOnPuschSegLLRs1Hndl)
{
    if(!uciOnPuschSegLLRs1Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschSegLLRs1* pUciOnPuschSegLLRs1 = static_cast<uciOnPuschSegLLRs1*>(uciOnPuschSegLLRs1Hndl);
    delete pUciOnPuschSegLLRs1;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciPolarDecoderGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPolarDecoderGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    polarDecoder::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePolarDecoder()

cuphyStatus_t CUPHYWINAPI cuphyCreatePolarDecoder(cuphyPolarDecoderHndl_t* pPolarDecoderHndl)
{
    if(!pPolarDecoderHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPolarDecoderHndl = nullptr;
    try
    {
        polarDecoder* pPolarDecoder = new polarDecoder;
        *pPolarDecoderHndl          = static_cast<cuphyPolarDecoderHndl_t>(pPolarDecoder);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPolarDecoder()

cuphyStatus_t CUPHYWINAPI cuphySetupPolarDecoder(cuphyPolarDecoderHndl_t       polarDecoderHndl,
                                                 uint16_t                      nPolCws,
                                                 __half**                      pCwTreeLLRsAddrs,
                                                 cuphyPolarCwPrm_t*            pCwPrmsGpu,
                                                 cuphyPolarCwPrm_t*            pCwPrmsCpu,
                                                 uint32_t**                    pPolCbEstAddrs,
                                                 bool**                        pListPolScratchAddrs,
                                                 uint8_t                       nPolarList,
                                                 uint8_t*                      pPolCrcErrorFlags,
                                                 bool                          enableCpuToGpuDescrAsyncCpy,
                                                 void*                         pCpuDynDescPolar,
                                                 void*                         pGpuDynDescPolar,
                                                 void*                         pCpuDynDescPolarLLRAddrs,
                                                 void*                         pCpuDynDescPolarCBAddrs,
                                                 void*                         pCpuDynDescListPolarScratchAddrs,
                                                 cuphyPolarDecoderLaunchCfg_t* pLaunchCfg,
                                                 cudaStream_t                  strm)
{
    if(!pCwTreeLLRsAddrs || !pCwPrmsGpu || !pCwPrmsCpu || !pPolCbEstAddrs || !pPolCrcErrorFlags || !pCpuDynDescPolar || !pGpuDynDescPolar || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    polarDecoder* pPolarDecoder = static_cast<polarDecoder*>(polarDecoderHndl);

    auto pCpuDynDesc                 = static_cast<polarDecoderDynDescr_t*>(pCpuDynDescPolar);
    pCpuDynDesc->cwTreeLLRsAddrs     = static_cast<__half**>(pCpuDynDescPolarLLRAddrs);
    pCpuDynDesc->polCbEstAddrs       = static_cast<uint32_t**>(pCpuDynDescPolarCBAddrs);
    pCpuDynDesc->listPolScratchAddrs = static_cast<bool**>(pCpuDynDescListPolarScratchAddrs);

    pPolarDecoder->setup(nPolCws,
                         pCwTreeLLRsAddrs,
                         pCwPrmsGpu,
                         pCwPrmsCpu,
                         pPolCbEstAddrs,
                         pListPolScratchAddrs,
                         nPolarList,
                         pPolCrcErrorFlags,
                         static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                         pCpuDynDesc,
                         pGpuDynDescPolar,
                         pLaunchCfg,
                         strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPolarDecoder()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolarDecoder(cuphyPolarDecoderHndl_t polarDecoderHndl)
{
    if(!polarDecoderHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    polarDecoder* pPolarDecoder = static_cast<polarDecoder*>(polarDecoderHndl);
    delete pPolarDecoder;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschSegLLrs2DescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs2GetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschSegLLRs2::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschSegLLRs2()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t* pUciOnPuschSegLLRs2Hndl)
{
    if(!pUciOnPuschSegLLRs2Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschSegLLRs2Hndl = nullptr;
    try
    {
        uciOnPuschSegLLRs2* pUciOnPuschSegLLRs2 = new uciOnPuschSegLLRs2;
        *pUciOnPuschSegLLRs2Hndl                = static_cast<cuphyUciOnPuschSegLLRs2Hndl_t>(pUciOnPuschSegLLRs2);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

// ////////////////////////////////////////////////////////////////////////
// // cuphySetupUciOnPuschSegLLRs2()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t       uciOnPuschSegLLRs2Hndl,
                                                       uint16_t                            nCsi2Ues,
                                                       uint16_t*                           pCsi2UeIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsGpu,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs2LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschSegLLRs2Hndl || !pCsi2UeIdxs || !pTbPrmsCpu || !pTbPrmsGpu || !pTensorPrmsEqOutLLRs || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschSegLLRs2* pUciOnPuschSegLLRs2 = static_cast<uciOnPuschSegLLRs2*>(uciOnPuschSegLLRs2Hndl);

    pUciOnPuschSegLLRs2->setup(nCsi2Ues,
                               pCsi2UeIdxs,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               nUeGrps,
                               pTensorPrmsEqOutLLRs,
                               pUeGrpPrmsCpu,
                               pUeGrpPrmsGpu,
                               static_cast<uciOnPuschSegLLRs2DynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               enableCpuToGpuDescrAsyncCpy,
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////

// cuphyDestroyUciOnPuschSegLLRs2()
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t uciOnPuschSegLLRs2Hndl)
{
    if(!uciOnPuschSegLLRs2Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschSegLLRs2* pUciOnPuschSegLLRs2 = static_cast<uciOnPuschSegLLRs2*>(uciOnPuschSegLLRs2Hndl);
    delete pUciOnPuschSegLLRs2;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschSegLLrs0DescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs0GetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschSegLLRs0::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschSegLLRs0()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t* pUciOnPuschSegLLRs0Hndl)
{
    if(!pUciOnPuschSegLLRs0Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschSegLLRs0Hndl = nullptr;
    try
    {
        uciOnPuschSegLLRs0* pUciOnPuschSegLLRs0 = new uciOnPuschSegLLRs0;
        *pUciOnPuschSegLLRs0Hndl                = static_cast<cuphyUciOnPuschSegLLRs0Hndl_t>(pUciOnPuschSegLLRs0);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupUciOnPuschSegLLrs0()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t       uciOnPuschSegLLRs0Hndl,
                                                       uint16_t                            nUciUes,
                                                       uint16_t*                           pUciUeIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsGpu,
                                                       cuphyUciToSeg_t                     uciToSeg,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs0LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschSegLLRs0Hndl || !pUciUeIdxs || !pTbPrmsCpu || !pTbPrmsGpu || !pTensorPrmsEqOutLLRs || !pUeGrpPrmsCpu || !pUeGrpPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschSegLLRs0* pUciOnPuschSegLLRs0 = static_cast<uciOnPuschSegLLRs0*>(uciOnPuschSegLLRs0Hndl);

    pUciOnPuschSegLLRs0->setup(nUciUes,
                               pUciUeIdxs,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               nUeGrps,
                               pTensorPrmsEqOutLLRs,
                               pUeGrpPrmsCpu,
                               pUeGrpPrmsGpu,
                               uciToSeg,
                               static_cast<uciOnPuschSegLLRs0DynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               enableCpuToGpuDescrAsyncCpy,
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschSegLLRs0()

cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t uciOnPuschSegLLRs0Hndl)
{
    if(!uciOnPuschSegLLRs0Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschSegLLRs0* pUciOnPuschSegLLRs0 = static_cast<uciOnPuschSegLLRs0*>(uciOnPuschSegLLRs0Hndl);
    delete pUciOnPuschSegLLRs0;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschCsi2CtrlDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschCsi2CtrlGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschCsi2Ctrl::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t* pUciOnPuschCsi2CtrlHndl)
{
    if(!pUciOnPuschCsi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschCsi2CtrlHndl = nullptr;
    try
    {
        uciOnPuschCsi2Ctrl* pUciOnPuschCsi2Ctrl = new uciOnPuschCsi2Ctrl;
        *pUciOnPuschCsi2CtrlHndl                = static_cast<cuphyUciOnPuschCsi2CtrlHndl_t>(pUciOnPuschCsi2Ctrl);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

// ////////////////////////////////////////////////////////////////////////
// // cuphySetupUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t       uciOnPuschCsi2CtrlHndl,
                                                       uint16_t                            nCsi2Ues,
                                                       uint16_t*                           pCsi2UeIdxsCpu,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschCellStatPrm_t*            pCellStatPrmsGpu,
                                                       cuphyUciOnPuschOutOffsets_t*        pUciOnPuschOutOffsetsCpu,
                                                       uint8_t*                            pUciPayloadsGpu,
                                                       uint16_t*                           pNumCsi2BitsGpu,
                                                       cuphyPolarUciSegPrm_t*              pCsi2PolarSegPrmsGpu,
                                                       cuphyPolarCwPrm_t*                  pCsi2PolarCwPrmsGpu,
                                                       cuphyRmCwPrm_t*                     pCsi2RmCwPrmsGpu,
                                                       cuphySimplexCwPrm_t*                pCsi2SpxCwPrmsGpu,
                                                       uint16_t                            forcedNumCsi2Bits,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschCsi2CtrlHndl || !pCsi2UeIdxsCpu || !pTbPrmsCpu || !pTbPrmsGpu || !pUeGrpPrmsCpu || !pCellStatPrmsGpu || !pUciOnPuschOutOffsetsCpu || !pUciPayloadsGpu || !pNumCsi2BitsGpu || !pCsi2PolarSegPrmsGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschCsi2Ctrl* pUciOnPuschCsi2Ctrl = static_cast<uciOnPuschCsi2Ctrl*>(uciOnPuschCsi2CtrlHndl);

    pUciOnPuschCsi2Ctrl->setup(nCsi2Ues,
                               pCsi2UeIdxsCpu,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               pUeGrpPrmsCpu,
                               pCellStatPrmsGpu,
                               pUciOnPuschOutOffsetsCpu,
                               pUciPayloadsGpu,
                               pNumCsi2BitsGpu,
                               pCsi2PolarSegPrmsGpu,
                               pCsi2PolarCwPrmsGpu,
                               pCsi2RmCwPrmsGpu,
                               pCsi2SpxCwPrmsGpu,
                               forcedNumCsi2Bits,
                               static_cast<uciOnPuschCsi2CtrlDynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t uciOnPuschCsi2CtrlHndl)
{
    if(!uciOnPuschCsi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschCsi2Ctrl* pUciOnPuschCsi2Ctrl = static_cast<uciOnPuschCsi2Ctrl*>(uciOnPuschCsi2CtrlHndl);
    delete pUciOnPuschCsi2Ctrl;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySrsChEst0GetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphySrsChEst0GetDescrInfo(size_t* pStatDescrSizeBytes, size_t* pStatDescrAlignBytes, size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    srsChEst0::getDescrInfo(*pStatDescrSizeBytes, *pStatDescrAlignBytes, *pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateSrsChEst0()

cuphyStatus_t CUPHYWINAPI cuphyCreateSrsChEst0(cuphySrsChEst0Hndl_t* pSrsChEst0Hndl,
                                               cuphySrsFilterPrms_t* pSrsFilterPrms,
                                               uint8_t               enableCpuToGpuDescrAsyncCpy,
                                               void*                 pCpuStatDesc,
                                               void*                 pGpuStatDesc,
                                               cudaStream_t          strm)
{
    if(!pSrsChEst0Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pSrsChEst0Hndl = nullptr;
    try
    {
        srsChEst0* pSrsChEst0 = new srsChEst0;
        *pSrsChEst0Hndl       = static_cast<cuphySrsChEst0Hndl_t>(pSrsChEst0);

        pSrsChEst0->init(pSrsFilterPrms,
                         (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                         static_cast<srsChEst0StatDescr_t*>(pCpuStatDesc),
                         pGpuStatDesc,
                         strm);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

// ////////////////////////////////////////////////////////////////////////
// // cuphySetupSrsChEst0()

cuphyStatus_t CUPHYWINAPI cuphySetupSrsChEst0(   cuphySrsChEst0Hndl_t          srsChEst0Hndl,
                                                 uint16_t                      nSrsUes,
                                                 cuphyUeSrsPrm_t*              h_srsUePrms,
                                                 uint16_t                      nCell,
                                                 cuphyTensorPrm_t*             pTDataRx,
                                                 cuphySrsCellPrms_t*           h_srsCellPrms,
                                                 float*                        d_rbSnrBuff,
                                                 uint32_t*                     h_rbSnrBuffOffsets,
                                                 cuphySrsReport_t*             d_pSrsReports,
                                                 cuphySrsChEstBuffInfo_t*      h_chEstBuffInfo,
                                                 void**                        d_addrsChEstToL2Buff,
                                                 cuphySrsChEstToL2_t*          h_chEstToL2,
                                                 uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                 void*                         pCpuDynDesc,
                                                 void*                         pGpuDynDesc,
                                                 cuphySrsChEst0LaunchCfg_t*    pLaunchCfg,
                                                 cudaStream_t                  strm)
{
    if(!srsChEst0Hndl || !h_srsUePrms || !h_srsCellPrms || !d_rbSnrBuff || !h_rbSnrBuffOffsets || !d_addrsChEstToL2Buff || !h_chEstToL2 || !d_pSrsReports || !h_chEstBuffInfo || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    srsChEst0* pSrsChEst0 = static_cast<srsChEst0*>(srsChEst0Hndl);


    return pSrsChEst0->setup(nSrsUes,
                            h_srsUePrms,
                            nCell,
                            pTDataRx,
                            h_srsCellPrms,
                            d_rbSnrBuff,
                            h_rbSnrBuffOffsets,
                            d_pSrsReports,
                            h_chEstBuffInfo,
                            d_addrsChEstToL2Buff,
                            h_chEstToL2,
                            static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                            static_cast<srsChEst0DynDescr_t*>(pCpuDynDesc),
                            pGpuDynDesc,
                            pLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroySrsChEst0()

cuphyStatus_t CUPHYWINAPI cuphyDestroySrsChEst0(cuphySrsChEst0Hndl_t srsChEst0Hndl)
{
    if(!srsChEst0Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    srsChEst0* pSrsChEst0 = static_cast<srsChEst0*>(srsChEst0Hndl);
    delete pSrsChEst0;
    return CUPHY_STATUS_SUCCESS;
}

