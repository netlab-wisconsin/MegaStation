/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "mex.h"
#include <vector>
#include <string>
#include <string.h>
#include <map>
#include <initializer_list>
#include "cuphy.h"

namespace
{
    

////////////////////////////////////////////////////////////////////////
// get_cuphy_type()
cuphyDataType_t get_cuphy_type(mxClassID classID, bool isComplex)
{
    switch(classID)
    {
    case mxUNKNOWN_CLASS:
    case mxCELL_CLASS:
    case mxSTRUCT_CLASS:
    case mxLOGICAL_CLASS:
    case mxVOID_CLASS:
    case mxCHAR_CLASS:
    case mxFUNCTION_CLASS:
    case mxINT64_CLASS:
    case mxUINT64_CLASS:
    default:
        return CUPHY_VOID;
    case mxDOUBLE_CLASS: return (isComplex ? CUPHY_C_64F : CUPHY_R_64F);
    case mxSINGLE_CLASS: return (isComplex ? CUPHY_C_32F : CUPHY_R_32F);
    case mxINT8_CLASS:   return (isComplex ? CUPHY_C_8I  : CUPHY_R_8I);
    case mxUINT8_CLASS:  return (isComplex ? CUPHY_C_8U  : CUPHY_R_8U);
    case mxINT16_CLASS:  return (isComplex ? CUPHY_C_16I : CUPHY_R_16I);
    case mxUINT16_CLASS: return (isComplex ? CUPHY_C_16U : CUPHY_R_16U);
    case mxINT32_CLASS:  return (isComplex ? CUPHY_C_32I : CUPHY_R_32I);
    case mxUINT32_CLASS: return (isComplex ? CUPHY_C_32U : CUPHY_R_32U);
    }
}

////////////////////////////////////////////////////////////////////////
// get_device_tensor_descriptor()
// Creates a cuPHY tensor descriptor to represent the data in either
// device or host memory, by copying the properties of the existing
// tensor descriptor. A stride flag can be provided to make the created
// descriptor have a different layout.
cuphyTensorDescriptor_t copy_tensor_descriptor(const cuphyTensorDescriptor_t srcDesc,
                                               unsigned int                  flags)
{
    cuphyDataType_t  dataType = CUPHY_VOID;
    std::vector<int> dims(CUPHY_DIM_MAX);
    int              numDims;
    cuphyGetTensorDescriptor(srcDesc,      // source descriptor
                             dims.size(),  // output dimension size
                             &dataType,    // data type (output)
                             &numDims,     // number of dimensions (output)
                             dims.data(),  // dimensions (output)
                             nullptr);     // strides
    cuphyTensorDescriptor_t newDesc = nullptr;
    cuphyCreateTensorDescriptor(&newDesc);
    if(newDesc)
    {
        if(CUPHY_STATUS_SUCCESS != cuphySetTensorDescriptor(newDesc,     // descriptor
                                                            dataType,    // data type
                                                            numDims,     // number of dimensions
                                                            dims.data(), // dimensions,
                                                            nullptr,     // strides
                                                            flags))
        {
            cuphyDestroyTensorDescriptor(newDesc);
            newDesc = nullptr;
        }
    }
    return newDesc;
}

////////////////////////////////////////////////////////////////////////
// get_mx_tensor_descriptor()
// Creates a cuPHY tensor descriptor to represent the data in a mxArray
// (on the host, owned by MATLAB).
cuphyTensorDescriptor_t get_mx_tensor_descriptor(const mxArray* mxA)
{
    cuphyTensorDescriptor_t tensorDesc = nullptr;
    bool                    isComplex = mxIsComplex(mxA);
    if(CUPHY_STATUS_SUCCESS != cuphyCreateTensorDescriptor(&tensorDesc))
    {
        return tensorDesc;
    }
    cuphyDataType_t  dataType = get_cuphy_type(mxGetClassID(mxA), isComplex);
    std::vector<int> dim(mxGetNumberOfDimensions(mxA));
    const mwSize*    mxDim = mxGetDimensions(mxA);
    for(size_t i = 0; i < dim.size(); ++i)
    {
        dim[i] = static_cast<int>(mxDim[i]);
    }
    if(CUPHY_STATUS_SUCCESS != cuphySetTensorDescriptor(tensorDesc,
                                                        dataType,
                                                        static_cast<int>(mxGetNumberOfDimensions(mxA)),
                                                        dim.data(),
                                                        nullptr,
                                                        0))
    {
        fprintf(stderr, "cuphySetTensorDescriptor() failure\n");
        cuphyDestroyTensorDescriptor(tensorDesc);
        return nullptr;
    }
    return tensorDesc;
}

////////////////////////////////////////////////////////////////////////
// allocate_device_tensor()
std::pair<cuphyTensorDescriptor_t, void*> allocate_device_tensor(std::initializer_list<int> ilist,
                                                                 cuphyDataType_t            dataType,
                                                                 unsigned int               alignmentFlags)
{
    std::vector<int>        dims(ilist.begin(), ilist.end());
    cuphyTensorDescriptor_t newDesc = nullptr;
    void*                   pv = nullptr;
    cuphyCreateTensorDescriptor(&newDesc);
    if(newDesc)
    {
        cuphyStatus_t sPHY = cuphySetTensorDescriptor(newDesc,         // descriptor
                                                      dataType,        // data type
                                                      dims.size(),     // number of dimensions
                                                      dims.data(),     // dimensions,
                                                      nullptr,         // strides
                                                      alignmentFlags); // alignment flags
        if(CUPHY_STATUS_SUCCESS != sPHY)
        {
            fprintf(stderr, "cuphySetTensorDescriptor() failure (%s)\n", cuphyGetErrorString(sPHY));
            cuphyDestroyTensorDescriptor(newDesc);
            newDesc = nullptr;
        }
        else
        {
            size_t sz = 0;
            cuphyGetTensorSizeInBytes(newDesc, &sz);
            cudaError_t sCUDA = cudaMalloc(&pv, sz);
            if(cudaSuccess != sCUDA)
            {
                fprintf(stderr, "cudaMalloc() error (%lu bytes) (%s)\n", sz, cudaGetErrorString(sCUDA));
            }
        }
    }
    return std::pair<cuphyTensorDescriptor_t, void*>(newDesc, pv);

}

////////////////////////////////////////////////////////////////////////
// initialize_device_tensor()
// Allocates device memory and initializes it with the contents with
// data from the mxArray. Returns the allocated address (which should
// be freed with cudaFree()) upon success, and nullptr on failure.
// Since we are supporting only the traditional split complex MATLAB
// format for now, this function will return nullptr for complex
// mxArrays.
void* initialize_device_tensor(cuphyTensorDescriptor_t tensorDesc,
                               cuphyTensorDescriptor_t mxtensorDesc,
                               const mxArray*          srcmx)
{
    void*  pvdevice = nullptr;
    void*  pvhost   = nullptr;
    size_t sz       = 0;
    if(mxIsComplex(srcmx))
    {
        return pvdevice;
    }
    if(CUPHY_STATUS_SUCCESS != cuphyGetTensorSizeInBytes(tensorDesc, &sz))
    {
        return pvdevice;
    }
    cudaError_t sCuda = cudaHostAlloc(&pvhost,
                                      mxGetNumberOfElements(srcmx) * mxGetElementSize(srcmx),
                                      cudaHostAllocWriteCombined);
    if(cudaSuccess != sCuda)
    {
        fprintf(stderr, "cudaHostAlloc() failure (%s)\n", cudaGetErrorString(sCuda));
        return pvdevice;
    }
    memcpy(pvhost, mxGetData(srcmx), mxGetNumberOfElements(srcmx) * mxGetElementSize(srcmx));
    if(cudaSuccess != cudaMalloc(&pvdevice, sz))
    {
        fprintf(stderr, "cudaMalloc() failure (%lu bytes)\n", sz);
        cudaFreeHost(pvhost);
        return pvdevice;
    }
    cuphyStatus_t scuPHY = cuphyConvertTensor(tensorDesc,    // dst tensor desc
                                              pvdevice,      // dst data
                                              mxtensorDesc,  // src tensor desc
                                              pvhost,        // src data address
                                              0);            // stream
    if(CUPHY_STATUS_SUCCESS != scuPHY)
    {
        fprintf(stderr, "cuphyConvertTensor() failure (%s)\n", cuphyGetErrorString(scuPHY));
        cudaFree(pvdevice);
        pvdevice = nullptr;
    }
    cudaFreeHost(pvhost);
    return pvdevice;
}

template <typename T>
void memcpy_split_to_interleaved(T* dst, const T* srcReal, const T* srcImag, size_t N)
{
    for(size_t i = 0; i < N; ++i)
    {
        dst[(i * 2) + 0] = srcReal[i];
        dst[(i * 2) + 1] = srcImag[i];
    }
}
    
template <typename T>
void memcpy_interleaved_to_split(T* dstReal, T* dstImag, const T* src, size_t N)
{
    for(size_t i = 0; i < N; ++i)
    {
        dstReal[i] = src[(i * 2) + 0];
        dstImag[i] = src[(i * 2) + 1];
    }
}
    
////////////////////////////////////////////////////////////////////////
// initialize_complex_device_tensor()
// Allocates device memory and initializes it with the contents with
// data from the mxArray. Returns the allocated address (which should
// be freed with cudaFree()) upon success, and nullptr on failure.
// Since we are supporting only the traditional split complex MATLAB
// format for now, this function will return nullptr for complex
// mxArrays.
void* initialize_complex_device_tensor(cuphyTensorDescriptor_t tensorDesc,
                                       cuphyTensorDescriptor_t mxtensorDesc,
                                       const mxArray*          srcmx)
{
    void*  pvdevice = nullptr;
    void*  pvhost   = nullptr;
    size_t sz       = 0;
    if(CUPHY_STATUS_SUCCESS != cuphyGetTensorSizeInBytes(tensorDesc, &sz))
    {
        return pvdevice;
    }
    // Multiply element size by 2 for complex data
    cudaError_t sCuda = cudaHostAlloc(&pvhost,
                                      mxGetNumberOfElements(srcmx) * mxGetElementSize(srcmx) * 2,
                                      cudaHostAllocWriteCombined);
    if(cudaSuccess != sCuda)
    {
        fprintf(stderr, "cudaHostAlloc() failure (%s)\n", cudaGetErrorString(sCuda));
        return pvdevice;
    }
    void* pr = mxGetPr(srcmx);
    void* pi = mxGetPi(srcmx);
    switch(mxGetClassID(srcmx))
    {
    case mxUNKNOWN_CLASS:
    case mxCELL_CLASS:
    case mxSTRUCT_CLASS:
    case mxLOGICAL_CLASS:
    case mxVOID_CLASS:
    case mxCHAR_CLASS:
    case mxFUNCTION_CLASS:
    case mxINT64_CLASS:
    case mxUINT64_CLASS:
    default:
        break;
    case mxDOUBLE_CLASS:
        memcpy_split_to_interleaved(static_cast<double*>(pvhost), static_cast<double*>(pr), static_cast<double*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxSINGLE_CLASS:
        memcpy_split_to_interleaved(static_cast<float*>(pvhost), static_cast<float*>(pr), static_cast<float*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxINT8_CLASS:
        memcpy_split_to_interleaved(static_cast<int8_t*>(pvhost), static_cast<int8_t*>(pr), static_cast<int8_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxUINT8_CLASS:
        memcpy_split_to_interleaved(static_cast<uint8_t*>(pvhost), static_cast<uint8_t*>(pr), static_cast<uint8_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxINT16_CLASS:
        memcpy_split_to_interleaved(static_cast<int16_t*>(pvhost), static_cast<int16_t*>(pr), static_cast<int16_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxUINT16_CLASS:
        memcpy_split_to_interleaved(static_cast<uint16_t*>(pvhost), static_cast<uint16_t*>(pr), static_cast<uint16_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxINT32_CLASS:
        memcpy_split_to_interleaved(static_cast<int32_t*>(pvhost), static_cast<int32_t*>(pr), static_cast<int32_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    case mxUINT32_CLASS:
        memcpy_split_to_interleaved(static_cast<uint32_t*>(pvhost), static_cast<uint32_t*>(pr), static_cast<uint32_t*>(pi), mxGetNumberOfElements(srcmx));
        break;
    }
    if(cudaSuccess != cudaMalloc(&pvdevice, sz))
    {
        fprintf(stderr, "cudaMalloc() failure (%lu bytes)\n", sz);
        cudaFreeHost(pvhost);
        return pvdevice;
    }
    cuphyStatus_t scuPHY = cuphyConvertTensor(tensorDesc,    // dst tensor desc
                                              pvdevice,      // dst data
                                              mxtensorDesc,  // src tensor desc
                                              pvhost,        // src data address
                                              0);            // stream
    if(CUPHY_STATUS_SUCCESS != scuPHY)
    {
        fprintf(stderr, "cuphyConvertTensor() failure (%s)\n", cuphyGetErrorString(scuPHY));
        cudaFree(pvdevice);
        pvdevice = nullptr;
    }
    cudaFreeHost(pvhost);
    return pvdevice;
}

////////////////////////////////////////////////////////////////////////
// allocate_device_tensor()
std::pair<cuphyTensorDescriptor_t, void*> allocate_device_tensor(const mxArray* mxA,
                                                                 unsigned int   alignmentFlags)
{
    cuphyTensorDescriptor_t newDesc = nullptr;
    void*                   pv      = nullptr;
    //------------------------------------------------------------------
    // Create a tensor descriptor from the mxArray
    cuphyTensorDescriptor_t tmx     = get_mx_tensor_descriptor(mxA);
    if(tmx)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make a copy, adding any provide alignment flags
        newDesc = copy_tensor_descriptor(tmx, alignmentFlags);
        if(newDesc)
        {
            pv = initialize_device_tensor(newDesc, tmx, mxA);
            if(!pv)
            {
                cuphyDestroyTensorDescriptor(newDesc);
                newDesc = nullptr;
            }
        }
        cuphyDestroyTensorDescriptor(tmx);
        tmx = nullptr;
    }
    return std::pair<cuphyTensorDescriptor_t, void*>(newDesc, pv);

}

////////////////////////////////////////////////////////////////////////
// allocate_complex_device_tensor()
std::pair<cuphyTensorDescriptor_t, void*> allocate_complex_device_tensor(const mxArray* mxA,
                                                                         unsigned int   alignmentFlags)
{
    cuphyTensorDescriptor_t newDesc = nullptr;
    void*                   pv      = nullptr;
    //------------------------------------------------------------------
    // Create a tensor descriptor from the mxArray
    cuphyTensorDescriptor_t tmx     = get_mx_tensor_descriptor(mxA);
    if(tmx)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make a copy, adding any provide alignment flags
        newDesc = copy_tensor_descriptor(tmx, alignmentFlags);
        if(newDesc)
        {
            pv = initialize_complex_device_tensor(newDesc, tmx, mxA);
            if(!pv)
            {
                cuphyDestroyTensorDescriptor(newDesc);
                newDesc = nullptr;
            }
        }
        cuphyDestroyTensorDescriptor(tmx);
        tmx = nullptr;
    }
    return std::pair<cuphyTensorDescriptor_t, void*>(newDesc, pv);
}

////////////////////////////////////////////////////////////////////////
// dimensions_are_not()
bool dimensions_are_not(const mxArray* mxA, std::initializer_list<int> initList)
{
    if(initList.size() != mxGetNumberOfDimensions(mxA))
    {
        return true;
    }
    const mwSize* dims = mxGetDimensions(mxA);
    for(auto it = initList.begin(); it != initList.end(); ++it)
    {
        if(*it != *dims++)
        {
            return true;
        }
    }
    return false;
}

////////////////////////////////////////////////////////////////////////
// device_tensor_to_complex_mxarray()
bool device_tensor_to_complex_mxarray(mxArray*                                  mxA,
                                      std::pair<cuphyTensorDescriptor_t, void*> tensor)
{
    // Create a duplicate tensor descriptor along with a device mapped
    // host buffer so that we can use the cuphyConvertTensor() function
    // to perform the copy.
    cuphyTensorDescriptor_t hostCopyDesc = copy_tensor_descriptor(tensor.first,
                                                                  CUPHY_TENSOR_ALIGN_TIGHT);
    if(hostCopyDesc)
    {
        void*  pvhost   = nullptr;
        size_t sz       = 0;
        if(CUPHY_STATUS_SUCCESS != cuphyGetTensorSizeInBytes(hostCopyDesc, &sz))
        {
            cuphyDestroyTensorDescriptor(hostCopyDesc);
            return false;
        }
        cudaError_t sCuda = cudaHostAlloc(&pvhost, sz, cudaHostAllocDefault);
        if(cudaSuccess != sCuda)
        {
            fprintf(stderr, "cudaHostAlloc() failure (%s)\n", cudaGetErrorString(sCuda));
            cuphyDestroyTensorDescriptor(hostCopyDesc);
            return false;
        }
        cuphyStatus_t scuPHY = cuphyConvertTensor(hostCopyDesc,  // dst tensor desc
                                                  pvhost,        // dst data
                                                  tensor.first,  // src tensor desc
                                                  tensor.second, // src data address
                                                  0);            // stream
        if(CUPHY_STATUS_SUCCESS != scuPHY)
        {
            fprintf(stderr, "cuphyConvertTensor() failure (%s)\n", cuphyGetErrorString(scuPHY));
            cuphyDestroyTensorDescriptor(hostCopyDesc);
            cudaFreeHost(pvhost);
            pvhost = nullptr;
            return false;
        }

        void* pr = mxGetPr(mxA);
        void* pi = mxGetPi(mxA);
        switch(mxGetClassID(mxA))
        {
        case mxUNKNOWN_CLASS:
        case mxCELL_CLASS:
        case mxSTRUCT_CLASS:
        case mxLOGICAL_CLASS:
        case mxVOID_CLASS:
        case mxCHAR_CLASS:
        case mxFUNCTION_CLASS:
        case mxINT64_CLASS:
        case mxUINT64_CLASS:
        default:
            break;
        case mxDOUBLE_CLASS:
            memcpy_interleaved_to_split(static_cast<double*>(pr), static_cast<double*>(pi), static_cast<double*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxSINGLE_CLASS:
            memcpy_interleaved_to_split(static_cast<float*>(pr), static_cast<float*>(pi), static_cast<float*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxINT8_CLASS:
            memcpy_interleaved_to_split(static_cast<int8_t*>(pr), static_cast<int8_t*>(pi), static_cast<int8_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxUINT8_CLASS:
            memcpy_interleaved_to_split(static_cast<uint8_t*>(pr), static_cast<uint8_t*>(pi), static_cast<uint8_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxINT16_CLASS:
            memcpy_interleaved_to_split(static_cast<int16_t*>(pr), static_cast<int16_t*>(pi), static_cast<int16_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxUINT16_CLASS:
            memcpy_interleaved_to_split(static_cast<uint16_t*>(pr), static_cast<uint16_t*>(pi), static_cast<uint16_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxINT32_CLASS:
            memcpy_interleaved_to_split(static_cast<int32_t*>(pr), static_cast<int32_t*>(pi), static_cast<int32_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        case mxUINT32_CLASS:
            memcpy_interleaved_to_split(static_cast<uint32_t*>(pr), static_cast<uint32_t*>(pi), static_cast<uint32_t*>(pvhost), mxGetNumberOfElements(mxA));
            break;
        }
        cudaFreeHost(pvhost);
        cuphyDestroyTensorDescriptor(hostCopyDesc);
        return true;
    }
    return false;
}
    
} // namespace

////////////////////////////////////////////////////////////////////////
// cuphy_mex_create()
void cuphy_mex_create(int            nlhs,   /* number of expected outputs */
                      mxArray*       plhs[], /* array of pointers to output arguments */
                      int            nrhs,   /* number of inputs */
                      const mxArray* prhs[]) /* array of pointers to input arguments */
{
    if(nlhs != 1)
    {
        mexErrMsgTxt("cuphy MEX create: One output expected.");
        return;
    }
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double* c = nullptr;
    c = mxGetPr(plhs[0]);
    // For now, just return an incrementing number
    static double s_create_count = 1.0;
    *c = s_create_count;
    s_create_count += 1.0f;
}

////////////////////////////////////////////////////////////////////////
// cuphy_mex_delete()
void cuphy_mex_delete(int            nlhs,   /* number of expected outputs */
                      mxArray*       plhs[], /* array of pointers to output arguments */
                      int            nrhs,   /* number of inputs */
                      const mxArray* prhs[]) /* array of pointers to input arguments */
{
    // No-op for now
}
        
////////////////////////////////////////////////////////////////////////
// cuphy_mex_channel_est_MMSE_1D()
void cuphy_mex_channel_est_MMSE_1D(int            nlhs,   /* number of expected outputs */
                                   mxArray*       plhs[], /* array of pointers to output arguments */
                                   int            nrhs,   /* number of inputs */
                                   const mxArray* prhs[]) /* array of pointers to input arguments */
{
    //------------------------------------------------------------------
    // Validate inputs
    if(nlhs != 1)
    {
        mexErrMsgTxt("cuphy MEX channelEstMMSE1D: One output expected.");
        return;
    }
    if(nrhs < 5)
    {
        mexErrMsgTxt("cuphy MEX channelEstMMSE1D: At least 5 inputs expected.");
        return;
    }
    //for(int i = 0; i < nrhs; ++i)
    //{
    //    printf("i: %i, type: %s, num_dim: %i\n", i, mxGetClassName(prhs[i]), mxGetNumberOfDimensions(prhs[i]));
    //}
    const mxArray* mxY               = prhs[0];
    const mxArray* mxWfreq           = prhs[1];
    const mxArray* mxWtime           = prhs[2];
    const mxArray* mxDMRS_index_freq = prhs[3];
    const mxArray* mxDMRS_index_time = prhs[4];
    if((3 != mxGetNumberOfDimensions(mxY))               ||
       (3 != mxGetNumberOfDimensions(mxWfreq))           ||
       (3 != mxGetNumberOfDimensions(mxWtime))           ||
       (2 != mxGetNumberOfDimensions(mxDMRS_index_freq)) ||
       (2 != mxGetNumberOfDimensions(mxDMRS_index_time)) ||
       (!mxIsComplex(mxY))                               ||
       (mxIsComplex(mxWfreq))                            ||
       (mxIsComplex(mxWtime))                            ||
       (mxIsComplex(mxDMRS_index_freq))                  ||
       (mxIsComplex(mxDMRS_index_time)))
    {
        mexErrMsgTxt("cuphy MEX channelEstMMSE1D: Invalid input types.");
        return;
    }
    //------------------------------------------------------------------
    // For now, only handle specific sizes
    if(dimensions_are_not(mxY,               {1248, 14, 16 } )   ||
       dimensions_are_not(mxWfreq,           {96,   32, 156} )   ||
       dimensions_are_not(mxWtime,           {14,   4,  156} )   ||
       dimensions_are_not(mxDMRS_index_freq, {32,   156    } )   ||
       dimensions_are_not(mxDMRS_index_time, {4,    156    } ))
    {
        mexErrMsgTxt("cuphy MEX channelEstMMSE1D: Unsupported sizes.");
    }
    typedef std::pair<cuphyTensorDescriptor_t, void*> tensor_pair_t;
    typedef std::map<const char*, tensor_pair_t>      tensor_map_t;
    tensor_map_t                                      tensors;
    tensors.insert(tensor_map_t::value_type("Hinterp",         allocate_device_tensor({96,14,16,156}, CUPHY_C_32F, CUPHY_TENSOR_ALIGN_COALESCE)));
    tensors.insert(tensor_map_t::value_type("Wfreq",           allocate_device_tensor(mxWfreq,                     CUPHY_TENSOR_ALIGN_COALESCE)));
    tensors.insert(tensor_map_t::value_type("Wtime",           allocate_device_tensor(mxWtime,                     CUPHY_TENSOR_ALIGN_COALESCE)));
    tensors.insert(tensor_map_t::value_type("DMRS_index_freq", allocate_device_tensor(mxDMRS_index_freq,           CUPHY_TENSOR_ALIGN_TIGHT)));
    tensors.insert(tensor_map_t::value_type("DMRS_index_time", allocate_device_tensor(mxDMRS_index_time,           CUPHY_TENSOR_ALIGN_TIGHT)));
    tensors.insert(tensor_map_t::value_type("Y",               allocate_complex_device_tensor(mxY,                 CUPHY_TENSOR_ALIGN_COALESCE)));

    int failCount = 0;
    for(auto it = tensors.begin(); it != tensors.end(); ++it)
    {
        tensor_pair_t& p = it->second;
        if(!p.first || !p.second)
        {
            mexErrMsgTxt("cuphy MEX channelEstMMSE1D: Tensor initialization failure");
            ++failCount;
        }
    }
    cuphyStatus_t s = cuphyChannelEst1DTimeFrequency(tensors["Hinterp"].first,
                                                     tensors["Hinterp"].second,
                                                     tensors["Y"].first,
                                                     tensors["Y"].second,
                                                     tensors["Wfreq"].first,
                                                     tensors["Wfreq"].second,
                                                     tensors["Wtime"].first,
                                                     tensors["Wtime"].second,
                                                     tensors["DMRS_index_freq"].first,
                                                     tensors["DMRS_index_freq"].second,
                                                     tensors["DMRS_index_time"].first,
                                                     tensors["DMRS_index_time"].second,
                                                     0); // stream
    if(CUPHY_STATUS_SUCCESS != s)
    {
        std::string msg("cuphy MEX channelEstMMSE1D: Error performing channel estimation (");
        msg.append(cuphyGetErrorString(s));
        msg.append(")");
        mexErrMsgTxt(msg.c_str());
    }
    else
    {
        // Copy result tensor to mxArray
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Hard coded sizes for now...
        std::vector<int> Hdims{96,14,16,156};
        plhs[0] = mxCreateNumericArray(Hdims.size(), Hdims.data(), mxSINGLE_CLASS, mxCOMPLEX);
        if(!device_tensor_to_complex_mxarray(plhs[0], tensors["Hinterp"]))
        {
            mexErrMsgTxt("cuphy MEX channelEstMMSE1D: Error copying data to mxArray.");
        }
        //double* c = nullptr;
        //c = mxGetPr(plhs[0]);
        //*c = 0.0;
    }

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Clean up tensors
    for(auto it = tensors.begin(); it != tensors.end(); ++it)
    {
        tensor_pair_t& p = it->second;
        if(p.first)
        {
            cuphyDestroyTensorDescriptor(p.first);
        }
        if(p.second)
        {
            cudaFree(p.second);
        }
    }
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //printf("SUCCESS\n");
}

////////////////////////////////////////////////////////////////////////
// mexFunction()
// MEX library "gateway" function
void mexFunction(int            nlhs,   /* number of expected outputs */
                 mxArray*       plhs[], /* array of pointers to output arguments */
                 int            nrhs,   /* number of inputs */
                 const mxArray* prhs[]) /* array of pointers to input arguments */
{
    //printf("CUPHY MEX mexFunction: nrhs = %i, nlhs = %i\n", nrhs, nlhs);
    //int        i;
    ///* Examine input (right-hand-side) arguments. */
    //mexPrintf("\nThere are %d right-hand-side argument(s).", nrhs);
    //for(int i = 0; i < nrhs; ++i)
    //{
    //    mexPrintf("\n\tInput Arg %i is of type:\t%s ", i, mxGetClassName(prhs[i]));
    //}
    //
    /* Examine output (left-hand-side) arguments. */
    //mexPrintf("\n\nThere are %d left-hand-side argument(s).\n", nlhs);
    //if (nlhs > nrhs)
    //  mexErrMsgIdAndTxt( "MATLAB:mexfunction:inputOutputMismatch",
    //          "Cannot specify more outputs than inputs.\n");
    //
    //for (i=0; i<nlhs; i++)  {
    //    plhs[i]=mxCreateDoubleMatrix(1,1,mxREAL);
    //    *mxGetPr(plhs[i])=(double)mxGetNumberOfElements(prhs[i]);
    //}
    //------------------------------------------------------------------
    // Validate inputs
    if((nrhs < 1) || (mxCHAR_CLASS != mxGetClassID(prhs[0])))
    {
        mexErrMsgTxt("First input should be a command string.");
        return;
    }
    //------------------------------------------------------------------
    // Get the command string
    size_t arg1N = mxGetN(prhs[0]);
    std::vector<char> commandString(arg1N + 1);
    if(0 != mxGetString(prhs[0], commandString.data(), arg1N + 1))
    {
        std::string msg("Error retrieving command string (length ");
        msg.append(std::to_string(arg1N));
        msg.append(")"); 
        mexErrMsgTxt(msg.c_str());
        return;
    }
    //------------------------------------------------------------------
    // Discard the string function name and the internal handle passed
    // as arguments
    int nrhs_fn             = nrhs - 2;
    const mxArray** prhs_fn = prhs + 2;
    if(0 == strcmp(commandString.data(), "create"))
    {
        cuphy_mex_create(nlhs, plhs, nrhs_fn, prhs_fn);
    }
    else if(0 == strcmp(commandString.data(), "delete"))
    {
        cuphy_mex_delete(nlhs, plhs, nrhs_fn, prhs_fn);
    }
    else if(0 == strcmp(commandString.data(), "channelEstMMSE1D"))
    {
        cuphy_mex_channel_est_MMSE_1D(nlhs, plhs, nrhs_fn, prhs_fn);
    }
    else
    {
        mexErrMsgTxt("Unknown command string.");
    }
    
}
