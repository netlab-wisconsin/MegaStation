/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define CUPHY_DEBUG 1

#include <assert.h>
#include "ldpc2.hpp"
#include "ldpc2_desc.cuh"
#include "ldpc2_reg_index_fp_desc_dyn_row_dep_sm80.hpp"
#include "ldpc2_bg_desc.hpp"

using namespace ldpc2;

extern const uint8_t  ldpc2_reg_index_fp_desc_dyn_row_dep_sm80_internal[];
extern unsigned int   ldpc2_reg_index_fp_desc_dyn_row_dep_sm80_internal_size;

namespace
{

const char* sm80_dyn_desc_row_dep_BG1_kernel_name    = "ldpc2_BG1_reg_index_fp_desc_dyn_row_dep_sm80";
const char* sm80_dyn_desc_row_dep_BG2_kernel_name    = "ldpc2_BG2_reg_index_fp_desc_dyn_row_dep_sm80";
const char* sm80_dyn_desc_row_dep_BG1_tb_kernel_name = "ldpc2_BG1_reg_index_fp_desc_dyn_row_dep_sm80_tb";
const char* sm80_dyn_desc_row_dep_BG2_tb_kernel_name = "ldpc2_BG2_reg_index_fp_desc_dyn_row_dep_sm80_tb";

////////////////////////////////////////////////////////////////////////
// launch_kernel_driver_api()
// Kernel launch wrapper function for the LDPC decoder kernel loaded
// from a .cubin file
template <typename             T,
          int                  BG,
          template <int> class BGDesc>
cuphyStatus_t launch_kernel_driver_api(CUfunction                f,
                                       const LDPC_kernel_params& params,
                                       const BGDesc<BG>&         bg_desc,
                                       const dim3&               grdDim,
                                       const dim3&               blkDim,
                                       cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                      params.Z,             // lifting size
                                                      sizeof(T));           // element size

    void* args[2] = {const_cast<void*>(static_cast<const void*>(&params)),
                     const_cast<void*>(static_cast<const void*>(&bg_desc))};
    CUresult e = cuLaunchKernel(f,
                                grdDim.x,
                                grdDim.y,
                                grdDim.z,
                                blkDim.x,
                                blkDim.y,
                                blkDim.z,
                                SHMEM_SIZE,
                                strm,
                                args,
                                nullptr);
    return (CUDA_SUCCESS == e) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// launch_tb_kernel_driver_api()
// Kernel launch wrapper function for the LDPC decoder kernel loaded
// from a .cubin file
template <typename             T,
          int                  BG,
          template <int> class BGDesc>
cuphyStatus_t launch_tb_kernel_driver_api(CUfunction                   f,
                                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                                          const BGDesc<BG>&            bg_desc,
                                          const dim3&                  grdDim,
                                          const dim3&                  blkDim,
                                          cudaStream_t                 strm)
{
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<BG>::value, // num shared memory nodes
                                                      decodeDesc.config.Z,                                            // lifting size
                                                      sizeof(T));                                                     // element size

    void* args[2] = {const_cast<void*>(static_cast<const void*>(&decodeDesc)),
                     const_cast<void*>(static_cast<const void*>(&bg_desc))};
    CUresult e = cuLaunchKernel(f,
                                grdDim.x,
                                grdDim.y,
                                grdDim.z,
                                blkDim.x,
                                blkDim.y,
                                blkDim.z,
                                SHMEM_SIZE,
                                strm,
                                args,
                                nullptr);
    return (CUDA_SUCCESS == e) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace


namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_row_dep_sm80::decode()
cuphyStatus_t reg_index_fp_desc_dyn_row_dep_sm80::decode(ldpc::decoder&                     dec,
                                                         LDPC_output_t&                     tDst,
                                                         const_tensor_pair&                 tLLR,
                                                         const cuphyLDPCDecodeConfigDesc_t& config,
                                                         cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::reg_index_fp_desc_dyn_row_dep_sm80::decode()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    dim3 grdDim(NUM_CW);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                const int         BG     = 1;
                CUfunction        f      = sm80_dyn_desc_row_dep_BG1_kernel_;
                const BG1_desc_t* bgdesc = get_BG_desc<__half, BG>(config.Z);
                if(!bgdesc) break;
                
                s = launch_kernel_driver_api<__half, BG, BG_desc>(f, params, *bgdesc, grdDim, blkDim, strm);
            }
            break;
        case 2:
            {
                const int         BG     = 2;
                CUfunction        f      = sm80_dyn_desc_row_dep_BG2_kernel_;
                const BG2_desc_t* bgdesc = get_BG_desc<__half, BG>(config.Z);
                if(!bgdesc) break;

                s = launch_kernel_driver_api<__half, BG, BG_desc>(f, params, *bgdesc, grdDim, blkDim, strm);
            }
            break;
        default:
            break;
        }
    }

    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_row_dep_sm80::decode_tb()
cuphyStatus_t reg_index_fp_desc_dyn_row_dep_sm80::decode_tb(ldpc::decoder&               dec,
                                                            const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::reg_index_fp_desc_dyn_row_dep_sm80::decode_tb()\n");
    //------------------------------------------------------------------
    dim3            grdDim(ldpc::decoder::get_total_num_codewords(decodeDesc));
    dim3            blkDim(decodeDesc.config.Z);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                const int         BG     = 1;
                CUfunction        f      = sm80_dyn_desc_row_dep_BG1_tb_kernel_;
                const BG1_desc_t* bgdesc = get_BG_desc<__half, BG>(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                s = launch_tb_kernel_driver_api<__half, BG, BG_desc>(f, decodeDesc, *bgdesc, grdDim, blkDim, strm);
            }
            break;
        case 2:
            {
                const int         BG     = 2;
                CUfunction        f      = sm80_dyn_desc_row_dep_BG2_tb_kernel_;
                const BG2_desc_t* bgdesc = get_BG_desc<__half, BG>(decodeDesc.config.Z);
                if(!bgdesc) break;

                s = launch_tb_kernel_driver_api<__half, BG, BG_desc>(f, decodeDesc, *bgdesc, grdDim, blkDim, strm);
            }
            break;
        default:
            break;
        }
    }
    return s;
}
////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_row_dep_sm80::get_workspace_size()
std::pair<bool, size_t> reg_index_fp_desc_dyn_row_dep_sm80::get_workspace_size(const ldpc::decoder&               dec,
                                                                               const cuphyLDPCDecodeConfigDesc_t& config,
                                                                               int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_row_dep_sm80::reg_index_fp_desc_dyn_sm80()
reg_index_fp_desc_dyn_row_dep_sm80::reg_index_fp_desc_dyn_row_dep_sm80(ldpc::decoder& dec)
{
    assert(dec.compute_cap() >= (8ULL << 32));
    //------------------------------------------------------------------
    // Load SM80 binaries for use with the CUDA driver API.
    //
    // SM8.0 support requires CUDA 11. Only attempt to load the
    // .cubin if the current device is SM80 or greater...
    int         cudaRuntimeVersion = 0;
    cudaError_t s = cudaRuntimeGetVersion(&cudaRuntimeVersion);
    if(cudaSuccess != s)
    {
        throw cuphy_i::cuda_exception(s);
    }
    if(cudaRuntimeVersion >= 11000)
    {
        sm80_dyn_desc_row_dep_module_.load(ldpc2_reg_index_fp_desc_dyn_row_dep_sm80_internal);
        sm80_dyn_desc_row_dep_BG1_kernel_    = sm80_dyn_desc_row_dep_module_.get_function(sm80_dyn_desc_row_dep_BG1_kernel_name/*,    std::nothrow_t()*/);
        sm80_dyn_desc_row_dep_BG2_kernel_    = sm80_dyn_desc_row_dep_module_.get_function(sm80_dyn_desc_row_dep_BG2_kernel_name/*,    std::nothrow_t()*/);
        sm80_dyn_desc_row_dep_BG1_tb_kernel_ = sm80_dyn_desc_row_dep_module_.get_function(sm80_dyn_desc_row_dep_BG1_tb_kernel_name/*, std::nothrow_t()*/);
        sm80_dyn_desc_row_dep_BG2_tb_kernel_ = sm80_dyn_desc_row_dep_module_.get_function(sm80_dyn_desc_row_dep_BG2_tb_kernel_name/*, std::nothrow_t()*/);
        //printf("sm80_dyn_desc_row_dep_BG1_kernel_    = %p\n", sm80_dyn_desc_row_dep_BG1_kernel_);
        //printf("sm80_dyn_desc_row_dep_BG2_kernel_    = %p\n", sm80_dyn_desc_row_dep_BG2_kernel_);
        //printf("sm80_dyn_desc_row_dep_BG1_tb_kernel_ = %p\n", sm80_dyn_desc_row_dep_BG1_tb_kernel_);
        //printf("sm80_dyn_desc_row_dep_BG2_tb_kernel_ = %p\n", sm80_dyn_desc_row_dep_BG2_tb_kernel_);
        const uint32_t MAX_VAR_NODES_BG1 = ldpc2::max_variable_nodes<1>::value;
        const uint32_t MAX_VAR_NODES_BG2 = ldpc2::max_variable_nodes<2>::value;
        //-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the maximum amount of shared memory that could be used
        // by a kernel
        const int MAX_BG1_SHMEM_SIZE = static_cast<int>(shmem_llr_buffer_size(MAX_VAR_NODES_BG1,           // num shared memory nodes
                                                                              CUPHY_LDPC_MAX_LIFTING_SIZE, // lifting size
                                                                              sizeof(__half)));            // element size
        const int MAX_BG2_SHMEM_SIZE = static_cast<int>(shmem_llr_buffer_size(MAX_VAR_NODES_BG2,           // num shared memory nodes
                                                                              CUPHY_LDPC_MAX_LIFTING_SIZE, // lifting size
                                                                              sizeof(__half)));            // element size
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        // For each kernel, set the maximum dynamic shared memory size
        typedef std::pair<CUfunction, int> func_attr_t;
        std::array<func_attr_t, 4> func_attrs =
        {
            func_attr_t(sm80_dyn_desc_row_dep_BG1_kernel_,    MAX_BG1_SHMEM_SIZE),
            func_attr_t(sm80_dyn_desc_row_dep_BG2_kernel_,    MAX_BG2_SHMEM_SIZE),
            func_attr_t(sm80_dyn_desc_row_dep_BG1_tb_kernel_, MAX_BG1_SHMEM_SIZE),
            func_attr_t(sm80_dyn_desc_row_dep_BG2_tb_kernel_, MAX_BG2_SHMEM_SIZE)
        };
        for(func_attr_t f_a : func_attrs)
        {
            CUresult e = cuFuncSetAttribute(f_a.first,
                                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                            f_a.second);
            if(CUDA_SUCCESS != e)
            {
                throw cuphy_i::cu_exception(e);
            }
        }
    }
    else
    {
        throw std::runtime_error("Invalid CUDA version for SM80 kernel");
    }
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_row_dep_sm80::get_launch_config()
cuphyStatus_t reg_index_fp_desc_dyn_row_dep_sm80::get_launch_config(const ldpc::decoder&           dec,
                                                                    cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)                  ?
                                 max_parity_nodes<1>::value :
                                 max_parity_nodes<2>::value;
    const int NUM_VAR_NODES    = ldpc::decoder::get_num_variable_nodes(BG,
                                                                       NUM_PARITY_NODES);
    //------------------------------------------------------------------
    // Validate input arguments
    if((Z < 2)                              ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)    ||
       (NUM_PARITY_NODES < 4)               ||
       (NUM_PARITY_NODES > MAX_PARITY_NODES))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    // Set up launch geometry and the kernel function (driver)
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codewords(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;
    launchConfig.kernel_node_params_driver.sharedMemBytes = shmem_llr_buffer_size(NUM_VAR_NODES,   // num shared memory nodes
                                                                                  Z,               // lifting size
                                                                                  sizeof(__half)); // element size
    launchConfig.kernel_node_params_driver.func = (1 == BG)                            ?
                                                  sm80_dyn_desc_row_dep_BG1_tb_kernel_ :
                                                  sm80_dyn_desc_row_dep_BG2_tb_kernel_;
    //------------------------------------------------------------------
    // Set kernel arguments:
    // arg 0: decode descriptor
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    // arg 1: base graph descriptor
    if(1 == BG)
    {
        const BG1_desc_t* bgdesc = get_BG_desc<__half, 1>(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    else
    {
        const BG2_desc_t* bgdesc = get_BG_desc<__half, 2>(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
