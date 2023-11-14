/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define CUPHY_DEBUG 1

#include <assert.h>
#include "ldpc2_split_index_fp_x2_desc_dyn_sm90.hpp"
#include "ldpc2.hpp"
#include "ldpc2_desc.cuh"
#include "ldpc2_bg_desc.hpp"
#include "ldpc2_c2v_x2.cuh"


// Note: the cubin was generated with this defined, so changing it here
// (without also changing the cubin) may cause the kernel to fail.
#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

extern const uint8_t  ldpc2_split_index_fp_x2_desc_dyn_sm90_internal[];
extern unsigned int   ldpc2_split_index_fp_x2_desc_dyn_sm90_internal_size;

namespace
{

const char* sm90_BG1_kernel_name    = "ldpc2_BG1_split_index_fp_x2_desc_dyn_sm90";
const char* sm90_BG2_kernel_name    = "ldpc2_BG2_split_index_fp_x2_desc_dyn_sm90";
const char* sm90_BG1_tb_kernel_name = "ldpc2_BG1_split_index_fp_x2_desc_dyn_sm90_tb";
const char* sm90_BG2_tb_kernel_name = "ldpc2_BG2_split_index_fp_x2_desc_dyn_sm90_tb";

////////////////////////////////////////////////////////////////////////
// Storing compressed check to variable (cC2V) data in registers may
// not be possible for all code rates. Furthermore, squeezing a
// larger number of parity node data into registers may actually
// decrease performance at high code rates.
// Note: These should not be changed without also changing the compiled
// cubin binaries.
template <int BG> struct num_reg_parity;
template <> struct num_reg_parity<1> { static constexpr int value = 17; };
template <> struct num_reg_parity<2> { static constexpr int value = 42; };
template <int BG> struct max_num_parity;
template <> struct max_num_parity<1> { static constexpr int value = 23; };
template <> struct max_num_parity<2> { static constexpr int value = 42; };

////////////////////////////////////////////////////////////////////////
// APP address calculation
// Compiled binaries use the "adjusted" base graph descriptor
// structures, BG_adj_desc<BG>.

////////////////////////////////////////////////////////////////////////
// get_app_c2v_shmem()
// Returns the number of bytes required for APP and C2V memory for
// this kernel.
template <int BG>
int get_app_c2v_shmem(int num_parity_nodes, int Z)
{
    const     int32_t NUM_VAR_NODES = ldpc2::max_info_nodes<BG>::value + num_parity_nodes;
    constexpr int32_t NUM_REG_NODES = num_reg_parity<BG>::value;
    const     int32_t APP_SIZE      = static_cast<int32_t>(shmem_llr_buffer_size(NUM_VAR_NODES,     // num shared memory nodes
                                                                                 Z,                 // lifting size
                                                                                 sizeof(__half2))); // element size
    // The first 'NUM_REG_NODES' of C2V data will reside in registers.
    // The remainder will be in shared memory.
    const int32_t C2V_SIZE = (num_parity_nodes > NUM_REG_NODES)                                          ?
                             (num_parity_nodes - NUM_REG_NODES) * Z * sizeof(cC2V_storage_x2_low_degree) :
                             0;
    // We need to pad the APP portion of shared memory to make sure
    // that C2V storage is aligned with the C2V type.
    int shmem_size = round_up_to_next(APP_SIZE, static_cast<int>(alignof(cC2V_storage_x2_low_degree))) +
                                      C2V_SIZE;
    return shmem_size;
}

////////////////////////////////////////////////////////////////////////
// get_shmem_required()
// Calculates the sum of the APP and C2V data storage.
int get_shmem_required(int BG,
                       int num_parity_nodes,
                       int Z)
{
    int shmem_size = (1 == BG) ? get_app_c2v_shmem<1>(num_parity_nodes, Z)
                               : get_app_c2v_shmem<2>(num_parity_nodes, Z);
#if LDPC_DECODE_USE_TB_SCAN
        // When using a scan algorithm to determine the codeword for a CTA,
        // extra shared memory for the token is required.
        shmem_size = round_up_to_next(shmem_size, static_cast<int>(alignof(tb_token))) +
                     sizeof(tb_token);
#endif
        return shmem_size;
}

////////////////////////////////////////////////////////////////////////
// launch_kernel_driver_api()
// Kernel launch wrapper function for the LDPC decoder kernel loaded
// from a .cubin file
template <int                  BG,
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
    const uint32_t SHMEM_SIZE = get_shmem_required(BG,                      // base graph
                                                   params.num_parity_nodes, // num parity nodes
                                                   params.Z);               // lifting size


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
template <int                  BG,
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
    const uint32_t SHMEM_SIZE = get_shmem_required(BG,                                 // base graph
                                                   decodeDesc.config.num_parity_nodes, // num shared memory nodes
                                                   decodeDesc.config.Z);               // lifting size

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
// split_index_fp_x2_desc_dyn_sm90::split_index_fp_x2_desc_dyn_sm90()
split_index_fp_x2_desc_dyn_sm90::split_index_fp_x2_desc_dyn_sm90(ldpc::decoder& dec)
{
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_SIZE = static_cast<int>(get_shmem_required(1,                             // BG
                                                                       max_num_parity<1>::value,      // max parity nodes
                                                                       CUPHY_LDPC_MAX_LIFTING_SIZE)); // lifting size
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(get_shmem_required(2,                             // BG
                                                                       max_num_parity<2>::value,      // max parity nodes
                                                                       CUPHY_LDPC_MAX_LIFTING_SIZE)); // lifting size
    //------------------------------------------------------------------
    // Maximum shared memory supported by the device
    const int MAX_SHMEM = dec.max_shmem_per_block_optin();
    //------------------------------------------------------------------
    // Load cubin binaries
    sm90_module_.load(ldpc2_split_index_fp_x2_desc_dyn_sm90_internal);
    sm90_BG1_kernel_    = sm90_module_.get_function(sm90_BG1_kernel_name/*,    std::nothrow_t()*/);
    sm90_BG2_kernel_    = sm90_module_.get_function(sm90_BG2_kernel_name/*,    std::nothrow_t()*/);
    sm90_BG1_tb_kernel_ = sm90_module_.get_function(sm90_BG1_tb_kernel_name/*, std::nothrow_t()*/);
    sm90_BG2_tb_kernel_ = sm90_module_.get_function(sm90_BG2_tb_kernel_name/*, std::nothrow_t()*/);
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<CUfunction, int> func_attr_t;
    std::array<func_attr_t, 4> func_attrs =
    {
        func_attr_t(sm90_BG1_kernel_,    std::min(MAX_BG1_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t(sm90_BG2_kernel_,    std::min(MAX_BG2_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t(sm90_BG1_tb_kernel_, std::min(MAX_BG1_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t(sm90_BG2_tb_kernel_, std::min(MAX_BG2_SHMEM_SIZE, MAX_SHMEM))
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

    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_split_index_fp_x2_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_split_index_fp_x2_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_split_index_fp_x2_desc_dyn_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_split_index_fp_x2_desc_dyn_tb);
}

////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn::get_workspace_size()
std::pair<bool, size_t> split_index_fp_x2_desc_dyn_sm90::get_workspace_size(const ldpc::decoder&               dec,
                                                                            const cuphyLDPCDecodeConfigDesc_t& config,
                                                                            int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn_sm90::can_decode_config()
bool split_index_fp_x2_desc_dyn_sm90::can_decode_config(const ldpc::decoder&               dec,
                                                        const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // Compare shared memory requirements to device maximum, as well as
    // the maximum that the kernel was compiled for.
    
    // Maximum number of parity nodes, as limited by compilation, to
    // limit register usage.
    const uint32_t MAX_NUM_PARITY = (1 == cfg.BG) ? max_num_parity<1>::value : max_num_parity<2>::value;
    // Calculate required shared memory
    const uint32_t SHMEM_BYTES    = get_shmem_required(cfg.BG,
                                                       cfg.num_parity_nodes,
                                                       cfg.Z);
    return (cfg.num_parity_nodes <= MAX_NUM_PARITY) &&
           (SHMEM_BYTES <= dec.max_shmem_per_block_optin());
}
////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn_sm90::decode()
cuphyStatus_t split_index_fp_x2_desc_dyn_sm90::decode(ldpc::decoder&                     dec,
                                                      LDPC_output_t&                     tDst,
                                                      const_tensor_pair&                 tLLR,
                                                      const cuphyLDPCDecodeConfigDesc_t& config,
                                                      cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_split_index_fp_x2_desc_dyn_sm90()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    dim3 grdDim(div_round_up(NUM_CW, 2));
    // We need to be mindful of the blockDim not being a multiple of 32.
    // The hard decision output writes 32-bit words. We may need to
    // revisit the output function to allow us to truncate the threads
    // that write to the next lowest multiple of 32, but that  may also
    // mean that we need to then have the output function LOOP.
    //dim3 blkDim(((config.Z + 31) / 32) * 32);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = get_shmem_required(config.BG,
                                                   config.num_parity_nodes,
                                                   config.Z);
    
    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                const int         BG         = 1;
                CUfunction        f          = sm90_BG1_kernel_;
                const BG_adj_desc<1>* bgdesc = get_adj_BG_desc<__half2, 1>(params.Z);
                if(!bgdesc) break;

                s = launch_kernel_driver_api<BG>(f, params, *bgdesc, grdDim, blkDim, strm);
            }
            break;
        case 2:
            {
                const int             BG     = 2;
                CUfunction            f      = sm90_BG2_kernel_;
                const BG_adj_desc<2>* bgdesc = get_adj_BG_desc<__half2, 2>(params.Z);
                if(!bgdesc) break;

                s = launch_kernel_driver_api<BG>(f, params, *bgdesc, grdDim, blkDim, strm);
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
// split_index_fp_x2_desc_dyn_sm90::decode_tb()
cuphyStatus_t split_index_fp_x2_desc_dyn_sm90::decode_tb(ldpc::decoder&               dec,
                                                         const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                         cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::split_index_fp_x2_desc_dyn_sm90::decode_tb()\n");
    
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        // We need to be mindful of the blockDim not being a multiple of 32.
        // The hard decision output writes 32-bit words. We may need to
        // revisit the output function to allow us to truncate the threads
        // that write to the next lowest multiple of 32, but that  may also
        // mean that we need to then have the output function LOOP.
        //dim3 blkDim(((config.Z + 31) / 32) * 32);
        dim3 blkDim(decodeDesc.config.Z);
        
        //------------------------------------------------------------------
        // Launch a CTA for each codeword pair. Note that the number of CTAs
        // may be more than the total number of codewords divided by 2 -
        // there may be transport blocks with odd numbers of codewords.
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                const int         BG         = 1;
                CUfunction        f          = sm90_BG1_tb_kernel_;
                const BG_adj_desc<1>* bgdesc = get_adj_BG_desc<__half2, 1>(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                s = launch_tb_kernel_driver_api<BG>(f, decodeDesc, *bgdesc, grdDim, blkDim, strm);
            }
            break;
        case 2:
            {
                const int         BG         = 2;
                CUfunction        f          = sm90_BG2_tb_kernel_;
                const BG_adj_desc<2>* bgdesc = get_adj_BG_desc<__half2, 2>(decodeDesc.config.Z);
                if(!bgdesc) break;

                s = launch_tb_kernel_driver_api<BG>(f, decodeDesc, *bgdesc, grdDim, blkDim, strm);
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
// split_index_fp_x2_desc_dyn_sm90::get_launch_config()
cuphyStatus_t split_index_fp_x2_desc_dyn_sm90::get_launch_config(const ldpc::decoder&           dec,
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

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codeword_pairs(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;

    const uint32_t SHMEM_SIZE = get_shmem_required(launchConfig.decode_desc.config.BG,
                                                   launchConfig.decode_desc.config.num_parity_nodes,
                                                   launchConfig.decode_desc.config.Z);
    launchConfig.kernel_node_params_driver.sharedMemBytes = SHMEM_SIZE;
    // Only tb interface supported by get_launch_config()
    launchConfig.kernel_node_params_driver.func = (1 == BG)           ?
                                                  sm90_BG1_tb_kernel_ :
                                                  sm90_BG2_tb_kernel_;

    //------------------------------------------------------------------
    // Set kernel arguments:
    // arg 0: decode descriptor
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    // arg 1: base graph descriptor
    if(1 == BG)
    {
        const BG_adj_desc<1>* bgdesc = get_adj_BG_desc<__half2, 1>(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    else
    {
        const BG_adj_desc<2>* bgdesc = get_adj_BG_desc<__half2, 2>(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
