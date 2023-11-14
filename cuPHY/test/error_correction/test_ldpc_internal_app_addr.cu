/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


// Decrease N_TEST_VALUES before enabling this... 
//#define CUPHY_DEBUG 1

#include <gtest/gtest.h>
#include <random>
#include <limits>
#include <type_traits>
#include "ldpc2.hpp"
#include "ldpc2.cuh"
#include "ldpc2_c2v_x2.cuh"
#include "cuphy.hpp"
#include "ldpc2_app_address.cuh"
#include "ldpc2_app_address_fp.cuh"
#include "ldpc2_app_address_fp_desc.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"

using namespace ldpc2;

template <int BG, int CHECK_IDX, class TAddr0, class TAddr1>
__device__
void address_pairs_compare_node(unsigned int*             dErrorCount,
                                TAddr0&                   addr0,
                                TAddr1&                   addr1)
{
    int app_addr_0[row_degree<BG, CHECK_IDX>::value];    // shared memory (byte) addresses
    int app_addr_1[row_degree<BG, CHECK_IDX>::value]; // shared memory (byte) addresses

    //if(0 == threadIdx.x)
    //{
    //    printf("address_pairs_compare_node(BG = %i, CHECK = %i)\n", BG, CHECK_IDX);
    //}
    
    addr0.generate<CHECK_IDX>(app_addr_0);
    addr1.generate<CHECK_IDX>(app_addr_1);

    for(int i = 0; i < row_degree<BG, CHECK_IDX>::value; ++i)
    {
        if(app_addr_0[i] != app_addr_1[i])
        {
            atomicAdd(dErrorCount, 1);
            printf("addr[%i, %u, %i] error: %i != %i (row_degree = %i)\n", CHECK_IDX, threadIdx.x, i, app_addr_0[i], app_addr_1[i], row_degree<BG, CHECK_IDX>::value);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// address_pairs_compare_node_unroll
template <int BG, int NODE_COUNT, class TAddr0, class TAddr1> struct address_pairs_compare_node_unroll;
template <int BG, class TAddr0, class TAddr1>
struct address_pairs_compare_node_unroll<BG, 1, TAddr0, TAddr1>
{
    __device__
    static void compare(unsigned int*             dErrorCount,
                        TAddr0&                   addr0,
                        TAddr1&                   addr1)
    {
        address_pairs_compare_node<BG, 0, TAddr0, TAddr1>(dErrorCount, addr0, addr1);
    }
};

template <int BG, int NODE_COUNT, class TAddr0, class TAddr1>
struct address_pairs_compare_node_unroll
{
    __device__
    static void compare(unsigned int*             dErrorCount,
                        TAddr0&                   addr0,
                        TAddr1&                   addr1)
    {
        address_pairs_compare_node_unroll<BG, NODE_COUNT-1, TAddr0, TAddr1>::compare(dErrorCount, addr0, addr1);
        address_pairs_compare_node<BG, NODE_COUNT-1, TAddr0, TAddr1>(dErrorCount, addr0, addr1);
    }
};

////////////////////////////////////////////////////////////////////////
// address_pairs_compare()
template <typename                           T,
          int                                BG,
          int                                Z,
          template<typename, int, int> class TAddr0,
          template<typename, int, int> class TAddr1,
          int                                MAX_NUM_PARITY>
__global__ __launch_bounds__(Z, 1)
void address_pairs_compare(unsigned int* dErrorCount)
{
    typedef TAddr0<T, BG, Z> addr0_t;
    typedef TAddr1<T, BG, Z> addr1_t;
    
    typedef address_pairs_compare_node_unroll<BG, MAX_NUM_PARITY, addr0_t, addr1_t> compare_unroll_t;
    
    addr0_t          addr0;
    addr1_t          addr1;
    compare_unroll_t comp;
    comp.compare(dErrorCount, addr0, addr1);
}

////////////////////////////////////////////////////////////////////////
// address_pairs_compare_desc()
// Compares address calculations, with the second generator using a
// base graph descriptor structure.
template <typename                           T,
          int                                BG,
          int                                Z,
          template<typename, int, int> class TAddr0,
          template<typename, int>      class TAddr1_Desc,
          template<int>                class BGDesc,
          int                                MAX_NUM_PARITY>
__global__ __launch_bounds__(Z, 1)
void address_pairs_compare_desc(unsigned int*            dErrorCount,
                                const LDPC_kernel_params params,
                                BGDesc<BG>               bgdesc)
{
    typedef TAddr0<T, BG, Z>   addr0_t;
    typedef TAddr1_Desc<T, BG> addr1_t;
    
    typedef address_pairs_compare_node_unroll<BG, MAX_NUM_PARITY, addr0_t, addr1_t> compare_unroll_t;
    
    addr0_t          addr0;
    addr1_t          addr1(params, bgdesc, threadIdx.x);
    compare_unroll_t comp;
    comp.compare(dErrorCount, addr0, addr1);
}

template <typename                           T,
          int                                BG,
          int                                Z,
          template<typename, int, int> class TAddr0,
          template<typename, int, int> class TAddr1,
          int                                MAX_NUM_PARITY>
void perform_address_compare(unsigned int* dErrorCount)
{
    //------------------------------------------------------------------
    // Reset the error count
    CUDA_CHECK(cudaMemset(dErrorCount, 0, sizeof(unsigned int)));

    const int mb = MAX_NUM_PARITY;
    const int Kb = max_info_nodes<BG>::value;
    
    //------------------------------------------------------------------
    // Initialize data structures describing the LDPC configuration
    //------------------------------------------------------------------
    // Initialize the LDPC configuration
    cuphy::LDPC_decode_config config(CUPHY_R_16F, // LLR type
                                     mb,          // num parity nodes
                                     Z,           // lifting size
                                     10,          // num iterations
                                     Kb,          // num info nodes
                                     1.0f,        // normalization
                                     0,           // flags
                                     BG,          // base graph
                                     0,           // algorithm index
                                     nullptr);    // workspace
    // We aren't using input/output addresses for address calculations,
    // so leave them NULL here
    LDPC_kernel_params params(config,             // LDPC config
                              (Kb + mb) * Z,      // input_stride_elem
                              nullptr,            // input_addr
                              (Kb * Z + 31) / 32, // output_stride_words
                              nullptr,            // output_addr
                              1);                 // number of codewords
    //------------------------------------------------------------------
    // Invoke the kernel
    address_pairs_compare<T, BG, Z, TAddr0, TAddr1, MAX_NUM_PARITY><<<1, Z>>>(dErrorCount);

    //------------------------------------------------------------------
    // Copy data back to the host to check
    unsigned int hErrorCount = 0;
    CUDA_CHECK(cudaMemcpy(&hErrorCount, dErrorCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hErrorCount, 0) << "Errors encountered for BG=" << BG << ", Z=";
}

template <typename                           T,
          int                                BG,
          int                                Z,
          template<typename, int, int> class TAddr0,
          template<typename, int> class      TAddr1_Desc,
          template<int>           class      BGDesc,
          int                                MAX_NUM_PARITY>
void perform_address_compare_desc(unsigned int* dErrorCount, const BGDesc<BG>& bgdesc)
{
    //------------------------------------------------------------------
    // Reset the error count
    CUDA_CHECK(cudaMemset(dErrorCount, 0, sizeof(unsigned int)));
    const int mb = MAX_NUM_PARITY;
    const int Kb = max_info_nodes<BG>::value;
    
    //------------------------------------------------------------------
    // Initialize data structures describing the LDPC configuration
    cuphy::LDPC_decode_config config(CUPHY_R_16F, // LLR type
                                     mb,          // num parity nodes
                                     Z,           // lifting size
                                     10,          // num iterations
                                     Kb,          // num info nodes
                                     1.0f,        // normalization
                                     0,           // flags
                                     BG,          // base graph
                                     0,           // algorithm index
                                     nullptr);    // workspace
    // We aren't using input/output addresses for address calculations,
    // so leave them NULL here
    LDPC_kernel_params params(config,             // LDPC config
                              (Kb + mb) * Z,      // input_stride_elem
                              nullptr,            // input_addr
                              (Kb * Z + 31) / 32, // output_stride_words
                              nullptr,            // output_addr
                              1);                 // num codewords
    //------------------------------------------------------------------
    // Invoke the kernel
    address_pairs_compare_desc<T, BG, Z, TAddr0, TAddr1_Desc, BGDesc, MAX_NUM_PARITY><<<1, Z>>>(dErrorCount, params, bgdesc);
    //------------------------------------------------------------------
    // Copy data back to the host to check
    unsigned int hErrorCount = 0;
    CUDA_CHECK(cudaMemcpy(&hErrorCount, dErrorCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hErrorCount, 0) << "Errors encountered for BG=" << BG << ", Z=";
}
#if 1
////////////////////////////////////////////////////////////////////////
// LDPCInternalAPPAddr.AddressPairsDescHalf
// Test to validate APP address calculation functions, where one of the
// address generators uses a base graph descriptor structure.
// Ideally, we would have a reference CPU implementation, but for now
// we are just testing "new" implementations that use denormalized
// floats (app_loc_address_fp) against the original implementation
// (app_loc_address).
TEST(LDPCInternalAPPAddr, AddressPairsDescHalf)
{
    //------------------------------------------------------------------
    // Allocate and initialize a device buffer with a variable to count
    // differences.
    cuphy::unique_device_ptr<unsigned int> dErrorCount = cuphy::make_unique_device<unsigned int>(1);

    //perform_address_compare_desc<__half, 1, 128, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z128_half);
    //perform_address_compare_desc<__half, 1, 160, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z160_half);
    //perform_address_compare_desc<__half, 1, 192, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z192_half);
    //perform_address_compare_desc<__half, 1, 224, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z224_half);
    //perform_address_compare_desc<__half, 1, 256, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z256_half);
    //perform_address_compare_desc<__half, 1, 288, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z288_half);
    //perform_address_compare_desc<__half, 1, 320, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z320_half);
    //perform_address_compare_desc<__half, 1, 352, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z352_half);
    perform_address_compare_desc<__half, 1, 384, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z384_half);

    //perform_address_compare_desc<__half, 2, 128, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z128_half);
    //perform_address_compare_desc<__half, 2, 160, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z160_half);
    //perform_address_compare_desc<__half, 2, 192, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z192_half);
    //perform_address_compare_desc<__half, 2, 224, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z224_half);
    //perform_address_compare_desc<__half, 2, 256, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z256_half);
    //perform_address_compare_desc<__half, 2, 288, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z288_half);
    //perform_address_compare_desc<__half, 2, 320, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z320_half);
    //perform_address_compare_desc<__half, 2, 352, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z352_half);
    //perform_address_compare_desc<__half, 2, 384, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_desc_Z384_half);

}
#endif
#if 1
////////////////////////////////////////////////////////////////////////
// LDPCInternalAPPAddr.AddressPairsAdjDescHalf
// Test to validate APP address calculation functions, where one of the
// address generators uses a base graph descriptor structure.
// Ideally, we would have a reference CPU implementation, but for now
// we are just testing "new" implementations that use denormalized
// floats (app_loc_address_fp) against the original implementation
// (app_loc_address).
TEST(LDPCInternalAPPAddr, AddressPairsAdjDescHalf)
{
    //------------------------------------------------------------------
     // Allocate and initialize a device buffer with a variable to count
    // differences.
    cuphy::unique_device_ptr<unsigned int> dErrorCount = cuphy::make_unique_device<unsigned int>(1);

    //perform_address_compare_desc<__half, 1, 128, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z128_half);
    //perform_address_compare_desc<__half, 1, 160, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z160_half);
    //perform_address_compare_desc<__half, 1, 192, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z192_half);
    //perform_address_compare_desc<__half, 1, 224, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z224_half);
    //perform_address_compare_desc<__half, 1, 256, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z256_half);
    //perform_address_compare_desc<__half, 1, 288, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z288_half);
    //perform_address_compare_desc<__half, 1, 320, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z320_half);
    //perform_address_compare_desc<__half, 1, 352, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z352_half);
    perform_address_compare_desc<__half, 1, 384, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z384_half);

    //perform_address_compare_desc<__half, 2, 128, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z128_half);
    //perform_address_compare_desc<__half, 2, 160, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z160_half);
    //perform_address_compare_desc<__half, 2, 192, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z192_half);
    //perform_address_compare_desc<__half, 2, 224, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z224_half);
    //perform_address_compare_desc<__half, 2, 256, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z256_half);
    //perform_address_compare_desc<__half, 2, 288, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z288_half);
    //perform_address_compare_desc<__half, 2, 320, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z320_half);
    //perform_address_compare_desc<__half, 2, 352, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z352_half);
    //perform_address_compare_desc<__half, 2, 384, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z384_half);

}
#endif
#if 1
////////////////////////////////////////////////////////////////////////
// LDPCInternalAPPAddr.AddressPairsAdjDescHalf2
// Test to validate APP address calculation functions, where one of the
// address generators uses a base graph descriptor structure.
// Ideally, we would have a reference CPU implementation, but for now
// we are just testing "new" implementations against the original
// implementation (app_loc_address).
TEST(LDPCInternalAPPAddr, AddressPairsAdjDescHalf2)
{
    //------------------------------------------------------------------
    // Allocate and initialize a device buffer with a variable to count
    // differences.
    cuphy::unique_device_ptr<unsigned int> dErrorCount = cuphy::make_unique_device<unsigned int>(1);

    //perform_address_compare_desc<__half2, 1, 128, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z128_half2);
    //perform_address_compare_desc<__half2, 1, 160, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z160_half2);
    //perform_address_compare_desc<__half2, 1, 192, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z192_half2);
    //perform_address_compare_desc<__half2, 1, 224, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z224_half2);
    //perform_address_compare_desc<__half2, 1, 256, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z256_half2);
    //perform_address_compare_desc<__half2, 1, 288, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z288_half2);
    //perform_address_compare_desc<__half2, 1, 320, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z320_half2);
    //perform_address_compare_desc<__half2, 1, 352, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z352_half2);
    perform_address_compare_desc<__half2, 1, 384, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_adj_desc_Z384_half2);

    //perform_address_compare_desc<__half2, 2, 128, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z128_half2);
    //perform_address_compare_desc<__half2, 2, 160, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z160_half2);
    //perform_address_compare_desc<__half2, 2, 192, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z192_half2);
    //perform_address_compare_desc<__half2, 2, 224, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z224_half2);
    //perform_address_compare_desc<__half2, 2, 256, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z256_half2);
    //perform_address_compare_desc<__half2, 2, 288, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z288_half2);
    //perform_address_compare_desc<__half2, 2, 320, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z320_half2);
    //perform_address_compare_desc<__half2, 2, 352, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z352_half2);
    //perform_address_compare_desc<__half2, 2, 384, app_loc_address, app_loc_address_fp_dp_desc, BG_adj_desc, max_parity_nodes<2>::value>(dErrorCount.get(), BG2_adj_desc_Z384_half2);

}
#endif
#if 0
////////////////////////////////////////////////////////////////////////
// LDPCInternalAPPAddr.AddressPairsDescFloat
// Test to validate APP address calculation functions, where one of the
// address generators uses a base graph descriptor structure.
// Ideally, we would have a reference CPU implementation, but for now
// we are just testing "new" implementations that use denormalized
// floats (app_loc_address_fp) against the original implementation
// (app_loc_address).
TEST(LDPCInternalAPPAddr, AddressPairsDescFloat)
{
    //------------------------------------------------------------------
    // Allocate and initialize a device buffer with a variable to count
    // differences.
    cuphy::unique_device_ptr<unsigned int> dErrorCount = cuphy::make_unique_device<unsigned int>(1);

    perform_address_compare_desc<float, 1, 128, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z128_half2);
    perform_address_compare_desc<float, 1, 160, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z160_half2);
    perform_address_compare_desc<float, 1, 192, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z192_half2);
    perform_address_compare_desc<float, 1, 224, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z224_half2);
    perform_address_compare_desc<float, 1, 256, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z256_half2);
    perform_address_compare_desc<float, 1, 288, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z288_half2);
    perform_address_compare_desc<float, 1, 320, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z320_half2);
    perform_address_compare_desc<float, 1, 352, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z352_half2);
    perform_address_compare_desc<float, 1, 384, app_loc_address, app_loc_address_fp_desc, BG_desc, max_parity_nodes<1>::value>(dErrorCount.get(), BG1_desc_Z384_half2);
}

#endif
#if 0
////////////////////////////////////////////////////////////////////////
// LDPCInternalAPPAddr.AddressPairsHalf
// Test to validate APP address calculation functions.
// Ideally, we would have a reference CPU implementation, but for now
// we are just testing "new" implementations that use denormalized
// floats (app_loc_address_fp) against the original implementation
// (app_loc_address).
TEST(LDPCInternalAPPAddr, AddressPairsHalf)
{
    //------------------------------------------------------------------
    // Allocate and initialize a device buffer with a variable to count
    // differences.
    cuphy::unique_device_ptr<unsigned int> dErrorCount = cuphy::make_unique_device<unsigned int>(1);

    perform_address_compare<__half, 1, 384, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 352, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 320, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 288, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 256, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 224, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 192, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 160, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 128, app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 96,  app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<__half, 1, 64,  app_loc_address, app_loc_address_fp, max_parity_nodes<1>::value>(dErrorCount.get());

    perform_address_compare<__half, 2, 384, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 352, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 320, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 288, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 256, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 224, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 192, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 160, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 128, app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 96,  app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<__half, 2, 64,  app_loc_address, app_loc_address_fp, max_parity_nodes<2>::value>(dErrorCount.get());
}
#endif
#if 0
////////////////////////////////////////////////////////////////////////
// LDPCInternalAPPAddr.AddressPairsFloat
// Test to validate APP address calculation functions.
// Note that the calculation of APP addresses for float APP values can
// also be used when working with a pair of fp16 values (because
// sizeof(float) = sizeof(__half2). (This might be the case for a kernel
// that decodes 2 codewords at a time.)
TEST(LDPCInternalAPPAddr, AddressPairsFloat)
{
    //------------------------------------------------------------------
    // Allocate and initialize a device buffer with a variable to count
    // differences.
    cuphy::unique_device_ptr<unsigned int> dErrorCount = cuphy::make_unique_device<unsigned int>(1);

    // BG1
    // Using (floor(2^16 / (Z * 4)) - 22) for max num parity nodes
    perform_address_compare<float, 1, 384, app_loc_address, app_loc_address_fp_imad, 20>                        (dErrorCount.get());
    perform_address_compare<float, 1, 352, app_loc_address, app_loc_address_fp_imad, 24>                        (dErrorCount.get());
    perform_address_compare<float, 1, 320, app_loc_address, app_loc_address_fp_imad, 29>                        (dErrorCount.get());
    perform_address_compare<float, 1, 288, app_loc_address, app_loc_address_fp_imad, 34>                        (dErrorCount.get());
    perform_address_compare<float, 1, 256, app_loc_address, app_loc_address_fp_imad, 42>                        (dErrorCount.get());
    perform_address_compare<float, 1, 224, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 192, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 160, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 128, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 96,  app_loc_address, app_loc_address_fp_imad, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 64,  app_loc_address, app_loc_address_fp_imad, max_parity_nodes<1>::value>(dErrorCount.get());

    // BG2
    // Using (floor(2^16 / (Z * 4)) - 10) for max num parity nodes
    perform_address_compare<float, 2, 384, app_loc_address, app_loc_address_fp_imad, 32>                        (dErrorCount.get());
    perform_address_compare<float, 2, 352, app_loc_address, app_loc_address_fp_imad, 36>                        (dErrorCount.get());
    perform_address_compare<float, 2, 320, app_loc_address, app_loc_address_fp_imad, 41>                        (dErrorCount.get());
    perform_address_compare<float, 2, 288, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 256, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 224, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 192, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 160, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 128, app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 96,  app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 64,  app_loc_address, app_loc_address_fp_imad, max_parity_nodes<2>::value>(dErrorCount.get());
}
#endif
#if 0
////////////////////////////////////////////////////////////////////////
// LDPCInternalAPPAddr.AddressPairsFloatLarge
// Test to validate APP address calculation functions.
// Note that the calculation of APP addresses for float APP values can
// also be used when working with a pair of fp16 values (because
// sizeof(float) = sizeof(__half2). (This might be the case for a kernel
// that decodes 2 codewords at a time.)
TEST(LDPCInternalAPPAddr, AddressPairsFloatLarge)
{
    //------------------------------------------------------------------
    // Allocate and initialize a device buffer with a variable to count
    // differences.
    cuphy::unique_device_ptr<unsigned int> dErrorCount = cuphy::make_unique_device<unsigned int>(1);

    // Only check BG/Z pairs for which 16 bits is not enough to store
    // all APP data (e.g. sizeof(T) = 4, lower code rates, ..)
    // BG1
    perform_address_compare<float, 1, 384, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 352, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 320, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 288, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<1>::value>(dErrorCount.get());
    perform_address_compare<float, 1, 256, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<1>::value>(dErrorCount.get());

    // BG2
    perform_address_compare<float, 2, 384, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 352, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<2>::value>(dErrorCount.get());
    perform_address_compare<float, 2, 320, app_loc_address, app_loc_address_fp_imad_lg, max_parity_nodes<2>::value>(dErrorCount.get());
}
#endif
////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
