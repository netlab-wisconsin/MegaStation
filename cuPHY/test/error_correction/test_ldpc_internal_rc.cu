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
#include "ldpc2_min_sum_update_half_0.cuh"
#include "ldpc2_sign_split.cuh"
#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_box_plus.cuh"
#include "cuphy.hpp"

#define CUDA_CHECK_LAST_ERROR_THROW()       \
    {                                       \
        cudaError_t e = cudaGetLastError(); \
        if(e != cudaSuccess)                \
        {                                   \
            throw cuphy::cuda_exception(e); \
        }                                   \
    }



using namespace ldpc2;

////////////////////////////////////////////////////////////////////////
// test_rc_fp16_kernel()
// Test kernel for the LDPC row_context class using fp16 APP data
template <int ROW_DEGREE, class TRowContext>
__global__ void test_rc_fp16_kernel(int            num_tests,
                                    __half2*       pout,
                                    const __half2* pin)
{
    //------------------------------------------------------------------
    const int      inputIdx  = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int      ROW_WORDS = (ROW_DEGREE + 1) / 2;
    const __half2* tIn       = pin  + (inputIdx * ROW_WORDS);
    __half2*       tOut      = pout + (inputIdx * ROW_WORDS);
    //------------------------------------------------------------------
    if(inputIdx >= num_tests)
    {
        return;
    }
    //------------------------------------------------------------------
    // Load data into registers
    word_t app[ROW_WORDS]; // Max
    for(int i = 0; i < ROW_WORDS; ++i)
    {
        // Setting elements with indices (i * 2) and ((i * 2) + 1)
        app[i].f16x2 = tIn[i];
        // Set the spare value to something that won't be min0 or min1
        if(((i * 2) + 1) >= ROW_DEGREE)
        {
            app[i].f16x2.y = 0x7C00;
        }
    }

    TRowContext rc(__float2half2_rn(1.0f), app, std::integral_constant<int, ROW_DEGREE>{});

    //KERNEL_PRINT("test_rc_fp16_kernel() [%i]: min1 = %f, min0 = %f, min0_index = %i, signs = 0x%X\n",
    //             inputIdx,
    //             __high2float(rc.min1_min0.f16x2),
    //             __low2float(rc.min1_min0.f16x2),
    //             rc.min0_index,
    //             rc.signs);
    //------------------------------------------------------------------
    // Extract values
    for(int i = 0; i < ROW_WORDS; ++i)
    {
        word_t w = rc.extract_pair(i); 
        tOut[i] = w.f16x2;
    }
}

typedef cC2V_row_context<__half2, sign_mgr_pair_src<true>, unused, cC2V_storage_x2_high_degree_split> rc_fp16x2_t;

////////////////////////////////////////////////////////////////////////
// test_rc_fp16x2_kernel()
// Test kernel for the LDPC row_context class using fp16x2 APP data to
// represent processing 2 codewords at a time.
__global__ void test_rc_fp16x2_kernel(int            num_tests,
                                      int            row_degree,
                                      __half2*       pout,
                                      const __half2* pin)
{
    const int      inputIdx  = (blockDim.x * blockIdx.x) + threadIdx.x;
    const __half2* tIn       = pin  + (inputIdx * row_degree);
    __half2*       tOut      = pout + (inputIdx * row_degree);
    //------------------------------------------------------------------
    if(inputIdx >= num_tests)
    {
        return;
    }
    //------------------------------------------------------------------
    // Load input data into registers
    word_t app[19]; // Maximum row degree
    for(int i = 0; i < row_degree; ++i)
    {
        app[i].f16x2 = tIn[i];
    }
    //------------------------------------------------------------------
    // Update the compressed min-sum representation
    rc_fp16x2_t rc;
    rc.init_row(app[0], app[1]);
    //KERNEL_PRINT("test_rc_fp16x2_kernel() [%i] init: min1 = [%f | %f], min0 = [%f | %f], min0_index = [%i | %i], signs = [0x%X | 0x%X]\n",
    //             inputIdx,
    //             __high2float(rc.min1.f16x2),
    //             __low2float(rc.min1.f16x2),
    //             __high2float(rc.min0.f16x2),
    //             __low2float(rc.min0.f16x2),
    //             rc.min0_index.u16x2.y,
    //             rc.min0_index.u16x2.x,
    //             rc_fp16x2_t::sign_mgr_t::debug_get_codeword_signs(rc.signs_0_9, rc.signs_10_, 1),
    //             rc_fp16x2_t::sign_mgr_t::debug_get_codeword_signs(rc.signs_0_9, rc.signs_10_, 0));
    for(int i = 2; i < row_degree; ++i)
    {
        rc.update(app[i], i);
        //KERNEL_PRINT("test_rc_fp16x2_kernel() [%i] update: min1 = [%f | %f], min0 = [%f | %f], min0_index = [%i | %i], signs = [0x%X | 0x%X]\n",
        //               inputIdx,
        //               __high2float(rc.min1.f16x2),
        //               __low2float(rc.min1.f16x2),
        //               __high2float(rc.min0.f16x2),
        //               __low2float(rc.min0.f16x2),
        //               rc.min0_index.u16x2.y,
        //               rc.min0_index.u16x2.x,
        //               rc_fp16x2_t::sign_mgr_t::debug_get_codeword_signs(rc.signs_0_9, rc.signs_10_, 1),
        //               rc_fp16x2_t::sign_mgr_t::debug_get_codeword_signs(rc.signs_0_9, rc.signs_10_, 0));
    }
    //KERNEL_PRINT("test_rc_fp16x2_kernel() [%i]: min1 = [%f | %f], min0 = [%f | %f], min0_index = [%i | %i], signs = [0x%X | 0x%X]\n",
    //             inputIdx,
    //             __high2float(rc.min1.f16x2),
    //             __low2float(rc.min1.f16x2),
    //             __high2float(rc.min0.f16x2),
    //             __low2float(rc.min0.f16x2),
    //             rc.min0_index.u16x2.y,
    //             rc.min0_index.u16x2.x,
    //             rc_fp16x2_t::sign_mgr_t::debug_get_codeword_signs(rc.signs_0_9, rc.signs_10_, 1),
    //             rc_fp16x2_t::sign_mgr_t::debug_get_codeword_signs(rc.signs_0_9, rc.signs_10_, 0));

    rc.finalize(__float2half2_rn(1.0f), row_degree);
    //------------------------------------------------------------------
    // Extract values
    for(int i = 0; i < row_degree; ++i)
    {
        word_t w = rc.extract_pair(i); 
        tOut[i] = w.f16x2;
    }
}

////////////////////////////////////////////////////////////////////////
// LDPCInternalHalf
// Test fixture with source data for half precision. Sequences of fp16
// values are stored combined to form a sequence of half2 instances.
class LDPCInternalHalf : public ::testing::Test
{
protected:
    static const size_t N_TEST_VALUES = 19 * 1024;
    //static const size_t N_TEST_VALUES = 40;
    static const size_t N_TEST_PAIRS = (N_TEST_VALUES + 1) / 2;
    typedef cuphy::buffer<__half2, cuphy::device_alloc> half2_buffer_device_t;
    //------------------------------------------------------------------
    // SetUp()
    void SetUp() override
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Random number generation. Choose an interval that will not
        // overflow fp16
        //std::random_device rd;
        //std::mt19937       e2(rd());
        std::mt19937                     e2;
        std::uniform_real_distribution<> dist(-60*1024.0f, 60*1024.0f);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Allocate host storage
        random_src.resize(((N_TEST_VALUES + 1) / 2) * 2);
        random_src_half2.resize(N_TEST_PAIRS);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate random data, but store as fp16
        for(size_t i = 0; i < random_src.size(); ++i)
        {
            random_src[i] = dist(e2);
            DEBUG_PRINTF("%lu: %.1f\n", i, __half2float(random_src[i]));
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Pair fp16 values into half2 values for use by the kernel
        for(size_t i = 0; i < random_src_half2.size(); ++i)
        {
            DEBUG_PRINTF("%lu: [%.1f | %.1f]\n", i, __half2float(random_src[i*2+1]), __half2float(random_src[i*2]));
            random_src_half2[i] = __floats2half2_rn(random_src[i * 2], random_src[i * 2 + 1]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize a device buffer with the src contents
        random_src_half2_device = half2_buffer_device_t(N_TEST_PAIRS);
        CUDA_CHECK(cudaMemcpy(random_src_half2_device.addr(),
                              random_src_half2.data(),
                              random_src_half2.size() * sizeof(random_src_half2[0]),
                              cudaMemcpyHostToDevice));
    }
    //------------------------------------------------------------------
    //void TearDown() override {}
    //------------------------------------------------------------------
    // Data
    std::vector<__half>   random_src;
    std::vector<__half2>  random_src_half2;
    half2_buffer_device_t random_src_half2_device;
};

////////////////////////////////////////////////////////////////////////
// get_elem_from_pairs()
// Retrieves a float element from a set of __half2 instances. The index
// is the "element" index, where each __half2 contains 2 elements.
float get_elem_from_pairs(const __half2* src, int index)
{
    const int pairIndex = index / 2;
    return (0 == (index % 2))           ?
           __low2float(src[pairIndex])  :
           __high2float(src[pairIndex]) ;
}

////////////////////////////////////////////////////////////////////////
// set_pair_elem()
// Sets an element in a set of __half2 instances. The index
// is the "element" index, where each __half2 contains 2 elements.
void set_pair_elem(__half2* dst, int index, float f)
{
    const int pairIndex = index / 2;
    __half2 __floats2half2_rn(const float a, const float b);
    dst[pairIndex] = (0 == (index % 2))                                 ?
                     __floats2half2_rn(f, __high2float(dst[pairIndex])) :
                     __floats2half2_rn(__low2float(dst[pairIndex]), f)  ;
}

////////////////////////////////////////////////////////////////////////
// host_min_sum()
void host_min_sum(int            N,
                  __half2*       dst,
                  const __half2* src)
{
    //------------------------------------------------------------------
    // Initialize min-sum representation
    float        min0       = std::numeric_limits<float>::infinity();
    float        min1       = std::numeric_limits<float>::infinity();
    int          min0_index = -1;
    unsigned int signs      = 0;
    //------------------------------------------------------------------
    // Update min-sum representation
    for(int i = 0; i < N; ++i)
    {
        float        f      = get_elem_from_pairs(src, i);
        unsigned int is_neg = (f < 0) ? 1 : 0;
        signs |= (is_neg << i);
        if(fabs(f) < min0)
        {
            min0_index = i;
            min1       = min0;
            min0       = fabsf(f);
        }
        else if(fabsf(f) < min1)
        {
            min1 = fabsf(f);
        }
    }
    //------------------------------------------------------------------
    // Extract values from min-sum representation
    uint32_t  sign_prod_is_neg = ((unsigned int)__builtin_popcount(signs) & 0x1);
    for(int i = 0; i < N; ++i)
    {
        float value  = (i == min0_index) ? min1 : min0;
        if(((signs >> i) & 0x1) ^ sign_prod_is_neg)
        {
            value = - value;
        }
        set_pair_elem(dst, i, value);
    }
}

////////////////////////////////////////////////////////////////////////
// min_sum_context
// Host struct to keep track of min0, min1, min0_index, and signs, and
// to provide the extracted sequence of values.
struct min_sum_context
{
    min_sum_context() :
        min0(std::numeric_limits<float>::infinity()),
        min1(std::numeric_limits<float>::infinity()),
        min0_index(-1),
        signs(0)
    {
    }
    void process(float f, int i)
    {
        unsigned int is_neg = (f < 0) ? 1 : 0;
        signs |= (is_neg << i);
        if(fabs(f) < min0)
        {
            min0_index = i;
            min1       = min0;
            min0       = fabsf(f);
        }
        else if(fabsf(f) < min1)
        {
            min1 = fabsf(f);
        }
    }
    float extract(int i)
    {
        uint32_t  sign_prod_is_neg = ((unsigned int)__builtin_popcount(signs) & 0x1);
        float value  = (i == min0_index) ? min1 : min0;
        //printf("extract[%i]: signs = 0x%X, popcnt = %i, sign_prod_is_neg = %u, check = %u\n",
        //       i,
        //       signs,
        //       __builtin_popcount(signs),
        //       sign_prod_is_neg,
        //       ((signs >> i) & 0x1));
        if(((signs >> i) & 0x1) ^ sign_prod_is_neg)
        {
            value = -value;
        }
        return value;
    }
    float min0;
    float min1;
    int   min0_index;
    int   signs;
};

////////////////////////////////////////////////////////////////////////
// host_min_sum_x2()
void host_min_sum_x2(int            N,
                     __half2*       dst,
                     const __half2* src)
{
    min_sum_context msctx[2];
    //------------------------------------------------------------------
    // Update min-sum representation
    for(int i = 0; i < N; ++i)
    {
        msctx[0].process(__low2float(src[i]), i);
        msctx[1].process(__high2float(src[i]), i);
    }
    //------------------------------------------------------------------
    // Extract values from min-sum representation
    for(int i = 0; i < N; ++i)
    {
        dst[i] = __floats2half2_rn(msctx[0].extract(i), msctx[1].extract(i));
    }
}

/////////////////////////////////////////////////////////////////////////
// compare_half_pairs()
// Compare two half precision output sequences (stored as half2 values)
// for equality.
void compare_half_pairs(const char*    desc,
                        int            testIndex,
                        int            rowDegree,
                        const __half2* hostRow,
                        const __half2* deviceRow)
{
    for(size_t j = 0; j < rowDegree; ++j)
    {
        const float hostValue   = get_elem_from_pairs(hostRow,   j);
        const float deviceValue = get_elem_from_pairs(deviceRow, j);
        if(hostValue != deviceValue)
        {
            DEBUG_PRINTF("FAILURE (%s): ROW_DEGREE = %i, TEST = %i, INDEX = %lu, HOST = %f, DEVICE = %f\n",
                         desc,
                         rowDegree,
                         testIndex,
                         j,
                         hostValue,
                         deviceValue);
        }
        else
        {
            DEBUG_PRINTF("MATCH (%s): ROW_DEGREE = %i, TEST = %i, INDEX = %lu, VALUE = %f\n",
                         desc,
                         rowDegree,
                         testIndex,
                         j,
                         hostValue);
        }
        EXPECT_EQ(hostValue, deviceValue) << "FAILURE (" << desc << "): ROW_DEGREE = " << rowDegree
                                          << ", TEST = " << testIndex
                                          << ", INDEX = " << j << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////
// Compare the output of a min-sum row context to a host reference
// implementation. fp16 values are stored in pairs "along" the row, such
// that for odd row degrees there will be an extra unused value in the
// last pair.
//
// The row context determines the two smallest values, the index of the
// smallest, and the signs of the inputs, and then generates an output
// sequence as dictated by those values.
TEST_F(LDPCInternalHalf, RowContext_fp16)
{
    std::vector<__half2> hostRef(random_src_half2.size());
    half2_buffer_device_t k_out;

    typedef sign_store_policy_split_dst<__half, split_sign_update_bit_ops, true>     sign_mgr_t;
    //typedef sign_store_policy_split_src<__half, split_sign_update_bit_ops, true>   sign_mgr_t;

    typedef C2V_storage_t<__half, 2>                                                 storage_t;
    typedef cC2V_row_context<__half, sign_mgr_t, min_sum_update_half_0, storage_t>   rc_fp16_t;
    //typedef cC2V_row_context<__half, sign_store_policy_split_dst<__half>>          rc_fp16_t;
    //typedef cC2V_row_context_sel_prmt<__half, sign_store_policy_split_src<__half>> rc_fp16_t;
    //------------------------------------------------------------------
    // Loop over all 5G row degrees (BG1 and BG2)
    std::array<int, 9> row_degrees{3, 4, 5, 6, 7, 8, 9, 10, 19};
    //std::array<int, 1> row_degrees{19};
    for(auto row_degree : row_degrees)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // How many tests can we run for the amount of allocated data,
        // for the current row degree?
        const size_t PAIRS_PER_ROW = (row_degree + 1) / 2;
        const size_t NUM_ROW_TESTS = (random_src.size() / 2) / PAIRS_PER_ROW;

        DEBUG_PRINTF("row_degree = %i, num_tests = %lu\n", row_degree, NUM_ROW_TESTS);
        
        hostRef.resize(NUM_ROW_TESTS * PAIRS_PER_ROW);
        k_out = half2_buffer_device_t(NUM_ROW_TESTS * PAIRS_PER_ROW);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate reference data on the host
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostSrc = random_src_half2.data() + (PAIRS_PER_ROW * i);
            __half2*       hostDst = hostRef.data()          + (PAIRS_PER_ROW * i);
            host_min_sum(row_degree, hostDst, hostSrc);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Execute the kernel on the device
        dim3 blkDim(32);
        dim3 grdDim((NUM_ROW_TESTS + (blkDim.x - 1)) / blkDim.x);
        switch(row_degree)
        {
        case 3:
            test_rc_fp16_kernel<3, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                  k_out.addr(),
                                                                  random_src_half2_device.addr());
            break;
        case 4:
            test_rc_fp16_kernel<4, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                  k_out.addr(),
                                                                  random_src_half2_device.addr());
            break;
        case 5:
            test_rc_fp16_kernel<5, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                  k_out.addr(),
                                                                  random_src_half2_device.addr());
            break;
        case 6:
            test_rc_fp16_kernel<6, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                  k_out.addr(),
                                                                  random_src_half2_device.addr());
            break;
        case 7:
            test_rc_fp16_kernel<7, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                  k_out.addr(),
                                                                  random_src_half2_device.addr());
            break;
        case 8:
            test_rc_fp16_kernel<8, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                  k_out.addr(),
                                                                  random_src_half2_device.addr());
            break;
        case 9:
            test_rc_fp16_kernel<9, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                  k_out.addr(),
                                                                  random_src_half2_device.addr());
            break;
        case 10:
            test_rc_fp16_kernel<10, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                   k_out.addr(),
                                                                   random_src_half2_device.addr());
            break;
        case 19:
            test_rc_fp16_kernel<19, rc_fp16_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                   k_out.addr(),
                                                                   random_src_half2_device.addr());
            break;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Transfer results back to the host
        std::vector<__half2> deviceResults(random_src_half2.size());
        CUDA_CHECK(cudaMemcpy(deviceResults.data(),
                              k_out.addr(),
                              k_out.size() * sizeof(half2_buffer_device_t::element_t),
                              cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Compare results
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostRow   = hostRef.data()       + (PAIRS_PER_ROW * i);
            const __half2* deviceRow = deviceResults.data() + (PAIRS_PER_ROW * i);
            compare_half_pairs("RowContext_fp16", i, row_degree, hostRow, deviceRow);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Compare the output of a min-sum row context to a host reference
// implementation. Two separate codewords are represented by the hi and
// lo values in the sequence, and the row context generates a pair of
// output sequences.
//
// For each codeword, the row context determines the two smallest values,
// the index of the smallest, and the signs of the inputs, and then
// generates an output sequence as dictated by those values.
TEST_F(LDPCInternalHalf, RowContext_fp16x2)
{
    std::vector<__half2> hostRef(random_src_half2.size());
    half2_buffer_device_t k_out;
    //------------------------------------------------------------------
    // Loop over all 5G row degrees (BG1 and BG2)
    std::array<int, 9> row_degrees{3, 4, 5, 6, 7, 8, 9, 10, 19};
    //std::array<int, 1> row_degrees{19};
    for(auto row_degree : row_degrees)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // How many tests can we run for the amount of allocated data,
        // for the current row degree?
        const size_t NUM_ROW_TESTS = random_src_half2.size() / row_degree;

        DEBUG_PRINTF("row_degree = %i, num_half2 = %lu, num_tests = %lu\n", row_degree, random_src_half2.size(), NUM_ROW_TESTS);

        hostRef.resize(NUM_ROW_TESTS * row_degree);
        k_out = half2_buffer_device_t(NUM_ROW_TESTS * row_degree);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate reference data on the host
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostSrc = random_src_half2.data() + (row_degree * i);
            __half2*       hostDst = hostRef.data()          + (row_degree * i);
            host_min_sum_x2(row_degree, hostDst, hostSrc);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Execute the kernel on the device
        dim3 blkDim(32);
        dim3 grdDim((NUM_ROW_TESTS + (blkDim.x - 1)) / blkDim.x);
        test_rc_fp16x2_kernel<<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                  row_degree,
                                                  k_out.addr(),
                                                  random_src_half2_device.addr());
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Transfer results back to the host
        std::vector<__half2> deviceResults(random_src_half2.size());
        CUDA_CHECK(cudaMemcpy(deviceResults.data(),
                              k_out.addr(),
                              k_out.size() * sizeof(half2_buffer_device_t::element_t),
                              cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Compare results
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostRow   = hostRef.data()       + (row_degree * i);
            const __half2* deviceRow = deviceResults.data() + (row_degree * i);
            for(size_t j = 0; j < row_degree; ++j)
            {
                const float hostValues[2]   = { __low2float(hostRow[j]),   __high2float(hostRow[j])   };
                const float deviceValues[2] = { __low2float(deviceRow[j]), __high2float(deviceRow[j]) };
                for(int c = 0; c < 2; ++c)
                {
                    if(hostValues[c] != deviceValues[c])
                    {
                        DEBUG_PRINTF("FAILURE: ROW_DEGREE = %i, TEST_IDX = %lu, ROW_INDEX = %lu.%s, HOST = %f, DEVICE = %f\n",
                                     row_degree,
                                     i,
                                     j,
                                     (0 == c) ? "LO" : "HI",
                                     hostValues[c],
                                     deviceValues[c]);
                    }
                    else
                    {
                        DEBUG_PRINTF("MATCH: ROW_DEGREE = %i, TEST_IDX = %lu, ROW_INDEX = %lu.%s, VALUE = %f\n",
                                     row_degree,
                                     i,
                                     j,
                                     (0 == c) ? "LO" : "HI",
                                     hostValues[c]);
                    }
                    EXPECT_EQ(hostValues[c], deviceValues[c]);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// test_box_plus_kernel()
template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TBoxPlusOp>
__global__ void test_box_plus_kernel(int            numTests,
                                     __half2*       out,
                                     const __half2* in)
{
    const int      inputIdx  = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int      ROW_WORDS = (ROW_DEGREE + 1) / 2;
    const __half2* tIn       = in + (inputIdx * ROW_WORDS);
    __half2*       tOut      = out + (inputIdx * ROW_WORDS);
    //------------------------------------------------------------------
    if(inputIdx >= numTests)
    {
        return;
    }
    //------------------------------------------------------------------
    // Load data into registers
    word_t app[ROW_WORDS];
    for(int i = 0; i < ROW_WORDS; ++i)
    {
        app[i].f16x2 = tIn[i];
        // Set the spare value for odd row degree cases
        if(((i*2) + 1) >= ROW_DEGREE)
        {
            app[i].f16x2.y = 0x7C00; // fp16(inf)
        }
    }
    //------------------------------------------------------------------
    // Determine the output sequence
    word_t seq[ROW_WORDS];

    typedef box_plus_seq_gen<__half, TBoxPlusOp, ROW_DEGREE, UPDATE_ROW_DEGREE> box_plus_seq_gen_t;
    
    box_plus_seq_gen_t::generate(seq, app);
    //------------------------------------------------------------------
    // Write to global memory
    for(int i = 0; i < ROW_WORDS; ++i)
    {
        tOut[i] = seq[i].f16x2;
    }
}

////////////////////////////////////////////////////////////////////////
// do_box_plus_test()
template <class TBoxPlusOp>
void do_box_plus_test(const std::vector<__half2>&                        input_half2,
                      const cuphy::buffer<__half2, cuphy::device_alloc>& input_half2_device)
{
    typedef cuphy::buffer<__half2, cuphy::device_alloc> half2_buffer_device_t;
    
    std::vector<__half2>  hostRef(input_half2.size());
    half2_buffer_device_t k_out;

    //------------------------------------------------------------------
    // Loop over all 5G row degrees (BG1 and BG2)
    std::array<int, 9> row_degrees{3, 4, 5, 6, 7, 8, 9, 10, 19};
    for(auto row_degree : row_degrees)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // How many tests can we run for the amount of allocated data,
        // for the current row degree?
        const size_t PAIRS_PER_ROW = (row_degree + 1) / 2;
        const size_t NUM_ROW_TESTS = input_half2.size() / PAIRS_PER_ROW;

        DEBUG_PRINTF("row_degree = %i, num_tests = %lu\n", row_degree, NUM_ROW_TESTS);
        
        hostRef.resize(NUM_ROW_TESTS * PAIRS_PER_ROW);
        k_out = half2_buffer_device_t(NUM_ROW_TESTS * PAIRS_PER_ROW);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate reference data on the host
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostSrc = input_half2.data() + (PAIRS_PER_ROW * i);
            __half2*       hostDst = hostRef.data()     + (PAIRS_PER_ROW * i);
            host_min_sum(row_degree, hostDst, hostSrc);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Execute the kernel on the device
        dim3 blkDim(32);
        dim3 grdDim((NUM_ROW_TESTS + (blkDim.x - 1)) / blkDim.x);
        typedef TBoxPlusOp box_plus_t;
        switch(row_degree)
        {
        default:
        case 3:
            test_box_plus_kernel<3, 3, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                       k_out.addr(),
                                                                       input_half2_device.addr());
            break;
        case 4:
            test_box_plus_kernel<4, 4, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                       k_out.addr(),
                                                                       input_half2_device.addr());
            break;
        case 5:
            test_box_plus_kernel<5, 5, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                       k_out.addr(),
                                                                       input_half2_device.addr());
            break;
        case 6:
            test_box_plus_kernel<6, 6, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                       k_out.addr(),
                                                                       input_half2_device.addr());
            break;
        case 7:
            test_box_plus_kernel<7, 7, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                       k_out.addr(),
                                                                       input_half2_device.addr());
            break;
        case 8:
            test_box_plus_kernel<8, 8, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                       k_out.addr(),
                                                                       input_half2_device.addr());
            break;
        case 9:
            test_box_plus_kernel<9, 9, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                       k_out.addr(),
                                                                       input_half2_device.addr());
            break;
        case 10:
            test_box_plus_kernel<10, 10, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                         k_out.addr(),
                                                                         input_half2_device.addr());
            break;
        case 19:
            test_box_plus_kernel<19, 19, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                         k_out.addr(),
                                                                         input_half2_device.addr());
            break;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Transfer results back to the host
        std::vector<__half2> deviceResults(input_half2.size());
        CUDA_CHECK(cudaMemcpy(deviceResults.data(),
                              k_out.addr(),
                              k_out.size() * sizeof(half2_buffer_device_t::element_t),
                              cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Compare results
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostRow   = hostRef.data()       + (PAIRS_PER_ROW * i);
            const __half2* deviceRow = deviceResults.data() + (PAIRS_PER_ROW * i);
            compare_half_pairs("BoxPlus_fp16", i, row_degree, hostRow, deviceRow);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// LDPCInternalHalf.BoxPlus_fp16
TEST_F(LDPCInternalHalf, BoxPlus_fp16)
{
    do_box_plus_test<ldpc2::box_plus_op>(random_src_half2,         // host data
                                         random_src_half2_device); // device data
}

////////////////////////////////////////////////////////////////////////
// test_box_plus_x2_kernel()
template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TBoxPlusOp>
__global__ void test_box_plus_x2_kernel(int            numTests,
                                        __half2*       out,
                                        const __half2* in)
{
    const int      inputIdx  = (blockIdx.x * blockDim.x) + threadIdx.x;
    const __half2* tIn       = in + (inputIdx * ROW_DEGREE);
    __half2*       tOut      = out + (inputIdx * ROW_DEGREE);
    //------------------------------------------------------------------
    if(inputIdx >= numTests)
    {
        return;
    }
    //------------------------------------------------------------------
    // Load data into registers
    word_t app[ROW_DEGREE];
    for(int i = 0; i < ROW_DEGREE; ++i)
    {
        app[i].f16x2 = tIn[i];
    }
    //------------------------------------------------------------------
    // Determine the output sequence
    word_t seq[ROW_DEGREE];

    typedef box_plus_seq_gen<__half2, TBoxPlusOp, ROW_DEGREE, UPDATE_ROW_DEGREE> box_plus_seq_gen_t;
    
    box_plus_seq_gen_t::generate(seq, app);
    //------------------------------------------------------------------
    // Write to global memory
    for(int i = 0; i < ROW_DEGREE; ++i)
    {
        tOut[i] = seq[i].f16x2;
    }
}

////////////////////////////////////////////////////////////////////////
// do_box_plus_x2_test()
template <class TBoxPlusOp>
void do_box_plus_x2_test(const std::vector<__half2>&                        input_half2,
                         const cuphy::buffer<__half2, cuphy::device_alloc>& input_half2_device)
{
    typedef cuphy::buffer<__half2, cuphy::device_alloc> half2_buffer_device_t;

    std::vector<__half2>  hostRef(input_half2.size());
    half2_buffer_device_t k_out;
    //------------------------------------------------------------------
    // Loop over all 5G row degrees (BG1 and BG2)
    std::array<int, 9> row_degrees{3, 4, 5, 6, 7, 8, 9, 10, 19};
    //std::array<int, 1> row_degrees{19};
    for(auto row_degree : row_degrees)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // How many tests can we run for the amount of allocated data,
        // for the current row degree?
        const size_t NUM_ROW_TESTS = input_half2.size() / row_degree;

        DEBUG_PRINTF("row_degree = %i, num_half2 = %lu, num_tests = %lu\n", row_degree, input_half2.size(), NUM_ROW_TESTS);

        hostRef.resize(NUM_ROW_TESTS * row_degree);
        k_out = half2_buffer_device_t(NUM_ROW_TESTS * row_degree);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate reference data on the host
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostSrc = input_half2.data() + (row_degree * i);
            __half2*       hostDst = hostRef.data()     + (row_degree * i);
            host_min_sum_x2(row_degree, hostDst, hostSrc);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Execute the kernel on the device
        dim3 blkDim(32);
        dim3 grdDim((NUM_ROW_TESTS + (blkDim.x - 1)) / blkDim.x);
        typedef TBoxPlusOp box_plus_t;
        switch(row_degree)
        {
        default:
        case 3:
            test_box_plus_x2_kernel<3, 3, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                          k_out.addr(),
                                                                          input_half2_device.addr());
            break;
        case 4:
            test_box_plus_x2_kernel<4, 4, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                          k_out.addr(),
                                                                          input_half2_device.addr());
            break;
        case 5:
            test_box_plus_x2_kernel<5, 5, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                          k_out.addr(),
                                                                          input_half2_device.addr());
            break;
        case 6:
            test_box_plus_x2_kernel<6, 6, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                          k_out.addr(),
                                                                          input_half2_device.addr());
            break;
        case 7:
            test_box_plus_x2_kernel<7, 7, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                          k_out.addr(),
                                                                          input_half2_device.addr());
            break;
        case 8:
            test_box_plus_x2_kernel<8, 8, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                          k_out.addr(),
                                                                          input_half2_device.addr());
            break;
        case 9:
            test_box_plus_x2_kernel<9, 9, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                          k_out.addr(),
                                                                          input_half2_device.addr());
            break;
        case 10:
            test_box_plus_x2_kernel<10, 10, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                            k_out.addr(),
                                                                            input_half2_device.addr());
            break;
        case 19:
            test_box_plus_x2_kernel<19, 19, box_plus_t><<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                                            k_out.addr(),
                                                                            input_half2_device.addr());
            break;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Transfer results back to the host
        std::vector<__half2> deviceResults(input_half2.size());
        CUDA_CHECK(cudaMemcpy(deviceResults.data(),
                              k_out.addr(),
                              k_out.size() * sizeof(half2_buffer_device_t::element_t),
                              cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Compare results
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostRow   = hostRef.data()       + (row_degree * i);
            const __half2* deviceRow = deviceResults.data() + (row_degree * i);
            for(size_t j = 0; j < row_degree; ++j)
            {
                const float hostValues[2]   = { __low2float(hostRow[j]),   __high2float(hostRow[j])   };
                const float deviceValues[2] = { __low2float(deviceRow[j]), __high2float(deviceRow[j]) };
                for(int c = 0; c < 2; ++c)
                {
                    if(hostValues[c] != deviceValues[c])
                    {
                        DEBUG_PRINTF("FAILURE: ROW_DEGREE = %i, TEST_IDX = %lu, ROW_INDEX = %lu.%s, HOST = %f, DEVICE = %f\n",
                                     row_degree,
                                     i,
                                     j,
                                     (0 == c) ? "LO" : "HI",
                                     hostValues[c],
                                     deviceValues[c]);
                    }
                    else
                    {
                        DEBUG_PRINTF("MATCH: ROW_DEGREE = %i, TEST_IDX = %lu, ROW_INDEX = %lu.%s, VALUE = %f\n",
                                     row_degree,
                                     i,
                                     j,
                                     (0 == c) ? "LO" : "HI",
                                     hostValues[c]);
                    }
                    EXPECT_EQ(hostValues[c], deviceValues[c]);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Compare the output of a min-sum row context to a host reference
// implementation. Two separate codewords are represented by the hi and
// lo values in the sequence, and the row context generates a pair of
// output sequences.
//
// For each codeword, the row context determines the two smallest values,
// the index of the smallest, and the signs of the inputs, and then
// generates an output sequence as dictated by those values.
TEST_F(LDPCInternalHalf, RowContext_BoxPlus_fp16x2)
{
    do_box_plus_x2_test<ldpc2::box_plus_op>(random_src_half2,         // host data
                                            random_src_half2_device); // device data
}

////////////////////////////////////////////////////////////////////////
// test_rc_fp16_signs_dp_kernel()
// Test kernel for dot product-based sign extraction
__global__ void test_rc_fp16_signs_dp_kernel(int            num_tests,
                                             int            row_degree,
                                             uint32_t*      pout,
                                             const __half2* pin)
{
    const int      inputIdx  = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int      row_words = (row_degree + 1) / 2;
    const __half2* tIn       = pin  + (inputIdx * row_words);
    uint32_t*      tOut      = pout + inputIdx;
    //------------------------------------------------------------------
    if(inputIdx >= num_tests)
    {
        return;
    }
    //------------------------------------------------------------------
    // Load input data into registers
    word_t app[19]; // Maximum row degree
    for(int i = 0; i < row_words; ++i)
    {
        app[i].f16x2 = tIn[i];
    }
    //------------------------------------------------------------------
    //uint32_t s = dp_signs_packed(app[0], app[1]);
    //uint32_t s = dp_signs_spread(app[0], app[1]);
    //printf("a[3] = %f a[2] =  %f, a[1] =  %f, a[0] =  %f, signs = 0x%X\n",
    //       __high2float(app[1].f16x2),
    //       __low2float(app[1].f16x2),
    //       __high2float(app[0].f16x2),
    //       __low2float(app[0].f16x2),
    //       s);
    //------------------------------------------------------------------
    uint32_t s;
    switch(row_degree)
    {
    case 3:
        s =   dp_signs_spread<3>(app[0], app[1]);
        break;
    case 4:
        s =   dp_signs_spread<4>(app[0], app[1]);
        break;
    case 5:
        s =   dp_signs_spread<4>(app[0], app[1]);
        s += (dp_signs_spread<1>(app[2], app[3]) * 256);
        break;
    case 6:
        s =   dp_signs_spread<4>(app[0], app[1]);
        s += (dp_signs_spread<2>(app[2], app[3]) * 256);
        break;
    case 7:
        s =   dp_signs_spread<4>(app[0], app[1]);
        s += (dp_signs_spread<3>(app[2], app[3]) * 256);
        break;
    case 8:
        s =   dp_signs_spread<4>(app[0], app[1]);
        s += (dp_signs_spread<4>(app[2], app[3]) * 256);
        break;
    case 9:
        s =   dp_signs_spread<4>(app[0], app[1]);
        s += (dp_signs_spread<4>(app[2], app[3]) * 256);
        s += (dp_signs_spread<1>(app[4], app[5]) * 65536);
        break;
    case 10:
        s =   dp_signs_spread<4>(app[0], app[1]);
        s += (dp_signs_spread<4>(app[2], app[3]) * 256);
        s += (dp_signs_spread<2>(app[4], app[5]) * 65536);
        break;
    case 19:
        s =   dp_signs_packed<4>(app[0], app[1]);
        s += (dp_signs_packed<4>(app[2], app[3]) * 16);
        s += (dp_signs_packed<4>(app[4], app[5]) * 256);
        s += (dp_signs_packed<4>(app[6], app[7]) * 4096);
        s += (dp_signs_packed<3>(app[8], app[9]) * 65536);
        break;
    }
    *tOut = s;
}

////////////////////////////////////////////////////////////////////////
// host_signs
// Host code to extract sign bits, for comparison with GPU implementation.
// Bits are "packed" for row degree 19, and "high bit spread" for all
// other row degrees.
void host_signs(int            N,
                uint32_t*      dst,
                const __half2* src)
{
    uint32_t s = 0;
    //------------------------------------------------------------------
    for(int i = 0; i < (N + 1) / 2; ++i)
    {
        float val = __low2float(src[i]);
        unsigned int sbit = signbit(val) ? 1 : 0;
        if(N == 19)
        {
            s |= (sbit << (i*2));
        }
        else
        {
            s |= (sbit << (i*4 + 1));
        }
        //printf("%i: %f, signbit = %u, s = 0x%X\n", i*2, val, sbit, s);
        if(((i * 2) + 1) < N)
        {
            val = __high2float(src[i]);
            sbit = signbit(val) ? 1 : 0;
            if(N == 19)
            {
                s |= (sbit << (i*2 + 1));
            }
            else
            {
                s |= (sbit << (i*4 + 3));
            }
            //printf("%i: %f, signbit = %u, s = 0x%X\n", i*2+1, val, sbit, s);
        }
    }
    *dst = s;
}

////////////////////////////////////////////////////////////////////////
// LDPCInternalHalf.DotProductSigns
// Test the device function that determines the signs of 3 fp16 inputs
// using a combination of permute and dot product functions
TEST_F(LDPCInternalHalf, DotProductSigns)
{
    std::vector<uint32_t> hostRef(random_src_half2.size());
    typedef cuphy::buffer<uint32_t, cuphy::device_alloc> uint32_buffer_device_t;
    uint32_buffer_device_t k_out;
    //------------------------------------------------------------------
    // Loop over all 5G row degrees (BG1 and BG2)
    std::array<int, 9> row_degrees{3, 4, 5, 6, 7, 8, 9, 10, 19};
    //std::array<int, 1> row_degrees{19};
    //std::array<int, 1> row_degrees{3};
    for(auto row_degree : row_degrees)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // How many tests can we run for the amount of allocated data,
        // for the current row degree?
        const size_t PAIRS_PER_ROW = (row_degree + 1) / 2;
        const size_t NUM_ROW_TESTS = random_src_half2.size() / PAIRS_PER_ROW;

        DEBUG_PRINTF("row_degree = %i, num_half2 = %lu, num_tests = %lu\n", row_degree, random_src_half2.size(), NUM_ROW_TESTS);

        k_out = uint32_buffer_device_t(NUM_ROW_TESTS);
        hostRef.resize(NUM_ROW_TESTS);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate reference data on the host
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            const __half2* hostSrc = random_src_half2.data() + (PAIRS_PER_ROW * i);
            uint32_t*      hostDst = hostRef.data()          + i;
            host_signs(row_degree, hostDst, hostSrc);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Execute the kernel on the device
        dim3 blkDim(32);
        dim3 grdDim((NUM_ROW_TESTS + (blkDim.x - 1)) / blkDim.x);
        test_rc_fp16_signs_dp_kernel<<<grdDim, blkDim>>>(NUM_ROW_TESTS,
                                                         row_degree,
                                                         k_out.addr(),
                                                         random_src_half2_device.addr());
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Transfer results back to the host
        std::vector<uint32_t> deviceResults(k_out.size());
        CUDA_CHECK(cudaMemcpy(deviceResults.data(),
                              k_out.addr(),
                              k_out.size() * sizeof(uint32_buffer_device_t::element_t),
                              cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Compare results
        for(size_t i = 0; i < NUM_ROW_TESTS; ++i)
        {
            //printf("row_degree = %i, test = %lu, device = 0x%X, host = 0x%X\n", row_degree, i, deviceResults[i], hostRef[i]);
            EXPECT_EQ(deviceResults[i], hostRef[i]);
        }
    }
}

#if CUPHY_INTERNAL_BUILD
#include "test_ldpc_internal_rc_sm80.cuh"
#include "test_ldpc_internal_rc_sm86.cuh"
#endif // CUPHY_INTERNAL_BUILD

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
