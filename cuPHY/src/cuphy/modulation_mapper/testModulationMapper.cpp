/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "cuphy.h"
#include "cuphy_internal.h"
#include "cuphy.hpp"

//using namespace std;
using namespace cuphy_i;

struct TestParams {
    int num_symbols;
    int modulation_order;
    unsigned int seed;
    float beta_qam;
};

template <typename T>
T symbol_modulation(uint32_t shifted_input_element, uint32_t modulation_order, float beta_qam) {

    uint32_t symbol_bits = shifted_input_element & ((1 << modulation_order) - 0x1U);
    T cpu_computed_symbol;

    std::vector<uint32_t> bit_vals(modulation_order);
    for (int i = 0; i < modulation_order; i++) {
        bit_vals[i] = ((symbol_bits >> i) & 0x1);
    }

    if (modulation_order == CUPHY_QAM_4) {
        // QPSK  modulation
	cpu_computed_symbol.x =  (int)(1 - 2*bit_vals[0]) / sqrt(2) * beta_qam;
	cpu_computed_symbol.y =  (int)(1 - 2*bit_vals[1]) / sqrt(2) * beta_qam;
    } else if (modulation_order == CUPHY_QAM_16) {
        // QAM16 modulation
	cpu_computed_symbol.x =  (int)((1 - 2*bit_vals[0]) * (1 + 2*bit_vals[2])) / sqrt(10) * beta_qam;
	cpu_computed_symbol.y =  (int)((1 - 2*bit_vals[1]) * (1 + 2*bit_vals[3])) / sqrt(10) * beta_qam;
    } else if (modulation_order == CUPHY_QAM_64) {
        // QAM64 modulation
        cpu_computed_symbol.x =  (int)((1 - 2*bit_vals[0]) * (4 - (1 - 2*bit_vals[2]) * (1 + 2*bit_vals[4]))) / sqrt(42) * beta_qam;
        cpu_computed_symbol.y =  (int)((1 - 2*bit_vals[1]) * (4 - (1 - 2*bit_vals[3]) * (1 + 2*bit_vals[5]))) / sqrt(42) * beta_qam;
    } else if (modulation_order == CUPHY_QAM_256) {
        // QAM256 modulation
        cpu_computed_symbol.x =  (int)((1 - 2*bit_vals[0]) * (8 - (1 - 2*bit_vals[2]) * (4 - (1 - 2*bit_vals[4]) * (1 + 2*bit_vals[6])))) / sqrt(170) * beta_qam;
        cpu_computed_symbol.y =  (int)((1 - 2*bit_vals[1]) * (8 - (1 - 2*bit_vals[3]) * (4 - (1 - 2*bit_vals[5]) * (1 + 2*bit_vals[7])))) / sqrt(170) * beta_qam;
    } else {
        std::cout << "Unsupported moudlation order " << modulation_order << std::endl;
        cpu_computed_symbol.x = 0.0;
        cpu_computed_symbol.y = 0.0;
    }

    return cpu_computed_symbol;
}


int reference_comparison(std::vector<uint32_t> & h_modulation_input, std::vector<__half2> & h_modulation_output, int modulation_order, int num_symbols, float beta_qam) {

    const int uint32_t_bits = 32;
    int mismatch_cnt = 0;
    const float tolerance = 0.0001f; //update comparison tolerance as needed.

    for (int symbol_id = 0; symbol_id < num_symbols; symbol_id++) {
        int input_element = (symbol_id * modulation_order) / uint32_t_bits;
        int symbol_start_bit = (symbol_id * modulation_order) % uint32_t_bits;
        uint32_t shifted_input_element = (h_modulation_input[input_element] >> symbol_start_bit);

        if (modulation_order == CUPHY_QAM_64) {
            if (symbol_start_bit == 28) {
                shifted_input_element &= 0x0FU;
                shifted_input_element |= ((h_modulation_input[input_element + 1] & 0x03U) << 4);
            } else if (symbol_start_bit == 30) {
                shifted_input_element &= 0x03U;
                shifted_input_element |= ((h_modulation_input[input_element + 1] & 0x0FU) << 2);
            }
        }

        __half2 cpu_computed_symbol = symbol_modulation<__half2>(shifted_input_element, modulation_order, beta_qam);

        if (!complex_approx_equal<__half2, __half>(cpu_computed_symbol, h_modulation_output[symbol_id], __float2half(beta_qam * tolerance))) {
            if (mismatch_cnt == 0) {
                //std::cout << "Mismatch for symbol " << symbol_id << " = " << std::hex << shifted_input_element;
                std::cout << "First Mismatch for symbol " << symbol_id << " = " << std::hex << shifted_input_element;
                std::cout << ": CPU val. " << (float) cpu_computed_symbol.x << " + i " << (float) cpu_computed_symbol.y;
                std::cout << " vs. GPU val. " << (float) h_modulation_output[symbol_id].x  << " + i " << (float) h_modulation_output[symbol_id].y << std::dec << std::endl;
            }

            mismatch_cnt += 1;
        }
    }

    std::cout << "Found " << mismatch_cnt << " mismatches out of " << num_symbols << " symbols." << std::endl;
    return mismatch_cnt;
}

void test_modulation(TestParams & test_params, int & gpu_mismatch, int num_iterations) {

    cudaStream_t strm = 0;

    // Randomly populate
    srand(test_params.seed);

    int num_symbols = test_params.num_symbols;
    int modulation_order = test_params.modulation_order;
    float beta_qam = test_params.beta_qam;

    if ((modulation_order != CUPHY_QAM_4) && (modulation_order != CUPHY_QAM_16) &&
       (modulation_order != CUPHY_QAM_64) && (modulation_order != CUPHY_QAM_256)) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Error: Invalid Modulation order {} is not supported", modulation_order);
        return;
    }

    int num_TBs = 1;
    int num_bits = num_symbols * modulation_order;
    int input_elements = div_round_up<uint32_t>(num_bits, 8*sizeof(uint32_t));
    unique_device_ptr<uint32_t> modulation_input = make_unique_device<uint32_t>(input_elements);
    std::vector<PdschPerTbParams> h_workspace(num_TBs);
    std::vector<PdschDmrsParams> h_pdschDmrsParams(num_TBs);
    for (int i = 0; i < num_TBs; i++) {
        h_workspace[i].G = num_bits;
        h_workspace[i].Qm = modulation_order;
        //TODO remaining h_workspace fields unitialized. Unused in modulation kernel.

        h_pdschDmrsParams[i].beta_qam = beta_qam;
        h_pdschDmrsParams[i].num_Rbs = 0; // force legacy flat input/output - we will likely deprecate that mode
        //TODO remaining h_pdschDmrsParams fields uninitialized. Unused in modulation kernel.
    }
    unique_device_ptr<PdschPerTbParams> d_workspace = make_unique_device<PdschPerTbParams>(num_TBs);
    unique_device_ptr<PdschDmrsParams> d_pdschDmrsParams = make_unique_device<PdschDmrsParams>(num_TBs);
    cuphyTensorDescriptor_t input_desc, output_desc;
    cuphyCreateTensorDescriptor(&input_desc);
    cuphyCreateTensorDescriptor(&output_desc);
    int input_dims[1] = {input_elements};
    int output_dims[1] = {num_symbols};
    cuphySetTensorDescriptor(input_desc, CUPHY_R_32U, 1, input_dims, nullptr, 0);
    cuphySetTensorDescriptor(output_desc, CUPHY_C_16F, 1, output_dims, nullptr, 0);

    unique_device_ptr<__half2> modulation_output = make_unique_device<__half2>(num_symbols);

    //Randomly populate modulation_input.
    std::vector<uint32_t> h_modulation_input(input_elements);
    for (int i = 0; i < input_elements; i++) {
        h_modulation_input[i] = rand();
    }
    CUDA_CHECK(cudaMemcpy(modulation_input.get(), h_modulation_input.data(), input_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace.get(), h_workspace.data(), num_TBs * sizeof(PdschPerTbParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pdschDmrsParams.get(), h_pdschDmrsParams.data(), num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice));

    int max_bits_per_layer = 0; // Does not matter when cuphyModulation's 1st arg. is nullptr. Otherwise it's max(G/Nl) across all TBs.

    // Allocate launch config struct.
    std::unique_ptr<cuphyModulationLaunchConfig> modulation_hndl = std::make_unique<cuphyModulationLaunchConfig>();

    // Allocate descriptors and setup modulation mapper component
    uint8_t desc_async_copy = 1; // Copy descriptor to the GPU during setup

    size_t desc_size=0, alloc_size=0;
    cuphyStatus_t status = cuphyModulationGetDescrInfo(&desc_size, &alloc_size);
    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("cuphyModulationGetDescrInfo error");
    }
    cuphy::unique_device_ptr<uint8_t> d_modulation_desc = cuphy::make_unique_device<uint8_t>(desc_size);
    cuphy::unique_pinned_ptr<uint8_t> h_modulation_desc = cuphy::make_unique_pinned<uint8_t>(desc_size);

    // Calling cuphySetupModulation with d_params == nullptr is permitted, and assumes
    // output is a contiguous buffer rather than a 3D {3276, 14, 4} tensor as in the PdschTx example.
    status = cuphySetupModulation(modulation_hndl.get(),
                                  //nullptr,
                                  d_pdschDmrsParams.get(),
                                  input_desc,
                                  modulation_input.get(),
                                  num_symbols,
                                  max_bits_per_layer,
                                  num_TBs,
                                  d_workspace.get(),
                                  output_desc,
                                  modulation_output.get(),
                                  h_modulation_desc.get(),
                                  d_modulation_desc.get(),
                                  desc_async_copy,
                                  strm);
    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cuphySetupModulation");
    }

    //A dummy call is needed or otherwise the event measurement for the first call is incorrect.
    CUresult r = launch_kernel(modulation_hndl.get()->m_kernelNodeParams, strm);

    float time1 = 0.0;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start, cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreate(&stop, cudaEventBlockingSync));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < num_iterations; iter++) {
         // Run Modulation
         r = launch_kernel(modulation_hndl.get()->m_kernelNodeParams, strm);
    }

    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "CUDA Error {}", cudaGetErrorString(cuda_error));
    }

    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time1, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cupphyRunModulation");
    }

    time1 /= num_iterations;


    printf("Modulation Mapper Kernel: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);

    std::vector<__half2> h_modulation_output(num_symbols);
    CUDA_CHECK(cudaMemcpy(h_modulation_output.data(), modulation_output.get(), num_symbols * sizeof(__half2), cudaMemcpyDeviceToHost));
    gpu_mismatch = reference_comparison(h_modulation_input, h_modulation_output, modulation_order, num_symbols, beta_qam);

}

class ModulationMapperTest: public ::testing::TestWithParam<TestParams> {
public:
    void basicTest() {
        params = ::testing::TestWithParam<TestParams>::GetParam();
        test_modulation(params, gpu_mismatch, num_iterations);
    }

    void SetUp() override {basicTest(); }

    void TearDown() override {
        gpu_mismatch = -1;
    }

protected:
    TestParams params;
    int gpu_mismatch = -1;
    int num_iterations = 20;
};

TEST_P(ModulationMapperTest, CONFIGS) {
    //EXPECT_EQ(0, gpu_mismatch);
    ASSERT_TRUE(gpu_mismatch == 0);
}

const std::vector<TestParams> CONFIGS = {
    /* number of symbols, modulation order, random seed to initialize input bits, scale factor */

    {100, CUPHY_QAM_4, 2019, 1.0},
    {235872, CUPHY_QAM_4, 1024, 1.0},
    {10000001, CUPHY_QAM_4, 1024, 1.0},

    {100, CUPHY_QAM_16, 2019, 1.0},
    {235872, CUPHY_QAM_16, 1024, 1.0},
    {10000001, CUPHY_QAM_16, 1024, 1.0},

    {100, CUPHY_QAM_64, 2019, 1.0},
    {235872, CUPHY_QAM_64, 1024, 1.0},
    {10000001, CUPHY_QAM_64, 1024, 1.0},

    {100, CUPHY_QAM_256, 2019, 1.0},
    {36036, CUPHY_QAM_256, 2019, 1.0},
    {235872, CUPHY_QAM_256, 1024, 1.0},
    {10000001, CUPHY_QAM_256, 1024, 1.0},

    // Test each constellation with an arbitrary large scale factor too
    {100, CUPHY_QAM_4, 2019, 6123.56},
    {100, CUPHY_QAM_16, 2019, 6123.56},
    {100, CUPHY_QAM_64, 2019, 6123.56},
    {100, CUPHY_QAM_256, 2019, 6123.56}
};

INSTANTIATE_TEST_CASE_P(ModulationMapperTests, ModulationMapperTest,
                        ::testing::ValuesIn(CONFIGS));

template <typename TOut>
void do_modulate_symbol_test(int LOG2_QAM, int NUM_BITS, int NUM_COLS)
{
    const int NUM_SYMBOLS = NUM_BITS / LOG2_QAM;

    //printf("NUM_BITS = %i, NUM_SYMBOLS = %i\n", NUM_BITS, NUM_SYMBOLS);
    
    const cuphyDataType_t SYM_TYPE = cuphy::type_to_cuphy_type<TOut>::value;

    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;
    typedef cuphy::typed_tensor<SYM_TYPE,  cuphy::pinned_alloc> tensor_sym_p;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    const std::array<int, 2> SRC_DIMS = {{NUM_BITS,    NUM_COLS}};
    const std::array<int, 2> DST_DIMS = {{NUM_SYMBOLS, NUM_COLS}};
    tensor_bit_p             tSrc(cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr), cuphy::tensor_flags::align_coalesce);
    tensor_sym_p             tDst(cuphy::tensor_layout(DST_DIMS.size(), DST_DIMS.data(), nullptr), cuphy::tensor_flags::align_coalesce);
    //printf("bit_p: %s\n", tSrc.desc().get_info().to_string().c_str());
    //printf("sym_p: %s\n", tDst.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize the source tensor with random bit values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Modulate the bits
    cuphy::modulate_symbol(tDst, tSrc, LOG2_QAM);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    int      NUM_WORDS = (NUM_BITS + 31) / 32;
    uint32_t MASK      = (1 << LOG2_QAM) - 1;
    for(int sym_idx = 0; sym_idx < NUM_SYMBOLS; ++sym_idx)
    {
        int      WORD_IDX    = (sym_idx * LOG2_QAM) / 32;
        int      WORD_OFFSET = (sym_idx * LOG2_QAM) % 32;
        for(int col = 0; col < SRC_DIMS[1]; ++col)
        {
            uint32_t lo          = tSrc(WORD_IDX, col);
            uint32_t hi          = tSrc(std::min(WORD_IDX + 1, NUM_WORDS), col);
            uint64_t hi_lo       = (static_cast<uint64_t>(hi) << 32) | lo;
            uint32_t symbol_bits = (hi_lo >> WORD_OFFSET) & MASK;
            TOut     ref_sym     = symbol_modulation<TOut>(symbol_bits, LOG2_QAM, 1.0f);
            float    eReal       = std::abs(static_cast<float>(ref_sym.x) - static_cast<float>(tDst(sym_idx, col).x));
            float    eImag       = std::abs(static_cast<float>(ref_sym.y) - static_cast<float>(tDst(sym_idx, col).y));
            EXPECT_LE(eReal, 1.0e-3f) << " sym_idx = "     << sym_idx
                                      << ", col = "        << col
                                      << ", bits = 0x"     << std::hex << symbol_bits << std::dec
                                      << ", expected = "   << static_cast<float>(ref_sym.x)
                                      << ", calculated = " << static_cast<float>(tDst(sym_idx, col).x)
                                      << std::endl;
            EXPECT_LE(eImag, 1.0e-3f) << " sym_idx = "     << sym_idx
                                      << ", col = "        << col
                                      << ", bits = 0x"     << std::hex << symbol_bits  << std::dec
                                      << ", expected = "   << static_cast<float>(ref_sym.y)
                                      << ", calculated = " << static_cast<float>(tDst(sym_idx, col).y)
                                      << std::endl;
            //printf("Symbol [%i, %i]: BITS: 0x%X  REF: %f %f TEST: %f %f\n",
            //       sym_idx, col,
            //       symbol_bits,
            //       static_cast<float>(ref_sym.x), static_cast<float>(ref_sym.y),
            //       static_cast<float>(tDst(sym_idx, col).x), static_cast<float>(tDst(sym_idx, col).y));

        }
    }
}

////////////////////////////////////////////////////////////////////////
// ModulationMapperTest.ModulateSymbol
// cuPHY API utility function test, not used during real-time operation
TEST(ModulationMapperTest, ModulateSymbol)
{
    do_modulate_symbol_test<__half2>  (CUPHY_QAM_4, CUPHY_QAM_4 * 1024, 11);
    do_modulate_symbol_test<cuComplex>(CUPHY_QAM_4, CUPHY_QAM_4 * 373,  2);
    
    do_modulate_symbol_test<__half2>  (CUPHY_QAM_16, CUPHY_QAM_16 * 2048, 1);
    do_modulate_symbol_test<cuComplex>(CUPHY_QAM_16, CUPHY_QAM_16 * 99,   32);

    do_modulate_symbol_test<__half2>  (CUPHY_QAM_64, CUPHY_QAM_64 * 16, 2);
    do_modulate_symbol_test<cuComplex>(CUPHY_QAM_64, CUPHY_QAM_64 * 1,  9);

    do_modulate_symbol_test<__half2>  (CUPHY_QAM_256, CUPHY_QAM_256 * 512, 18);
    do_modulate_symbol_test<cuComplex>(CUPHY_QAM_256, CUPHY_QAM_256 * 11,  2);
}

//TODO add test cases + code to test non identical TB configs too

int main(int argc, char** argv) {

#if 1
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
#else // To debug an individual config
    int gpu_mismatch = -1;
    TestParams params = {36036, CUPHY_QAM_256, 2019};
    test_modulation(params, gpu_mismatch, 1);

    return 0;
#endif
}
