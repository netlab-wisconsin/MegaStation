/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "cuphy_internal.h"

#include "ofdmMod.cuh"
#include "ofdmDemod.cuh"

#define USE_HALF16_PRECISION_ // whether half16 precision is used or not

#ifdef USE_HALF16_PRECISION_
#define OFDM_TEST_CUPHY_TENSOR_DATA_TYPE CUPHY_C_16F
#else
#define OFDM_TEST_CUPHY_TENSOR_DATA_TYPE CUPHY_C_32F
#endif

/**
 * @brief read caerrier info from h5 file data set element
 * 
 * @param CarrierPrms struct to hold carrier info
 * @param dset_elem dataset element
 */
inline void carrier_pars_from_dataset_elem(cuphyCarrierPrms * CarrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_elem)
{
    CarrierPrms -> N_sc                   = dset_elem["N_sc"].as<uint16_t>();
    CarrierPrms -> N_FFT                  = dset_elem["N_FFT"].as<uint16_t>();
    CarrierPrms -> N_txLayer              = dset_elem["N_txLayer"].as<uint16_t>();
    CarrierPrms -> N_rxLayer              = dset_elem["N_rxLayer"].as<uint16_t>();
    CarrierPrms -> id_slot                = dset_elem["id_slot"].as<uint16_t>();
    CarrierPrms -> id_subFrame            = dset_elem["id_subFrame"].as<uint16_t>();
    CarrierPrms -> mu                     = dset_elem["mu"].as<uint16_t>();
    CarrierPrms -> cpType                 = dset_elem["cpType"].as<uint16_t>();
    CarrierPrms -> f_c                    = dset_elem["f_c"].as<uint32_t>();
    CarrierPrms -> f_samp                 = dset_elem["f_samp"].as<uint32_t>();
    CarrierPrms -> N_symble_slot          = dset_elem["N_symble_slot"].as<uint16_t>();
    CarrierPrms -> kappa_bits             = dset_elem["kappa_bits"].as<uint16_t>();
    CarrierPrms -> ofdmWindowLen          = dset_elem["ofdmWindowLen"].as<uint16_t>();
    CarrierPrms -> rolloffFactor          = dset_elem["rolloffFactor"].as<float>();
}


/**
 * @brief check ofdm results
 * if TV is used, we will compare both time and freq samples
 * otherwise, only compare freq tx with freq rx, whether they match
 * 
 * @param dataIn CPU buffer for comparison
 * @param dataOut CPU buffer for comparison
 * @param compareLen length of comparison
 * @return true if dataIn match with dataOut with tolerance
 * @return false if dataIn does match with dataOut with tolerance
 */
template<typename Tscalar, typename Tcomplex>
bool checkOfdmRes(Tcomplex * dataIn, Tcomplex * dataOut, int compareLen)
{
    const Tscalar tolerance = static_cast<Tscalar>(0.001f); //update tolerance as needed.
    for(int i=0; i<compareLen; i++)
    {
        if(!(complex_approx_equal<Tcomplex, Tscalar>(dataIn[i], dataOut[i], tolerance)))
        {
            printf("input and output samples do not match! starting %d\n", i);
            printf("In: %f + %f i, out %f + %f i\n", float(dataIn[i].x), float(dataIn[i].y), float(dataOut[i].x), float(dataOut[i].y));
            return false;
        }
    }
    return true;
}

/**
 * @brief printout usage message
 * 
 */
void usage() {
    std::cout << "OFDM modulation and demodulation [options]" << std::endl;
    std::cout << "  Options:" << std::endl;

    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename  num_iterations  (Input HDF5 filename, Number of iterations)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./ofdm_mod_demod_ex" << std::endl;
    std::cout << "      ./ofdm_mod_demod_ex /home/shaoranl/TVs/ofdm_tdl/ofdm_text_tv.h5 10" << std::endl;

}

/**
 * @brief test OFDM fucntion
 * using this function for quickly change from float32 to float16
 * 
 * @param argc, argv input arguments
 */
template<typename Tscalar, typename Tcomplex>
void test_OFDM(int argc, char* argv[])
{
    srand(time(NULL));
    int num_iterations;

    switch (argc)
    {
    case 1:
        num_iterations = 10;
        break;
    
    case 3:
        num_iterations = std::__cxx11::stoi(argv[2]);
        if (num_iterations <= 0) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations: {}. Should be > 0", num_iterations);
            exit(1);
        }
        break;

    default:
        usage();
        exit(1);
        break;
    }

    cuphy::stream cuStrmMain;

    bool prach = 0;

    cuphyCarrierPrms * CarrierPrms = new cuphyCarrierPrms;
    /* ----   Config carrierPrms based on TV ---------*/
    if(argc > 1)
    {
        hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(argv[1]);
        hdf5hpp::hdf5_dataset dset_gnb  = input_file.open_dataset("carrier_pars");
        carrier_pars_from_dataset_elem(CarrierPrms, dset_gnb[0]);
    }

    uint  blocks = (CarrierPrms -> N_txLayer) * (CarrierPrms -> N_symble_slot);
    CarrierPrms -> ofdmWindowLen = 0; // ofdm windowing effect not tested
    const uint  freqDataSize = (CarrierPrms -> N_sc) * blocks;
    // creat freqDataInGpu tensor
    cuphy::tensor_device freqDataInGpu(OFDM_TEST_CUPHY_TENSOR_DATA_TYPE, freqDataSize, cuphy::tensor_flags::align_tight);
    cuphy::tensor_device freqDataOutGpu(OFDM_TEST_CUPHY_TENSOR_DATA_TYPE, freqDataSize, cuphy::tensor_flags::align_tight);

    ofdm_modulate::ofdmModulate<Tscalar, Tcomplex> * ofdmMod = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(CarrierPrms, static_cast<Tcomplex*>(freqDataInGpu.addr()), cuStrmMain.handle());

    const uint timeDataSize = ofdmMod -> getTimeDateLen();
    Tcomplex * freqDataInCpu = new Tcomplex[freqDataSize];
    Tcomplex * freqDataOutCpu = new Tcomplex[freqDataSize];

    // printf("timeDataSize = %d \n", timeDataSize);
    Tcomplex * timeDataOutCpu = new Tcomplex[timeDataSize];
    Tcomplex * timeDataOutCpu_ref = new Tcomplex[timeDataSize];

    if(argc > 1)
    {
        // Read input HDF5 file to read rate-matching output.
        printf("Using TV to test OFDM, time and frequency domain ref check enabled \n");
        hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(argv[1]);
        hdf5hpp::hdf5_dataset Xtf_dataset = input_file.open_dataset("X_tf");
        Xtf_dataset.read(freqDataInCpu);//freqDataInGpu.addr());
        cudaMemcpyAsync(freqDataInGpu.addr(), freqDataInCpu, sizeof(Tcomplex)*freqDataSize, cudaMemcpyHostToDevice, cuStrmMain.handle());

        // read reference time domain date
        hdf5hpp::hdf5_dataset Xt_dataset = input_file.open_dataset("X_t");
        Xt_dataset.read(timeDataOutCpu_ref);//freqDataInGpu.addr());        
    }
    else
    {
        // use random generated data as frequency input
        printf("Using random input to test OFDM, only frequency domain ref check enabled \n");
        cuphy::rng rng;
        cuComplex m32_0      = make_cuFloatComplex(0.0f, 0.0f);
        cuComplex stddev32_1 = make_cuFloatComplex(sqrt(0.5f), sqrt(0.5f));
        rng.normal(freqDataInGpu, m32_0, stddev32_1, cuStrmMain.handle());
    }
    cudaStreamSynchronize(cuStrmMain.handle());

    // get time signal from OFDM modulate for OFDM demodulation
    Tcomplex * timeDataInGpu = ofdmMod -> getTimeDataOut();

    ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex> * ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(CarrierPrms, timeDataInGpu, static_cast<Tcomplex*>(freqDataOutGpu.addr()), prach, cuStrmMain.handle());

    // measure time
    float elapsedTime;
    std::vector<float> measureTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int sta=0; sta<num_iterations; sta++)
    {
        cudaEventRecord(start, cuStrmMain.handle());

        // OFDM modulation
        ofdmMod -> run(cuStrmMain.handle());
        
        // OFDM demodulation
        ofdmDeMod -> run(cuStrmMain.handle());

        #ifdef PRINT_SAVE_SAMPLE_OFDM_DATA_
        ofdmMod -> printTimeSample();
        ofdmDeMod -> printFreqSample();
        #endif

        cudaEventRecord(stop, cuStrmMain.handle());
        CUDA_CHECK(cudaEventSynchronize(start));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&elapsedTime, start, stop);
        measureTime.push_back(elapsedTime);
    }

    /*-----------------        check results match       --------------*/
    // check whether ofdm modulation input vs ofdm demodulation out 
    cudaMemcpy(freqDataInCpu, freqDataInGpu.addr(), freqDataSize * sizeof(Tcomplex), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaMemcpy(freqDataOutCpu, ofdmDeMod -> getFreqDataOut(), freqDataSize * sizeof(Tcomplex), cudaMemcpyDeviceToHost));
    if(checkOfdmRes<Tscalar, Tcomplex>(freqDataInCpu, freqDataOutCpu, freqDataSize))
    {
        printf("OFDM test PASS, frequency input and output samples match \n");
    }
    else
    {
        printf("OFDM test FAIL, frequency input and output samples do not match \n");
    }

    // checkofdm modulation output vs ref TV 
    if(argc > 1)
    {
        CUDA_CHECK(cudaMemcpy(timeDataOutCpu, timeDataInGpu, timeDataSize * sizeof(Tcomplex), cudaMemcpyDeviceToHost));
        if(checkOfdmRes<Tscalar, Tcomplex>(timeDataOutCpu_ref, timeDataOutCpu, timeDataSize))
        {
            printf("OFDM test PASS, time input and output samples match \n");
        }
        else
        {
            printf("OFDM test FAIL, time input and output samples do not match \n");
        }
    }

    // print running time info
    printf("Total OFDM modulation and demodulation time (include add and remove CP) cost: %f millisecond (avg over %ld runs) \n", std::reduce(measureTime.begin(), measureTime.end())/float(measureTime.size()), measureTime.size());
    
    // release allocated buffers
    delete[] freqDataInCpu;
    delete[] freqDataOutCpu;
    delete[] timeDataOutCpu;
    delete[] timeDataOutCpu_ref;

    delete CarrierPrms;
    delete ofdmMod;
    delete ofdmDeMod;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[])
{  
    #ifdef USE_HALF16_PRECISION_
    test_OFDM<__half, __half2>(argc, argv);  
    #else
    test_OFDM<float, cuComplex>(argc, argv);  
    #endif

    return 0;
}
