/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "tdl_chan.cuh"
#include <cassert>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"

#define USE_HALF16_PRECISION_ // whether half16 precision is used or not

#ifdef USE_HALF16_PRECISION_
#define TDL_TEST_CUPHY_TENSOR_DATA_TYPE_REAL CUPHY_R_16F
#define TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX CUPHY_C_16F
#else
#define TDL_TEST_CUPHY_TENSOR_DATA_TYPE_REAL CUPHY_R_32F
#define TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX CUPHY_C_32F
#endif

/**
 * @brief This function saves the tdl data into h5 file, for verification in matlab
 * 
 * @param tdlCfg tdl configurations
 * @param tdlChanPtr pointer to tdl channel class
 * @param strm cuda stream
 */
template <typename Tscalar, typename Tcomplex> 
void saveTdlChanToFile(tdlConfig_t* tdlCfg, tdlChan<Tscalar, Tcomplex> * tdlChanPtr, cudaStream_t strm);

template<typename Tscalar, typename Tcomplex>
void test_TDL(int argc, char* argv[])
{
    cudaStream_t cuMainStrm;
    cudaStreamCreate(&cuMainStrm);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    tdlConfig_t * tdlCfg = new tdlConfig_t;
    
    // change defualt paramters if needed
    // tdlCfg -> useSimplifiedPpd = false;
    // tdlCfg -> delayProfile = 'C';
    // tdlCfg -> delaySpread = 300;
    // tdlCfg -> maxDopplerShift = 100;
    // tdlCfg -> f_samp = 8192 * 15e3;
    // tdlCfg -> mimoCorrMat = NULL;
    // tdlCfg -> nTxAnt = 4;
    // tdlCfg -> nRxAnt = 4;
    // tdlCfg -> normChannOutput = true;
    // tdlCfg -> fBatch = 15e3;
    // tdlCfg -> numPath = 48;
    tdlCfg -> timeSigLenPerAnt = int(8192*1.5); // an example of ofdm output
    tdlCfg -> runMode = 1; // set to 1 will also generate freq channel

    cuphy::tensor_ref txTimeSigInTensor;
    cuphy::rng rng;
    cuComplex m32_0      = make_cuFloatComplex(0.0f, 0.0f);
    cuComplex stddev32_1 = make_cuFloatComplex(sqrt(0.5f), sqrt(0.5f));

    if(tdlCfg -> timeSigLenPerAnt) // randomly generate test input signal
    {
        uint txSigSize = (tdlCfg -> timeSigLenPerAnt) * (tdlCfg -> nTxAnt);
        cudaMalloc((void**) &(tdlCfg -> txTimeSigIn), sizeof(cuComplex) * txSigSize);

        /*  Set random input time signals   */

        txTimeSigInTensor.desc().set(TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX,  txSigSize, cuphy::tensor_flags::align_tight);
        txTimeSigInTensor.set_addr(tdlCfg -> txTimeSigIn);
    }
    else
    {
        tdlCfg -> txTimeSigIn = NULL;
    }
    
    /*---------------    Below tests multiple TDL channel class       --------------------*/
    // by default only the first one is inspected
    const int N_array = 1;
    tdlChan<Tscalar, Tcomplex> ** tdlChanTestArray = new tdlChan<Tscalar, Tcomplex> *[N_array];
    uint16_t randSeed = 0; // time(NULL)
    for(int i=0; i<N_array; i++)
    {
        tdlChanTestArray[i] = new tdlChan<Tscalar, Tcomplex>(tdlCfg, randSeed, cuMainStrm);
    }
    const int N_TTI = 5; 
    const float ttiLen = 0.0005;
    cudaEventRecord(start);
    for(int i=0; i<N_array; i++)
    {
        auto tdlChanTest = tdlChanTestArray[i];
        for(int ttiIdx=0; ttiIdx<N_TTI; ttiIdx++)
        {
            if(tdlCfg -> timeSigLenPerAnt)
            {
                rng.normal(txTimeSigInTensor, m32_0, stddev32_1, cuMainStrm);
            }
            tdlChanTest -> run(ttiIdx * ttiLen); // use current TTI time as reference time
            #ifdef PRINT_SAVE_SAMPLE_TDL_DATA_
            printf("TTI Idx = %d \n", ttiIdx);
            /*---------------    Below are optinonal save to file       --------------------*/
            if(ttiIdx == 0 || ttiIdx == N_TTI/2 || ttiIdx == N_TTI-1)
            {
                saveTdlChanToFile<Tscalar, Tcomplex>(tdlCfg, tdlChanTestArray[0], ttiIdx, cuMainStrm);
            }
            #endif
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Done! Time for generation input signal, tdl channel coes and process signals: " << milliseconds/N_TTI <<" ms"<< std::endl;

    #ifdef PRINT_SAVE_SAMPLE_TDL_DATA_
    /*---------------    Below are optinonal printout message       --------------------*/
    tdlChanTestArray[0] -> printTimeChan();
    if(tdlCfg -> timeSigLenPerAnt)
    {
        tdlChanTestArray[0] -> printRxTimeSig();
    }
    
    if(tdlCfg -> runMode)
    {
        tdlChanTestArray[0] -> printFreqSCChan();
        tdlChanTestArray[0] -> printFreqPRBGChan();
    }
    #endif
    /*---------------    clean up memory       --------------------*/
    cudaStreamDestroy(cuMainStrm);
    cudaFree(tdlCfg -> txTimeSigIn);
    delete tdlCfg;
    for(int i=0; i<N_array; i++)
    {
        delete tdlChanTestArray[i];
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] tdlChanTestArray;
}

int main(int argc, char* argv[])
{
    #ifdef USE_HALF16_PRECISION_
    test_TDL<__half, __half2>(argc, argv);  
    #else
    test_TDL<float, cuComplex>(argc, argv);  
    #endif

    return 0;
}

template <typename Tscalar, typename Tcomplex> 
void saveTdlChanToFile(tdlConfig_t * tdlCfg, tdlChan<Tscalar, Tcomplex> * tdlChanPtr, int & ttiIdx, cudaStream_t strm)
{
    tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescrCpu = (tdlChanPtr -> getTdlDynDescrCpu());
    int timeChanSize = tdlChanPtr -> getTimeChanSize();
    uint timeSigInLen = (tdlDynDescrCpu -> timeSigLenPerAnt)*(tdlDynDescrCpu -> nTxAnt);
    uint timeSigOutLen = (tdlDynDescrCpu -> timeSigLenPerAnt)*(tdlDynDescrCpu -> nRxAnt);
    std::string outFilename;
    outFilename = "tdlChan_" + std::to_string(tdlDynDescrCpu -> nTxAnt) + "x" + std::to_string(tdlDynDescrCpu -> nRxAnt) + "_" + tdlCfg -> delayProfile + std::to_string(int(tdlCfg -> delaySpread)) + "_dopp" + std::to_string(int(tdlCfg -> maxDopplerShift)) + "_cfo" + std::to_string(int(tdlCfg -> cfoHz)) + "_TTI" + std::to_string(ttiIdx) + ".h5";

    std::unique_ptr<hdf5hpp::hdf5_file> tdlFile;
    hdf5hpp::hdf5_file  tdlHdf5File;
    if(!outFilename.empty())
    {
        tdlFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outFilename.c_str())));
        tdlHdf5File = hdf5hpp::hdf5_file::open(outFilename.c_str());
    }
    
    // save tdl FIR and time channel coes
    cuphy::tensor_ref firNzIdxGpu_tensor;
    firNzIdxGpu_tensor.desc().set(CUPHY_R_16U, tdlDynDescrCpu -> firNzLen, cuphy::tensor_flags::align_tight);
    firNzIdxGpu_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> firNzIdx));
    cuphy::write_HDF5_dataset(tdlHdf5File, firNzIdxGpu_tensor,  "firNzIdx", strm);

    cuphy::tensor_ref firNzPwGpu_tensor;
    firNzPwGpu_tensor.desc().set(TDL_TEST_CUPHY_TENSOR_DATA_TYPE_REAL, tdlDynDescrCpu -> firNzLen, cuphy::tensor_flags::align_tight);
    firNzPwGpu_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> firNzPw));
    cuphy::write_HDF5_dataset(tdlHdf5File, firNzPwGpu_tensor, "firNzPw", strm);

    cuphy::tensor_ref timeChanGpu_tensor;
    timeChanGpu_tensor.desc().set(TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX, timeChanSize, cuphy::tensor_flags::align_tight);
    timeChanGpu_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> timeChan));
    cuphy::write_HDF5_dataset(tdlHdf5File, timeChanGpu_tensor, "timeChan", strm);

    // save frequency channel if exists
    if(tdlDynDescrCpu -> freqChanSize)
    {
        cuphy::tensor_ref freqChanSC_tensor;
        freqChanSC_tensor.desc().set(TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX, tdlDynDescrCpu -> freqChanSize, cuphy::tensor_flags::align_tight);
        freqChanSC_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> freqChanSC));
        cuphy::write_HDF5_dataset(tdlHdf5File, freqChanSC_tensor, "freqChanSC", strm);

        cuphy::tensor_ref freqChanPRBG_tensor;
        freqChanPRBG_tensor.desc().set(TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX, (tdlDynDescrCpu -> freqChanSize) / (tdlDynDescrCpu -> N_sc_PRBG), cuphy::tensor_flags::align_tight);
        freqChanPRBG_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> freqChanPRBG));
        cuphy::write_HDF5_dataset(tdlHdf5File, freqChanPRBG_tensor, "freqChanPRBG", strm);
    }

    // save time processing samples
    if(tdlDynDescrCpu -> timeSigLenPerAnt)
    {
        
        cuphy::tensor_ref txTimeSigIn_tensor;
        txTimeSigIn_tensor.desc().set(TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX, timeSigInLen, cuphy::tensor_flags::align_tight);
        txTimeSigIn_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> txTimeSigIn));
        cuphy::write_HDF5_dataset(tdlHdf5File, txTimeSigIn_tensor, "txTimeSigIn", strm);

        cuphy::tensor_ref rxTimeSigOut_tensor;
        rxTimeSigOut_tensor.desc().set(TDL_TEST_CUPHY_TENSOR_DATA_TYPE_COMPLEX, timeSigOutLen, cuphy::tensor_flags::align_tight);
        rxTimeSigOut_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> rxTimeSigOut));
        cuphy::write_HDF5_dataset(tdlHdf5File, rxTimeSigOut_tensor, "rxTimeSigOut", strm);
    }

    // save random numbers used to generate tdl
    uint randSize = (tdlDynDescrCpu -> nTxAnt) * (tdlDynDescrCpu -> nRxAnt) * (tdlDynDescrCpu -> firNzLen) * (tdlDynDescrCpu -> nPath);

    cuphy::tensor_ref thetaRand_tensor;
    thetaRand_tensor.desc().set(CUPHY_R_32F, randSize*2, cuphy::tensor_flags::align_tight);
    thetaRand_tensor.set_addr(static_cast<void*>(tdlDynDescrCpu -> thetaRand));
    cuphy::write_HDF5_dataset(tdlHdf5File, thetaRand_tensor, "thetaRand", strm);

    CUDA_CHECK(cudaStreamSynchronize(strm));

    tdlHdf5File.close();
}