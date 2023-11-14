/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "fading_chan.cuh"

/**
 * @brief dispaly usage of fading channel test
 * 
 */
void usage() {
    std::cout << "Fading channel test [options]" << std::endl;
    std::cout << "  Options:" << std::endl;

    std::cout << "     inputFilename fadingMode randSeed nIter useFp16Flag (Input HDF5 filename, AWGN<0> or TDL<1>, random seed, num of iterations, use FP16<1> or FP32<0>)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./fading_chan_ex <path to TV> 0 0 10 1(AWGN, seed 0, 10 iterations, using FP16)" << std::endl;
    std::cout << "      ./fading_chan_ex <path to TV> 1 0 10 0 (TDL, seed 0, 10 iterations, using FP16)" << std::endl;
    std::cout << "      ./fading_chan_ex <path to TV> 1 0 10 0 (TDL, seed 0, 10 iterations, using FP32)" << std::endl;
}

/**
 * @brief main test function for fading channel
 * 
 * @tparam Tcomplex Template for complext number, cuPHY tensor type and the scalar type will be automatically decided
 * @param iFileName input TV file name to read carier pars, channel pars, freq rx smaples
 * @param nIter number of iteration to measure time
 */
template<typename Tcomplex>
void test_fadeChan(std::string iFileName, int nIter = 10, uint8_t fadingMode = 1, uint16_t randSeed = 0, uint8_t phyChannType = 1)
{
    cuphy::stream cuStrmMain;
    // Input file
    hdf5hpp::hdf5_file inputFile = hdf5hpp::hdf5_file::open(iFileName.c_str());
    /*------------------------- Creat buffer --------------------------------*/
    hdf5hpp::hdf5_dataset dset_carrier  = inputFile.open_dataset("carrier_pars");
    hdf5hpp::hdf5_dataset_elem dset_elem = dset_carrier[0];
    uint16_t N_sc                   = dset_elem["N_sc"].as<uint16_t>();
    uint16_t N_txLayer              = dset_elem["N_txAnt"].as<uint16_t>();
    uint16_t N_rxLayer              = dset_elem["N_rxAnt"].as<uint16_t>();
    uint16_t N_symble_slot          = dset_elem["N_symble_slot"].as<uint16_t>();
    // creat freqData Gpu tensor
    // get cuPHY tensor type based on template parameters
    cuphyDataType_t fadingTestCuphyTensorType;
    if(typeid(Tcomplex) == typeid(__half2))
    {
        fadingTestCuphyTensorType = CUPHY_C_16F;
        #ifdef PRINT_FADING_CHAN_TIME_
        printf("Using float 16 precision \n");
        #endif
    }
    else if(typeid(Tcomplex) == typeid(cuComplex))
    {
        fadingTestCuphyTensorType = CUPHY_C_32F;
        #ifdef PRINT_FADING_CHAN_TIME_
        printf("Using float 32 precision \n");
        #endif
    }
    else
    {
        printf("Error: Unsopported precision, only FP32 and FP16 are supported \n");
        exit(1);
    }

    cuphy::tensor_device freqTxGpu(fadingTestCuphyTensorType, N_sc, N_symble_slot, N_txLayer, cuphy::tensor_flags::align_tight);
    cuphy::tensor_device freqRxGpu(fadingTestCuphyTensorType, N_sc, N_symble_slot, N_rxLayer, cuphy::tensor_flags::align_tight);

    /*------------------------ Creat fading channel ------------------------*/
    // uint8_t fadingMode, uint16_t randSeed
    fadingChan<Tcomplex> * fadeChanPtr = new fadingChan<Tcomplex>(static_cast<Tcomplex*>(freqTxGpu.addr()), static_cast<Tcomplex*>(freqRxGpu.addr()), cuStrmMain.handle(), fadingMode, randSeed, phyChannType);

    fadeChanPtr -> setup(inputFile);

    /*------------------------ measure time ------------------------*/
    float elapsedTime;
    std::vector<float> elapsedTimeVec; // save run time per TTI
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float TTIlen = 0.0005f; // legnth of each TTI, assuming numorology 1
    float targetSNR = 10.0f; // target SNR in dB

    for(int TTIIdx=0; TTIIdx<nIter; TTIIdx++)
    {
        #ifdef PRINT_FADING_CHAN_TIME_
        printf("Running TTI %d \n", TTIIdx);
        #endif
        cudaEventRecord(start, cuStrmMain.handle());

        fadeChanPtr -> run(TTIlen * TTIIdx, targetSNR);

        cudaEventRecord(stop, cuStrmMain.handle());
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaStreamSynchronize(cuStrmMain.handle());
        cudaEventElapsedTime(&elapsedTime, start, stop);
        elapsedTimeVec.push_back(elapsedTime);

        // optional test: to check SNR per antenna
        // report average SNR over all antennas & save SNRs to "SNR.txt" file during savefadingChanToFile() if called;
        fadeChanPtr -> calSnr(13, 0, 240);
    }

    #ifdef PRINT_FADING_CHAN_TIME_
    // print running time info
    printf("Total fading channel process cost: %f milisecond (avg over %d runs) \n", std::reduce(elapsedTimeVec.begin(), elapsedTimeVec.end())/float(elapsedTimeVec.size()), elapsedTimeVec.size());

    for(auto x: elapsedTimeVec)
    {
      printf("%f, ", x);
    }
    printf("\n");
    #endif

    /*-----------------Save to Rx freq to file -------------------*/
    fadeChanPtr -> savefadingChanToFile();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Fail: CUDA error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Success: fading channel runs without errors\n");
    }

    delete fadeChanPtr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[])
{  
    std::string iFileName; // input TV name
    uint8_t fadingMode = 1; // fading mode: 0 for AWGN, 1 for TDL
    uint16_t randSeed = 0; // random seeds, used for TDL and generating noise
    int nIter = 10;  // number of iterations
    bool useFp16Flag = false; // 1 for use FP16, 0 for use FP32
    uint8_t phyChannType = 1; // 0 - PUSCH, 1 - PUCCH, 2 - PRACH

    // arguments parser
    // inputFilename fadingMode randSeed nIter useFp16Flag
    switch (argc)
    {
        case 2: 
            iFileName.assign(argv[1]);
            break;
        
        case 3: 
            iFileName.assign(argv[1]);
            fadingMode = std::__cxx11::stoi(argv[2]);
            break;

        case 4: 
            iFileName.assign(argv[1]);
            fadingMode = std::__cxx11::stoi(argv[2]);
            randSeed = std::__cxx11::stoi(argv[3]);
            break;

        case 5: 
            iFileName.assign(argv[1]);
            fadingMode = std::__cxx11::stoi(argv[2]);
            randSeed = std::__cxx11::stoi(argv[3]);
            nIter = std::__cxx11::stoi(argv[4]);
            break;

        case 6: 
            iFileName.assign(argv[1]);
            fadingMode = std::__cxx11::stoi(argv[2]);
            randSeed = std::__cxx11::stoi(argv[3]);
            nIter = std::__cxx11::stoi(argv[4]);
            useFp16Flag = std::__cxx11::stoi(argv[5]);
            break;

        default: {
            usage();
            exit(1);
        }
            break;
    }

    if(nIter <= 0) 
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations: {}. Should be > 0", nIter);
        exit(1);
    }

    // start testing based on TV, iter, and precision 
    if(useFp16Flag)
    {
        printf("FadingChan test: Using 16 bits precision, %s, random seed = %d, %d iterations \n", fadingMode ? "TDL" : "AWGN", randSeed, nIter);
        test_fadeChan<__half2>(iFileName, nIter, fadingMode, randSeed, phyChannType);        
    }
    else
    {
        printf("FadingChan test: Using 32 bits precision, %s, random seed = %d, %d iterations \n", fadingMode ? "TDL" : "AWGN", randSeed, nIter);
        test_fadeChan<cuComplex>(iFileName, nIter, fadingMode, randSeed, phyChannType);
    }

    return 0;
}
