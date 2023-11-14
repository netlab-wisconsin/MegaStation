/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <fstream>
#include <math.h>
#include <random>
#include "cuComplex.h"

#define MAX_NZ_TAPS_ 24 // max number of taps according to 3GPP specs
#define MAX_TX_RX_ANT_ 64 // max number supported for nTxAnt * nRxAnt
#define BLOCK_SAMPLE_ 128 // threads per block for processing tx samples. Note: no more than int(1024/tdlDynDescrCpu -> nRxAnt)
#define USE_MEMOERY_FFT_SHIFT_ // use fft shift in memeory or multiplication
#define FFTs_PER_BLOCK_CONST_ 1 // FFTs per block when generating freq domain channels on SC and PRBG, changing this will affect run times, should be a divisor of OFDM_SYMBOLS_PER_SLOT (14)
// #define PRINT_SAVE_SAMPLE_TDL_DATA_ // define whether sample TDL channel data is printed out and saved to h5 file

// macros for normalization
// By default, no need for post-generation normalization, ensure over long term E{||chanMatrix(:, rxAntIdx, :, batchIdx)||_2^2} = 1 for all (rxAntIdx, batchIdx) for all (rxAntIdx, batchIdx), see detials in "aerial_sdk/5GModel/nr_matlab/channel/genChan.m"
// #define ENABLE_NORMALIZATION_ // normalize tdl time chan coes per slot
#define THREADS_PER_BLOCK_NORMALIZATION_ 128 // # of threads per block in time channel normalization

/**
 * @brief struct to config tdl model
 * 
 * @todo no MIMO antenna correlations are added
 */
struct tdlConfig_t{
    bool useSimplifiedPdp = true; // true for simplified pdp in 38.141, false for 38.901
    char delayProfile = 'A';
    float delaySpread = 30;
    float maxDopplerShift = 5;
    float f_samp = 8192 * 15e3;
    uint16_t nTxAnt = 4;
    uint16_t nRxAnt = 4;
    uint fBatch = 15e3; // update rate of quasi-static channel
    uint numPath = 48;
    float cfoHz = 200;
    float delay = 0; // delay in second
    long timeSigLenPerAnt = 4096;
    void * txTimeSigIn; // GPU address of tx time signal 
    uint16_t freqConvertN_sc = 1200; // max 273 PRBs, setting this to 0 will not provide freq channel
    uint16_t N_sc_PRBG = 4*12; // # of PRBs per PRBG
    uint8_t runMode = 0; 
    // runMode 0: TDL time channel and processing tx sig
    // runMode 1: TDL time channel and frequency channel on SC and PRBG
};

/**
 * @brief TDL dynamic descriptor
 * 
 */
template <typename Tscalar, typename Tcomplex> 
struct tdlDynDescr_t
{
    // cell config params
    uint nTxAnt;
    uint nRxAnt;
    uint16_t nPath; // number of sins to add up
    uint16_t nBatch; // batch of channel coefficients to genearate
    uint nBatchSamp; // data samples per each batch in quasi-static tdl channel
    float tBatch;  // update rate of quasi-static channel
    float cfoHz; // CFO in Hz
    float cfoPhaseSamp; // sample period
    float maxDopplerShift;
    bool LosTap; // 0: first tap is NLOS, 1: first tap is LOS
    // channel input and output
    long timeSigLenPerAnt; // lenght of tx time signal per antenna
    Tcomplex * txTimeSigIn;
    Tcomplex * rxTimeSigOut;
    // FIR related params, only store non-zero parameters, as sparse matrix
    Tscalar * firNzPw; // non-zero coefficient of FIR, calculated on CPU
    uint16_t * firNzIdx; // non-zero indexes
    float * thetaRand; // rand phase for real and imag part due to doppler
    float PI_4_nPath; // a constant pi/4/nPath, calculated on CPU
    uint16_t firNzLen; // number of non-zero taps
    uint16_t firMaxLen; // maximum number of taps
    // TDL time channel coefficient
    Tcomplex * timeChan; // GPU memeory address of tdl channel in time domain, index: from big-> small: [nBatch, nRx, nTx, firNzLen]; E.g., index 1 is for(0,0,0,1);   index ((1*nRx+2)*nTx+3)*firNzLen+4 is (1 batch, 2 Rx, 3 Tx, 4 firNzIdx); Only non-zero chan coe are stored to save memory
    uint timeChanSize; // length of time domain coes
    // TDL freq channel coefficient
    uint N_FFT; // FFT size to generate freq channel
    uint N_sc;  // number of SC, read from " tdlConfig_t -> freqConvertN_sc ""
    uint N_PRBG; // number of PRBG, calculated by tdlChan class
    uint N_sc_PRBG; // number of SCs per PRBG, will be used to calculate N_PRBG
    float freqChanNormalizeCoe; // to normalize the frequency channel power
    Tcomplex * freqChanSC; // TDL frequency channel on SC, index: from big-> small: [nTx, nRx, nSC]; E.g., index 1 is for(0,0,1);  index (1*nRX+2)*nSc+3 is (1 Tx, 2 Rx, 3 SC)
    Tcomplex * freqChanPRBG; // TDL frequency channel on PRBG, index: from big-> small: [nTx, nRx, nPRBG]; 
    uint freqChanSize; // length of freq domain coes
    // for adding delay
    uint nDelaySample; // delay / T_sample 
};

/**
* @brief TDL channel class
*/
template <typename Tscalar, typename Tcomplex> class tdlChan{
public:
    /**
     * @brief Construct a new tdl Chan object
     * 
     * @param tdlConfig TDL chan configurations
     * @param randSeed random seed to generate tdl channel
     * @param strm cuda stream during config setup
     */
    tdlChan(tdlConfig_t * tdlCfg, uint16_t randSeed, cudaStream_t strm);
    ~tdlChan();

    /**
     * @brief generate tdl time chan and process the tx time signals
     * will generate tdl freq chan if runMode=1
     * will process tx time samples if TimeSigLenPerAnt > 0 
     * support time correlation so the time stamp should be input
     * 
     * @param refTime0 the time stamp for the start of tx symbol
     */
    void run(float refTime0 = 0.0f);
    /**
     * @brief update the random mag and phase in tdl chann
     * this help control the time domain correlation
     * by default it only runs during tdl chan setup
     */
    void updateTapPathRand();

    // generate channels
    void genTdltimeChan();
    void genTdlFreqChan();

    // process tx time domain signals
    void processTxSig();

    // obtain channel memory address
    Tcomplex * getTdltimeChan() {return tdlDynDescrCpu -> timeChan;}; 
    Tcomplex * getTdlfreqChanSC() {return tdlDynDescrCpu -> freqChanSC;}; 
    Tcomplex * getTdlfreqChanPRBG() {return tdlDynDescrCpu -> freqChanPRBG;};
    Tcomplex * getRxTimeSigOut(){return tdlDynDescrCpu -> rxTimeSigOut;}; // get output signals
    tdlDynDescr_t<Tscalar, Tcomplex> * getTdlDynDescrCpu() {return tdlDynDescrCpu;};
    int getTimeChanSize(){return m_timeChanSize;};

    // for printout samples
    void printTimeChan(int printLen = 10);
    void printFreqSCChan(int printLen = 10);
    void printFreqPRBGChan(int printLen = 10);
    void printRxTimeSig(int printLen = 10);

private:
    cudaStream_t m_strm; // cuda stream
    float m_delaySpread; 
    float m_maxDopplerShift;
    float m_f_samp; // sampling frequency
    float * m_mimoCorrMat; // MIMO correlation matrix, currently not used
    bool m_LosTap; // whether the first tap is LOS
    curandGenerator_t m_Rng; // random number generator
    uint m_randSize; // size of random numbers
    float * m_pdp; // read power delay profile from tables stored in tdl_pdp_table.h
    char m_delayProfile; // delay profile
    uint8_t m_numTaps; // numbe of taps in pdp table
    int m_timeChanSize; // time channel size
    std::vector<Tscalar> m_firNzPw; // CPU buffer to calculate and store non-zero FIR coefficients
    std::vector<uint16_t> m_firNzIdx; // CPU buffer to calculate and store non-zero FIR indexes
    uint16_t m_firNzLen; // number of non-zero coefficients
    float m_refTime0; // the time stamp for the start of tx symbol
    uint8_t m_runMode; // run mode, control frequency channel generation
    // dynamic descriptor
    tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescrCpu;
    tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescrGpu;
    
    // launch kernel drives
    dim3 m_gridDim, m_blockDim;
    void *m_args[2];
    cudaFunction_t m_functionPtr;
};

// Explicitly instantiate the template to resovle "undefined functions"
    template class tdlChan<__half, __half2>;
    template class tdlChan<float, cuComplex>;

/**
 * @brief CUDA kernel to generate time domain channel
 * 
 * @param tdlDynDescr TDL dynamic descriptor
 * @param refTime0 the time stamp for the start of tx symbol
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void genTdlChanCoeKernel(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr, float refTime0);

/**
 * @brief normalize the tdl channel coes in time domain per TTI, controlled by macro ENABLE_NORMALIZATION_
 * 
 * @param tdlDynDescr TDL dynamic descriptor
 *
 * @note can use thread shuffle to do fast parallel reduction but it requires Kelper arch
 * ref: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 */
 template <typename Tscalar, typename Tcomplex> 
 static __global__ void normalizeTimeChan(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr);

/**
 * @brief CUDA kernel to process tx time data using the generated time channel 
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void processInputKernel(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr, float refTime0);

/**
 * @brief CUDA kernel to generate freq channel on SC based on the time channel
 */
template<typename FFT, typename Tscalar, typename Tcomplex>
//__launch_bounds__(FFT::max_threads_per_block)
static __global__ void tdl_fft_kernel(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr);

template<typename Tscalar, typename Tcomplex>
using fftKernelHandle = void (*)(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr);

// Choose FFT kernel
template<typename Tscalar, typename Tcomplex, unsigned int FftSize, unsigned int Arch>
fftKernelHandle<Tscalar, Tcomplex> tdl_get_fft_param(dim3& block_dim, uint& shared_memory_size);

/**
 * @brief CUDA kernel to calculate freq chan on PRBG from freq chan on SC
 */
template<typename Tscalar, typename Tcomplex>
static __global__ void convertSCtoPRBG(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr);