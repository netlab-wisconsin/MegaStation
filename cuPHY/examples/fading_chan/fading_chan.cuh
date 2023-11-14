/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

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
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.h"
#include <type_traits>
#include <typeinfo>

#include "ofdmMod.cuh"
#include "ofdmDemod.cuh"
#include "tdl_chan.cuh"

template<class T> struct getScalarType {};
float type_convert(getScalarType<cuComplex>);
__half type_convert(getScalarType<__half2>);

// #define PRINT_FADING_CHAN_TIME_
/**
 * @brief fading channel class
 * this connects tdl channel and ofdm class
 *
 * @note Support data type FP16 and FP32 through <typename Tcomplex>
 * valid data types for <typename Tcomplex> are __half2 and cuComplex:
 *      __half2:   using FP16 data format, cuPHY signal should use CUPHY_C_16F
 *      cuComplex: using FP32 data format, cuPHY signal should use CUPHY_C_32F
 * @note the data buffers Tx and freqRx have to be allocated outside, fadingChan do not check their dimensions
 */
template <typename Tcomplex> 
class fadingChan{
public:
    /**
     * @brief Construct a new fading Chan object
     * 
     * @param Tx GPU memory for freq/time-domain tx samples
     * @param freqRx GPU memory for freq rx samples
     * @param fadingMode fading mode, currently support 0: AWGN and 1: TDL
     * @param randSeed random seed to generate 
     * @param strm cuda stream to run fading chan 
     * @param phyChannType indicator for physical channel type: 0 - PUSCH, 1 - PUCCH, 2 - PRACH
     */
    fadingChan(Tcomplex* Tx, Tcomplex* freqRx, cudaStream_t strm, uint8_t fadingMode = 1, uint16_t randSeed = 0, uint8_t phyChannType = 0);
    ~fadingChan();
    fadingChan(fadingChan const&) = delete;
    fadingChan& operator=(fadingChan const&) = delete;

    /**
     * @brief setup fadingChan class using a TV file
     * 
     * @param inputFile input TV file with chan_pars, carrier_pars and, timeTx (for TDL testing), freqRx (for AWGN testing)
     */
    void setup(hdf5hpp::hdf5_file& inputFile);

    /**
     * @brief run fading channel
     * perform ofdm mod, generate tdl time chan, apply tdl chan to tx sample, add nosie, perform ofdm demod
     * 
     * @param refTime0  the time stamp for the start of tx symbol
     * @param targetSNR target SNR for adding noise
     */
    void run(float refTime0 = 0.0f, float targetSNR = 0.0f); // run fading channel pipleline

    /**
     * @brief save freq tx,rxNoisy, rxNoiseFree, noise data, and estimated SNR (if exists) to "fadingChanData.h5" file
     */
    void savefadingChanToFile();

    /**
     * @brief estimate SNR from a specific OFDM symbol and a set of SCs, based on m_freqRxNoiseFree and m_noise 
     * 
     * @param ofdmSymIdx index of OFDM symbol
     * @param startSC start SC index, inclusive
     * @param endSC end SC index, exclusive; total SCs used is [endSC , startSC)
     * 
     * @note report average SNR over all antennas in command line & save SNRs to "SNR.txt" file during savefadingChanToFile()
     */
    void calSnr(uint16_t ofdmSymIdx, uint16_t startSC, uint16_t endSC);

private:

    /**
     * @brief add noise to rx time samples
     * 
     * @param targetSNR SNR to calculate noise samples, default 0.0f
     */
    void addNoiseFreq(float targetSNR = 0.0f); // add noise

    /**
     * @brief read carrier and channel parameters from TV file into m_carrierPrms and m_tdlCfg
     * 
     * @param inputFile input TV file
     */
    void readCarrierChanPar(hdf5hpp::hdf5_file& inputFile);

    /**
     * @brief read samples from TV
     * for AWGN, read rx freq data from TV into m_freqRx
     * for TDL, read tx freq data from TV m_freqTx
     * 
     * @param inputFile  input TV file
     * @param freqDataSize data size of freq samples
     */
    void read_Xtf(hdf5hpp::hdf5_file& inputFile);

    /**
     * @brief read frequency/time-domain samples from TV for PRACH
     * @param inputFile  input TV file
     */
    void read_Xtf_prach(hdf5hpp::hdf5_file& inputFile);

    /* -----  tdl, ofdm classes ---------*/
    // configuration
    cuphyCarrierPrms_t* m_carrierPrms;
    tdlConfig_t* m_tdlCfg;

    // class declaration
    using myTscalar = decltype(type_convert(getScalarType<Tcomplex>{}));
    tdlChan<myTscalar, Tcomplex> * m_tdl_chan;  // ptr to tdl channel class
    ofdm_modulate::ofdmModulate<myTscalar, Tcomplex> * m_ofdmMod; // ptr to ofdm modulation class
    ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex> * m_ofdmDeMod; // ptr to ofdm demodualation class

    /* -----  sample buffers in GPU ---------*/
    Tcomplex * m_Tx;
    Tcomplex * m_timeTx;
    Tcomplex * m_timeRx;
    Tcomplex * m_freqRxNoiseFree;
    Tcomplex * m_freqRxNoisy;
    uint m_freqTxDataSize; // length of frequency-domain tx sample
    uint m_timeTxDataSize; // length of time-domain tx sample
    uint m_freqRxDataSize; // length of frequency-domain rx sample, for adding noise
    
    //rng and noise buffer for adding noise
    Tcomplex * m_noise;
    cuphy::rng * m_rng;
    /**
     * @param  m_phyChannType physical channel type
     * 0: PUSCH
     * 1: PUCCH
     * 2: PRACH
     */
    uint8_t m_phyChannType;

    /**
     * @param  m_prach indicator for PRACH
     * true: PRACH
     * false: non-PRACH
     */
    bool m_prach;

    /**
     * @param  m_fadingMode fading mode, currently support AWGN and TDL
     * 0: AWGN
     * 1: TDL based on TV
     */
    uint8_t m_fadingMode;
    uint16_t m_randSeed;
    cudaStream_t m_strm; // stream to perform all processing
    cuphyDataType_t m_cuphyTensorType; // cuPHY tensor type for adding noise and save to file

    std::vector<float> m_SNR; // for tracking the SNR over time
};

// Explicitly instantiate the template to resovle "undefined functions"
template class fadingChan<__half2>;
template class fadingChan<cuComplex>; 

