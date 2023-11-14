/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define USE_MEMOERY_FFT_SHIFT_ // use fft shift in memeory or multiplication
#define OFDM_FFTs_PER_BLOCK_CONST_ 2 // FFTs per block in ofdm modulation and demodulation, changing this will affect run times, should be a divisor of OFDM_SYMBOLS_PER_SLOT (14)
// #define PRINT_SAVE_SAMPLE_OFDM_DATA_ // whether OFDM data will be print
#pragma once
/**
 * @brief the carriers parameters with default values
 * 
 */
typedef struct cuphyCarrierPrms
{
    uint16_t N_sc = 3276; // 12 * num of RBs
    uint16_t N_FFT = pow(2, ceilf(log2(N_sc)));  // also N_IFFT
    uint16_t N_txLayer = 4; 
    uint16_t N_rxLayer = 4;
    uint16_t id_slot = 0;  // per sub frame
    uint16_t id_subFrame = 0; // per frame
    uint16_t mu = 1; // numerology
    uint16_t cpType = 0; // 0 for normal CP, 1 for extended CP
    uint f_c = 480e3 * 4096; // delta_f_max * N_f based on 38.211
    float T_c = 5.0863e-10;
    uint f_samp = 15e3 * 8192; // 1ee3 * 2^mu * Nfft
    uint16_t N_symble_slot = OFDM_SYMBOLS_PER_SLOT; // 14 OFDMs per slot
    uint16_t kappa_bits = 6; // kappa = 64 (2^6); constants defined in 38.211
    uint ofdmWindowLen = 0; // ofdm windowing
    float rolloffFactor = 0.5; // rolloff factor for rcos in windowing
    uint32_t N_samp_slot = 61440;
    uint16_t k_const = 64;
    uint32_t N_u_mu = 65536;

    // PRACH parameters
    uint32_t startRaSym = 0;
    uint32_t delta_f_RA = 30000;
    uint32_t N_CP_RA = 29952;
    uint32_t K = 1;
    int32_t  k1 = -1638;
    uint32_t kBar = 2;
    uint32_t N_u = 786432;
    uint32_t L_RA = 139;
    uint32_t n_slot_RA_sel = 0;
    uint32_t N_rep = 12;
} cuphyCarrierPrms_t;


/**
 * @brief Get the cuda device arch object to be used in FFT kernel
 * 
 * @return unsigned int
 */
static unsigned int get_cuda_device_arch() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
}

/**
 * @brief launch configuration structure, includes driver info and kernel input variables
 * 
 */
typedef struct {
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} launchCfg_t;
