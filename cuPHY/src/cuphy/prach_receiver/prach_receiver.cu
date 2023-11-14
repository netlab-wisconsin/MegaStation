/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cuphy_internal.h"
#include "cuphy.hpp"
#include "prach_receiver.hpp"
#include <iostream>
#include "tensor_desc.hpp"
#include "type_convert.hpp"

#ifdef USE_CUFFTDX
#include <cufftdx.hpp>
#endif

#define NUM_THREAD 256

using namespace cuphy_i;

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


template<typename Tcomplex>
__device__ Tcomplex complex_mult2(Tcomplex num1, Tcomplex num2);

template<>
__device__ cuFloatComplex complex_mult2(cuFloatComplex num1, cuFloatComplex num2) {
    return cuCmulf(num1, num2);
};

template<>
__device__ __half2 complex_mult2(__half2 num1, __half2 num2) {
    return __hmul2(num1, num2);
};

/** @brief: Do coherent combining for repetitive preambles samples and calculate the correlation between 
 *          the received (averaged) preamble and local reference preamble.
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_compute_correlation(const PrachInternalDynParamPerOcca* d_dynParam,
                                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                            uint max_ant_u,
                                            uint16_t nOccaProc) {

    int batchIndex = blockIdx.x / max_ant_u;
    int ant_u_idx = blockIdx.x - batchIndex * max_ant_u;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_ant = prach_params->N_ant;
    const int Nfft = prach_params->Nfft;
    const int uCount = prach_params->uCount;

    if(threadIdx.x >= Nfft || ant_u_idx >= N_ant*uCount)
        return;

    const int L_RA = prach_params->L_RA;
    const int N_rep = prach_params->N_rep;
    const int N_nc = prach_params->N_nc;
    const int kBar = prach_params->kBar;

    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);

    // O-RAN FH sends 144 or 864 samples instead of 139 or 839 samples
    const int L_ORAN = (L_RA == 139) ? 144 : 864;

    if (threadIdx.x < L_RA) {
        const __half2* d_prach_rx = d_dynParam[batchIndex].dataRx;
        const __half2* d_y_u_ref = staticOccaParam.d_y_u_ref;

        int uIdx = ant_u_idx % uCount;
        int antIdx = ant_u_idx / uCount;
        int idx_y_ref = uIdx * L_RA + threadIdx.x;
        Tscalar x = d_y_u_ref[idx_y_ref].x;
        Tscalar y = -d_y_u_ref[idx_y_ref].y;
        Tcomplex y_ref_val = make_complex<Tcomplex>::create(x, y);
        int rep_start = 0;
        int step = 0;
        // for each non-coherent combining group
        for (int idxNc = 0; idxNc < N_nc; idxNc ++) {
            Tcomplex y_rx_val = make_complex<Tcomplex>::create(0.0, 0.0); 
            if (idxNc < N_nc-1) {
                step = N_rep/N_nc;
            }
            else {
                step = N_rep - (N_nc-1)*N_rep/N_nc;
            }
            // average over repetitive preambles
            for (int idxRep = rep_start; idxRep < rep_start + step; idxRep ++) {
                // int idx_y_rx = (antIdx *N_rep + idxRep) * L_RA + threadIdx.x;
                // Need to skip the first kBar guard subcarriers for O-RAN FH samples
                int idx_y_rx = (antIdx *N_rep + idxRep) * L_ORAN + threadIdx.x + kBar;
                y_rx_val.x = ((Tscalar) d_prach_rx[idx_y_rx].x) + y_rx_val.x;
                y_rx_val.y = ((Tscalar) d_prach_rx[idx_y_rx].y) + y_rx_val.y;
            }
            y_rx_val.x = ((Tscalar) y_rx_val.x)/((Tscalar) step);
            y_rx_val.y = ((Tscalar) y_rx_val.y)/((Tscalar) step);
            rep_start = rep_start + step;
            // freq domain multiplication
            Tcomplex z_u  = complex_mult2<Tcomplex>(y_rx_val, y_ref_val);
            int idx_fft = (ant_u_idx*N_nc+idxNc)*Nfft + threadIdx.x;
            d_fft[idx_fft] = z_u;
        }            
    }
    else {
        for (int idxNc = 0; idxNc < N_nc; idxNc ++) {
            int idx_fft = (ant_u_idx*N_nc+idxNc)*Nfft + threadIdx.x;
            // pad zero
            d_fft[idx_fft] = make_complex<Tcomplex>::create(0.0, 0.0);
        }
    }
}

#ifdef USE_CUFFTDX
template<class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void block_fft_kernel(cuFloatComplex* d_fft) {
    using namespace cufftdx;

    using complex_type = typename FFT::value_type;
    complex_type* data = (complex_type*)d_fft;

    static_assert(FFT::storage_size == 2);
    // Local array and copy data into it
    complex_type thread_data[FFT::storage_size];

    const int stride = size_of<FFT>::value / FFT::elements_per_thread;

    int batchIndex = blockIdx.x;

    // generic index calculation
    //int index = threadIdx.x + threadIdx.y * size_of<FFT>::value + blockDim.y * size_of<FFT>::value * batchIndex;

    // index value when single FFT is computed per block
    int index = threadIdx.x + size_of<FFT>::value * batchIndex;

    thread_data[0].x = data[index].x;
    thread_data[0].y = data[index].y;

    thread_data[1].x = data[index + stride].x;
    thread_data[1].y = data[index + stride].y;

    extern __shared__ complex_type shared_mem[];

    // Execute FFT
    FFT().execute(thread_data, shared_mem);

    // Save results
    data[index].x = thread_data[0].x;
    data[index].y = thread_data[0].y;

    data[index + stride].x = thread_data[1].x;
    data[index + stride].y = thread_data[1].y;
}

using FftKernelHandle = void (*)(cuFloatComplex* d_fft);

template<typename Tscalar, unsigned int FftSize, unsigned int Arch>
FftKernelHandle prach_get_fft_param(dim3& block_dim, uint& shared_memory_size) { 

    using namespace cufftdx;

    using FFT = decltype(Size<FftSize>() + Precision<Tscalar>() + Type<fft_type::c2c>()
                        + Direction<fft_direction::inverse>() + FFTsPerBlock<1>()
                        + ElementsPerThread<2>() + SM<Arch>() + Block());

    block_dim = FFT::block_dim;
    shared_memory_size = FFT::shared_memory_size;

    return block_fft_kernel<FFT>;
 }

 template<typename Tscalar>
FftKernelHandle prach_get_fft_param(unsigned int Nfft, unsigned int cudaDeviceArch, dim3& block_dim, uint& shared_memory_size) { 

    if(Nfft == 256)
    {
        switch(cudaDeviceArch)
        {
            // All SM supported by cuFFTDx
            case 700: return prach_get_fft_param<Tscalar, 256, 700>(block_dim, shared_memory_size);
            case 720: return prach_get_fft_param<Tscalar, 256, 720>(block_dim, shared_memory_size);
            case 750: return prach_get_fft_param<Tscalar, 256, 750>(block_dim, shared_memory_size);
            case 800: return prach_get_fft_param<Tscalar, 256, 800>(block_dim, shared_memory_size);
            case 860: return prach_get_fft_param<Tscalar, 256, 860>(block_dim, shared_memory_size);
            case 870: return prach_get_fft_param<Tscalar, 256, 870>(block_dim, shared_memory_size);
            case 890: return prach_get_fft_param<Tscalar, 256, 890>(block_dim, shared_memory_size);
            case 900: return prach_get_fft_param<Tscalar, 256, 900>(block_dim, shared_memory_size);
            default: assert(false); return nullptr;
        }
    }
    else if(Nfft == 1024)
    {
        switch(cudaDeviceArch)
        {
            // All SM supported by cuFFTDx
            case 700: return prach_get_fft_param<Tscalar, 1024, 700>(block_dim, shared_memory_size);
            case 720: return prach_get_fft_param<Tscalar, 1024, 720>(block_dim, shared_memory_size);
            case 750: return prach_get_fft_param<Tscalar, 1024, 750>(block_dim, shared_memory_size);
            case 800: return prach_get_fft_param<Tscalar, 1024, 800>(block_dim, shared_memory_size);
            case 860: return prach_get_fft_param<Tscalar, 1024, 860>(block_dim, shared_memory_size);
            case 870: return prach_get_fft_param<Tscalar, 1024, 870>(block_dim, shared_memory_size);
            case 890: return prach_get_fft_param<Tscalar, 1024, 890>(block_dim, shared_memory_size);
            case 900: return prach_get_fft_param<Tscalar, 1024, 900>(block_dim, shared_memory_size);
            default: assert(false); return nullptr;
        }
    }
    else
    {
        assert(false);
    }
    return nullptr;
 }
 
#else
/** @brief: Perform ifft for correlation output
 *  @param[in] plan: FFT plan.
 *  @param[in, out] d_fft: pointer to FFT buffer.
 *  @param[in] strm: stream for cuFFT.
 */
 template<typename Tcomplex, typename Tscalar>
 cuphyStatus_t prach_compute_ifft(cufftHandle plan, Tcomplex * __restrict__ d_fft, cudaStream_t strm) { 
    cufftResult res = cufftSetStream(plan, strm);
    if (CUFFT_SUCCESS != res) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUFFT error: SetStream failed with code {}", res);
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    res = cufftExecC2C(plan, (cufftComplex *)d_fft, (cufftComplex *)d_fft, CUFFT_INVERSE);
    if (CUFFT_SUCCESS != res){
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUFFT error: ExecC2C Forward failed with code {}", res);
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    return CUPHY_STATUS_SUCCESS;
 }
#endif

 /** @brief: Do non-coherent combining and find the power, peak value and peak location 
 *          for each preamble zone. 
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_compute_pdp(const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    int zoneSizeExt) {

    int batchIndex = blockIdx.y;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_ant = prach_params->N_ant;

    int NzonePerBlock = NUM_THREAD/zoneSizeExt;
    int zoneIdxInBlock = threadIdx.x/zoneSizeExt;
    int global_idxZone = blockIdx.x * NzonePerBlock + zoneIdxInBlock;
    int antIdx = global_idxZone / CUPHY_PRACH_RX_NUM_PREAMBLE;

    if(antIdx >= N_ant)
        return;

    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
 
    const int L_RA = prach_params->L_RA;
    const int N_CS = prach_params->N_CS;

    int zoneSize = (N_CS*Nfft+L_RA-1)/L_RA;
    
    __shared__ Tscalar local_power[NUM_THREAD];
    __shared__ Tscalar local_max[NUM_THREAD];
    __shared__ int local_loc[NUM_THREAD];

    int zoneSearchGap = (int) Nfft/L_RA;    
    int C_v = 0;
    int zone_start = 0;
    int prmbCount = global_idxZone & (CUPHY_PRACH_RX_NUM_PREAMBLE-1); // & => mod
    int NzonePerU = L_RA / N_CS;
    int uIdx = prmbCount / NzonePerU;
    int tIdx = threadIdx.x;
    int idxInZone = tIdx & (zoneSizeExt-1); // & => mod

    // copy abs(dfft)^2 to shared memory for each zone
    if (idxInZone < zoneSize) {
        C_v = (prmbCount % NzonePerU) * N_CS;
        zone_start = (C_v*Nfft+L_RA-1)/L_RA;                       
        zone_start = (Nfft - zone_start) & (Nfft-1);  // & => mod
        // compute abs()^2 and do non-coherent combining
        Tscalar val = 0.0;        
        for (int idxNc = 0; idxNc < N_nc; idxNc ++) {
            int idx_fft = ((antIdx * uCount + uIdx) * N_nc + idxNc) * Nfft + ((zone_start + idxInZone - zoneSearchGap + Nfft) & (Nfft-1)); //& => mod
            Tscalar x = ((Tscalar) d_fft[idx_fft].x)/((Tscalar) L_RA);
            Tscalar y = ((Tscalar) d_fft[idx_fft].y)/((Tscalar) L_RA);
            val = x*x + y*y + val;           
        }
        val = val/Tscalar(N_nc);      

        local_power[tIdx] = val;
        local_max[tIdx] = val;
        local_loc[tIdx] = idxInZone;        
    }
    else {
        local_power[tIdx] = 0;
        local_max[tIdx] = 0;
        local_loc[tIdx] = 0; 
    }
    __syncthreads();

    // compute sum and find max/loc for each zone
    for (unsigned int s=zoneSizeExt/2; s>0; s>>=1) {
        if (idxInZone < s) {
            local_power[tIdx] = local_power[tIdx] + local_power[tIdx + s];
            if (local_max[tIdx] < local_max[tIdx + s]) {
                local_max[tIdx] = local_max[tIdx + s];
                local_loc[tIdx] = local_loc[tIdx + s];
            }
        }
        __syncthreads();
    }

    // copy results from shared memory back to d_pdp
    if (idxInZone == 0) {
        local_power[tIdx] = local_power[tIdx]/((Tscalar)zoneSize);
        int pdp_index = antIdx * CUPHY_PRACH_RX_NUM_PREAMBLE + prmbCount;
        d_pdp[pdp_index].power = local_power[tIdx];
        d_pdp[pdp_index].max = local_max[tIdx];
        d_pdp[pdp_index].loc = local_loc[tIdx] - zoneSearchGap;   
    }
 }


 /** @brief: Estimate noise power and detect preambles based on threshold.
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_search_pdp(const PrachInternalDynParamPerOcca* d_dynParam,
                                const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                uint32_t * __restrict__  num_detectedPrmb_addr_arr,
                                uint32_t * __restrict__  prmbIndex_estimates_addr_arr,
                                float * __restrict__ prmbDelay_estimates_addr_arr,
                                float * __restrict__ prmbPower_estimates_addr_arr,
                                float * __restrict__ interference_addr_arr) {
    uint batchIndex = blockIdx.x;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;
    uint16_t occaPrmDynIdx = d_dynParam[batchIndex].occaPrmDynIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_ant = prach_params->N_ant;

    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    float* d_ant_rssi = (float*)(d_det + 1); 
    float* d_rssiLin = d_ant_rssi + N_ant;

    uint32_t* num_detectedPrmb_addr = num_detectedPrmb_addr_arr + occaPrmDynIdx;
    uint32_t* prmbIndex_estimates_addr = prmbIndex_estimates_addr_arr + occaPrmDynIdx * CUPHY_PRACH_RX_NUM_PREAMBLE;
    float* prmbDelay_estimates_addr = prmbDelay_estimates_addr_arr + occaPrmDynIdx * CUPHY_PRACH_RX_NUM_PREAMBLE;
    float* prmbPower_estimates_addr = prmbPower_estimates_addr_arr + occaPrmDynIdx * CUPHY_PRACH_RX_NUM_PREAMBLE;

    float* interference_addr = interference_addr_arr + occaPrmDynIdx;
      
    const int delta_f_RA = prach_params->delta_f_RA;
    int detIdx = 0, prmbIdx = threadIdx.x;

    Tscalar thr0 = d_dynParam[batchIndex].thr0;

    __shared__ Tscalar np1, thr1, np2, thr2;
    __shared__ int shared_detIdx;
    __shared__ struct  local_pdp_t
                        {
                            Tscalar power;
                            int cnt;
                        } local_pdp [CUPHY_PRACH_RX_NUM_PREAMBLE];

    if (prmbIdx == 0) {
        shared_detIdx = 0;
    }

    // average pdp over antennas
    if (threadIdx.x < CUPHY_PRACH_RX_NUM_PREAMBLE) {        
        int maxLoc = d_pdp[prmbIdx].loc;
        Tscalar maxVal = d_pdp[prmbIdx].max;
        for (int antIdx = 1; antIdx < N_ant; antIdx ++) {
            int global_index = antIdx * CUPHY_PRACH_RX_NUM_PREAMBLE + prmbIdx;
            d_pdp[prmbIdx].power = d_pdp[prmbIdx].power + d_pdp[global_index].power;
            d_pdp[prmbIdx].max = d_pdp[prmbIdx].max + d_pdp[global_index].max;
            if (((Tscalar) d_pdp[global_index].max) > maxVal) {
                maxVal = d_pdp[global_index].max;
                maxLoc = d_pdp[global_index].loc;
            }                                 
        }
        d_pdp[prmbIdx].power = d_pdp[prmbIdx].power/((Tscalar) N_ant);
        d_pdp[prmbIdx].max = d_pdp[prmbIdx].max/((Tscalar) N_ant);
        d_pdp[prmbIdx].loc = maxLoc;            
        local_pdp[prmbIdx].power = d_pdp[prmbIdx].power; 
        local_pdp[prmbIdx].cnt = 1;        
    }
    __syncthreads();

    // calculate the sum of power over all preamble indices
    for (unsigned int s=CUPHY_PRACH_RX_NUM_PREAMBLE/2; s>0; s>>=1) {
        // Overrunning array 
        if (prmbIdx < s && (prmbIdx + s) < CUPHY_PRACH_RX_NUM_PREAMBLE) {
            local_pdp[prmbIdx].power = local_pdp[prmbIdx].power + local_pdp[prmbIdx + s].power;
            local_pdp[prmbIdx].cnt = local_pdp[prmbIdx].cnt + local_pdp[prmbIdx + s].cnt;
        }
        __syncthreads();
    }

    // calculate the average power and update threshold "thr1"
    if (prmbIdx == 0) {
        np1 = ((Tscalar) local_pdp[0].power)/((Tscalar) local_pdp[0].cnt);
        thr1 = thr0*np1;
    }    
    __syncthreads();

    // find the preamble indices with peak < thr1 (as noise)and record their power
    if (((Tscalar) d_pdp[prmbIdx].max) < thr1) {
        local_pdp[prmbIdx].power = d_pdp[prmbIdx].power;
        local_pdp[prmbIdx].cnt = 1; 
    }
    else
    {
        local_pdp[prmbIdx].power = 0;
        local_pdp[prmbIdx].cnt = 0;
    }   
    __syncthreads();

    // calculate sum of noise power
    for (unsigned int s=CUPHY_PRACH_RX_NUM_PREAMBLE/2; s>0; s>>=1) {
        if (prmbIdx < s) {
            local_pdp[prmbIdx].power = local_pdp[prmbIdx].power + local_pdp[prmbIdx + s].power;
            local_pdp[prmbIdx].cnt = local_pdp[prmbIdx].cnt + local_pdp[prmbIdx + s].cnt;
        }
        __syncthreads();
    }

    // calculate the average noise power and update threshold thr2
    if (prmbIdx == 0) {
        np2 = ((Tscalar) local_pdp[0].power)/((Tscalar)local_pdp[0].cnt);
        if(np2 == (Tscalar)0 || np1 == (Tscalar)0)
        {
            *interference_addr = -100;
        }
        else
        {
            *interference_addr = 10*log10((float)np2);
        }

        Tscalar thr2_min = (*d_rssiLin >= 1 ?  1 : *d_rssiLin) * 1e-2;
        if (((Tscalar) np2*thr0) > thr2_min) 
            thr2 = np2*thr0;
        else
            thr2 = thr2_min;
    }
    __syncthreads();
  
    // find the preamble indices with peak > thr2  (as detected)
    if (((Tscalar) d_pdp[prmbIdx].max) > ((Tscalar) thr2)) {       
        detIdx = atomicAdd(&shared_detIdx, 1);        
        //TBD: may change d_det from struct of array to array of struct to make memory access faster
        d_det->power[detIdx] = d_pdp[prmbIdx].max;
        d_det->prmbIdx[detIdx] = prmbIdx;
        d_det->loc[detIdx] = (d_pdp[prmbIdx].loc > 0)?d_pdp[prmbIdx].loc:0;                                       
    }    
    __syncthreads();

    // pass the dection results 
    if (prmbIdx == 0) {                              
        d_det->Ndet = shared_detIdx;
        * num_detectedPrmb_addr = d_det->Ndet;        
        for (int i = 0; i < d_det->Ndet; i++) {
            prmbIndex_estimates_addr[i] = (uint32_t) d_det->prmbIdx[i];
            prmbDelay_estimates_addr[i] = ((Tscalar) d_det->loc[i])/(((Tscalar) Nfft)*((Tscalar) delta_f_RA));  
            prmbPower_estimates_addr[i] = (Tscalar) d_det->power[i];    
        }
    }   
    __syncthreads();
 }

/** @brief: Compute average power for each antenna and average power over all antennas
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_compute_rssi(const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    float* __restrict__ d_rssiDbArray,
                                    uint max_l_oran_ant)
 {
    __shared__ bool isLastBlockDone;

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint batchIndex = blockIdx.y;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;
    uint16_t occaPrmDynIdx = d_dynParam[batchIndex].occaPrmDynIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_rep = prach_params->N_rep;
    const int N_ant = prach_params->N_ant;
    const int L_RA = prach_params->L_RA;

    // O-RAN FH sends 144 or 864 samples instead of 139 or 839 samples
    const int L_ORAN = ((L_RA == 139) ? 144 : 864) * N_rep;

    // align L_ORAN so that same warp doesn't have samples for two different antennas
    // this allows us to use shuffle reduction
    unsigned int align_l_oran = ((L_ORAN + 31) >> 5) << 5;

    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    float* d_ant_rssi = (float*)(d_det + 1); 
    float* d_rssiLin = d_ant_rssi + N_ant;
    unsigned int* d_count = (unsigned int*)(d_rssiLin + 1);

    const __half2* d_prach_rx = d_dynParam[batchIndex].dataRx;
    float* d_rssiDb = d_rssiDbArray + occaPrmDynIdx;
    
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int threadId = cta.thread_rank();

    if (threadId == 0) 
    {
        isLastBlockDone = false;
    }

    cta.sync();

    // Handle to tile in thread block
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    int antIdx = gid / align_l_oran;
    int idxOran = gid -  antIdx * align_l_oran;

    float absRx = 0.0f;

    if(idxOran < L_ORAN && antIdx < N_ant)
    {
        __half2 rx = d_prach_rx[antIdx * L_ORAN + idxOran];
        absRx = rx.x * rx.x + rx.y * rx.y;
    }

    // shuffle redux
    absRx = cg::reduce(tile, absRx, cg::plus<float>());
    cg::sync(tile);
    if (tile.thread_rank() == 0 && antIdx < N_ant) 
    {
      atomicAdd(&d_ant_rssi[antIdx], absRx);
    }

    if (threadId == 0)
    {
        // make sure d_ant_rssi values are updated before we modify d_count
        __threadfence();

        unsigned int value = atomicInc(d_count, gridDim.x);
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    // make sure each thread reads correct value of isLastBlockDone
    cta.sync();

    // number of antennas expected to be <= 32
    assert(N_ant <= warpSize);

    if (isLastBlockDone) 
    {
        // take average of d_ant_rssi values
        if(threadId < warpSize)
        {
            absRx = 0.0f;
            if(threadId < N_ant)
            {
                absRx = d_ant_rssi[threadId];
                if(absRx == 0)
                {
                    d_ant_rssi[threadId] = -100;
                }
                else
                {
                    d_ant_rssi[threadId] = 10*log10(absRx/L_ORAN);
                }
            }

            // shuffle redux over all antenna power
            absRx = cg::reduce(tile, absRx, cg::plus<float>());
            cg::sync(tile);

            if (threadId == 0) 
            {
                if(absRx == 0)
                {
                    d_rssiDb[0] = -100;
                }
                else
                {
                    absRx = absRx/(L_ORAN * N_ant);
                    d_rssiDb[0] = 10*log10(absRx);
                }

                *d_rssiLin = absRx;
            }
        }
    }
 }

 template<typename Tcomplex, typename Tscalar>
 __global__ void memsetRssi(const PrachInternalDynParamPerOcca* d_dynParam,
                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                            uint16_t maxAntenna,
                            uint16_t nOccaProc)
 {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     int batchIndex = index / (maxAntenna + 2);
     if(batchIndex >= nOccaProc)
        return;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_ant = prach_params->N_ant;

    int antIndex = index - batchIndex * (maxAntenna + 2);

    if(antIndex >= N_ant + 2)
        return;

    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    uint* d_ant_rssi = (uint*)(d_det + 1); 

    d_ant_rssi[antIndex] = 0;
 }

  template<typename Tcomplex, typename Tscalar>
 __global__ void memcpyRssi(float* ant_rssi_addr, 
                            const PrachInternalDynParamPerOcca* d_dynParam,
                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                            uint16_t maxAntenna,
                            uint16_t nOccaProc)
 {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     int batchIndex = index / maxAntenna;
     if(batchIndex >= nOccaProc)
        return;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_ant = prach_params->N_ant;

    int antIndex = index - batchIndex * maxAntenna;

    if(antIndex >= N_ant)
        return;

    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    float* d_ant_rssi = (float*)(d_det + 1); 

    uint16_t occaPrmDynIdx = d_dynParam[batchIndex].occaPrmDynIdx;
    float* pinned_ant_rssi = ant_rssi_addr + occaPrmDynIdx * MAX_N_ANTENNAS_SUPPORTED;

    pinned_ant_rssi[antIndex] = d_ant_rssi[antIndex];
 }


template<typename Tcomplex, typename Tscalar>
cuphyStatus_t launch_templated_prach_receiver_kernels(const PrachInternalDynParamPerOcca* d_dynParam,
                                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                            const PrachInternalDynParamPerOcca* h_dynParam,
                                            const PrachInternalStaticParamPerOcca* h_staticParam,
                                            uint32_t* num_detectedPrmb_addr,
                                            uint32_t* prmbIndex_estimates_addr,
                                            float* prmbDelay_estimates_addr,
                                            float* prmbPower_estimates_addr,
                                            float* ant_rssi_addr,
                                            float* rssi_addr,
                                            float* interference_addr,
                                            uint16_t nOccaProc,
                                            uint16_t maxAntenna,
                                            uint max_l_oran_ant,
                                            uint max_ant_u,
                                            uint max_nfft,
                                            int max_zoneSizeExt,
                                            uint cudaDeviceArch,
                                            cudaStream_t strm) {

    // intialize d_ant_rssi, d_rssiLin, d_count with 0
    dim3 block_dim(128, 1, 1);
    dim3 grid_dim((nOccaProc * (maxAntenna + 2) + block_dim.x - 1) / block_dim.x);

    memsetRssi<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, maxAntenna, nOccaProc);

    grid_dim = dim3((max_l_oran_ant + block_dim.x - 1) / block_dim.x, nOccaProc);
    prach_compute_rssi<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, rssi_addr, max_l_oran_ant);

    grid_dim = dim3((nOccaProc * maxAntenna + block_dim.x - 1) / block_dim.x);
    memcpyRssi<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(ant_rssi_addr, d_dynParam, d_staticParam, maxAntenna, nOccaProc);

    block_dim = dim3(max_nfft);
    grid_dim = dim3(max_ant_u * nOccaProc);
    prach_compute_correlation<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, max_ant_u, nOccaProc);

    for(int i = 0; i < nOccaProc; ++i)
    {
        uint16_t occaPrmStaticIdx = h_dynParam[i].occaPrmStatIdx;

        const PrachInternalStaticParamPerOcca& staticOccaParam = h_staticParam[occaPrmStaticIdx];
        Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer.addr());

#ifdef USE_CUFFTDX
        const PrachParams& prach_params = staticOccaParam.prach_params;
        grid_dim = dim3(prach_params.N_ant * prach_params.uCount * prach_params.N_nc, 1, 1);
        uint shared_memory_size = 0;
        auto kernelPtr = prach_get_fft_param<Tscalar>(prach_params.Nfft, cudaDeviceArch, block_dim, shared_memory_size);
        if(kernelPtr == nullptr)
        {
            return CUPHY_STATUS_ARCH_MISMATCH;
        }
        void *kernelArgs[1] = {(void *)&d_fft};
        cudaError_t error = cudaLaunchKernel(reinterpret_cast<void*>(kernelPtr), grid_dim, block_dim, 
                                                (void**)(kernelArgs), shared_memory_size, strm);
        if (cudaSuccess != error) {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Error: {}", cudaGetErrorString(error));
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
#else
        cufftHandle fft_plan = staticOccaParam.fft_plan;
        cuphyStatus_t status = prach_compute_ifft<Tcomplex, Tscalar>(fft_plan, d_fft, strm);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            return status;
        }
#endif
    }

    assert(max_zoneSizeExt <= 512);
    block_dim = dim3(max_zoneSizeExt > NUM_THREAD ? max_zoneSizeExt : NUM_THREAD);
    int Nzone = block_dim.x/max_zoneSizeExt;
    grid_dim = dim3((maxAntenna * CUPHY_PRACH_RX_NUM_PREAMBLE + Nzone-1) / Nzone, nOccaProc);
    prach_compute_pdp<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, max_zoneSizeExt);

    dim3 search_pdp_block_dim(CUPHY_PRACH_RX_NUM_PREAMBLE);
    dim3 search_pdp_grid_dim(nOccaProc);
    prach_search_pdp<Tcomplex, Tscalar><<<search_pdp_grid_dim, search_pdp_block_dim, 0, strm>>>(d_dynParam, d_staticParam,
                                                    num_detectedPrmb_addr,
                                                    prmbIndex_estimates_addr,
                                                    prmbDelay_estimates_addr,
                                                    prmbPower_estimates_addr,
                                                    interference_addr);

    cudaError_t cuda_error = cudaGetLastError();
    if (cudaSuccess != cuda_error) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Error: {}", cudaGetErrorString(cuda_error));
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachCreateGraph(cudaGraph_t* graph, cudaGraphExec_t* graphInstance, std::vector<cudaGraphNode_t>& nodes,  cudaStream_t strm,
                                    const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    const PrachInternalStaticParamPerOcca* h_staticParam,
                                    uint32_t* num_detectedPrmb_addr,
                                    uint32_t* prmbIndex_estimates_addr,
                                    float* prmbDelay_estimates_addr,
                                    float* prmbPower_estimates_addr,
                                    float* ant_rssi_addr,
                                    float* rssi_addr,
                                    float* interference_addr,
                                    uint16_t nTotCellOcca,
                                    uint16_t nMaxOccasions,
                                    uint16_t maxAntenna,
                                    uint max_l_oran_ant,
                                    uint max_ant_u,
                                    uint max_nfft,
                                    int max_zoneSizeExt,
                                    std::vector<char>& activeOccasions,
                                    uint cudaDeviceArch)
{
    using Tcomplex = cuFloatComplex;
    using Tscalar = float;

    CUDA_CHECK_EXCEPTION(cudaGraphCreate(graph, 0));

    {
        dim3 block_dim(128, 1, 1);
        cudaKernelNodeParams kernelNodeParams = {0};
        // Update: nMaxOccasions -> nOccaProc
        void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &maxAntenna,
                            &nTotCellOcca};

        kernelNodeParams.func = (void *)memsetRssi<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((nTotCellOcca * (maxAntenna + 2) + block_dim.x - 1) / block_dim.x, 1, 1);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::MemsetRSSI], *graph, nullptr,
                                0, &kernelNodeParams));
    }

    {
        dim3 block_dim(128, 1, 1);
        cudaKernelNodeParams kernelNodeParams = {0};
        void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &rssi_addr,
                            &max_l_oran_ant};

        kernelNodeParams.func = (void *)prach_compute_rssi<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((max_l_oran_ant + block_dim.x - 1) / block_dim.x, nTotCellOcca, 1);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::ComputeRSSI], *graph, &nodes[GraphNodeType::MemsetRSSI],
                                1, &kernelNodeParams));
    }

    {
        dim3 block_dim(128, 1, 1);
        cudaKernelNodeParams kernelNodeParams = {0};
        // Update: nMaxOccasions -> nOccaProc
        void *kernelArgs[5] = {(void *)&ant_rssi_addr, (void *)&d_dynParam, &d_staticParam,
                            &maxAntenna, &nMaxOccasions};

        kernelNodeParams.func = (void *)memcpyRssi<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((nMaxOccasions * maxAntenna + block_dim.x - 1) / block_dim.x, 1, 1);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::MemcpyRSSI], *graph, &nodes[GraphNodeType::ComputeRSSI],
                                1, &kernelNodeParams));
    }

    {
        cudaKernelNodeParams kernelNodeParams = {0};
        // Update: nMaxOccasions -> nOccaProc
        void *kernelArgs[4] = {(void *)&d_dynParam, &d_staticParam,
                            &max_ant_u, &nMaxOccasions};

        kernelNodeParams.func = (void *)prach_compute_correlation<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3(max_ant_u * nMaxOccasions);
        kernelNodeParams.blockDim = dim3(max_nfft);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::ComputeCorrelationNode], *graph, nullptr,
                                0, &kernelNodeParams));
    }
    int nodeIdx = 0;
    for(int i = 0; i < nMaxOccasions; ++i)
    {
        if(activeOccasions[i] != 0) // Only add nodes for configured occasions
        {
            const PrachInternalStaticParamPerOcca& staticOccaParam = h_staticParam[i];

            Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer.addr());

#ifdef USE_CUFFTDX
            const PrachParams& prach_params = staticOccaParam.prach_params;
            int numBlocks = prach_params.N_ant * prach_params.uCount * prach_params.N_nc;
            uint shared_memory_size = 0;
            dim3 block_dim;
            auto kernelPtr = prach_get_fft_param<Tscalar>(prach_params.Nfft, cudaDeviceArch, block_dim, shared_memory_size);

            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[4] = {(void *)&d_fft};

            kernelNodeParams.func = (void*)kernelPtr;
            kernelNodeParams.gridDim = dim3(numBlocks);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = shared_memory_size;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphAddKernelNode(&nodes[GraphNodeType::FFTNode + nodeIdx], *graph, &nodes[GraphNodeType::ComputeCorrelationNode],
                                    1, &kernelNodeParams));

#else // we can't use this path for now until CUDA supports disabling/enabling child nodes at runtime
            cudaGraph_t childGraph;
            
            cufftHandle fft_plan = staticOccaParam.fft_plan;

            CUDA_CHECK_EXCEPTION(cudaStreamBeginCapture(strm, cudaStreamCaptureModeThreadLocal));

            // TBD: use device side FFT (cuFFTDx) instead of host side FFT (cuFFT) and merge into other kernel
            cuphyStatus_t status = prach_compute_ifft<Tcomplex, Tscalar>(fft_plan, d_fft, strm);
            if(status != CUPHY_STATUS_SUCCESS)
            {
                return status;
            }

            CUDA_CHECK_EXCEPTION(cudaStreamEndCapture(strm, &childGraph));

            CUDA_CHECK_EXCEPTION(
            cudaGraphAddChildGraphNode(&nodes[GraphNodeType::FFTNode + nodeIdx], *graph, &nodes[GraphNodeType::ComputeCorrelationNode],
                                    1, childGraph));
#endif
            nodeIdx++;
        }
    }

    {
        assert(max_zoneSizeExt <= 512);
        dim3 block_dim(max_zoneSizeExt > NUM_THREAD ? max_zoneSizeExt : NUM_THREAD);
        int Nzone = block_dim.x/max_zoneSizeExt;

        cudaKernelNodeParams kernelNodeParams = {0};
        void *kernelArgs[3] = {(void *)&d_dynParam, &d_staticParam,
                            &max_zoneSizeExt};

        kernelNodeParams.func = (void *)prach_compute_pdp<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((maxAntenna * CUPHY_PRACH_RX_NUM_PREAMBLE + Nzone-1) / Nzone, nTotCellOcca);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::ComputePDPNode], *graph, &nodes[GraphNodeType::FFTNode],
                                nTotCellOcca, &kernelNodeParams));
    }

    {
        cudaKernelNodeParams kernelNodeParams = {0};
        void *kernelArgs[7] = {(void *)&d_dynParam, &d_staticParam,
                            &num_detectedPrmb_addr, &prmbIndex_estimates_addr, &prmbDelay_estimates_addr,
                            &prmbPower_estimates_addr, &interference_addr};

        kernelNodeParams.func = (void *)prach_search_pdp<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3(nMaxOccasions);
        kernelNodeParams.blockDim =dim3(CUPHY_PRACH_RX_NUM_PREAMBLE);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::SearchPDPNode], *graph, &nodes[GraphNodeType::ComputePDPNode],
                                1, &kernelNodeParams));
    }

    CUDA_CHECK_EXCEPTION(cudaGraphInstantiate(graphInstance, *graph, NULL, NULL, 0));

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachUpdateGraph(cudaGraphExec_t graphInstance, std::vector<cudaGraphNode_t>& nodes,
                                    const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    const PrachInternalDynParamPerOcca* h_dynParam,
                                    uint32_t* num_detectedPrmb_addr,
                                    uint32_t* prmbIndex_estimates_addr,
                                    float* prmbDelay_estimates_addr,
                                    float* prmbPower_estimates_addr,
                                    float* ant_rssi_addr,
                                    float* rssi_addr,
                                    float* interference_addr,
                                    uint32_t*& prev_num_detectedPrmb_addr,
                                    uint32_t*& prev_prmbIndex_estimates_addr,
                                    float*& prev_prmbDelay_estimates_addr,
                                    float*& prev_prmbPower_estimates_addr,
                                    float*& prev_ant_rssi_addr,
                                    float*& prev_rssi_addr,
                                    float*& prev_interference_addr,
                                    uint16_t nTotCellOcca,
                                    uint16_t& nPrevOccaProc,
                                    uint16_t nOccaProc,
                                    uint16_t maxAntenna,
                                    uint max_l_oran_ant,
                                    uint max_ant_u,
                                    uint max_nfft,
                                    int max_zoneSizeExt,
                                    std::vector<char>& activeOccasions,
                                    std::vector<char>& prevActiveOccasions)
{
    using Tcomplex = cuFloatComplex;
    using Tscalar = float;

    for(int i = 0; i < nOccaProc; ++i)
    {
        uint16_t occaPrmStaticIdx = h_dynParam[i].occaPrmStatIdx;
        if(activeOccasions[occaPrmStaticIdx] != prevActiveOccasions[occaPrmStaticIdx])
        {
            cudaGraphNodeSetEnabled(graphInstance, nodes[GraphNodeType::FFTNode + i], activeOccasions[occaPrmStaticIdx]);
            prevActiveOccasions[occaPrmStaticIdx] = activeOccasions[occaPrmStaticIdx];
        }
    }

    if(nPrevOccaProc != nOccaProc || prev_num_detectedPrmb_addr != num_detectedPrmb_addr ||
        prev_prmbIndex_estimates_addr != prmbIndex_estimates_addr || prev_prmbDelay_estimates_addr != prmbDelay_estimates_addr ||
        prev_prmbPower_estimates_addr != prmbPower_estimates_addr || prev_ant_rssi_addr != ant_rssi_addr ||
        prev_rssi_addr != rssi_addr || prev_interference_addr != interference_addr)
    {
        if(nPrevOccaProc != nOccaProc)
        {
            dim3 block_dim(128, 1, 1);
            cudaKernelNodeParams kernelNodeParams = {0};
            // Update: nMaxOccasions -> nOccaProc
            void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &maxAntenna,
                                &nOccaProc};

            kernelNodeParams.func = (void *)memsetRssi<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((nOccaProc * (maxAntenna + 2) + block_dim.x - 1) / block_dim.x, 1, 1);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::MemsetRSSI],
                                            &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || rssi_addr != prev_rssi_addr)
        {
            dim3 block_dim(128, 1, 1);
            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &rssi_addr,
                                &max_l_oran_ant};

            kernelNodeParams.func = (void *)prach_compute_rssi<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((max_l_oran_ant + block_dim.x - 1) / block_dim.x, nOccaProc, 1);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::ComputeRSSI],
                                            &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || ant_rssi_addr != prev_ant_rssi_addr)
        {
            dim3 block_dim(128, 1, 1);
            cudaKernelNodeParams kernelNodeParams = {0};
            // Update: nMaxOccasions -> nOccaProc
            void *kernelArgs[5] = {(void *)&ant_rssi_addr, (void *)&d_dynParam, &d_staticParam,
                                &maxAntenna, &nOccaProc};

            kernelNodeParams.func = (void *)memcpyRssi<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((nOccaProc * maxAntenna + block_dim.x - 1) / block_dim.x, 1, 1);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::MemcpyRSSI],
                                            &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || ant_rssi_addr != prev_ant_rssi_addr)
        {
            cudaKernelNodeParams kernelNodeParams = {0};
            // Update: nMaxOccasions -> nOccaProc
            void *kernelArgs[4] = {(void *)&d_dynParam, &d_staticParam,
                                &max_ant_u, &nOccaProc};

            kernelNodeParams.func = (void *)prach_compute_correlation<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3(max_ant_u * nOccaProc);
            kernelNodeParams.blockDim = dim3(max_nfft);
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::ComputeCorrelationNode],
                                            &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc)
        {
            assert(max_zoneSizeExt <= 512);
            dim3 block_dim(max_zoneSizeExt > NUM_THREAD ? max_zoneSizeExt : NUM_THREAD);
            int Nzone = block_dim.x/max_zoneSizeExt;

            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[3] = {(void *)&d_dynParam, &d_staticParam,
                                &max_zoneSizeExt};

            kernelNodeParams.func = (void *)prach_compute_pdp<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((maxAntenna * CUPHY_PRACH_RX_NUM_PREAMBLE + Nzone-1) / Nzone, nOccaProc);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
                cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::ComputePDPNode],
                                                &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || prev_num_detectedPrmb_addr != num_detectedPrmb_addr ||
        prev_prmbIndex_estimates_addr != prmbIndex_estimates_addr || prev_prmbDelay_estimates_addr != prmbDelay_estimates_addr ||
        prev_prmbPower_estimates_addr != prmbPower_estimates_addr || prev_interference_addr != interference_addr)
        {
            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[7] = {(void *)&d_dynParam, &d_staticParam,
                                &num_detectedPrmb_addr, &prmbIndex_estimates_addr, &prmbDelay_estimates_addr,
                                &prmbPower_estimates_addr, &interference_addr};

            kernelNodeParams.func = (void *)prach_search_pdp<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3(nOccaProc);
            kernelNodeParams.blockDim =dim3(CUPHY_PRACH_RX_NUM_PREAMBLE);
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
                cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::SearchPDPNode],
                                                &kernelNodeParams));
        }

        nPrevOccaProc = nOccaProc;
        prev_num_detectedPrmb_addr = num_detectedPrmb_addr;
        prev_prmbIndex_estimates_addr = prmbIndex_estimates_addr;
        prev_prmbDelay_estimates_addr = prmbDelay_estimates_addr;
        prev_prmbPower_estimates_addr = prmbPower_estimates_addr;
        prev_ant_rssi_addr = ant_rssi_addr;
        prev_rssi_addr = rssi_addr;
        prev_interference_addr = interference_addr;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachLaunchGraph(cudaGraphExec_t graphInstance, cudaStream_t strm)
{
    CUDA_CHECK_EXCEPTION(cudaGraphLaunch(graphInstance, strm));
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachReceiver(const PrachInternalDynParamPerOcca* d_dynParam,
                        const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                        const PrachInternalDynParamPerOcca* h_dynParam,
                        const PrachInternalStaticParamPerOcca* h_staticParam,
                        uint32_t* num_detectedPrmb_addr,
                        uint32_t* prmbIndex_estimates_addr,
                        float* prmbDelay_estimates_addr,
                        float* prmbPower_estimates_addr,
                        float* ant_rssi_addr,
                        float* rssi_addr,
                        float* interference_addr,
                        uint16_t nOccaProc,
                        uint16_t maxAntenna,
                        uint max_l_oran_ant,
                        uint max_ant_u,
                        uint max_nfft,
                        int max_zoneSizeExt,
                        uint cudaDeviceArch,
                        cudaStream_t strm) {

    return launch_templated_prach_receiver_kernels<cuFloatComplex, float>(d_dynParam,
                                                                    d_staticParam,
                                                                    h_dynParam,
                                                                    h_staticParam,
                                                                    num_detectedPrmb_addr,
                                                                    prmbIndex_estimates_addr,
                                                                    prmbDelay_estimates_addr,
                                                                    prmbPower_estimates_addr,
                                                                    ant_rssi_addr,
                                                                    rssi_addr,
                                                                    interference_addr,
                                                                    nOccaProc,
                                                                    maxAntenna,
                                                                    max_l_oran_ant,
                                                                    max_ant_u,
                                                                    max_nfft,
                                                                    max_zoneSizeExt,
                                                                    cudaDeviceArch,
                                                                    strm);

    return CUPHY_STATUS_NOT_SUPPORTED;
}
