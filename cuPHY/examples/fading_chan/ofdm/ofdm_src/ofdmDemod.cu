/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
 
#include "ofdmDemod.cuh"
#include <cufftdx.hpp>
using namespace ofdm_demodulate;

/**
 * @brief main kernel for ofdm demodulation
 * 
 * @tparam FFT FFT configurations, see cuFFTdx documents for detals
 * @tparam Tscalar scalar template, must match with Tcomplex
 * @tparam Tcomplex comlex template, must match with Tscalar
 * @param ofdmDeModdynDescpr ofdm demodulation dynamic descriptor
 * 
 * @param GridDim m_ofdmDeModdynDescprCpu -> N_rxLayer, cuphyCarrierPrms -> N_symble_slot / OFDM_FFTs_PER_BLOCK_CONST_, 1
 * @param BlockDim defuallt set by cuFFTdx
 */
template<typename FFT, typename Tscalar, typename Tcomplex>
__launch_bounds__(FFT::max_threads_per_block)
static __global__ void ofdmDeMod_fft_kernel(ofdmDeModDynDescr_t<Tscalar, Tcomplex> * ofdmDeModdynDescpr)
{
    using namespace cufftdx;
    
    // Registers
    cuComplex thread_data[FFT::storage_size];
    Tcomplex * timeDataIn = ofdmDeModdynDescpr -> timeDataIn;
    Tcomplex * freqDataOut = ofdmDeModdynDescpr -> freqDataOut;
    uint N_sc_over_2 = ofdmDeModdynDescpr -> N_sc >> 1; // divide by 2
    uint N_FFT = ofdmDeModdynDescpr -> N_FFT;
    // Local batch id of this FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Global batch id of this FFT in CUDA grid is equal to number of batches per CUDA block (ffts_per_block)
    // times CUDA block id, plus local batch id.
    // const unsigned int global_fft_id = (blockIdx.x * FFT::ffts_per_block) + local_fft_id;
    const unsigned int global_fft_id = (blockIdx.x * gridDim.y + blockIdx.y) * FFT::ffts_per_block + local_fft_id;

    /*-------------------   remove CPs from tx dand load data into shared memoery for FFT-------------------*/
    // Load freq data from global memory to registers
    
    // calculate read data positions
    const unsigned int freq_offset = ofdmDeModdynDescpr -> N_sc * global_fft_id;
    const unsigned int CP_current = ofdmDeModdynDescpr -> cpInfo[blockIdx.y * FFT::ffts_per_block + threadIdx.y]; // CP length for current OFDM symbol
    const unsigned int CP_offset  = ofdmDeModdynDescpr -> cpInfo[blockIdx.y * FFT::ffts_per_block + threadIdx.y + (ofdmDeModdynDescpr -> N_symble_slot)]; // CP_offset in this layer
    const unsigned int time_offset = cufftdx::size_of<FFT>::value * global_fft_id + CP_offset + (ofdmDeModdynDescpr -> cpInfo[(ofdmDeModdynDescpr -> N_symble_slot) * 2 - 1]) * blockIdx.x; // FFT size + CP offset in this layer + CP offset in previous layers
    const unsigned int stride = FFT::stride;
    unsigned int       index  = threadIdx.x + time_offset;

#pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        // Make sure not to go out-of-bounds
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
        {
            #ifdef USE_MEMOERY_FFT_SHIFT_  // use fftshift later
            thread_data[i].x = timeDataIn[index].x;
            thread_data[i].y = timeDataIn[index].y;
            #else // times 1 or -1 to real part if no fftshift
            if(index & 1) // last bit is 1
            {
                thread_data[i].x =  - timeDataIn[index].x;
                thread_data[i].y =  - timeDataIn[index].y;
            }
            else
            {
                thread_data[i].x = timeDataIn[index].x;
                thread_data[i].y = timeDataIn[index].y;
            }
            #endif
            index += stride;
        }
    }

    // FFT::shared_memory_size bytes of shared memory
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];

    // Execute FFT
    FFT().execute(thread_data, shared_mem);

    /*-------------------   Extract freq signals and save to global memory  -------------------*/
    index = threadIdx.x;
#pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
        {
            thread_data[i].x = thread_data[i].x * (ofdmDeModdynDescpr -> sqrt_N_FFT_inverse);  // normalize by sqrt(N_FFT)
            thread_data[i].y = thread_data[i].y * (ofdmDeModdynDescpr -> sqrt_N_FFT_inverse);  // normalize by sqrt(N_FFT)
            #ifdef USE_MEMOERY_FFT_SHIFT_ 
            if(index < N_sc_over_2)
            {
                freqDataOut[index + freq_offset + N_sc_over_2].x = thread_data[i].x;;
                freqDataOut[index + freq_offset + N_sc_over_2].y = thread_data[i].y;;
            }
            else if(index >= (N_FFT - N_sc_over_2))
            {
                freqDataOut[index + freq_offset - N_FFT + N_sc_over_2].x = thread_data[i].x;
                freqDataOut[index + freq_offset - N_FFT + N_sc_over_2].y = thread_data[i].y;
            }
            #else
            if(index >= ((N_FFT >> 1) - N_sc_over_2) && index < ((N_FFT >> 1) + N_sc_over_2) )  // Middle part
            {
                freqDataOut[index + freq_offset - (N_FFT >> 1) + N_sc_over_2].x = thread_data[i].x;
                freqDataOut[index + freq_offset - (N_FFT >> 1) + N_sc_over_2].y = thread_data[i].y;
            }
            #endif
            index += stride;
        }
        // printf("FFT out: threadIdx.x=%d, i=%d, (i * stride + threadIdx.x) = %d: thread_data[i].x = %f, thread_data[i].y = %f \n", threadIdx.x, i, (i * stride + threadIdx.x), thread_data[i].x, thread_data[i].y);
    }
}

/**
 * @brief  kernel for ofdm demodulation for PRACH
 * 
 * @tparam FFT FFT configurations, see cuFFTdx documents for detals
 * @tparam Tscalar scalar template, must match with Tcomplex
 * @tparam Tcomplex comlex template, must match with Tscalar
 * @param ofdmDeModdynDescpr ofdm demodulation dynamic descriptor
 * 
 * @param GridDim m_ofdmDeModdynDescprCpu -> N_rxLayer, cuphyCarrierPrms -> N_rep / OFDM_FFTs_PER_BLOCK_CONST_, 1
 * @param BlockDim defuallt set by cuFFTdx
 */
template<typename FFT, typename Tscalar, typename Tcomplex>
__launch_bounds__(FFT::max_threads_per_block)
static __global__ void ofdmDeMod_fft_prach_kernel(ofdmDeModDynDescr_t<Tscalar, Tcomplex> * ofdmDeModdynDescpr)
{
    using namespace cufftdx;
    
    // Registers
    cuComplex thread_data[FFT::storage_size];
    Tcomplex * timeDataIn               = ofdmDeModdynDescpr->timeDataIn;
    Tcomplex * freqDataOut              = ofdmDeModdynDescpr->freqDataOut;

    uint     N_FFT                      = ofdmDeModdynDescpr->Nfft_RA;
    uint32_t L_RA                       = ofdmDeModdynDescpr->L_RA;
    uint32_t startSC                    = ofdmDeModdynDescpr->startSC;
    uint32_t cpLenPrach                 = ofdmDeModdynDescpr->cpLenPrach;
    uint32_t preambleSampStart          = ofdmDeModdynDescpr->preambleSampStart;
    uint32_t N_samp_slot                = ofdmDeModdynDescpr->N_samp_slot;
    uint32_t Nrep                       = ofdmDeModdynDescpr->Nrep;
    uint32_t Nsamp_oran                 = ofdmDeModdynDescpr->Nsamp_oran;
    float    sqrt_N_FFT_inverse         = ofdmDeModdynDescpr->sqrt_N_FFT_inverse;
    uint32_t kBar                       = ofdmDeModdynDescpr->kBar;
    const unsigned int stride           = FFT::stride;

    // Local batch id of this FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id     = threadIdx.y;
    // Global batch id of this FFT in CUDA grid 
    const unsigned int global_fft_id    = (blockIdx.x * gridDim.y + blockIdx.y) * FFT::ffts_per_block + local_fft_id;

    /*-------------------   remove CP from tx dand load data into shared memoery for FFT-------------------*/
    // Load freq data from global memory to registers
    
    // calculate read data positions
    const unsigned int CP_offset        = cpLenPrach;

    // repitition index in this layer
    const unsigned int repIdxThisLayer  = blockIdx.y * FFT::ffts_per_block + threadIdx.y;

    // Rx antenna index
    const unsigned int rxAntIdx         = blockIdx.x;
    
    // time offset: starting sample index for this rx antenna + starting sample index for preamble + CP offset in this layer + FFT size*repitition index
    const unsigned int time_offset      = N_samp_slot * rxAntIdx + preambleSampStart + CP_offset + repIdxThisLayer * cufftdx::size_of<FFT>::value;

    unsigned int       index            = threadIdx.x + time_offset;

#pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        // Make sure not to go out-of-bounds
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
        {
            thread_data[i].x = timeDataIn[index].x;
            thread_data[i].y = timeDataIn[index].y;

            index += stride;
        }
    }

    // FFT::shared_memory_size bytes of shared memory
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];

    // Execute FFT
    FFT().execute(thread_data, shared_mem);

    /*-------------------   Extract freq signals and save to global memory  -------------------*/
    index = threadIdx.x;
    
    const unsigned int freq_offset      = Nsamp_oran * global_fft_id;

    const unsigned int loadFftOutStart  = startSC % N_FFT;
    const unsigned int loadFftOutend    = (startSC+L_RA-1) % N_FFT;
    const unsigned int Nsamp_right      = Nsamp_oran - L_RA - kBar;

    if (index == 0) {
        for (int outIdx = 0; outIdx < kBar; outIdx++) {
            freqDataOut[freq_offset + outIdx].x = 0;
            freqDataOut[freq_offset + outIdx].y = 0;
        }

        for (int outIdx = 0; outIdx< Nsamp_right; outIdx++) {
            freqDataOut[freq_offset + kBar + L_RA + outIdx].x = 0;
            freqDataOut[freq_offset + kBar + L_RA + outIdx].y = 0;
        }
    }

#pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
        {
            thread_data[i].x = thread_data[i].x * sqrt_N_FFT_inverse;  // normalize by sqrt(N_FFT)
            thread_data[i].y = thread_data[i].y * sqrt_N_FFT_inverse;  // normalize by sqrt(N_FFT)

            if (loadFftOutStart < loadFftOutend) {
                if (index >= loadFftOutStart && index <= loadFftOutend) {
                    freqDataOut[freq_offset + index - loadFftOutStart + kBar].x = thread_data[i].x;
                    freqDataOut[freq_offset + index - loadFftOutStart + kBar].y = thread_data[i].y;
                }
            } else { // loadFftOutend < loadFftOutStart
                if (index >= loadFftOutStart) {
                    freqDataOut[freq_offset + index - loadFftOutStart + kBar].x = thread_data[i].x;
                    freqDataOut[freq_offset + index - loadFftOutStart + kBar].y = thread_data[i].y;
                } else if (index <= loadFftOutend) {
                    freqDataOut[freq_offset + index + N_FFT - loadFftOutStart + kBar].x = thread_data[i].x;
                    freqDataOut[freq_offset + index + N_FFT - loadFftOutStart + kBar].y = thread_data[i].y;
                }
            }

            index += stride;
        }
    }
}

/**
 * @brief config FFT class for later get fft kernel handle
 */
template<typename Tscalar, typename Tcomplex>
using fftKernelHandle = void (*)(ofdmDeModDynDescr_t<Tscalar, Tcomplex> * ofdmDeModdynDescpr);
// Choose FFT kernel
template<typename Tscalar, typename Tcomplex, unsigned int FftSize, unsigned int Arch>
fftKernelHandle<Tscalar, Tcomplex> ofdmDeMod_get_fft_param(dim3& block_dim, uint& shared_memory_size, const bool prach) 
{ 
    using namespace cufftdx;

    // use predefined numbers
    using FFT = decltype(Size<FftSize>() + Precision<float>() + Type<fft_type::c2c>()
                                + Direction<fft_direction::forward>()
                                + FFTsPerBlock<OFDM_FFTs_PER_BLOCK_CONST_>()// + ElementsPerThread<8>()
                                + SM<Arch>() + Block());

    block_dim = FFT::block_dim;
    shared_memory_size = FFT::shared_memory_size;

    if (prach) { // for PRACH
        return ofdmDeMod_fft_prach_kernel<FFT, Tscalar, Tcomplex>;
    } else { // for non-PRACH
        return ofdmDeMod_fft_kernel<FFT, Tscalar, Tcomplex>;
    }
 }

 /**
  * @brief get fft kernel handles
  * 
  * @param Nfft FFT size
  * @param cudaDeviceArch GPU device arch
  * @param block_dim auto config by cuFFTdx
  * @param shared_memory_size auto config by cuFFTdx
  * @param prach indicator for PRACH
  * @return fftKernelHandle<Tscalar, Tcomplex> 
  *
  * @note To conserve memeory, only selected FFT size and cudaDeviceArch are added. If your Nfft and cudaDeviceArch are not in the below list, please add them and retry the build
  */
template<typename Tscalar, typename Tcomplex>
fftKernelHandle<Tscalar, Tcomplex> ofdmDeMod_get_fft_param(const int Nfft, unsigned int cudaDeviceArch, dim3& block_dim, uint& shared_memory_size, const bool prach) 
{ 
    // current only support cudaDeviceArch = 800
    switch(Nfft) 
    {
        case 256:
            return ofdmDeMod_get_fft_param<Tscalar, Tcomplex,  512, 800>(block_dim, shared_memory_size, prach);
            break;
        case 512:
            return ofdmDeMod_get_fft_param<Tscalar, Tcomplex,  512, 800>(block_dim, shared_memory_size, prach);
            break;
        case 1024:
            return ofdmDeMod_get_fft_param<Tscalar, Tcomplex,  1024, 800>(block_dim, shared_memory_size, prach);
            break;
        case 2048:
            return ofdmDeMod_get_fft_param<Tscalar, Tcomplex,  2048, 800>(block_dim, shared_memory_size, prach);
            break;
        case 4096:
            return ofdmDeMod_get_fft_param<Tscalar, Tcomplex,  4096, 800>(block_dim, shared_memory_size, prach);
            break;
        default:
            printf("Unsupported FFT length %d or cudaDeviceArch %d in OFDM demodulation, please add your Nfft or cudaDeviceArch into ofdmDeMod_get_fft_param and retry\n", Nfft, cudaDeviceArch); 
            assert(false);
            return nullptr;
    }
    return nullptr;
}

template <typename Tscalar, typename Tcomplex> 
ofdmDeModulate<Tscalar, Tcomplex>::ofdmDeModulate(cuphyCarrierPrms_t * cuphyCarrierPrms, Tcomplex * timeDataIn, Tcomplex * freqDataOut, bool prach, cudaStream_t strm)
{
    m_prach = prach;

    uint mu = cuphyCarrierPrms -> mu;
    uint N_symble_slot = cuphyCarrierPrms -> N_symble_slot;
    //m_N_FFT = cuphyCarrierPrms -> N_FFT;
    m_ofdmDeModdynDescprCpu = new ofdmDeModDynDescr_t<Tscalar, Tcomplex>;
    m_ofdmDeModdynDescprCpu -> N_FFT = cuphyCarrierPrms -> N_FFT;
    m_ofdmDeModdynDescprCpu -> sqrt_N_FFT_inverse = 1.0f/sqrt(cuphyCarrierPrms -> N_FFT);
    m_ofdmDeModdynDescprCpu -> N_sc = cuphyCarrierPrms -> N_sc;
    m_ofdmDeModdynDescprCpu -> N_rxLayer = cuphyCarrierPrms -> N_rxLayer;
    m_ofdmDeModdynDescprCpu -> mu = mu;
    m_ofdmDeModdynDescprCpu -> N_symble_slot = N_symble_slot;

    m_cpInfoCpu = nullptr;
    m_cpInfoGpu = nullptr;

    if (m_prach) {// for PRACH
        determinePrachParam(cuphyCarrierPrms);
    } else {// for non-PRACH 
        // calculate CP length
        uint16_t cpInfoLen = (N_symble_slot << 1);
        m_cpInfoCpu = new uint16_t[cpInfoLen]; // [CP info, accumCP] 

        uint symbol0IdxPerSubFrame = (cuphyCarrierPrms -> id_slot) * N_symble_slot;
    
        float T_c_over_T_samp = float(cuphyCarrierPrms->f_samp)/float(cuphyCarrierPrms->f_c);

        if(cuphyCarrierPrms -> cpType == 0) // normal CP length
        {
            uint16_t lenCP0 = (((144 >> mu) + 16) << (cuphyCarrierPrms -> kappa_bits))*T_c_over_T_samp; //(144+16)/2048*Nfft;
            uint16_t lenCP1 = ((144 >> mu) << (cuphyCarrierPrms -> kappa_bits))*T_c_over_T_samp; // 144/2048*Nfft;
        
            for(uint8_t symbolIdx = 0; symbolIdx < N_symble_slot; symbolIdx++)
            {
                m_cpInfoCpu[symbolIdx] = lenCP1;
            }
            if(mu == 0 && symbol0IdxPerSubFrame == 0)
            {
                m_cpInfoCpu[0] = lenCP0;
                m_cpInfoCpu[7] = lenCP0;
            } // check number of OFDM symbols per layer
            else if(mu != 0 && (symbol0IdxPerSubFrame == 0 || symbol0IdxPerSubFrame == (7 << mu)))
            {
                m_cpInfoCpu[0] = lenCP0;
            }
        } else {
            if(mu != 2)
            {
                printf("Error! Extended CP only applible in numerology 2! \n");
                exit(1);
            }
            uint lenCP1 = ((512 >> mu) << (cuphyCarrierPrms -> kappa_bits))*T_c_over_T_samp;
            for(uint8_t symbolIdx = 0; symbolIdx < N_symble_slot; symbolIdx++)
            {
                m_cpInfoCpu[symbolIdx] = lenCP1;
            }
        }
    
        // calculate assumualte CP
        m_cpInfoCpu[N_symble_slot] = m_cpInfoCpu[0];
        for(uint8_t symbolIdx = 1; symbolIdx < N_symble_slot; symbolIdx++)
        {
            m_cpInfoCpu[symbolIdx + N_symble_slot] = m_cpInfoCpu[symbolIdx] + m_cpInfoCpu[symbolIdx + N_symble_slot -1];
        }
        // copy CP info
        cudaMalloc((void**)&m_cpInfoGpu, sizeof(uint16_t) * cpInfoLen);
        m_ofdmDeModdynDescprCpu -> cpInfo = m_cpInfoGpu;
        cudaMemcpy(m_cpInfoGpu, m_cpInfoCpu, sizeof(uint16_t) * cpInfoLen, cudaMemcpyHostToDevice);
    }
    
    m_ofdmDeModdynDescprCpu -> timeDataIn = timeDataIn;
    m_ofdmDeModdynDescprCpu -> freqDataOut = freqDataOut;

    // copy dynamic descriptor to GPU
    cudaMalloc((void**)&m_ofdmDeModdynDescprGpu, sizeof(ofdmDeModDynDescr_t<Tscalar, Tcomplex>));
    cudaMemcpy(m_ofdmDeModdynDescprGpu, m_ofdmDeModdynDescprCpu, sizeof(ofdmDeModDynDescr_t<Tscalar, Tcomplex>), cudaMemcpyHostToDevice);

    // Kernel launch config
    m_pLaunchCfg = new launchCfg_t;

    // set up kernel
    using namespace cufftdx;
    uint shared_memory_size = 0;
    dim3 grid_dim;

    if (m_prach) {// for PRACH
        grid_dim = dim3(m_ofdmDeModdynDescprCpu -> N_rxLayer, cuphyCarrierPrms->N_rep / OFDM_FFTs_PER_BLOCK_CONST_, 1); 
    } else {// for non-PRACH
        grid_dim = dim3(m_ofdmDeModdynDescprCpu -> N_rxLayer, cuphyCarrierPrms -> N_symble_slot / OFDM_FFTs_PER_BLOCK_CONST_, 1); // OFDM_SYMBOLS_PER_SLOT = 14)
    }
     
    dim3 block_dim;
    const uint cudaDeviceArch = get_cuda_device_arch();
    auto kernelPtr = ofdmDeMod_get_fft_param<Tscalar, Tcomplex>(m_ofdmDeModdynDescprCpu -> N_FFT, cudaDeviceArch, block_dim, shared_memory_size, m_prach);
    m_pLaunchCfg->kernelArgs[0] = &m_ofdmDeModdynDescprGpu;

    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pLaunchCfg->kernelNodeParamsDriver;
    
    CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, reinterpret_cast<void*>(kernelPtr)));
    kernelNodeParamsDriver.blockDimX = block_dim.x;
    kernelNodeParamsDriver.blockDimY = block_dim.y;
    kernelNodeParamsDriver.blockDimZ = block_dim.z;

    kernelNodeParamsDriver.gridDimX = grid_dim.x;
    kernelNodeParamsDriver.gridDimY = grid_dim.y;
    kernelNodeParamsDriver.gridDimZ = grid_dim.z;
    kernelNodeParamsDriver.sharedMemBytes = shared_memory_size;
    kernelNodeParamsDriver.kernelParams = &(m_pLaunchCfg->kernelArgs[0]);
    kernelNodeParamsDriver.extra = NULL;

    // pre load FFT kernel to avoid first run timing issue
    run(strm);
}

template <typename Tscalar, typename Tcomplex> 
ofdmDeModulate<Tscalar, Tcomplex>::~ofdmDeModulate()
{
    if (m_cpInfoCpu) delete[] m_cpInfoCpu;
    if (m_cpInfoGpu) cudaFree(m_cpInfoGpu);
    
    cudaFree(m_ofdmDeModdynDescprGpu);
    delete m_ofdmDeModdynDescprCpu;
    delete m_pLaunchCfg;
}

template <typename Tscalar, typename Tcomplex> 
void ofdmDeModulate<Tscalar, Tcomplex>::determinePrachParam(cuphyCarrierPrms_t* cuphyCarrierPrms)
{
    uint32_t startRaSym     = cuphyCarrierPrms->startRaSym + cuphyCarrierPrms->n_slot_RA_sel*14;
    uint32_t k_const        = cuphyCarrierPrms->k_const;
    uint32_t N_CP_l_mu      = 0;
    uint32_t t_start_RA;
    uint32_t t_start_l_mu   = 0;
    uint32_t mu             = cuphyCarrierPrms->mu;
    uint32_t N_u            = cuphyCarrierPrms->N_u;
    uint32_t N_CP_RA        = cuphyCarrierPrms->N_CP_RA;
    uint32_t temp           = static_cast<uint32_t>(pow(2, mu));
    float T_c               = cuphyCarrierPrms->T_c;
    uint f_samp             = cuphyCarrierPrms->f_samp;
    uint32_t delta_f_RA     = cuphyCarrierPrms->delta_f_RA;

    uint32_t N_CP_l_RA;

    // find starting time for preamble
    for (int ll = 0; ll<=startRaSym; ll++) {
        // find t_start_RA
        if (ll == 0) {
            t_start_l_mu    = 0;
            N_CP_l_mu       = 144*k_const*temp + 16*k_const;
        } else {
            if (ll-1 == 0 || ll-1 == 7*temp) {
                // increase CP length for symbol 0 and 7
                N_CP_l_mu   = 144*k_const*temp + 16*k_const;
            } else {
                N_CP_l_mu   = 144*k_const*temp;
            }
            t_start_l_mu   += (cuphyCarrierPrms->N_u_mu + N_CP_l_mu)*T_c;
        }
    }
    t_start_RA              = t_start_l_mu;

    // find CP length for preamble
    uint32_t n;
    if (delta_f_RA == 1250 || delta_f_RA == 5000) {
        n = 0;
    } else if (delta_f_RA == 15000 || delta_f_RA == 30000 || delta_f_RA == 60000 || delta_f_RA == 120000) {
        n = 0;
        float tempf = t_start_RA + (N_u + N_CP_RA)*T_c;
        if (t_start_RA <= 0 && tempf > 0) {
            n++;
        }

        if (t_start_RA <= 0.5e-3 && tempf > 0.5e-3) {
            n++;
        }
    }

    //increase cp length based on n
    N_CP_l_RA                                   = N_CP_RA + n*16*k_const;  
    
    uint32_t preambleSampStart                  = round(static_cast<float>(t_start_RA)*f_samp); // starting point right before CP
    // To fit for per slot process
    preambleSampStart = preambleSampStart % cuphyCarrierPrms->N_samp_slot;
    uint32_t preambleSampEnd                    = floor(preambleSampStart + (N_CP_l_RA + N_u)*T_c*f_samp - 1);
    
    m_ofdmDeModdynDescprCpu->preambleSampStart  = preambleSampStart;
    m_ofdmDeModdynDescprCpu->preambleSampEnd    = preambleSampEnd;

    // currently only support K == 1 so that Nfft_RA = cuphyCarrierPrms->N_FFT
    uint32_t Nfft_RA                            = cuphyCarrierPrms->N_FFT;
    m_ofdmDeModdynDescprCpu->Nfft_RA            = Nfft_RA;

    uint32_t N_samp_RA                          = static_cast<uint32_t>(N_u*T_c*f_samp);
    uint32_t Nrep                               = N_samp_RA/Nfft_RA;
    m_ofdmDeModdynDescprCpu->Nrep               = Nrep;

    uint32_t cpLenPrach                         = N_CP_l_RA*T_c*f_samp;
    m_ofdmDeModdynDescprCpu->cpLenPrach         = cpLenPrach;

    m_ofdmDeModdynDescprCpu->L_RA               = cuphyCarrierPrms->L_RA;

    int startSC                                 = (cuphyCarrierPrms->K * cuphyCarrierPrms->k1 + cuphyCarrierPrms->kBar) % Nfft_RA;
    startSC = startSC >= 0? startSC:(startSC+Nfft_RA);
    startSC++; // skip DC
    m_ofdmDeModdynDescprCpu->startSC            = startSC;

    m_ofdmDeModdynDescprCpu->N_samp_slot        = cuphyCarrierPrms->N_samp_slot;

    m_ofdmDeModdynDescprCpu->Nsamp_oran         = cuphyCarrierPrms->L_RA == 139? 144 : 864;

    m_ofdmDeModdynDescprCpu->kBar               = cuphyCarrierPrms->kBar;
}

template <typename Tscalar, typename Tcomplex> 
void ofdmDeModulate<Tscalar, Tcomplex>::run(cudaStream_t strm)
{
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pLaunchCfg->kernelNodeParamsDriver;

    CUresult runStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
                                        kernelNodeParamsDriver.gridDimX,
                                        kernelNodeParamsDriver.gridDimY, 
                                        kernelNodeParamsDriver.gridDimZ,
                                        kernelNodeParamsDriver.blockDimX, 
                                        kernelNodeParamsDriver.blockDimY, 
                                        kernelNodeParamsDriver.blockDimZ,
                                        kernelNodeParamsDriver.sharedMemBytes,
                                        strm,
                                        kernelNodeParamsDriver.kernelParams,
                                        kernelNodeParamsDriver.extra);
}

template <typename Tscalar, typename Tcomplex> 
void ofdmDeModulate<Tscalar, Tcomplex>::printFreqSample(int printLen)
{
    Tcomplex * temp_CPU_buffer = new Tcomplex[printLen];

    cudaMemcpy(temp_CPU_buffer, m_ofdmDeModdynDescprCpu -> freqDataOut, printLen * sizeof(Tcomplex), cudaMemcpyDeviceToHost);

    for (int index=0; index< printLen; index++)
    {
        printf("index: %d: %1.4e + %1.4e  i\n", index, float(temp_CPU_buffer[index].x), float(temp_CPU_buffer[index].y));
    }
    printf("Done printing output freq domain signal from GPU \n");

    delete[] temp_CPU_buffer;
}