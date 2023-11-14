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
#include "tdl_pdp_table.h"
#include <cufftdx.hpp>
#include "cuphy.hpp"

template <typename Tscalar, typename Tcomplex> 
tdlChan<Tscalar, Tcomplex>::tdlChan(tdlConfig_t * tdlCfg, uint16_t randSeed, cudaStream_t strm)
{
    m_strm = strm;
    m_runMode = tdlCfg -> runMode;
    // Setup dynamic descriptor
    tdlDynDescrCpu = new  tdlDynDescr_t<Tscalar, Tcomplex>;
    // cell config params
    tdlDynDescrCpu -> nTxAnt = tdlCfg -> nTxAnt;
    tdlDynDescrCpu -> nRxAnt = tdlCfg -> nRxAnt;
    tdlDynDescrCpu -> nPath = tdlCfg -> numPath;
    tdlDynDescrCpu -> nBatchSamp = round(tdlCfg -> f_samp/ tdlCfg -> fBatch);
    float tSample =  1.0f / (tdlCfg -> f_samp);
    tdlDynDescrCpu -> cfoHz = (tdlCfg -> cfoHz);
    tdlDynDescrCpu -> cfoPhaseSamp = tSample * 2 * M_PI * (tdlCfg -> cfoHz);
    tdlDynDescrCpu -> tBatch = tSample * (tdlDynDescrCpu -> nBatchSamp) ;
    tdlDynDescrCpu -> maxDopplerShift = tdlCfg -> maxDopplerShift;
    tdlDynDescrCpu -> nDelaySample = roundf((tdlCfg -> delay) / tSample);
    // FIR related params, only store non-zero parameters
    // set TDL model parameters
    m_delayProfile = tdlCfg -> delayProfile;
    m_f_samp = tdlCfg -> f_samp;
    m_mimoCorrMat = NULL; // no correlation for now

    /*-------------------   read pdp from tables in tdl_pdp_table.h   -------------------*/
    if(tdlCfg -> useSimplifiedPdp) // set delay profile based on TS 38.141
    {
        switch(m_delayProfile)
        {
            case 'A': // TLDA30
                m_delaySpread = 30.0f;
                m_pdp = &pdp_38141_const[0];
                m_numTaps = 12;
                m_LosTap = false;
            break;
            case 'B': // TLDB100
                m_delaySpread = 100.0f;
                m_pdp = &pdp_38141_const[0] + 12*2;
                m_numTaps = 12;
                m_LosTap = false;
            break;
            case 'C': // TDLC300
                m_delaySpread = 300.0f;
                m_pdp = &pdp_38141_const[0] + 24*2;
                m_numTaps = 12;
                m_LosTap = false;
            break;
            default: // TLDA30 default
                m_delaySpread = 30.0f;
                m_pdp = &pdp_38141_const[0];
                m_numTaps = 12;
                m_LosTap = false;
            break;
        }
    }
    else // set delay profile based on TS 38.901
    {
        m_delaySpread = tdlCfg -> delaySpread; // customize delay spread
        switch(m_delayProfile)
        {
            case 'A':
                m_pdp = &pdp_38901_const[0];
                m_numTaps = 23;
                m_LosTap = false;
            break;

            case 'B':
                m_pdp = &pdp_38901_const[0] + 23*2;
                m_numTaps = 23;
                m_LosTap = false;
            break;

            case 'C':
                m_pdp = &pdp_38901_const[0] + 46*2;
                m_numTaps = 24;
                m_LosTap = false;
            break;

            case 'D':
                m_pdp = &pdp_38901_const[0] + 70*2;
                m_numTaps = 14;
                m_LosTap = true;
            break;

            case 'E':
                m_pdp = &pdp_38901_const[0] + 84*2;
                m_numTaps = 14;
                m_LosTap = true;
            break;

            default:
                m_pdp = &pdp_38901_const[0];
                m_numTaps = 23;
                m_LosTap = false;
            break;
        }
    }
    tdlDynDescrCpu -> LosTap = m_LosTap;
    // currently does not support TDL-D and TDL-E with LOS path
    // @todo will added later
    if(m_LosTap)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: TDL with LOS path is not supported yet");
        exit(1);
    }
    /*-------------------   calcualte FIR based on tdl config   -------------------*/
    // buffer for FIR filter 
    uint16_t maxlenFir = round(m_pdp[(m_numTaps-1)*2] * 1e-9 * m_f_samp * m_delaySpread) + 1;
    for(int tapIdx=0; tapIdx<m_numTaps; tapIdx++)
    {
        uint16_t firIdx = round(m_pdp[tapIdx*2] * 1e-9 * m_f_samp * m_delaySpread);
        if((tapIdx != 0) && (m_firNzIdx.back() == firIdx)) // if taps combined into one FIR coe
        {
            m_firNzPw.back() = float(m_firNzPw.back()) + float(pow(10, m_pdp[tapIdx *2 + 1] * 0.1f));
        }
        else // a new FIR coe
        {
            m_firNzPw.push_back(pow(10, m_pdp[tapIdx *2 + 1] * 0.1f));
            m_firNzIdx.push_back(firIdx);
        }
        // printf("firIdx = %d, m_firNzPw[firIdx] = %f\n", firIdx, m_firNzPw[firIdx]);
    }
    m_firNzLen = m_firNzIdx.size();
    tdlDynDescrCpu -> firNzLen = m_firNzLen;
    tdlDynDescrCpu -> firMaxLen = m_firNzIdx.back() + 1;
    // normalize firPW
    float sum_firNzPw = 0.0f;
    for(auto x : m_firNzPw)
    {
        sum_firNzPw += float(x);
    }
    // normalization targe: 1/sqrt(nPath)
    sum_firNzPw = 1.0f/(sum_firNzPw * (tdlDynDescrCpu -> nPath));
    for(int firIdx = 0; firIdx<m_firNzLen; firIdx ++)
    {
        // take sqrt to be multiplied for chan coe
        m_firNzPw[firIdx] = sqrt(float(m_firNzPw[firIdx]) * sum_firNzPw);
    }
    
    // copy to GPU
    cudaMalloc((void**)&(tdlDynDescrCpu -> firNzPw), sizeof(Tscalar) * m_firNzLen);
    cudaMalloc((void**)&(tdlDynDescrCpu -> firNzIdx), sizeof(uint16_t) * m_firNzLen);

    cudaMemcpyAsync(tdlDynDescrCpu -> firNzPw, m_firNzPw.data(), sizeof(Tscalar) * m_firNzLen, cudaMemcpyHostToDevice, m_strm);
    cudaMemcpyAsync(tdlDynDescrCpu -> firNzIdx, m_firNzIdx.data(), sizeof(uint16_t) * m_firNzLen, cudaMemcpyHostToDevice, m_strm);

    /*-------------------   setup channel input and output buffers   -------------------*/
    tdlDynDescrCpu -> timeSigLenPerAnt = tdlCfg -> timeSigLenPerAnt;
    tdlDynDescrCpu -> nBatch = (tdlDynDescrCpu -> timeSigLenPerAnt + tdlDynDescrCpu -> nBatchSamp - 1)/(tdlDynDescrCpu -> nBatchSamp);
    tdlDynDescrCpu -> nBatch = max(1, tdlDynDescrCpu -> nBatch);
    if(tdlDynDescrCpu -> timeSigLenPerAnt) // need to perform tx signal processing
    {
        tdlDynDescrCpu -> txTimeSigIn = reinterpret_cast<Tcomplex*>(tdlCfg -> txTimeSigIn);
        cudaMalloc((void**) &(tdlDynDescrCpu -> rxTimeSigOut), sizeof(Tcomplex) * (tdlDynDescrCpu -> timeSigLenPerAnt) * tdlDynDescrCpu -> nRxAnt);
    }
    else
    {
        tdlDynDescrCpu -> txTimeSigIn = NULL;
        tdlDynDescrCpu -> rxTimeSigOut = NULL;
    }
    
    // buffer for output channel
    m_timeChanSize = (tdlDynDescrCpu -> nTxAnt) * (tdlDynDescrCpu -> nRxAnt) * m_firNzLen * (tdlDynDescrCpu -> nBatch);
    tdlDynDescrCpu -> timeChanSize = m_timeChanSize;
    cudaMalloc((void**)&(tdlDynDescrCpu -> timeChan), sizeof(Tcomplex) * m_timeChanSize);

    // for curand states
    curandCreateGenerator(&m_Rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(m_Rng, randSeed);//random seed applied here
    m_randSize = (tdlDynDescrCpu -> nTxAnt) * (tdlDynDescrCpu -> nRxAnt) * m_firNzLen * (tdlDynDescrCpu -> nPath);
    cudaMalloc((void**) &(tdlDynDescrCpu -> thetaRand), 2 * m_randSize * sizeof(float));
    tdlDynDescrCpu -> PI_4_nPath = M_PI/4.0f/(tdlDynDescrCpu -> nPath); // calculate a constant pi/4/nPath on CPU
    
    // tdl freq channel coefficient, only when m_runMode=1
    tdlDynDescrCpu -> N_sc = tdlCfg -> freqConvertN_sc;
    if(m_runMode == 1)
    {
        tdlDynDescrCpu -> N_FFT = 4096;
        tdlDynDescrCpu -> freqChanNormalizeCoe = 1.0f * sqrt(tdlDynDescrCpu -> nTxAnt);// * sqrt(tdlDynDescrCpu -> N_sc) / sqrt(float(tdlDynDescrCpu -> N_FFT));
        tdlDynDescrCpu -> freqChanSize = (tdlDynDescrCpu -> nTxAnt) * (tdlDynDescrCpu -> nRxAnt) * (tdlDynDescrCpu -> N_sc);
        tdlDynDescrCpu -> N_sc_PRBG = tdlCfg -> N_sc_PRBG;
        tdlDynDescrCpu -> N_PRBG = (tdlDynDescrCpu -> N_sc)/(tdlDynDescrCpu -> N_sc_PRBG);
        if((tdlDynDescrCpu -> N_sc) != (tdlDynDescrCpu -> N_PRBG)*(tdlDynDescrCpu -> N_sc_PRBG))
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Error! N_sc ({}) is not a multiple of N_sc_PRBG ({})", tdlDynDescrCpu -> N_sc, tdlDynDescrCpu -> N_sc_PRBG);
            exit(1);
        }

        cudaMalloc((void**) &(tdlDynDescrCpu -> freqChanSC), sizeof(Tcomplex) * (tdlDynDescrCpu -> freqChanSize));
        cudaMalloc((void**) &(tdlDynDescrCpu -> freqChanPRBG), sizeof(Tcomplex) * (tdlDynDescrCpu -> freqChanSize) / (tdlDynDescrCpu -> N_sc_PRBG));
    }
    else
    {
        tdlDynDescrCpu -> freqChanSize = 0;
        tdlDynDescrCpu -> freqChanSC = NULL;
        tdlDynDescrCpu -> freqChanPRBG = NULL;
    }
    

    //copy dyndescriptor to GPU
    cudaMalloc((void**)&(tdlDynDescrGpu), sizeof(tdlDynDescr_t<Tscalar, Tcomplex>));
    cudaMemcpyAsync(tdlDynDescrGpu, tdlDynDescrCpu, sizeof(tdlDynDescr_t<Tscalar, Tcomplex>), cudaMemcpyHostToDevice, m_strm);

    // get kernel inputs
    m_refTime0 = 0.0f;
    m_args[0] = &tdlDynDescrGpu;
    m_args[1] = &m_refTime0;

    updateTapPathRand();
}

template <typename Tscalar, typename Tcomplex> 
tdlChan<Tscalar, Tcomplex>::~tdlChan()
{
    cudaFree(tdlDynDescrCpu -> firNzPw);
    cudaFree(tdlDynDescrCpu -> firNzIdx);
    cudaFree(tdlDynDescrCpu -> timeChan);
    cudaFree(tdlDynDescrCpu -> thetaRand);

    if(tdlDynDescrCpu -> timeSigLenPerAnt)
    {
        cudaFree(tdlDynDescrCpu -> rxTimeSigOut);
    }

    if(m_runMode == 1)
    {
        cudaFree(tdlDynDescrCpu -> freqChanSC);
        cudaFree(tdlDynDescrCpu -> freqChanPRBG);
    }

    cudaFree(tdlDynDescrGpu);
    delete tdlDynDescrCpu;

    curandDestroyGenerator(m_Rng);
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::run(float refTime0)
{
    m_refTime0 = refTime0;
    genTdltimeChan();
    if(tdlDynDescrCpu -> timeSigLenPerAnt) // has input signal
    {
        processTxSig();
    }
    if(m_runMode == 1) // freq channel required
    {
        genTdlFreqChan();
    }
    cudaStreamSynchronize(m_strm);
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::updateTapPathRand()
{
    curandStatus_t curandResult;
    curandResult = curandGenerateUniform(m_Rng, tdlDynDescrCpu -> thetaRand, 2*m_randSize); // for phase 
    if (curandResult != CURAND_STATUS_SUCCESS) 
    {
        std::string msg("Could not generate random number thetaRand: ");
        throw std::runtime_error(msg);
    }
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::genTdltimeChan()
{
    // generate time domain channel
    m_gridDim = {tdlDynDescrCpu->nBatch,1,1};
    m_blockDim = {tdlDynDescrCpu -> firNzLen, tdlDynDescrCpu -> nTxAnt, tdlDynDescrCpu -> nRxAnt};
    cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(genTdlChanCoeKernel<Tscalar, Tcomplex>));
    
    CUresult status = cuLaunchKernel(m_functionPtr, m_gridDim.x, m_gridDim.y, m_gridDim.z, m_blockDim.x, m_blockDim.y, m_blockDim.z, 0, m_strm, m_args, NULL);
    assert(status == CUDA_SUCCESS);

    // normalize the tdl time channel per TTI if macro ENABLE_NORMALIZATION_ is defined, not used by default
    #ifdef ENABLE_NORMALIZATION_
    m_gridDim = {1,1,1};
    m_blockDim = {THREADS_PER_BLOCK_NORMALIZATION_, 1, 1};
    cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(normalizeTimeChan<Tscalar, Tcomplex>));
    status = cuLaunchKernel(m_functionPtr, m_gridDim.x, m_gridDim.y, m_gridDim.z, m_blockDim.x, m_blockDim.y, m_blockDim.z, 0, m_strm, m_args, NULL);
    assert(status == CUDA_SUCCESS);
    #endif 
    // printf("TDL channel coefficients generated \n");
}

template <typename Tscalar, typename Tcomplex>
void tdlChan<Tscalar, Tcomplex>::processTxSig()
{
    m_gridDim = {tdlDynDescrCpu->nBatch,1,1};
    m_blockDim = {BLOCK_SAMPLE_, tdlDynDescrCpu -> nRxAnt, 1};

    /**
     * @todo: try to use dynamic config shared memory
     * 
     */

    // uint16_t firMaxLen = tdlDynDescrCpu -> firMaxLen;
    // uint16_t firNzLen = m_firNzLen;
    // __shared__ extern uint16_t firNzIdx[]; // firNzLen
    // uint shareMemBytes = sizeof(Tcomplex) * (firMaxLen + BLOCK_SAMPLE_);
    // __shared__ extern Tcomplex txTimeSigInBlock[]; // firMaxLen + BLOCK_SAMPLE_
    // shareMemBytes += sizeof(Tcomplex) * firNzLen * (tdlDynDescrCpu -> nRxAnt) * (tdlDynDescrCpu -> nTxAnt);
    // __shared__ extern Tcomplex chanCoeLocal[]; // firNzLen * nRxAnt * nTxAnt
    // shareMemBytes += sizeof(uint16_t) * firNzLen;

    cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(processInputKernel<Tscalar, Tcomplex>));
    CUresult status = cuLaunchKernel(m_functionPtr, m_gridDim.x, m_gridDim.y, m_gridDim.z, m_blockDim.x, m_blockDim.y, m_blockDim.z, 0, m_strm, m_args, NULL);
    assert(status == CUDA_SUCCESS);
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::printTimeChan(int printLen)
{
    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];
    cudaMemcpyAsync(tempCpuBuffer, tdlDynDescrCpu -> timeChan, sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int chanIdx=0; chanIdx<printLen; chanIdx++)
    {
        printf("chanIdx %d: %f + %f i \n", chanIdx, float(tempCpuBuffer[chanIdx].x), float(tempCpuBuffer[chanIdx].y));
    }
    printf("Done print tdl time channel! \n");
    delete[] tempCpuBuffer;
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::printFreqSCChan(int printLen)
{
    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];
    cudaMemcpyAsync(tempCpuBuffer, tdlDynDescrCpu -> freqChanSC, sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int chanIdx=0; chanIdx<printLen; chanIdx++)
    {
        printf("chanIdx %d: %f + %f i \n", chanIdx, float(tempCpuBuffer[chanIdx].x), float(tempCpuBuffer[chanIdx].y));
    }
    printf("Done print tdl freq channel on subcarriers! \n");
    delete[] tempCpuBuffer;
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::printFreqPRBGChan(int printLen)
{
    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];
    cudaMemcpyAsync(tempCpuBuffer, tdlDynDescrCpu -> freqChanPRBG, sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int chanIdx=0; chanIdx<printLen; chanIdx++)
    {
        printf("chanIdx %d: %f + %f i \n", chanIdx, float(tempCpuBuffer[chanIdx].x), float(tempCpuBuffer[chanIdx].y));
    }
    printf("Done print tdl freq channel on PRBGs! \n");
    delete[] tempCpuBuffer;
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::printRxTimeSig(int printLen)
{
    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];
    cudaMemcpyAsync(tempCpuBuffer, tdlDynDescrCpu -> rxTimeSigOut, sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int rxSigIdx=0; rxSigIdx<printLen; rxSigIdx++)
    {
        printf("rxSigIdx %d: %f + %f i \n", rxSigIdx, float(tempCpuBuffer[rxSigIdx].x), float(tempCpuBuffer[rxSigIdx].y));
    }
    printf("Done print rx time out signal! \n");
    delete[] tempCpuBuffer;
}

// get device arch for FFT kernel
static unsigned int get_cuda_device_arch() {
    int device;
    cudaGetDevice(&device);

    int major = 0;
    int minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::genTdlFreqChan()
// generate frequency domain chanel for both SC level or PRBG level
{
    // set up kernel
    // using namespace cufftdx;
    uint shared_memory_size = 0;
    //dim3 grid_dim = dim3((m_ofdmDeModdynDescprCpu -> N_txLayer) * (cuphyCarrierPrms -> N_symble_slot)); // OFDM_SYMBOLS_PER_SLOT = 14)
    dim3 gridDim = dim3(tdlDynDescrCpu -> nTxAnt, tdlDynDescrCpu -> nRxAnt / FFTs_PER_BLOCK_CONST_, 1); // OFDM_SYMBOLS_PER_SLOT = 14)
    dim3 blockDim;
    // const uint N_FFT = m_ofdmDeModdynDescprCpu -> N_FFT;
    const uint cudaDeviceArch = get_cuda_device_arch();
    auto kernelPtr = tdl_get_fft_param<Tscalar, Tcomplex>( tdlDynDescrCpu -> N_FFT, cudaDeviceArch, blockDim, shared_memory_size);
    
    // launch kernel for freq domain channel per SC
    cudaFunction_t functionPtr;
    cudaGetFuncBySymbol(&functionPtr, reinterpret_cast<void*>(kernelPtr));
    CUresult status = cuLaunchKernel(functionPtr, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, shared_memory_size, m_strm, m_args, NULL);
    assert(status == CUDA_SUCCESS);
    
    gridDim = dim3(tdlDynDescrCpu -> nTxAnt, 1, 1); // OFDM_SYMBOLS_PER_SLOT = 14)
    blockDim = dim3(tdlDynDescrCpu -> N_PRBG, tdlDynDescrCpu -> nRxAnt, 1);
    kernelPtr = convertSCtoPRBG<Tscalar, Tcomplex>;
    cudaGetFuncBySymbol(&functionPtr, reinterpret_cast<void*>(kernelPtr));
    status = cuLaunchKernel(functionPtr, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, m_strm, m_args, NULL);
    assert(status == CUDA_SUCCESS);
}

template<typename FFT, typename Tscalar, typename Tcomplex>
__launch_bounds__(FFT::max_threads_per_block)
static __global__ void tdl_fft_kernel(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr)
{
    // GRID(tdlDynDescrCpu -> nTxAnt, tdlDynDescrCpu -> nRxAnt / FFTs_PER_BLOCK_CONST_, 1)
    // BLOCK(by FFT default)
    using namespace cufftdx;
    
    // Registers
    cuComplex thread_data[FFT::storage_size];
    uint N_sc_over_2 = tdlDynDescr -> N_sc >> 1; // divide by 2
    uint N_FFT = tdlDynDescr -> N_FFT;
    uint16_t firNzLen = tdlDynDescr -> firNzLen;
    uint16_t firMaxLen = tdlDynDescr -> firMaxLen;
    // Local batch id of this FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Global batch id of this FFT in CUDA grid is equal to number of batches per CUDA block (ffts_per_block)
    // times CUDA block id, plus local batch id.
    const unsigned int global_fft_id = (blockIdx.x * gridDim.y + blockIdx.y) * FFT::ffts_per_block + local_fft_id;

    // Load freq data from global memory to registers
    const unsigned int freq_offsetSC = tdlDynDescr -> N_sc * global_fft_id;
    const unsigned int time_offset = firNzLen * global_fft_id; 
    const unsigned int stride = FFT::stride;
    unsigned int       index  = time_offset;

    // output buffer
    Tcomplex * freqChanSC = tdlDynDescr -> freqChanSC;
    Tcomplex * freqChanPRBG = tdlDynDescr -> freqChanPRBG;
    Tcomplex * timeChan = tdlDynDescr -> timeChan;
    
    // get input sinnals
    uint16_t * firNzIdx = tdlDynDescr -> firNzIdx;

    // FFT::shared_memory_size bytes of shared memory
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];// assuming FFT shared memoery size if much higher than firMaxLen
        
    // extern __shared__ Tcomplex shared_mem[];
    // Tcomplex * FFT_shared_mem = shared_mem + (FFT::ffts_per_block)*firMaxLen;
    // if(threadIdx.x == 0)
    // {
    //     printf("FFT::storage_size = %d, FFT::elements_per_thread = %d, stride = %d \n ", FFT::storage_size, FFT::elements_per_thread, stride);
    // }
    for(uint resetMemIdx = threadIdx.x; resetMemIdx < firMaxLen; resetMemIdx += blockDim.x)
    {
        shared_mem[firMaxLen * local_fft_id + resetMemIdx].x = 0.0;
        shared_mem[firMaxLen * local_fft_id + resetMemIdx].y = 0.0;
    }
    __syncthreads();

    if(threadIdx.x < firNzLen) // copy NZ taps into the shared
    {
        shared_mem[firMaxLen * local_fft_id + firNzIdx[threadIdx.x]].x = timeChan[time_offset + threadIdx.x].x;
        shared_mem[firMaxLen * local_fft_id + firNzIdx[threadIdx.x]].y = timeChan[time_offset + threadIdx.x].y;
    }
    __syncthreads();

    // Make sure not to go out-of-bounds
    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        if (i * stride + threadIdx.x < cufftdx::size_of<FFT>::value) 
        {
            #ifdef USE_MEMOERY_FFT_SHIFT_  // use fftshift later
            if(i * stride + threadIdx.x < firMaxLen)
            {
                thread_data[i].x = shared_mem[firMaxLen * local_fft_id + i * stride + threadIdx.x].x;
                thread_data[i].y = shared_mem[firMaxLen * local_fft_id + i * stride + threadIdx.x].y;
            }
            else
            {
                thread_data[i].x = 0.0f;
                thread_data[i].y = 0.0f;
            }
            #else // times 1 or -1 to real part if no fftshift
            if(i * stride + threadIdx.x < firNzLen)
            {
                if(index & 1) // last bit is 1
                {
                    thread_data[i].x =  - shared_mem[firMaxLen * local_fft_id + threadIdx.x].x;
                    thread_data[i].y =  - shared_mem[firMaxLen * local_fft_id + threadIdx.x].y;
                }
                else
                {
                    thread_data[i].x = shared_mem[firMaxLen * local_fft_id + threadIdx.x].x;
                    thread_data[i].y = shared_mem[firMaxLen * local_fft_id + threadIdx.x].y;
                }
            }
            else
            {
                thread_data[i].x = 0.0f;
                thread_data[i].y = 0.0f;
            }
            #endif
            index += stride;
        }
        // printf("FFT in: threadIdx.x=%d, i=%d, (i * stride + threadIdx.x) = %d: thread_data[i].x = %f, thread_data[i].y = %f \n", threadIdx.x, i, (i * stride + threadIdx.x), thread_data[i].x, thread_data[i].y);
    }

    // Execute IFFT
    FFT().execute(thread_data, shared_mem);

    // Extract per SC channel coefficients
    index = threadIdx.x;
#pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
        {
            thread_data[i].x = thread_data[i].x * (tdlDynDescr -> freqChanNormalizeCoe);  // normalize by sqrt(N_IFFT)
            thread_data[i].y = thread_data[i].y * (tdlDynDescr -> freqChanNormalizeCoe);  // normalize by sqrt(N_IFFT)
            #ifdef USE_MEMOERY_FFT_SHIFT_ 
            if(index < N_sc_over_2)
            {
                freqChanSC[index + freq_offsetSC + N_sc_over_2].x = thread_data[i].x;;
                freqChanSC[index + freq_offsetSC + N_sc_over_2].y = thread_data[i].y;;
            }
            else if(index >= (N_FFT - N_sc_over_2))
            {
                freqChanSC[index + freq_offsetSC - N_FFT + N_sc_over_2].x = thread_data[i].x;
                freqChanSC[index + freq_offsetSC - N_FFT + N_sc_over_2].y = thread_data[i].y;
            }
            #else
            if(index >= ((N_FFT >> 1) - N_sc_over_2) && index < ((N_FFT >> 1) + N_sc_over_2) )  // Middle part
            {
                freqChanSC[index + freq_offsetSC - (N_FFT >> 1) + N_sc_over_2].x = thread_data[i].x;
                freqChanSC[index + freq_offsetSC - (N_FFT >> 1) + N_sc_over_2].y = thread_data[i].y;
            }
            #endif
            index += stride;
        }
        //printf("FFT out: threadIdx.x=%d, i=%d, (i * stride + threadIdx.x) = %d: thread_data[i].x = %f, thread_data[i].y = %f \n", threadIdx.x, i, (i * stride + threadIdx.x), thread_data[i].x, thread_data[i].y);
    } 
}

// template<typename Tscalar, typename Tcomplex>
// using fftKernelHandle = void (*)(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr);

// Choose FFT kernel
template<typename Tscalar, typename Tcomplex, unsigned int FftSize, unsigned int Arch>
fftKernelHandle<Tscalar, Tcomplex> tdl_get_fft_param(dim3& block_dim, uint& shared_memory_size) 
{ 
    using namespace cufftdx;

    // use predefined numbers
    using FFT = decltype(Size<FftSize>() + Precision<float>() + Type<fft_type::c2c>()
                                + Direction<fft_direction::forward>()
                                + FFTsPerBlock<FFTs_PER_BLOCK_CONST_>() // + ElementsPerThread<FFT_ELEMENTS_PER_THREAD_CONST_>()
                                + SM<Arch>() + Block());
    
    // use cuFFTdx configurations
    // Base of the FFT description
    // using FFT_base = decltype(Size<FftSize>() + Precision<Tscalar>() + Type<fft_type::c2c>()
    // + Direction<fft_direction::forward>()
    // /* Notice lack of ElementsPerThread and FFTsPerBlock operators */
    // + SM<Arch>() + Block());
    // // FFT description with suggested FFTs per CUDA block for the default (optimal) elements per thread
    // using FFT = decltype(FFT_base() + FFTsPerBlock<1>());

    block_dim = FFT::block_dim;
    shared_memory_size = FFT::shared_memory_size;

    return tdl_fft_kernel<FFT, Tscalar, Tcomplex>;
}

template<typename Tscalar, typename Tcomplex>
fftKernelHandle<Tscalar, Tcomplex> tdl_get_fft_param(const int Nfft, unsigned int cudaDeviceArch, dim3& block_dim, uint& shared_memory_size) 
{ 
    if((Nfft == 4096) && (cudaDeviceArch == 800))
    {
        return tdl_get_fft_param<Tscalar, Tcomplex,  4096, 800>(block_dim, shared_memory_size);
    }
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Unsupported FFT length or cudaDeviceArch in TDL frequency channel, please add your Nfft or cudaDeviceArch into tdl_get_fft_param and retry");
        assert(false);
        return nullptr;
    }
    return nullptr;
}


template <typename Tscalar, typename Tcomplex> 
static __global__ void genTdlChanCoeKernel(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr, float refTime0)
{
    // m_gridDim = {tdlDynDescrCpu->nBatch,1,1};
    // m_blockDim = {tdlDynDescrCpu -> firNzLen, tdlDynDescrCpu -> nTxAnt, tdlDynDescrCpu -> nRxAnt};

    uint16_t batchIdx = blockIdx.x;
    uint16_t firNzIdx = threadIdx.x;
    uint16_t txAntIdx = threadIdx.y;
    uint16_t rxAntIdx = threadIdx.z;
    uint16_t firNzLen = tdlDynDescr -> firNzLen;
    uint nPath = tdlDynDescr -> nPath;
    uint pathOffset = nPath * ((rxAntIdx*blockDim.y + txAntIdx)*blockDim.x + firNzIdx);
    uint globalChanIdx = ((batchIdx*blockDim.z + rxAntIdx)*blockDim.y + txAntIdx)*blockDim.x + firNzIdx; // [1firNzLen, 2nTx, 3nRx, 4nBatch]
    Tcomplex * timeChan = tdlDynDescr -> timeChan;
    float * thetaRand = tdlDynDescr -> thetaRand; // thetaRand in (0.0f, 1.0f]
    float PI_4_nPath = tdlDynDescr -> PI_4_nPath; // a constant pi/4/nPath
    float timeStamp = (tdlDynDescr -> tBatch) * batchIdx + refTime0;
    float maxDopplerShift = tdlDynDescr -> maxDopplerShift;
    Tcomplex tempChanCoe = {0.0f, 0.0f};
    
    Tcomplex chanPathAdd; // for superimpose of nPath sins
    float freqReal, freqImag; // for doppler of real and imag
    for(uint pathIdx = 0; pathIdx < nPath; pathIdx++)
    {
        // calculate doppler frequency
        float alpha_0 = PI_4_nPath * (firNzIdx+1) / (firNzLen+2);
        freqReal = maxDopplerShift * cos(2.0f * PI_4_nPath * (pathIdx + 0.5) + alpha_0);
        freqImag = maxDopplerShift * cos(2.0f * PI_4_nPath * (pathIdx + 0.5) - alpha_0);
        // calculate channel per path
        chanPathAdd.x = cos(2 * M_PI * (timeStamp * freqReal + thetaRand[(pathOffset + pathIdx)*2]));
        chanPathAdd.y = cos(2 * M_PI * (timeStamp * freqImag + thetaRand[(pathOffset + pathIdx)*2 + 1]));

        // @todo, for TDL-D and TDL-E, the first tap is LOS
        // if((tdlDynDescr -> LosTap) && (firNzIdx==0)) // first tap, LOS in TDL-D and TLD-E

        // add the channel per path to TDL time channel
        tempChanCoe.x = tempChanCoe.x + chanPathAdd.x;
        tempChanCoe.y = tempChanCoe.y + chanPathAdd.y;
    }
    // multiple by FIR and save to gloabl memory; firNzPw has already been normalized
    timeChan[globalChanIdx].x = (tdlDynDescr -> firNzPw[firNzIdx]) * tempChanCoe.x;
    timeChan[globalChanIdx].y = (tdlDynDescr -> firNzPw[firNzIdx]) * tempChanCoe.y;
}

template <typename Tscalar, typename Tcomplex> 
static __global__ void normalizeTimeChan(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr)
{
    // m_gridDim = {1,1,1};
    // m_blockDim = {THREADS_PER_BLOCK_NORMALIZATION_, 1, 1};
    Tcomplex * timeChan = tdlDynDescr -> timeChan;
    uint timeChanSize = tdlDynDescr -> timeChanSize;
    int tid = threadIdx.x;

    // shared memeory for calcualte local sums and save normalization coe at localSum[0];
    __shared__ Tscalar localSum[THREADS_PER_BLOCK_NORMALIZATION_];

    // calculate temporary sum for each thread
    localSum[tid] = 0.0f;
    for(int timeChanIdx = tid; timeChanIdx < timeChanSize; timeChanIdx += blockDim.x)
    {
        localSum[tid] += timeChan[timeChanIdx].x * timeChan[timeChanIdx].x + timeChan[timeChanIdx].y * timeChan[timeChanIdx].y;
    }
    __syncthreads();

    // obtain normalziation coe using parallel reduction
    uint16_t h = THREADS_PER_BLOCK_NORMALIZATION_;
    uint16_t s = ceilf(h*0.5f);
    #pragma unroll
    while(s > 1)
    {
        if(tid < h-s)
        {
            localSum[tid] += localSum[tid + s];
        }
        h = s; s = ceilf(h*0.5f);
        __syncthreads();
    }
    if(tid == 0)
    {
        localSum[0] += localSum[1];
        Tscalar normalizeTarget = (tdlDynDescr -> nRxAnt) * (tdlDynDescr -> nBatch);
        localSum[0] = sqrt(float(normalizeTarget / localSum[0]));
    }
    __syncthreads();

    // apply normalization
    for(int timeChanIdx = tid; timeChanIdx < timeChanSize; timeChanIdx += blockDim.x)
    {
        timeChan[timeChanIdx].x *= localSum[0];
        timeChan[timeChanIdx].y *= localSum[0];
    }
}

template<typename Tscalar, typename Tcomplex>
static __global__ void convertSCtoPRBG(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr)
{
    // GRID(nTx)
    // BLOCK(nPRBG, nRx)

    uint N_sc = tdlDynDescr -> N_sc;
    uint N_PRBG = tdlDynDescr -> N_PRBG;
    uint N_sc_PRBG = tdlDynDescr -> N_sc_PRBG;
    uint nRxAnt = tdlDynDescr -> nRxAnt;
    Tcomplex * freqChanSC = tdlDynDescr -> freqChanSC;
    Tcomplex * freqChanPRBG = tdlDynDescr -> freqChanPRBG;

    uint txAntIdx = blockIdx.x;
    uint rxAntIdx = threadIdx.y;
    uint prbgIdx = threadIdx.x;

    uint sc_offset = (txAntIdx * nRxAnt + rxAntIdx) * N_sc + prbgIdx * N_sc_PRBG;
    uint prbg_offet = (txAntIdx * nRxAnt + rxAntIdx) * N_PRBG + prbgIdx;

    Tcomplex tempSum;

    // the first symbol set
    tempSum = freqChanSC[sc_offset];
    for(uint rbgIdx = 1; rbgIdx < N_sc_PRBG; rbgIdx++)
    {
        sc_offset++;
        tempSum.x = tempSum.x + freqChanSC[sc_offset].x;
        tempSum.y = tempSum.y + freqChanSC[sc_offset].y;
    }
    freqChanPRBG[prbg_offet].x = float(tempSum.x) / float(N_sc_PRBG);
    freqChanPRBG[prbg_offet].y = float(tempSum.y) / float(N_sc_PRBG);
}

template <typename Tscalar, typename Tcomplex> 
static __global__ void processInputKernel(tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr, float refTime0)
{
    // m_gridDim = {tdlDynDescrCpu->nBatch,1,1};
    // m_blockDim = {BLOCK_SAMPLE_, tdlDynDescrCpu -> nRxAnt, 1};
    // BLOCK_SAMPLE_ should be less than int(1024/tdlDynDescrCpu -> nRxAnt)
    // timeChan saved in [firNzLen, nTx, nRx, nBatch]
    // from launch configuration
    uint16_t batchIdx = blockIdx.x;
    uint16_t chanCoeIdx = threadIdx.x;
    uint16_t rxAntIdx = threadIdx.y;
    
    // from dyn descriptor
    uint16_t firNzLen = tdlDynDescr -> firNzLen;
    uint16_t firMaxLen = tdlDynDescr -> firMaxLen; // maxmimum delay tap
    uint nBatchSamp = tdlDynDescr -> nBatchSamp;
    uint nTxAnt = tdlDynDescr -> nTxAnt;
    uint nRxAnt = tdlDynDescr -> nRxAnt;
    long timeSigLenPerAnt = tdlDynDescr -> timeSigLenPerAnt;
    Tcomplex * timeChan = tdlDynDescr -> timeChan;
    Tcomplex * txTimeSigIn =  tdlDynDescr -> txTimeSigIn;
    Tcomplex * rxTimeSigOut = tdlDynDescr -> rxTimeSigOut;
    float cfoPhaseRef = 2 * M_PI * refTime0 * (tdlDynDescr -> cfoHz); 
    float cfoPhaseSamp = tdlDynDescr -> cfoPhaseSamp;
    uint nDelaySample = tdlDynDescr -> nDelaySample;
    /**
     * @todo: try to use dynamic config shared memory
     * 
     */
    // extern __shared__ char shareData[]; // all shared memory data pointer
    // Tcomplex * txTimeSigInBlock = (Tcomplex *) (shareData); // firMaxLen + BLOCK_SAMPLE_
    // Tcomplex * chanCoeLocal = (Tcomplex *)(shareData + sizeof(Tcomplex) * (firMaxLen + BLOCK_SAMPLE_)); // firNzLen * nRxAnt * nTxAnt
    // uint16_t * firNzIdx = (uint16_t*) (shareData + sizeof(Tcomplex) * (firMaxLen + BLOCK_SAMPLE_ + firNzLen * nRxAnt * nTxAnt)); // firNzLen
    // Tcomplex * txTimeSigInBlock = (Tcomplex *) (shareData); // firMaxLen + BLOCK_SAMPLE_
    //Tcomplex * chanCoeLocal = (Tcomplex *)(shareData + sizeof(Tcomplex) * (firMaxLen + BLOCK_SAMPLE_)); // firNzLen * nRxAnt * nTxAnt


    __shared__ Tcomplex txTimeSigInBlock[BLOCK_SAMPLE_];
    __shared__ Tcomplex chanCoeLocal[MAX_TX_RX_ANT_ * MAX_NZ_TAPS_];
    __shared__ uint16_t firNzIdx[MAX_NZ_TAPS_]; // no more than 24 NZ taps based on 3GPP 38.901 

    // copy all channel coeficient into shared memory
    uint localCoeMatOffset =  firNzLen * rxAntIdx * nTxAnt;
    uint gloablCoeMatOffset = firNzLen * batchIdx * nRxAnt * nTxAnt + localCoeMatOffset; // start of TxAnt = 0, read firNzLen * nTxAnt * nRxAnt * 1 (batch)
    // read channCoe for all Tx and Rx, use for the whole precedure
    if(chanCoeIdx < firNzLen) // suppose (firNzLen<BLOCK_SAMPLE_ < int(1024/tdlDynDescrCpu -> nRxAnt)
    {
        #pragma unroll
        for(uint txAntIdx = 0; txAntIdx < nTxAnt; txAntIdx ++) // add sum of signals from each Tx antenna
        {
            chanCoeLocal[localCoeMatOffset + firNzLen * txAntIdx + chanCoeIdx].x = timeChan[gloablCoeMatOffset + firNzLen * txAntIdx + chanCoeIdx].x; // blockDim.y = nRxAnt
            chanCoeLocal[localCoeMatOffset + firNzLen * txAntIdx + chanCoeIdx].y = timeChan[gloablCoeMatOffset + firNzLen * txAntIdx + chanCoeIdx].y; // blockDim.y = nRxAnt
        }
        if(rxAntIdx == 0)
        {
            firNzIdx[chanCoeIdx] = tdlDynDescr -> firNzIdx[chanCoeIdx];
        }
    }
    __syncthreads();

    // read input signal and provide output signal
    uint antTxSampOffset, globalTxSampOffset; // antTxSampOffset: tx sample offset for this antenna, i.e., [0, timeSigLenPerAnt-1]; globalTxSampOffset = antTxSampOffset + txAntIdx * timeSigLenPerAnt
    uint antRxSampOffset, globalRxSampOffset; // antRxSampOffset: rx sample offset for this antenna, i.e., [0, timeSigLenPerAnt-1], consider delay; globalRxSampOffset = antRxSampOffset + rxAntIdx * timeSigLenPerAnt
    Tcomplex rxSigRegNoCfo;
    Tcomplex tempTxSig, tempChanCoe;
    for(uint16_t chunkIdx = 0; chunkIdx < ceilf(float(nBatchSamp)/BLOCK_SAMPLE_); chunkIdx ++) // each chunk for processing
    {
        antTxSampOffset = nBatchSamp * batchIdx + BLOCK_SAMPLE_ * chunkIdx + chanCoeIdx;
        globalTxSampOffset = antTxSampOffset;        
        rxSigRegNoCfo.x = 0.0f; rxSigRegNoCfo.y = 0.0f;
        #pragma unroll
        for(uint txAntIdx = 0; txAntIdx < nTxAnt; txAntIdx ++) // add sum of signals from each Tx antenna
        {
            // read tx singal for this tx antenna and batch
            // format: [firNzLen BLOCK_SAMPLE_]
            if(rxAntIdx == 0)  
            {
                if(antTxSampOffset < timeSigLenPerAnt)
                {
                    txTimeSigInBlock[chanCoeIdx] = txTimeSigIn[globalTxSampOffset];
                }
                else // out of bound, no need to read
                {
                    txTimeSigInBlock[chanCoeIdx].x = 0.0f;
                    txTimeSigInBlock[chanCoeIdx].y = 0.0f;
                }
            }
            __syncthreads();    
            // apply the channel coes to tx symbols, add to local register
            if(antTxSampOffset < timeSigLenPerAnt)
            {
                for(uint channSumIdx = 0; channSumIdx < firNzLen; channSumIdx++)
                {
                    if(chanCoeIdx >= firNzIdx[channSumIdx]) 
                    // if tx samples in shared memory, read from shared memmory
                    // else read from global memory
                    {
                        tempTxSig = txTimeSigInBlock[chanCoeIdx - firNzIdx[channSumIdx]]; // current TxSig for sum
                    }
                    else if(antTxSampOffset >= firNzIdx[channSumIdx])
                    {
                        tempTxSig = txTimeSigIn[globalTxSampOffset - firNzIdx[channSumIdx]]; // current TxSig for sum
                    }
                    else
                    {
                        tempTxSig.x = 0.0f;
                        tempTxSig.y = 0.0f;
                    }
                    
                    tempChanCoe = chanCoeLocal[firNzLen * (rxAntIdx * nTxAnt + txAntIdx) + channSumIdx]; // current chanCoe for sum
                    
                    // rxSigRegNoCfo += tempTxSig * tempChanCoe
                    rxSigRegNoCfo.x = rxSigRegNoCfo.x + (tempTxSig.x * tempChanCoe.x - tempTxSig.y * tempChanCoe.y);
                    rxSigRegNoCfo.y = rxSigRegNoCfo.y + (tempTxSig.x * tempChanCoe.y + tempTxSig.y * tempChanCoe.x);
                }
            }
            // update gloal channel index
            globalTxSampOffset += timeSigLenPerAnt;
            __syncthreads(); // wait for the processing of sampels to finish
        }
        // add delay and cfo, save the rx data sample to global memory
        
        // get position to write rx sample with delay
        // same with 5GModel using cyclic shift ref: 5GModel/nr_matlab/channel/Channel.m
        // @note it could be the same to just use zero padding at the begining; no impact since it falls into CP position, will be discarded anyway

        if(antTxSampOffset < timeSigLenPerAnt - nDelaySample) // samples to be shift later by nDelaySample, new position no more than timeSigLenPerAnt
        {
            antRxSampOffset = antTxSampOffset + nDelaySample;
        }
        else if(antTxSampOffset < timeSigLenPerAnt) // samples padded to the begin, following cyclicshift
        {
            antRxSampOffset = antTxSampOffset + nDelaySample - timeSigLenPerAnt;
        }
        else // no opreations needed for samples
        {
            antRxSampOffset = antTxSampOffset;
        }

        if(antRxSampOffset < timeSigLenPerAnt)
        {  
            globalRxSampOffset = rxAntIdx * timeSigLenPerAnt + antRxSampOffset;
            float cfoPhaseTotal = cfoPhaseRef + cfoPhaseSamp * antRxSampOffset; //
            Tcomplex cfoRotation;

            cfoRotation.x = cos(cfoPhaseTotal);
            cfoRotation.y = sin(cfoPhaseTotal);
            // no CFO involved
            // rxTimeSigOut[globalRxSampOffset].x = rxSigRegNoCfo.x;
            // rxTimeSigOut[globalRxSampOffset].y = rxSigRegNoCfo.y;

            // Apply CFO
            rxTimeSigOut[globalRxSampOffset].x = rxSigRegNoCfo.x * cfoRotation.x - rxSigRegNoCfo.y * cfoRotation.y;
            rxTimeSigOut[globalRxSampOffset].y = rxSigRegNoCfo.x * cfoRotation.y + rxSigRegNoCfo.y * cfoRotation.x;
        }
    }
}

