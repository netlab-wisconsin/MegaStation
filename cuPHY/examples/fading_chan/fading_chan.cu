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
#include "cuphy.hpp"

template <typename Tcomplex>
fadingChan<Tcomplex>::fadingChan(Tcomplex* Tx, Tcomplex* freqRx, cudaStream_t strm, uint8_t fadingMode, uint16_t randSeed, uint8_t phyChannType)
{
    m_strm = strm;
    m_Tx = Tx; // frequency-domain TX signal for non-PRACH; time-domain TX signal for PRACH
    m_freqRxNoisy = freqRx;
    m_fadingMode = fadingMode;
    m_randSeed = randSeed;
    m_phyChannType = phyChannType;

    m_prach = m_phyChannType == 2? true : false;
    

    // get carrier and channel params
    m_carrierPrms = new cuphyCarrierPrms_t;
    m_tdlCfg = new tdlConfig_t;

    // get cuPHY tensor type
    if(typeid(Tcomplex) == typeid(__half2))
    {
        m_cuphyTensorType = CUPHY_C_16F;
    }
    else if(typeid(Tcomplex) == typeid(cuComplex))
    {
        m_cuphyTensorType = CUPHY_C_32F;
    }

    m_rng = new cuphy::rng(m_randSeed, 0, m_strm);
}

template <typename Tcomplex>
fadingChan<Tcomplex>::~fadingChan()
{
    // free up configuration
    if (m_carrierPrms) delete m_carrierPrms;
    if (m_tdlCfg) delete m_tdlCfg;

    // free up tdl and ofdm classes
    if (m_tdl_chan) delete m_tdl_chan;
    if (m_ofdmMod) delete m_ofdmMod;
    if (m_ofdmDeMod) delete m_ofdmDeMod;

    // free up noise and freq rx noise free buffer
    delete m_rng;
    cudaFree(m_noise);
    cudaFree(m_freqRxNoiseFree);
}

inline void carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier);
inline void prach_carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier, const hdf5hpp::hdf5_dataset_elem& prachParams);
inline void tdl_pars_from_dataset_elem(tdlConfig_t * tdlCfg, const hdf5hpp::hdf5_dataset_elem& dset_elem);

template <typename Tcomplex>
void fadingChan<Tcomplex>::setup(hdf5hpp::hdf5_file& inputFile)
{
    using myTscalar = decltype(type_convert(getScalarType<Tcomplex>{}));

    // get configurations from TV file
    readCarrierChanPar(inputFile); 
    
    if (m_prach) { // PRACH
        uint32_t Nsamp_oran = m_carrierPrms->L_RA == 139? 144 : 864;
        m_freqRxDataSize = Nsamp_oran * m_carrierPrms->N_rep * m_carrierPrms->N_rxLayer;
        cudaMalloc((void**)&m_freqRxNoiseFree, sizeof(Tcomplex)*m_freqRxDataSize);
        cudaMalloc((void**)&m_noise, sizeof(Tcomplex)*m_freqRxDataSize);

        // read data from TV
        read_Xtf_prach(inputFile);

        if (m_fadingMode == 1) { // TDL
            m_tdlCfg -> timeSigLenPerAnt = m_carrierPrms -> N_samp_slot;
            m_tdlCfg -> txTimeSigIn = m_Tx;
            m_tdl_chan = new tdlChan<myTscalar, Tcomplex>(m_tdlCfg, m_randSeed + 1, m_strm);
            // printf("tdl chan created\n");

            /*-----------------------  OFDM demodulation     ---------------------*/
            // get time domain output address
            m_timeRx = m_tdl_chan -> getRxTimeSigOut();
            m_ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex>(m_carrierPrms, m_timeRx, m_freqRxNoiseFree, m_prach, m_strm);
            // printf("OFDM de modulation created\n");
        } else {
            m_tdl_chan = nullptr;
            m_ofdmDeMod = nullptr;
        }
        m_ofdmMod = nullptr;
    } else {
        // allocate buffer for noise free freq rx and noise
        m_freqRxDataSize = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_rxLayer) * (m_carrierPrms -> N_symble_slot);
        cudaMalloc((void**)&m_freqRxNoiseFree, sizeof(Tcomplex)*m_freqRxDataSize);
        cudaMalloc((void**)&m_noise, sizeof(Tcomplex)*m_freqRxDataSize);

        // read data from TV
        read_Xtf(inputFile);

        if(m_fadingMode == 1)
        {
            /*-----------------------  OFDM modulation     ---------------------*/
            m_ofdmMod = new ofdm_modulate::ofdmModulate<myTscalar, Tcomplex>(m_carrierPrms, m_Tx, m_strm);
            // printf("OFDM modulation created\n");

            /*-----------------------  TDL channel modulation  ---------------------*/
            // get total time sample length
            uint timeTxLen = m_ofdmMod -> getTimeDateLen();
            m_tdlCfg -> timeSigLenPerAnt = timeTxLen / (m_carrierPrms -> N_txLayer);
            // get time GPU address
            m_timeTx = m_ofdmMod -> getTimeDataOut(); // tx time data after ofdm modulation
            m_tdlCfg -> txTimeSigIn = m_timeTx;
            m_tdl_chan = new tdlChan<myTscalar, Tcomplex>(m_tdlCfg, m_randSeed + 1, m_strm);
            // printf("tdl chan created\n");

            /*-----------------------  OFDM demodulation     ---------------------*/
            // get time domain output address
            m_timeRx = m_tdl_chan -> getRxTimeSigOut();
            m_ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex>(m_carrierPrms, m_timeRx, m_freqRxNoiseFree, m_prach, m_strm);
            // printf("OFDM de modulation created\n");
        }
        else
        {
            m_ofdmMod = nullptr;
            m_tdl_chan = nullptr;
            m_ofdmDeMod = nullptr;
        }
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::readCarrierChanPar(hdf5hpp::hdf5_file& inputFile)
{
    hdf5hpp::hdf5_dataset dset_carrier  = inputFile.open_dataset("carrier_pars");
    if (m_prach) {
        hdf5hpp::hdf5_dataset prachParams   = inputFile.open_dataset("prachParams_0");
        prach_carrier_pars_from_dataset_elem(m_carrierPrms, dset_carrier[0], prachParams[0]);
    } else {
        carrier_pars_from_dataset_elem(m_carrierPrms, dset_carrier[0]);
    }
    
    hdf5hpp::hdf5_dataset dset_chan  = inputFile.open_dataset("chan_pars");
    tdl_pars_from_dataset_elem(m_tdlCfg, dset_chan[0]);

    // obtain the tx layers, check the parameters setting
    if (!m_prach) {
        hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("X_tf_transmitted_from_UE_0");
        hdf5hpp::hdf5_dataspace XtfDataSpace = Xtf_dataset.get_dataspace();
        int ndims = XtfDataSpace.get_rank();
        std::vector<hsize_t> dims = XtfDataSpace.get_dimensions();

        // automatically change # of input layers
        if(ndims == 2)
        {
            m_carrierPrms -> N_txLayer = 1;
        }
        else if(ndims == 3)
        {
            m_carrierPrms -> N_txLayer = dims[0];
        }
        else
        {
            printf("Input Xtf format error with rank %d, dims: [ ", ndims);
            for(auto i : dims)
            {
                printf("%lld ", i);
            }
            printf("]\n");
            exit(1);
        }
    }
    
    assert(m_carrierPrms -> N_txLayer <= m_tdlCfg -> nTxAnt); // # of tx layers must be no more than nTxAnt
    m_tdlCfg -> nTxAnt = m_carrierPrms -> N_txLayer;

    /*--------------------------Below is for overwriting parameters, use caution---------------------------------*/
    // m_carrierPrms -> N_sc = 3276; // 12 * num of RBs
    // m_carrierPrms -> N_FFT = pow(2, ceilf(log2(N_sc)));  // also N_IFFT
    // m_carrierPrms -> N_txLayer = 1;
    // m_carrierPrms -> N_rxLayer = 1;
    // m_carrierPrms -> id_slot = 0;  // per sub frame
    // m_carrierPrms -> id_subFrame = 0; // per frame
    // m_carrierPrms -> mu = 1; // numerology
    // m_carrierPrms -> cpType = 0;
    // m_carrierPrms -> f_c = 480e3 * 4096; // delta_f_max * N_f based on 38.211
    // m_carrierPrms -> f_samp = 15e3 * 8192; // 1ee3 * 2^mu * Nfft
    // m_carrierPrms -> N_symble_slot = OFDM_SYMBOLS_PER_SLOT; // 14 OFDMs per slot
    // m_carrierPrms -> kappa_bits = 6; // kappa = 64 (2^6); constants defined in 38.211
    // m_carrierPrms -> ofdmWindowLen = 0; // ofdm windowing, not used
    // m_carrierPrms -> rolloffFactor = 0.5; // ofdm windowing, not used

    // change defualt paramters
    // m_tdlCfg -> useSimplifiedPpd = false;
    // m_tdlCfg -> delayProfile = 'A';
    // m_tdlCfg -> delaySpread = 30;
    // m_tdlCfg -> maxDopplerShift = 5;
    // m_tdlCfg -> f_samp = m_carrierPrms -> f_samp; //8192 * 15e3;
    // m_tdlCfg -> mimoCorrMat = NULL;
    // m_tdlCfg -> nTxAnt = m_carrierPrms -> N_txLayer;
    // m_tdlCfg -> nRxAnt = m_carrierPrms -> N_rxLayer;
    // m_tdlCfg -> normChannOutput = true;
    // m_tdlCfg -> fBatch = 15e3;
    // m_tdlCfg -> numPath = 48;
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::read_Xtf(hdf5hpp::hdf5_file& inputFile)
{    
    m_freqTxDataSize = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_txLayer) * (m_carrierPrms -> N_symble_slot);

    switch(m_fadingMode)
    {
        case 0: // AWGN, read freq rx from TV, float32 in TV
        {
            float * readOutBuffer = new float[m_freqRxDataSize * 2];
            Tcomplex * freqRxCpu = new Tcomplex[m_freqRxDataSize];
            // Read input HDF5 file to read rate-matching output.
            printf("Reading %d freq rx symbols from TV \n", m_freqRxDataSize);

            if (m_phyChannType == 0) { // PUSCH
                hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("X_tf");
                Xtf_dataset.read(readOutBuffer);
            } else if (m_phyChannType == 1) { // PUCCH
                hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("DataRx");
                Xtf_dataset.read(readOutBuffer);
            }

            
            
            for(int i=0; i<m_freqRxDataSize; i++)
            {
                if(typeid(Tcomplex) == typeid(__half2)) // float16 is used, type conversion needed
                {   
                    freqRxCpu[i].x = __float2half(readOutBuffer[i*2]);
                    freqRxCpu[i].y = __float2half(readOutBuffer[i*2 + 1]);
                }
                else
                {
                    freqRxCpu[i].x = readOutBuffer[i*2];
                    freqRxCpu[i].y = readOutBuffer[i*2 + 1];  
                }
            }
            cudaMemcpyAsync(m_freqRxNoiseFree, freqRxCpu, sizeof(Tcomplex)*m_freqRxDataSize, cudaMemcpyHostToDevice, m_strm);
            delete[] readOutBuffer; 
            delete[] freqRxCpu;
            break;
        }

        case 1: // TDL, read freq tx from TV, double in TV
        {
            double * readOutBuffer = new double[m_freqTxDataSize * 2];
            Tcomplex * freqTxCpu = new Tcomplex[m_freqTxDataSize];
            // Read input HDF5 file to read rate-matching output.
            printf("Reading %d freq tx symbols from TV \n", m_freqTxDataSize);
            hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("X_tf_transmitted_from_UE_0");
            Xtf_dataset.read(readOutBuffer);
            for(int i=0; i<m_freqTxDataSize; i++)
            {
                if(typeid(Tcomplex) == typeid(__half2)) // float16 is used, type conversion needed
                {   
                    freqTxCpu[i].x = __double2half(readOutBuffer[i*2]);
                    freqTxCpu[i].y = __double2half(readOutBuffer[i*2 + 1]);
                }
                else
                {
                    freqTxCpu[i].x = float(readOutBuffer[i*2]);
                    freqTxCpu[i].y = float(readOutBuffer[i*2 + 1]);
                }
            }
            cudaMemcpyAsync(m_Tx, freqTxCpu, sizeof(Tcomplex)*m_freqTxDataSize, cudaMemcpyHostToDevice, m_strm);  
            delete[] readOutBuffer; 
            delete[] freqTxCpu;
            break;
        }
        default: // report error
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Error: unsupported fading mode");
            exit(1);
        }
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::read_Xtf_prach(hdf5hpp::hdf5_file& inputFile)
{
    if (m_fadingMode == 1) { // TDL
        m_timeTxDataSize = m_carrierPrms -> N_samp_slot * m_carrierPrms -> N_txLayer;

        double * readOutBuffer = new double[m_timeTxDataSize * 2];
        Tcomplex * timeTxCpu = new Tcomplex[m_timeTxDataSize];

        printf("Reading %d time-domain tx samples from TV \n", m_timeTxDataSize);
        hdf5hpp::hdf5_dataset Xt_dataset = inputFile.open_dataset("X_t_transmitted_from_UE_0");
        Xt_dataset.read(readOutBuffer);
        for(int i=0; i<m_timeTxDataSize; i++) {
            if(typeid(Tcomplex) == typeid(__half2)) { // float16 is used, type conversion needed
                timeTxCpu[i].x = __double2half(readOutBuffer[i*2]);
                timeTxCpu[i].y = __double2half(readOutBuffer[i*2 + 1]);
            } else {
                timeTxCpu[i].x = float(readOutBuffer[i*2]);
                timeTxCpu[i].y = float(readOutBuffer[i*2 + 1]);
            }
        }
        cudaMemcpyAsync(m_Tx, timeTxCpu, sizeof(Tcomplex)*m_timeTxDataSize, cudaMemcpyHostToDevice, m_strm);  
        delete[] readOutBuffer; 
        delete[] timeTxCpu;
    } else { // AWGN
        float * readOutBuffer = new float[m_freqRxDataSize * 2];
        Tcomplex * freqRxCpu = new Tcomplex[m_freqRxDataSize];
        // Read input HDF5 file to read rate-matching output.
        printf("Reading %d frequency-domain rx symbols from TV \n", m_freqRxDataSize);

        hdf5hpp::hdf5_dataset Xtf_dataset = inputFile.open_dataset("y_uv_rx_0");
        Xtf_dataset.read(readOutBuffer);
            
        for(int i=0; i<m_freqRxDataSize; i++)
            {
                if(typeid(Tcomplex) == typeid(__half2)) // float16 is used, type conversion needed
                {   
                    freqRxCpu[i].x = __float2half(readOutBuffer[i*2]);
                    freqRxCpu[i].y = __float2half(readOutBuffer[i*2 + 1]);
                }
                else
                {
                    freqRxCpu[i].x = readOutBuffer[i*2];
                    freqRxCpu[i].y = readOutBuffer[i*2 + 1];  
                }
            }
            cudaMemcpyAsync(m_freqRxNoiseFree, freqRxCpu, sizeof(Tcomplex)*m_freqRxDataSize, cudaMemcpyHostToDevice, m_strm);
            delete[] readOutBuffer; 
            delete[] freqRxCpu;
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::run(float refTime0, float targetSNR)
{
    if(m_fadingMode == 1) // only run in TDL mode
    {
        if (!m_prach) {
            // OFDM modulation
            m_ofdmMod -> run(m_strm);
        }
           
        // apply TDL
        m_tdl_chan -> run(refTime0);
        // OFDM demodulation
        m_ofdmDeMod -> run(m_strm);
    }

    // add noise in freq domain
    addNoiseFreq(targetSNR);
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::addNoiseFreq(float targetSNR)
{
    // currently generated and add separately
    float noiseVariance  = pow(10.0, -targetSNR / 10.0);
    cuComplex m32_0      = make_cuFloatComplex(0, 0);
    cuComplex stddev32_1 = make_cuFloatComplex(sqrt(noiseVariance / 2), sqrt(noiseVariance / 2));

    // Add noise to slot: convert GPU memory address to cuPHY tensors
    cuphy::tensor_ref freqRxNoisyTensor;
    cuphy::tensor_ref freqRxNoiseFreeTensor;
    cuphy::tensor_ref noiseTensor;

    // for non-PRACH m_freqRxDataSize = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_rxLayer) * (m_carrierPrms -> N_symble_slot);
    // old codes using (nSubcarriers, OFDM_SYMBOLS_PER_SLOT, nGnbAnt)
    // for PRACH m_freqRxDataSize = Nsamp_oran * m_carrierPrms->N_rep * m_carrierPrms->N_rxLayer;

    freqRxNoisyTensor.desc().set(m_cuphyTensorType,  m_freqRxDataSize, cuphy::tensor_flags::align_tight);
    freqRxNoiseFreeTensor.desc().set(m_cuphyTensorType, m_freqRxDataSize, cuphy::tensor_flags::align_tight);
    noiseTensor.desc().set(m_cuphyTensorType, m_freqRxDataSize, cuphy::tensor_flags::align_tight);

    freqRxNoisyTensor.set_addr(m_freqRxNoisy);
    freqRxNoiseFreeTensor.set_addr(m_freqRxNoiseFree);
    noiseTensor.set_addr(m_noise);

    m_rng -> normal(noiseTensor, m32_0, stddev32_1, m_strm);
    cuphyStatus_t s = cuphyTensorElementWiseOperation(freqRxNoisyTensor.desc().handle(),     
                                                      freqRxNoisyTensor.addr(),             
                                                      freqRxNoiseFreeTensor.desc().handle(),
                                                      freqRxNoiseFreeTensor.addr(),       
                                                      nullptr,            
                                                      noiseTensor.desc().handle(),  
                                                      noiseTensor.addr(),           
                                                      nullptr,            
                                                      CUPHY_ELEMWISE_ADD, 
                                                      m_strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy::cuphy_fn_exception(s, "cuphyTensorElementWiseOperation()");
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::savefadingChanToFile()
{
    std::string outFilename = "fadingChanData.h5";
    std::unique_ptr<hdf5hpp::hdf5_file> fadingChanFile;
    hdf5hpp::hdf5_file  fadingChanHdf5File;
    if(!outFilename.empty())
    {
        fadingChanFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outFilename.c_str())));
        fadingChanHdf5File = hdf5hpp::hdf5_file::open(outFilename.c_str());
    }
    // Add noise to slot: convert GPU memory address to cuPHY tensors
    cuphy::tensor_ref TxTensor;
    cuphy::tensor_ref freqRxNoisyTensor;
    cuphy::tensor_ref freqRxNoiseFreeTensor;
    cuphy::tensor_ref noiseTensor;

    // m_freqRxDataSize = (m_carrierPrms -> N_sc) * (m_carrierPrms -> N_rxLayer) * (m_carrierPrms -> N_symble_slot);
    // old codes using (nSubcarriers, OFDM_SYMBOLS_PER_SLOT, nGnbAnt)
    uint16_t N_sc           = m_carrierPrms -> N_sc;
    uint16_t N_symble_slot  = m_carrierPrms -> N_symble_slot;
    uint16_t N_txLayer      = m_carrierPrms -> N_txLayer;
    uint16_t N_rxLayer      = m_carrierPrms -> N_rxLayer;

    TxTensor.desc().set(m_cuphyTensorType,  N_sc, N_symble_slot, N_txLayer, cuphy::tensor_flags::align_tight);
    freqRxNoisyTensor.desc().set(m_cuphyTensorType,  N_sc, N_symble_slot, N_rxLayer, cuphy::tensor_flags::align_tight);
    freqRxNoiseFreeTensor.desc().set(m_cuphyTensorType, N_sc, N_symble_slot, N_rxLayer, cuphy::tensor_flags::align_tight);
    noiseTensor.desc().set(m_cuphyTensorType, N_sc, N_symble_slot, N_rxLayer, cuphy::tensor_flags::align_tight);

    TxTensor.set_addr(m_Tx);
    freqRxNoisyTensor.set_addr(m_freqRxNoisy);
    freqRxNoiseFreeTensor.set_addr(m_freqRxNoiseFree);
    noiseTensor.set_addr(m_noise);

    cuphy::write_HDF5_dataset(fadingChanHdf5File, TxTensor,  "Tx", m_strm);
    cuphy::write_HDF5_dataset(fadingChanHdf5File, freqRxNoisyTensor,  "freqRxNoisy", m_strm);
    cuphy::write_HDF5_dataset(fadingChanHdf5File, freqRxNoiseFreeTensor,  "freqRxNoiseFree", m_strm);
    cuphy::write_HDF5_dataset(fadingChanHdf5File, noiseTensor,  "noise", m_strm);
    CUDA_CHECK(cudaStreamSynchronize(m_strm));
    fadingChanHdf5File.close();

    if(!m_SNR.empty())
    {
        std::ofstream outputFile("SNR.txt");
    
        if (outputFile.is_open())
        {
            for (int i = 0; i < m_SNR.size(); ++i)
            {
                outputFile << m_SNR[i] << '\t';
            }
            
            outputFile.close();
        }
        else
        {
            std::cout << "Unable to open file." << std::endl;
        }
        printf("Average SNR: %f, (avg over %d iterations)\n", std::reduce(m_SNR.begin(), m_SNR.end())/float(m_SNR.size()), int(m_SNR.size()/(m_carrierPrms -> N_rxLayer)));

        outputFile.close();
    }
}

template <typename Tcomplex>
void fadingChan<Tcomplex>::calSnr(uint16_t ofdmSymIdx, uint16_t startSC, uint16_t endSC)
{
    uint16_t nRxAnt = m_carrierPrms -> N_rxLayer;
    uint16_t N_sc = m_carrierPrms -> N_sc;
    uint16_t N_symble_slot = m_carrierPrms -> N_symble_slot;

    Tcomplex * tempFreqNoiseFreeBuffer = new Tcomplex[m_freqRxDataSize];
    Tcomplex * tempNoiseBuffer = new Tcomplex[m_freqRxDataSize];
    cudaMemcpy(tempFreqNoiseFreeBuffer, m_freqRxNoiseFree, sizeof(Tcomplex) * m_freqRxDataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(tempNoiseBuffer, m_noise, sizeof(Tcomplex) * m_freqRxDataSize, cudaMemcpyDeviceToHost);

    for(uint8_t rxAntIdx = 0; rxAntIdx < nRxAnt; rxAntIdx++)
    {
        uint scOffset = (rxAntIdx * N_symble_slot + ofdmSymIdx) * N_sc;
        float sigalSum = 0.0f;
        float noiseSum = 0.0f;
        Tcomplex tempSamp, tempNoise;
        for(uint scIdx = startSC; scIdx < endSC; scIdx++)
        {
            tempSamp = tempFreqNoiseFreeBuffer[scOffset + scIdx];
            sigalSum += float(tempSamp.x) * float(tempSamp.x) + float(tempSamp.y) * float(tempSamp.y);
        }
        sigalSum /= (endSC - startSC);
        for(uint scIdx = 0; scIdx < N_sc; scIdx++)
        {
            tempNoise = tempNoiseBuffer[scOffset + scIdx];
            noiseSum += float(tempNoise.x) * float(tempNoise.x) + float(tempNoise.y) * float(tempNoise.y);
        }
        noiseSum /= N_sc;
        m_SNR.push_back(sigalSum / noiseSum);
    }

    delete[] tempFreqNoiseFreeBuffer;
    delete[] tempNoiseBuffer;
}

/*-----------------------Below are configurations for carrier and chan_pars---------------------------------------*/
// use caution when overwriting the params read from TV
inline void carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier)
{
    carrierPrms -> N_sc                   = dset_carrier["N_sc"].as<uint16_t>();
    carrierPrms -> N_FFT                  = dset_carrier["N_FFT"].as<uint16_t>();
    carrierPrms -> N_txLayer              = dset_carrier["N_txAnt"].as<uint16_t>();
    carrierPrms -> N_rxLayer              = dset_carrier["N_rxAnt"].as<uint16_t>();
    carrierPrms -> id_slot                = dset_carrier["id_slot"].as<uint16_t>();
    carrierPrms -> id_subFrame            = dset_carrier["id_subFrame"].as<uint16_t>();
    carrierPrms -> mu                     = dset_carrier["mu"].as<uint16_t>();
    carrierPrms -> cpType                 = dset_carrier["cpType"].as<uint16_t>();
    carrierPrms -> f_c                    = dset_carrier["f_c"].as<uint32_t>();
    carrierPrms -> f_samp                 = dset_carrier["f_samp"].as<uint32_t>();
    carrierPrms -> N_symble_slot          = dset_carrier["N_symble_slot"].as<uint16_t>();
    carrierPrms -> kappa_bits             = dset_carrier["kappa_bits"].as<uint16_t>();
    carrierPrms -> ofdmWindowLen          = 0;//dsedset_carriert_elem["ofdmWindowLen"].as<uint16_t>();
    carrierPrms -> rolloffFactor          = 0.5;//dset_carrier["rolloffFactor"].as<float>();
}

inline void prach_carrier_pars_from_dataset_elem(cuphyCarrierPrms * carrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_carrier, const hdf5hpp::hdf5_dataset_elem& prachParams)
{
    carrierPrms -> N_sc                   = dset_carrier["N_sc"].as<uint16_t>();
    carrierPrms -> N_FFT                  = dset_carrier["N_FFT"].as<uint16_t>();
    carrierPrms -> N_txLayer              = dset_carrier["N_txAnt"].as<uint16_t>();
    carrierPrms -> N_rxLayer              = dset_carrier["N_rxAnt"].as<uint16_t>();
    carrierPrms -> id_slot                = dset_carrier["id_slot"].as<uint16_t>();
    carrierPrms -> id_subFrame            = dset_carrier["id_subFrame"].as<uint16_t>();
    carrierPrms -> mu                     = dset_carrier["mu"].as<uint16_t>();
    carrierPrms -> cpType                 = dset_carrier["cpType"].as<uint16_t>();
    carrierPrms -> f_c                    = dset_carrier["f_c"].as<uint32_t>();
    carrierPrms -> f_samp                 = dset_carrier["f_samp"].as<uint32_t>();
    carrierPrms -> N_symble_slot          = dset_carrier["N_symble_slot"].as<uint16_t>();
    carrierPrms -> kappa_bits             = dset_carrier["kappa_bits"].as<uint16_t>();
    carrierPrms -> ofdmWindowLen          = 0;//dsedset_carriert_elem["ofdmWindowLen"].as<uint16_t>();
    carrierPrms -> rolloffFactor          = 0.5;//dset_carrier["rolloffFactor"].as<float>();
    carrierPrms -> T_c                    = 1.0/float(carrierPrms -> f_c);
    carrierPrms -> N_samp_slot            = dset_carrier["N_samp_slot"].as<uint32_t>();
    carrierPrms -> k_const                = dset_carrier["k_const"].as<uint16_t>();
    carrierPrms -> N_u_mu                 = dset_carrier["N_u_mu"].as<uint32_t>();
    carrierPrms -> startRaSym             = prachParams["startRaSym"].as<uint32_t>();
    carrierPrms -> delta_f_RA             = prachParams["delta_f_RA"].as<uint32_t>();
    carrierPrms -> N_CP_RA                = prachParams["N_CP_RA"].as<uint32_t>();
    carrierPrms -> K                      = prachParams["K"].as<uint32_t>();
    carrierPrms -> k1                     = prachParams["k1"].as<int32_t>();
    carrierPrms -> kBar                   = prachParams["kBar"].as<uint32_t>();
    carrierPrms -> N_u                    = prachParams["N_u"].as<uint32_t>();
    carrierPrms -> L_RA                   = prachParams["L_RA"].as<uint32_t>();
    carrierPrms -> n_slot_RA_sel          = prachParams["n_slot_RA_sel"].as<uint32_t>();
    carrierPrms -> N_rep                  = prachParams["N_rep"].as<uint32_t>();
}


inline void tdl_pars_from_dataset_elem(tdlConfig_t * tdlCfg, const hdf5hpp::hdf5_dataset_elem& dset_elem)
{
    tdlCfg -> useSimplifiedPdp = dset_elem["useSimplifiedPdp"].as<uint8_t>(); // true for simplified pdp in 38.141, false for 38.901
    tdlCfg -> delayProfile = dset_elem["delayProfile"].as<uint8_t>() + 'A';
    tdlCfg -> delaySpread = dset_elem["delaySpread"].as<float>();
    tdlCfg -> maxDopplerShift = dset_elem["maxDopplerShift"].as<float>();
    tdlCfg -> f_samp = dset_elem["f_samp"].as<uint32_t>();
    tdlCfg -> nTxAnt = dset_elem["numTxAnt"].as<uint16_t>();
    tdlCfg -> nRxAnt = dset_elem["numRxAnt"].as<uint16_t>();
    tdlCfg -> fBatch = dset_elem["fBatch"].as<uint32_t>(); // update rate of quasi-static channel
    tdlCfg -> numPath = dset_elem["numPath"].as<uint32_t>();
    tdlCfg -> cfoHz = dset_elem["CFO"].as<float>();
    tdlCfg -> delay = dset_elem["delay"].as<float>(); // NOT USED FOR NOW, placeholder 
    /*  Below are are frequency domain chan generation, not used for cuPHY testing*/
    tdlCfg -> freqConvertN_sc = 0; // max 273 PRBs, setting this to 0 will not provide freq channel
    tdlCfg -> N_sc_PRBG = 0; // # of PRBs per PRBG
    tdlCfg -> runMode = 0; // set to 1 will also generate freq channel
};