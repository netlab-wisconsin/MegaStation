#  Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.palettes import d3
import aerial_mcore as NRSimulator
import matlab

from cuda import cudart

import ctypes as C
import pycuphy
from pycuphy import check_cuda_errors

output_notebook()

NUM_RX_ANTENNAS = 4
START_RB = 0
NUM_RBS = 1
START_SYM = 0
NUM_SYMS = 14
DMRS_POSITION = [0,0,1,0,0,0,0,0,0,0,0,1,0,0] # [0,0,1,1,0,0,0,0,0,0,1,1,0,0] [0,0,1,0,0,0,0,1,0,0,0,1,0,0]
DMRS_ADDLN_POS = 1
DMRS_MAX_LEN = 1 # Note that cuPHY uses 'DMRS_ADDLN_POS' and 'DMRS_MAX_LEN' to derive the num of DMRS symbols. So make sure those two params are set correctly.

class PuschTxRx:

    def __init__(self):
        self.eng = NRSimulator.initialize()
        self.simCtrl = self.eng.cfgSimCtrl()
        self.simCtrl['alg']['TdiMode'] = 0.
        self.simCtrl['alg']['enableIrc'] = 0.
        self.eng.setSimCtrl(self.simCtrl,nargout=0)

        testAlloc = {
        "dl": 0,
        "ul": 1,
        "pusch": 1
        }
        self.SysPar = self.eng.initSysPar(testAlloc);

        self.SysPar['SimCtrl']['fp16AlgoSel'] = 2
        self.SysPar['SimCtrl']['N_frame'] = 1
        self.SysPar["carrier"]["Nant_gNB"] = float(NUM_RX_ANTENNAS)
        self.SysPar["carrier"]["Nant_UE"] = 1.
        self.SysPar["carrier"]["N_ID_CELL"] = 41

    def setPuschConfig(self,puschConfig):
        self.SysPar["pusch"] = [self.eng.cfgPusch()]
        self.SysPar["pusch"][0].update(puschConfig)

    def tx(self):
        # Update SysPar using helper functions for UE
        self.SysPar_UE = self.SysPar
        self.SysPar_UE['carrier'] = self.eng.updateCarrier(self.SysPar['carrier'])
        self.SysPar_UE = self.eng.updateAlloc(self.SysPar_UE)
        idxUE = 1
        self.UE_in = self.eng.initUE(self.SysPar_UE, idxUE)

        UE_out = self.eng.UEtransmitter(self.UE_in)
        tx_tensor = np.array(UE_out['Phy']['tx']['Xtf'])

        return tx_tensor, UE_out

    def plotTx(self,tx_tensor):
        p = figure(title="Tx Tensor Constellation")
        reStart = int(self.SysPar_UE["pusch"][0]["rbStart"])*12
        reEnd = reStart + int(self.SysPar_UE["pusch"][0]["rbSize"])*12
        startSymIndex = int(self.SysPar_UE["pusch"][0]["StartSymbolIndex"])
        stopSymIndex = startSymIndex + int(self.SysPar_UE["pusch"][0]["NrOfSymbols"])
        for sym in range(startSymIndex,stopSymIndex,1):
           if self.SysPar_UE["pusch"][0]["DmrsSymbPos"][0][sym] == 0.0:
              p.circle(x=tx_tensor[reStart:reEnd,sym].real, y=tx_tensor[reStart:reEnd,sym].imag, size=5, color=d3['Category20b'][14][sym])
           else:
              # Every other RE in range contains DMRS for this symbol
              p.diamond(x=tx_tensor[reStart:reEnd:2,sym].real, y=tx_tensor[reStart:reEnd:2,sym].imag, size=15, color=d3['Category20b'][14][sym])

        handle = show(p, notebook_handle=True)

    def matlab_rx_init(self):
        self.SysPar_gNB = self.SysPar
        self.SysPar_gNB['carrier'] = self.eng.updateCarrier(self.SysPar['carrier'])
        self.SysPar_gNB = self.eng.updateAlloc(self.SysPar_gNB)
        self.SysPar_gNB['chan_BF'] = []
        self.SysPar_gNB['Chan_UL'] = []
        self.SysPar_gNB['Chan_DL'] = []

        self.gNB = self.eng.initgNB(self.SysPar_gNB)
        return self.gNB

    def matlab_rx(self,matlabRxSamp):
        rxSamp = matlabRxSamp[0]
        rxSamp_prach = matlabRxSamp[1]
        rxSamp_noNoise = matlabRxSamp[2]
        rxSamp_prach_noNoise = matlabRxSamp[3]

        gNB = self.eng.gNBreceiver(self.gNB, rxSamp, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise)
        return gNB


#
# Channel - for now just implement an AWGN channel on non-PRACH, and passthrough for PRACH
#
def awgn_channel(tx_tensor, sigma):
    # Noise for non-PRACH
    n = np.random.normal(loc=0,scale=sigma,size=(3276,14,NUM_RX_ANTENNAS)) + 1j*np.random.normal(loc=0,scale=sigma,size=(3276,14,NUM_RX_ANTENNAS))
    rx_tensor = n
    for antIdx in range(NUM_RX_ANTENNAS):
        rx_tensor[:,:,antIdx] = rx_tensor[:,:,antIdx] + tx_tensor

    return rx_tensor


def matlab_rx_helper(rx_tensor, UE_out):
    txSamp = UE_out['Phy']['tx']['Xtf']
    txSamp_prach = UE_out['Phy']['tx']['Xtf_prach']

    #Hack to convert np.ndarray back into matlab of the correct shape
    if (NUM_RX_ANTENNAS == 1):
        rxSamp_flat = np.swapaxes(rx_tensor,0,1).reshape((3276*14*NUM_RX_ANTENNAS))
    else:
        rxSamp_flat = np.swapaxes(rx_tensor,0,2).reshape((3276*14*NUM_RX_ANTENNAS))

    #rxSamp_flat = rx_tensor.reshape((3276*14*NUM_RX_ANTENNAS))
    # Note that the "matlab" library used to be called mlarray
    rxSamp_mlarray = matlab.double(list(rxSamp_flat),is_complex=True)
    rxSamp_mlarray.reshape((3276,14,NUM_RX_ANTENNAS))

    # Matlab rx implementation also requires separate PRACH tensor.
    # Since we aren't using it in this example, just pass the tx tensor through
    rxSamp_prach = txSamp_prach

    # Matlab rx implemenation also requires no-noise samples for some genie-aided debug diagnostics
    # TODO can we just pass in a zero vector if we're not using the debug diagnostics?
    rxSamp_noNoise = txSamp
    rxSamp_prach_noNoise = txSamp_prach

    return (rxSamp_mlarray, rxSamp_prach, rxSamp_noNoise, rxSamp_prach_noNoise)


def print_status_header():
    print("Es/No     TBs      TB       BLER   Seconds/TB")
    print(" (dB)          Errors")
    print("---------------------------------------------")

def print_status(EsNo_dB, cur_tb_count, cur_tb_err_count, t_delta, newline):
    if (newline is True):
        newline_char = '\n'
    else:
        newline_char = '\r'

    print(f"{EsNo_dB:5.2f}   {cur_tb_count:5d}   {cur_tb_err_count:5d}   {(cur_tb_err_count/cur_tb_count):5.2e}        {(t_delta.total_seconds()/cur_tb_count):4.2f}",end=newline_char)


def test_PUSCH_Rx(xtf, stream):
    '''
    An example/unit test: DL transmssion 1 gNB -> four UEs (no channel impact)
    '''

    # cudart.cudaSetDevice(gpu_id)
#     stream = check_cuda_errors(cudart.cudaStreamCreate()) # move the CUDA stream creation out of the function

    # gNB
    #d_txTensorMem = check_cuda_errors(cudart.cudaMalloc(273*12 * 14 * 4 * 4))
    #h_tx_tensor_mem = check_cuda_errors(cudart.cudaMallocHost(273*12 * 14 * 4 * 8))


    # set pdsch dyn prms
    rntis = [20000]
    nUE = 1
    scid = [0]
    data_scid = [41]
    layers = [1]
    mcs = [10]
    #dmrs_port = [int('01230000', 16)]
    dmrs_port = [int('00000000', 16)]

    qam = []
    coderate = []
    for i in range(nUE):
        qam_i, coderate_i = pycuphy.get_mcs(mcs[i])
        qam.append(qam_i)
        coderate.append(coderate_i)

    # Now UE part (no channel impact)
    pdsch_rx_pipe = []
    pdsch_rx_handle = []
    pdsch_rx_dyn_params = []
    tbOutput = []
    for i in range(nUE):

        # create pipeline object
        pdsch_rx_pipe.append(pycuphy.PuschPipeline())

        # set static prms
        pdsch_rx_stat_prms = pycuphy.get_pusch_stat_prms(
            cell_id=41,
            num_rx_ant=4,
            num_tx_ant=4
        )
        pdsch_rx_handle.append(pdsch_rx_pipe[i].create_pusch_rx(pdsch_rx_stat_prms, stream))

        # set dyn prms
        currentSnr = -36 # set it to a small value for now. Eventually we need to fix the LLR clipping effect in cuPHY
        N0_ref = 10 ** (-currentSnr/10)
        nBSAnts = pdsch_rx_stat_prms.cellStatPrms[0].nRxAnt
        RB_num = pdsch_rx_stat_prms.cellStatPrms[0].nPrbUlBwp
        RwwInv  = (1/N0_ref) * np.eye(nBSAnts)
        tNoisePwr = (np.tile(RwwInv.flatten(), RB_num).reshape(nBSAnts, nBSAnts, RB_num, order='F') + 0j).astype(np.complex64)

        pdsch_rx_dyn_params.append(pycuphy.get_pusch_dyn_prms(
            stream=stream,
            num_ue=1,
            layers=[layers[i]],
            xtf_list=[xtf[:,:,i*4: (i+1)*4]],
            noise_power_list=[tNoisePwr],
            rnti_list=[rntis[i]],
            qam_list=[qam[i]],
            coderate_list=[coderate[i] * 10],
            scid_list=[scid[i]],
            data_scid_list=[data_scid[i]],
            dmrs_port_list=[dmrs_port[i]],
            dmrs_list=DMRS_POSITION,
            dmrs_add_ln_pos=DMRS_ADDLN_POS,
            dmrs_max_len=DMRS_MAX_LEN,
            rb_start=START_RB,
            rb_size=NUM_RBS,
            start_symbol_index=START_SYM,
            num_symbols=NUM_SYMS
        ))

        # run setup phase 1
        pdsch_rx_pipe[i].setup_pusch_rx(pdsch_rx_handle[i], pdsch_rx_dyn_params[i])

        # run setup phase 2
        harq_buffer_size = pdsch_rx_dyn_params[i].dataOut.harqBufferSizeInBytes[0]
        # harq_buffer = check_cuda_errors(cudart.cudaHostAlloc(harq_buffer_size * 1, cudart.cudaHostAllocPortable | cudart.cudaHostAllocMapped))
        harq_buffer = check_cuda_errors(cudart.cudaMalloc(harq_buffer_size))
        check_cuda_errors(cudart.cudaMemsetAsync(harq_buffer, 0, harq_buffer_size * 1, stream))
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))

        pdsch_rx_dyn_params[i] = pycuphy.get_dyn_prms_phase_2(pdsch_rx_dyn_params[i], harq_buffer)
        pdsch_rx_pipe[i].setup_pusch_rx(pdsch_rx_handle[i], pdsch_rx_dyn_params[i])

        # run pipeline
        pdsch_rx_pipe[i].run_pusch_rx(pdsch_rx_handle[i], stream)

        # check crc
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))
        crc = pdsch_rx_dyn_params[i].dataOut.tbCrcs[0]
        #assert crc == 0
        #print(crc, pdsch_rx_dyn_params[i].dataOut.tbPayloads)
        tbOutput.append(pdsch_rx_dyn_params[i].dataOut.tbPayloads)

        # if choose to store harq buffer in host
        if True:
            h_harq_buffer = check_cuda_errors(cudart.cudaMallocHost(harq_buffer_size))
            check_cuda_errors(cudart.cudaMemcpyAsync(h_harq_buffer, harq_buffer, harq_buffer_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
            check_cuda_errors(cudart.cudaStreamSynchronize(stream))

            buffer_point = C.cast(h_harq_buffer,C.POINTER(C.c_uint8))
            buffer_array = np.ctypeslib.as_array(buffer_point, shape=(harq_buffer_size,))

    # print("success.\n")

    # clean up
    for i in range(nUE):
        pdsch_rx_pipe[i].destroy_pusch_rx(pdsch_rx_handle[i])

    return tbOutput, crc


def test_PDSCH_Tx_to_PUSCH_Rx():
    '''
    An example/unit test: DL transmssion 1 gNB -> four UEs (no channel impact)
    '''

    np.random.seed(0)

    # cudart.cudaSetDevice(gpu_id)
    stream = check_cuda_errors(cudart.cudaStreamCreate())

    # gNB
    d_txTensorMem = check_cuda_errors(cudart.cudaMalloc(273*12 * 14 * 16 * 4))
    h_tx_tensor_mem = check_cuda_errors(cudart.cudaMallocHost(273*12 * 14 * 16 * 8))

    # create pipeline object
    pdsch_tx_pipe = pycuphy.PdschPipeline()

    # set pdsch stat prms
    pdsch_tx_stat_prms = pycuphy.get_pdsch_stat_prms(
        cell_id=41,
        num_rx_ant=4,
        num_tx_ant=16
    )
    print(f"pdsch_tx_stat_prms: {pdsch_tx_stat_prms}")
    pdsch_tx_handle = pdsch_tx_pipe.create_pdsch_tx(pdsch_tx_stat_prms)

    # set pdsch dyn prms
    rntis = [20000, 1001, 1002, 1003]
    nUE = 1
    scid = [0, 0, 1, 1]
    data_scid = [41, 42, 43, 44]
    layers = [1, 1, 1, 1]
    #Previously MCS index per UE was 27 and MCS table 1 (default option). These values correspond to QAM=8 and target code rate of 9480.
    target_code_rate = [9480, 9480, 9480, 9480]
    qam = [8, 8, 8, 8]
    dmrs_port = [int('01230000', 16), int('45670000', 16), int('01230000', 16), int('45670000', 16)]

    coderate = [] # values should have been divided by 1024 but following get_mcs function.
    tb_input = []
    for i in range(nUE):
        coderate_i = target_code_rate[i] / (10);
        qam_i = qam[i]

        tb_input_i = pycuphy.random_tb(qam_i, coderate_i, DMRS_POSITION, NUM_RBS, NUM_SYMS, layers[i])
        coderate.append(coderate_i)
        tb_input.append(tb_input_i)

    pdsch_tx_dyn_params = pycuphy.get_pdsch_dyn_prms(
        stream=stream,
        device_tx_tensor_mem=d_txTensorMem,
        num_ue=nUE,
        layers=layers,
        rnti_list=rntis,
        tb_input_list=tb_input,
        target_code_rate_list=target_code_rate,
        qam_list=qam,
        scid_list=scid,
        data_scid_list=data_scid,
        dmrs_port_list=dmrs_port,
        dmrs_list=DMRS_POSITION,
        rb_start=START_RB,
        rb_size=NUM_RBS,
        start_symbol_index=START_SYM,
        num_symbols=NUM_SYMS,
        dl_dmrs_scrambling_id=41
    )
    print(f"pdsch_tx_dyn_params: {pdsch_tx_dyn_params}")
    pdsch_tx_pipe.setup_pdsch_tx(pdsch_tx_handle, pdsch_tx_dyn_params)

    # run pdsch
    pdsch_tx_pipe.run_pdsch_tx(pdsch_tx_handle)

    # get xtf from cuphy
    check_cuda_errors(cudart.cudaStreamSynchronize(stream))
    xtf = pycuphy.return_xtf(d_txTensorMem, h_tx_tensor_mem, [273*12, 14, 16], stream)

    # Now UE part (no channel impact)
    pdsch_rx_pipe = []
    pdsch_rx_handle = []
    pdsch_rx_dyn_params = []
    for i in range(nUE):

        # create pipeline object
        pdsch_rx_pipe.append(pycuphy.PuschPipeline())

        # set static prms
        pdsch_rx_stat_prms = pycuphy.get_pusch_stat_prms(
            cell_id=41,
            num_rx_ant=4,
            num_tx_ant=4
        )
        print(f"pusch_rx_stat_prms: {pdsch_rx_stat_prms}")
        pdsch_rx_handle.append(pdsch_rx_pipe[i].create_pusch_rx(pdsch_rx_stat_prms, stream))

        # set dyn prms
        currentSnr = 36
        N0_ref = 10 ** (-currentSnr/10)
        nBSAnts = pdsch_rx_stat_prms.cellStatPrms[0].nRxAnt
        RB_num = pdsch_rx_stat_prms.cellStatPrms[0].nPrbUlBwp
        RwwInv  = (1/N0_ref) * np.eye(nBSAnts)
        tNoisePwr = (np.tile(RwwInv.flatten(), RB_num).reshape(nBSAnts, nBSAnts, RB_num, order='F') + 0j).astype(np.complex64)

        pdsch_rx_dyn_params.append(pycuphy.get_pusch_dyn_prms(
            stream=stream,
            num_ue=1,
            layers=[layers[i]],
            xtf_list=[xtf[:,:,i*4: (i+1)*4]],
            noise_power_list=[tNoisePwr],
            rnti_list=[rntis[i]],
            qam_list=[qam[i]],
            coderate_list=[coderate[i] * 10],
            scid_list=[scid[i]],
            data_scid_list=[data_scid[i]],
            dmrs_port_list=[dmrs_port[i]],
            dmrs_list=DMRS_POSITION,
            dmrs_add_ln_pos=DMRS_ADDLN_POS,
            dmrs_max_len=DMRS_MAX_LEN,
            rb_start=START_RB,
            rb_size=NUM_RBS,
            start_symbol_index=START_SYM,
            num_symbols=NUM_SYMS
        ))
        print(f"pusch_rx_dyn_params phase 1: {pdsch_rx_dyn_params}")

        # run setup phase 1
        pdsch_rx_pipe[i].setup_pusch_rx(pdsch_rx_handle[i], pdsch_rx_dyn_params[i])

        # run setup phase 2
        harq_buffer_size = pdsch_rx_dyn_params[i].dataOut.harqBufferSizeInBytes[0]
        # harq_buffer = check_cuda_errors(cudart.cudaHostAlloc(harq_buffer_size * 1, cudart.cudaHostAllocPortable | cudart.cudaHostAllocMapped))
        harq_buffer = check_cuda_errors(cudart.cudaMalloc(harq_buffer_size))
        check_cuda_errors(cudart.cudaMemsetAsync(harq_buffer, 0, harq_buffer_size * 1, stream))
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))

        pdsch_rx_dyn_params[i] = pycuphy.get_dyn_prms_phase_2(pdsch_rx_dyn_params[i], harq_buffer)
        print(f"pusch_rx_dyn_params phase 2: {pdsch_rx_dyn_params}")
        pdsch_rx_pipe[i].setup_pusch_rx(pdsch_rx_handle[i], pdsch_rx_dyn_params[i])

        # run pipeline
        pdsch_rx_pipe[i].run_pusch_rx(pdsch_rx_handle[i], stream)

        # check crc
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))
        crc = pdsch_rx_dyn_params[i].dataOut.tbCrcs[0]
        #assert crc == 0
        print(crc, tb_input[i], pdsch_rx_dyn_params[i].dataOut.tbPayloads)

        # if choose to store harq buffer in host
        if True:
            h_harq_buffer = check_cuda_errors(cudart.cudaMallocHost(harq_buffer_size))
            check_cuda_errors(cudart.cudaMemcpyAsync(h_harq_buffer, harq_buffer, harq_buffer_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
            check_cuda_errors(cudart.cudaStreamSynchronize(stream))

            buffer_point = C.cast(h_harq_buffer,C.POINTER(C.c_uint8))
            buffer_array = np.ctypeslib.as_array(buffer_point, shape=(harq_buffer_size,))

    # print("success.\n")

    # clean up
    pdsch_tx_pipe.destroy_pdsch_tx(pdsch_tx_handle)
    for i in range(nUE):
        pdsch_rx_pipe[i].destroy_pusch_rx(pdsch_rx_handle[i])

    check_cuda_errors(cudart.cudaFree(d_txTensorMem))
    check_cuda_errors(cudart.cudaFreeHost(h_tx_tensor_mem))

    print(f"\n\nFinal result crc: {crc}")

    return xtf
