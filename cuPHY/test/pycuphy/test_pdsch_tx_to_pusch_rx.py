# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test PDSCH Tx and PUSCH Rx."""
from cuda import cudart
import numpy as np

import pycuphy


CUPHY_LDPC_MAX_LIFTING_SIZE = 384
CUPHY_MAX_N_CBS_PER_TB_SUPPORTED = 152
CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES = 66
NUM_RE_PER_PRB = 12
MAX_NUM_PRB = 273
NUM_TX_ANT = 16
NUM_SYM_PER_SLOT = 14


def test_pdsch_tx_to_pusch_rx():
    """An example/unit test: DL transmssion 1 gNB -> four UEs (no channel impact)."""
    np.random.seed(0)

    stream = pycuphy.check_cuda_errors(cudart.cudaStreamCreate())

    num_re = MAX_NUM_PRB * NUM_RE_PER_PRB * NUM_SYM_PER_SLOT * NUM_TX_ANT
    # Half-precision, complex values, 4 bytes per element.
    device_tx_tensor_mem = pycuphy.check_cuda_errors(cudart.cudaMalloc(num_re * 4))

    # Full precision, complex values, 8 bytes per element.
    host_tx_tensor_mem = pycuphy.check_cuda_errors(cudart.cudaMallocHost(num_re * 8))

    # Maximum number of bits, one bit expressed as a float.
    max_num_bits = CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * \
        CUPHY_LDPC_MAX_LIFTING_SIZE * CUPHY_MAX_N_CBS_PER_TB_SUPPORTED
    host_tx_ldpc_mem = pycuphy.check_cuda_errors(cudart.cudaMallocHost(max_num_bits * 4))

    # Create a PDSCH pipeline object.
    pdsch_tx_pipe = pycuphy.PdschPipeline()

    # Set PdschTxStatPrms.
    pdsch_tx_stat_prms = pycuphy.get_pdsch_stat_prms(cell_id=41, num_rx_ant=4, num_tx_ant=4)
    pdsch_tx_handle = pdsch_tx_pipe.create_pdsch_tx(pdsch_tx_stat_prms)

    # Set PdschTxDynPrms.
    rntis = [1000, 1001, 1002, 1003]
    num_ue = 4
    scid = [0, 0, 1, 1]
    data_scid = [41, 42, 43, 44]
    dmrs_position = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    layers = [4, 4, 4, 4]
    #Previously MCS index per UE was 27 and MCS table 1 (default option). These values correspond to QAM=8 and target code rate of 9480.
    target_code_rate = [9480, 9480, 9480, 9480]
    qam = [8, 8, 8, 8]
    dmrs_port = [
        int('01230000', 16), int('45670000', 16), int('01230000', 16), int('45670000', 16)
    ]

    coderate = [] # Values should technically be divided by 1024 to get code rate
    tb_input = []
    for i in range(num_ue):
        coderate_i = target_code_rate[i] / 10.0; # should have been divided by 1024 too but following get_mcs function.
        qam_i = qam[i]
        tb_input_i = pycuphy.random_tb(qam_i, coderate_i, dmrs_position, 273, 12, layers[i])

        coderate.append(coderate_i)
        tb_input.append(tb_input_i)

    pdsch_tx_dyn_params = pycuphy.get_pdsch_dyn_prms(
        stream=stream,
        device_tx_tensor_mem=device_tx_tensor_mem,
        num_ue=num_ue,
        layers=layers,
        rnti_list=rntis,
        tb_input_list=tb_input,
        target_code_rate_list=target_code_rate,
        qam_list=qam,
        scid_list=scid,
        data_scid_list=data_scid,
        dmrs_port_list=dmrs_port
    )
    pdsch_tx_pipe.setup_pdsch_tx(pdsch_tx_handle, pdsch_tx_dyn_params)

    # Run PDSCH.
    pdsch_tx_pipe.run_pdsch_tx(pdsch_tx_handle)
    pycuphy.check_cuda_errors(cudart.cudaStreamSynchronize(stream))

    # Get LDPC output and check that the size matches, i.e. basically checking
    # that this still runs.
    cell_idx = 0
    for tb_idx in range(4):
        ldpc_output = pdsch_tx_pipe.get_ldpc_output(pdsch_tx_handle, cell_idx, tb_idx, host_tx_ldpc_mem)
        assert ldpc_output.shape == (25344, 92)

    # Get xtf.
    xtf = pycuphy.device_to_numpy(device_tx_tensor_mem, host_tx_tensor_mem, [273 * 12, 14, 16], stream)

    # Now UE part (no channel impact), PDSCH Rx = PUSCH Rx here.
    pdsch_rx_pipe = []
    pdsch_rx_handle = []
    pdsch_rx_dyn_params = []
    for i in range(num_ue):

        # Create pipeline object.
        current_pdsch_pipe = pycuphy.PuschPipeline()
        pdsch_rx_pipe.append(current_pdsch_pipe)

        # Set static prms.
        # If debugging, set h5 file for dump, if not debug, set debug_dump_file = None.
        if i == 0 or i == 1:
            debug_dump_file = f"debug_dump_{i}.h5"  # Debug first two PUSCH pipelines.
        else:
            debug_dump_file = None  # Not debugging.

        pdsch_rx_stat_prms = pycuphy.get_pusch_stat_prms(
            cell_id=41,
            num_rx_ant=4,
            num_tx_ant=4,
            debug_file_name=debug_dump_file
        )
        pdsch_rx_handle.append(current_pdsch_pipe.create_pusch_rx(pdsch_rx_stat_prms, stream))

        # Set dynamic prms.
        pdsch_rx_dyn_params.append(pycuphy.get_pusch_dyn_prms(
            stream=stream,
            num_ue=1,
            layers=[layers[i]],
            xtf_list=[xtf[:, :, i * 4:(i + 1) * 4]],
            rnti_list=[rntis[i]],
            qam_list=[qam[i]],
            coderate_list=[coderate[i] * 10],
            scid_list=[scid[i]],
            data_scid_list=[data_scid[i]],
            dmrs_port_list=[dmrs_port[i]],
            dmrs_sym=dmrs_position
        ))

        # Run setup phase 1.
        current_pdsch_pipe.setup_pusch_rx(pdsch_rx_handle[i], pdsch_rx_dyn_params[i])

        # Run setup phase 2.
        harq_buffer_size = pdsch_rx_dyn_params[i].dataOut.harqBufferSizeInBytes[0]
        harq_buffer = pycuphy.check_cuda_errors(cudart.cudaMalloc(harq_buffer_size))
        pycuphy.check_cuda_errors(
            cudart.cudaMemsetAsync(harq_buffer, 0, harq_buffer_size * 1, stream)
        )
        pycuphy.check_cuda_errors(cudart.cudaStreamSynchronize(stream))

        pdsch_rx_dyn_params[i] = pycuphy.get_dyn_prms_phase_2(pdsch_rx_dyn_params[i], harq_buffer)
        current_pdsch_pipe.setup_pusch_rx(pdsch_rx_handle[i], pdsch_rx_dyn_params[i])

        # Run pipeline.
        current_pdsch_pipe.run_pusch_rx(pdsch_rx_handle[i], stream)

        # Dump debug file.
        if i == 0 or i == 1:
            pdsch_rx_pipe[i].write_dbg_buf_synch(pdsch_rx_handle[i], stream)

        # Check CRC.
        pycuphy.check_cuda_errors(cudart.cudaStreamSynchronize(stream))
        crc = pdsch_rx_dyn_params[i].dataOut.tbCrcs[0]
        assert crc == 0

        pycuphy.check_cuda_errors(cudart.cudaFree(harq_buffer))

    # Clean up.
    pdsch_tx_pipe.destroy_pdsch_tx(pdsch_tx_handle)
    for i in range(num_ue):
        pdsch_rx_pipe[i].destroy_pusch_rx(pdsch_rx_handle[i])

    pycuphy.check_cuda_errors(cudart.cudaFree(device_tx_tensor_mem))
    pycuphy.check_cuda_errors(cudart.cudaFreeHost(host_tx_tensor_mem))
    pycuphy.check_cuda_errors(cudart.cudaFreeHost(host_tx_ldpc_mem))
    pycuphy.check_cuda_errors(cudart.cudaStreamDestroy(stream))

    print("PASS!")