# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Python bindings for cuPHY - cuPHY API parameter helper functions."""
from typing import Optional, List
from typing import Union

import numpy as np

from pycuphy.types import (  # type: ignore
    T_PdschTxStatPrms, T_CellStatPrm, T_PdschCellDynPrm, T_PdschUeGrpPrm,
    T_PdschUePrm, T_PdschCwPrm, T_PdschCellGrpDynPrm, T_PdschDataIn,
    T_CuPHYTensor, T_PdschDataOut, T_PdschTxDynPrms, T_PuschRxStatPrms,
    T_PuschRxStatDbgPrms, T_PuschRxCellDynPrm, T_PuschRxUeGrpPrm,
    T_PuschRxDmrsPrm, T_PuschRxUePrm, T_PuschRxCellGrpDynPrm,
    T_PuschRxDataIn, T_PuschRxDataOut, T_PuschRxDataInOut,
    T_PuschRxDynPrms, T_PuschRxDynDbgPrms, T_CuPHYDataType, T_PuschSetupPhase, T_PuschRunPhase, T_PuschEqCoefAlgoType, T_CuPHYTracker
)
from pycuphy.chest_filters import (  # type: ignore
    w_freq_array, w_freq4_array, w_freq_small_array, shift_seq_array,
    unshift_seq_array, shift_seq4_array, unshift_seq4_array
)
from pycuphy.util import bit_list_to_uint16  # type: ignore
from pycuphy.util import dmrs_api_convert  # type: ignore

__all__ = [
    "get_pdsch_stat_prms",
    "get_pdsch_dyn_prms",
    "get_pusch_stat_prms",
    "get_pusch_dyn_prms",
    "get_dyn_prms_phase_2",
]

# Constant definitions.
NUM_RE_PER_PRB = 12
NUM_PRB_MAX = 273
NUM_SYMBOLS = 14


def get_pdsch_stat_prms(
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_prb_ul_bwp: int = NUM_PRB_MAX,
        num_prb_dl_bwp: int = NUM_PRB_MAX,
        mu: int = 1)  -> T_PdschTxStatPrms:
    """Get a PdschTxStatPrms object based on given parameters.

    Args:
        cell_id (int): Physical cell ID.
        num_rx_ant (int): Number of receive antennas.
        num_tx_ant (int): Number of transmit antennas.
        num_prb_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
            Default: 273.
        num_prb_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
            Default: 273.
        mu (int): Numerology. Values in [0, 3]. Default: 1.

    Returns:
        T_PdschTxStatPrms: The PdschTxStatPrms object.
    """
    cell_stat_prm = T_CellStatPrm(
        phyCellId=cell_id,
        nRxAnt=int(num_rx_ant),
        nTxAnt=int(num_tx_ant),
        nPrbUlBwp=num_prb_ul_bwp,
        nPrbDlBwp=num_prb_dl_bwp,
        mu=mu
    )

    cuphy_tracker = T_CuPHYTracker(
        memoryFootprint=[]
    )

    pdsch_tx_stat_prms = T_PdschTxStatPrms(
        outInfo=[cuphy_tracker],
        cellStatPrms=[cell_stat_prm],
        dbg=None,
        read_TB_CRC=False,
        full_slot_processing=True,
        stream_priority=0,
        nMaxCellsPerSlot=1,
        nMaxUesPerCellGroup=8,
        nMaxCBsPerTB=0,
        nMaxPrb=NUM_PRB_MAX
    )

    return pdsch_tx_stat_prms


def get_pdsch_dyn_prms(
        stream: int,
        device_tx_tensor_mem: int,
        num_ue: int,
        layers: List[int],
        rnti_list: List[int],
        tb_input_list: List[np.ndarray],
        target_code_rate_list: List[int],
        qam_list: List[int],
        scid_list: List[int],
        data_scid_list: List[int],
        dmrs_port_list: List[int],
        slot: int = 0,
        dmrs_sym: Optional[List[int]] = None,
        resource_alloc: int = 1,
        rb_bitmap: List[int] = list(36*[0]),
        rb_start: int = 0,
        rb_size: int = 273,
        start_symbol_index: int = 2,
        num_symbols: int = 12,
        dl_dmrs_scrambling_id: int = 41
    ) -> T_PdschTxDynPrms:
    """Get a PdschTxDynPrms object based on given parameters.

    Args:
        stream (int): CUDA stream on which pipeline is launched.
        device_tx_tensor_mem (int): Raw pointer to the tensor buffer.
        num_ue (int): Number of UEs.
        layers (List[int]): Number of layers for each UE.
        rnti_list (List[int]) RNTI for each UE.
        tb_input_list (List[np.ndarray]): Transport blocks in bytes for each UE.
        target_code_rate_list (List[int]): target code rate for each UE.
        qam_list (List[int]): modulation order for each UE.
        scid_list (List[int]): DMRS sequence initialization for each UE
            [TS38.211, sec 7.4.1.1.2].
        data_scid_list (List[int]): Data scrambling IDs for each UE, more precisely
            `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
        dmrs_port_list (List[int]): DMRS ports for each UE. 4 bits per layer, up to 8 layers, i.e.
            32 bits per UE.
        slot (int): Slot number.
        dmrs_sym (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        resource_alloc (int): Resource allocation type
        rb_bitmap (List[int]): array of bytes indicating bitmask for allocated RBs
        rb_start (int): Start PRB index for the UE group.
        rb_size (int): Number of allocated PRBs for the UE group.
        start_symbol_index (int): Start OFDM symbol index of the UE group allocation.
        num_symbols (int): Number of symbols in the allocation, starting from
            `start_symbol_index`.
        dl_dmrs_scrambling_id (int): Downlink DMRS scrambling ID.
    """
    # Set the default value.
    dmrs_sym = dmrs_sym or [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    cell_dyn_prm = T_PdschCellDynPrm(
        nCsiRsPrms=0,
        csiRsPrmsOffset=0,
        cellPrmStatIdx=0,
        cellPrmDynIdx=0,
        slotNum=slot,
        testModel=0
    )

    ue_grp_prm = T_PdschUeGrpPrm(
        cellPrmIdx=0,
        resourceAlloc=resource_alloc,
        rbBitmap=rb_bitmap,
        rbStart=rb_start,
        rbSize=rb_size,
        dlDmrsSymbPos=np.uint16(bit_list_to_uint16(dmrs_sym)),
        StartSymbolIndex=int(start_symbol_index),
        NrOfSymbols=int(num_symbols),
        uePrmIdxs=list(range(num_ue)),
        numDmrsCdmGrpsNoData=int(2),
        dlDmrsScramblingId=int(dl_dmrs_scrambling_id)
    )

    ue_prm = []
    for i in range(num_ue):
        dmrs_port_bitmask = 0
        for j in range(int(layers[i])):
            dmrs_port_bitmask |= (1 << ((dmrs_port_list[i] >> (28 - 4 *j)) & 0x000F))
        ue_prm.append(T_PdschUePrm(
            ueGrpPrmIdx=0,
            SCID=int(scid_list[i]),
            nrOfLayers=int(layers[i]),
            dmrsPortBmsk=np.uint16(dmrs_port_bitmask),
            BWPStart=int(0),
            refPoint=0,
            beta_dmrs=1.0,
            beta_qam=1.0,
            RNTI=int(rnti_list[i]),
            dataScramblingId=int(data_scid_list[i]),
            cwIdxs=[i],
            enablePrcdBf=False,
            pmwPrmIdx=None
        ))

    cw_prm = []
    tb_offset = 0
    for i in range(num_ue):
        cw_prm.append(T_PdschCwPrm(
            uePrmIdx=i,
            targetCodeRate=np.uint16(target_code_rate_list[i]),
            qamModOrder = np.uint8(qam_list[i]),
            rvIndex=int(0),
            tbStartOffset=tb_offset,
            TBSize=int(tb_input_list[i].size),
            n_PRB_LBRM=int(NUM_PRB_MAX),
            maxLayers=int(4),
            maxQm=int(8),
        ))
        tb_offset += tb_input_list[i].size

    cell_grp_dyn_prm = T_PdschCellGrpDynPrm(
        cellPrms=[cell_dyn_prm],
        ueGrpPrms=[ue_grp_prm],
        uePrms=ue_prm,  # A list.
        cwPrms=cw_prm,  # A list.
        csiRsPrms=None,
        pmwPrms=None
    )

    data_in = T_PdschDataIn(
        tbInput=[np.concatenate(tb_input_list)]
    )

    cuphy_tensor = T_CuPHYTensor(
        dimensions=[NUM_PRB_MAX * NUM_RE_PER_PRB, NUM_SYMBOLS, 16],
        strides=[NUM_PRB_MAX * NUM_RE_PER_PRB, NUM_SYMBOLS, 16],
        dataType=T_CuPHYDataType.CUPHY_C_32F,
        pAddr=device_tx_tensor_mem
    )

    data_out = T_PdschDataOut(
        dataTx=[cuphy_tensor]
    )

    pdsch_tx_dyn_prms = T_PdschTxDynPrms(
        cuStream=stream,
        procModeBmsk=4,  # Enable inter-cell batching.
        cellGrpDynPrm=cell_grp_dyn_prm,
        dataIn=data_in,
        tbCRCDataIn=None,
        dataOut=data_out
    )

    return pdsch_tx_dyn_prms


def get_pusch_stat_prms(
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_prb_ul_bwp: int = 273,
        num_prb_dl_bwp: int = 273,
        mu: int = 1,
        debug_file_name: Optional[str] = None) -> T_PuschRxStatPrms:
    """Get a PuschRxStatPrms object based on given parameters.

    Args:
        cell_id (int): Physical cell ID.
        num_rx_ant (int): Number of receive antennas.
        num_tx_ant (int): Number of transmit antennas.
        num_prb_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
            Default: 273.
        num_prb_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
            Default: 273.
        mu (int): Numerology. Values in [0, 3]. Default: 1.
        debug_file_name (str): Debug dump filename. Default: None (no debugging).

    Returns:
        T_PuschRxStatPrms: The PuschRxStatPrms object.
    """
    cell_stat_prm = T_CellStatPrm(
        phyCellId=cell_id,
        nRxAnt=num_rx_ant,
        nTxAnt=num_tx_ant,
        nPrbUlBwp=num_prb_ul_bwp,
        nPrbDlBwp=num_prb_dl_bwp,
        mu=mu)

    cuphy_tracker = T_CuPHYTracker(
        memoryFootprint=[]
    )

    pusch_rx_stat_prms = T_PuschRxStatPrms(
        outInfo=[cuphy_tracker],
        cellStatPrms = [cell_stat_prm],
        enableCfoCorrection = np.uint8(0),
        enablePuschTdi = np.uint8(0),
        enableDftSOfdm = np.uint8(0),  # disable this feature now
        enableTbSizeCheck = np.uint8(1),
        ldpcnIterations = np.uint8(10),
        ldpcEarlyTermination = np.uint8(0),
        ldpcUseHalf = np.uint8(1),
        ldpcAlgoIndex = np.uint8(0),
        ldpcFlags = np.uint32(0),
        ldpcKernelLaunch = np.uint32(1),
        enableEqIrc = np.uint8(0),
        eqCoeffAlgo = T_PuschEqCoefAlgoType.PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
        enableRssiMeasurement = np.uint8(0),
        enableSinrMeasurement = np.uint8(0),
        stream_priority = 0,
        nCells = np.uint16(1),
        nMaxCells = np.uint16(1),
        nMaxCellsPerSlot = np.uint16(1),
        nMaxTbs = np.uint32(0),
        nMaxCbsPerTb = np.uint32(0),
        nMaxTotCbs = np.uint32(0),
        nMaxRx = np.uint32(0),
        nMaxPrb = np.uint32(273),
        WFreq = w_freq_array,
        WFreq4 = w_freq4_array,
        WFreqSmall = w_freq_small_array,
        ShiftSeq = shift_seq_array,
        UnShiftSeq = unshift_seq_array,
        ShiftSeq4 = shift_seq4_array,
        UnShiftSeq4 = unshift_seq4_array,
        dbg = T_PuschRxStatDbgPrms(
            outFileName = debug_file_name,
            descrmOn = 1,
            enableApiLogging = 0),
        )

    return pusch_rx_stat_prms


def get_pusch_dyn_prms(
        stream: int,
        num_ue: int,
        layers: List[int],
        xtf_list: List[np.ndarray],
        rnti_list: List[int],
        qam_list: List[int],
        coderate_list: List[float],
        scid_list: List[int],
        data_scid_list: List[int],
        dmrs_port_list: List[int],
        slot: int = 0,
        dmrs_sym: Optional[List[int]] = None,
        dmrs_scrm_id: int = 41,
        dmrs_max_len: int = 2,
        dmrs_add_ln_pos: int = 1,
        num_dmrs_cdm_grps_no_data: int = 2,
        rb_start: int = 0,
        rb_size: int = 273,
        start_symbol_index: int = 2,
        num_symbols: int = 12
    ) -> T_PuschRxDynPrms:
    """Get a PuschTxDynPrms object based on given parameters.

    Args:
        stream (int): CUDA stream on which pipeline is launched.
        num_ue (int): Number of UEs.
        layers (List[int]): Number of layers for each UE.
        xtf_list (List[np.ndarray]): List of tensors with each tensor (indexed by
            `cellPrmDynIdx`) representing the receive slot buffer of a cell in the cell group.
        rnti_list (List[int]) RNTI for each UE.
        qam_list (List[int]): Modulation order for each UE.
        coderate_list (List[float]): Code rate for each UE. This is the number of information bits
            per 1024 coded bits expressed in 0.1 bit units.
        scid_list (List[int]): DMRS sequence initialization for each UE
            [TS38.211, sec 7.4.1.1.2].
        data_scid_list (List[int]): Data scrambling IDs for each UE, more precisely
            `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
        dmrs_port_list (List[int]): DMRS ports for each UE.
        slot (int): Slot number.
        dmrs_sym (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        dmrs_scrm_id (int): DMRS scrambling ID.
        dmrs_max_len (int):
        dmrs_add_ln_pos (int):
        num_dmrs_cdm_grps_no_data (int):
        rb_start (int): Start PRB index of the UE group allocation.
        rb_size (int): Number of allocated PRBs for the UE group.
        start_symbol_index (int): Start OFDM symbol index for the UE group allocation.
        num_symbols (int): Number of symbols in the UE group allocation.

    Returns:
        T_PuschRxDynPrms: PUSCH Tx dynamic parameters.
    """
    # Set the default value.
    dmrs_sym = dmrs_sym or [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    cell_dyn_pPrm = T_PuschRxCellDynPrm(
        cellPrmStatIdx = np.uint16(0),
        cellPrmDynIdx = np.uint16(0),
        slotNum = np.uint16(slot)
    )

    ue_grp_prm = T_PuschRxUeGrpPrm(
        cellPrmIdx = 0,
        dmrsDynPrm = T_PuschRxDmrsPrm(
            dmrsAddlnPos = np.uint8(dmrs_add_ln_pos),
            dmrsMaxLen = np.uint8(dmrs_max_len),
            numDmrsCdmGrpsNoData = np.uint8(num_dmrs_cdm_grps_no_data),
            ulDmrsScramblingId = np.uint8(dmrs_scrm_id)
        ),
        rbStart=np.uint16(rb_start),
        rbSize = np.uint16(rb_size),
        StartSymbolIndex = np.uint8(start_symbol_index),
        NrOfSymbols = np.uint8(num_symbols),
        dmrsSymLocBmsk = np.uint16(bit_list_to_uint16(dmrs_sym)),
        rssiSymLocBmsk = np.uint16(bit_list_to_uint16(dmrs_sym)),
        uePrmIdxs = list(range(num_ue)),
    )

    ue_prm = []
    for i in range(num_ue):
        ue_prm.append(T_PuschRxUePrm(
            pduBitmap = np.uint16(1),
            ueGrpPrmIdx = np.uint16(0),
            SCID = np.uint8(scid_list[i]),
            # E.g., [1+2, 4+8] UE1 uses layer 1 2, UE2 uses layer 3 4
            dmrsPortBmsk = np.uint16(dmrs_api_convert(dmrs_port_list[i], layers[i])),
            mcsTable = np.uint8(0),  # Not used.
            mcsIndex = np.uint8(0),  # Not used.
            TBSize   = np.uint32(96321),
            targetCodeRate = np.uint16(coderate_list[i]),
            qamModOrder = np.uint8(qam_list[i]),
            rvIndex = np.uint8(0),
            RNTI = np.uint16(rnti_list[i]),
            dataScramblingId = np.uint16(data_scid_list[i]),
            nrOfLayers = np.uint8(layers[i]),
            newDataIndicator = np.uint8(1),
            harqProcessId = np.uint8(0),
            i_lbrm = np.uint8(1),
            maxLayers = np.uint8(4),
            maxQm = np.uint8(8),
            n_PRB_LBRM = np.uint16(273),
            enableTfPrcd = np.uint8(0), # disable this feature now
            uciPrms = None
        ))

    cell_grp_dyn_pPrm = T_PuschRxCellGrpDynPrm(
        cellPrms = [cell_dyn_pPrm],
        ueGrpPrms = [ue_grp_prm],
        uePrms = ue_prm,
    )

    data_in = T_PuschRxDataIn(
        tDataRx = xtf_list,
    )

    data_out = T_PuschRxDataOut(
        totNumTbs = np.zeros([1], dtype=np.uint32),
        totNumCbs = np.zeros([1], dtype=np.uint32),
        totNumPayloadBytes = np.zeros([1], dtype=np.uint32),
        totNumUciSegs = np.zeros([1], dtype=np.uint16),
        harqBufferSizeInBytes = np.zeros([num_ue], dtype=np.uint32),
        cbCrcs = np.ones([1000], dtype=np.uint32),
        tbCrcs = np.ones([num_ue], dtype=np.uint32),
        tbPayloads = np.zeros([200000], dtype=np.uint8),
        uciPayloads = None,
        uciCrcFlags = None,
        pUciDTXs = None,
        numCsi2Bits = None,
        startOffsetsCbCrc = np.zeros([num_ue], dtype=np.uint32),
        startOffsetsTbCrc = np.zeros([num_ue], dtype=np.uint32),
        startOffsetsTbPayload = np.zeros([num_ue], dtype=np.uint32),
        taEsts = np.zeros([num_ue], dtype=float),
        rssi = np.zeros([1], dtype=float),
        rsrp = np.zeros([num_ue], dtype=float),
        noiseVarPreEq = np.zeros([1], dtype=float),
        noiseVarPostEq = np.zeros([num_ue], dtype=float),
        sinrPreEq = np.zeros([num_ue], dtype=float),
        sinrPostEq = np.zeros([num_ue], dtype=float),
        cfoHz = np.zeros([num_ue], dtype=float),
        uciOnPuschOutOffsets = None,
        HarqDetectionStatus = np.zeros([num_ue], dtype=np.uint8),
        CsiP1DetectionStatus = np.zeros([num_ue], dtype=np.uint8),
        CsiP2DetectionStatus = np.zeros([num_ue], dtype=np.uint8),
    )

    data_in_out = T_PuschRxDataInOut(
        harqBuffersInOut=[]
    )

    pusch_rx_dyn_pPrms = T_PuschRxDynPrms(
        cuStream = stream,
        setupPhase = T_PuschSetupPhase.PUSCH_SETUP_PHASE_1,
        procModeBmsk = 0,
        cellGrpDynPrm = cell_grp_dyn_pPrm,
        dataIn = data_in,
        dataOut = data_out,
        dataInOut = data_in_out,
        cpuCopyOn = 1,
        dbg = T_PuschRxDynDbgPrms(
            enableApiLogging = 0
        ))

    return pusch_rx_dyn_pPrms


def get_dyn_prms_phase_2(
        pusch_rx_dyn_prms_phase1: T_PuschRxDynPrms,
        harq_buffer: Union[int, List[int]]
    ) -> T_PuschRxDynPrms:
    """Get dynamic PUSCH phase 2 setup parameters."""
    if isinstance(harq_buffer, int):
        harq_buffer = [harq_buffer]

    pusch_rx_dyn_prms_phase2 = pusch_rx_dyn_prms_phase1._replace(
        setupPhase=T_PuschSetupPhase.PUSCH_SETUP_PHASE_2,
        dataInOut=T_PuschRxDataInOut(harqBuffersInOut=harq_buffer)
    )
    return pusch_rx_dyn_prms_phase2
