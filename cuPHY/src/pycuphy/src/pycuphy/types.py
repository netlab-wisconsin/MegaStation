# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Python bindings for cuPHY - cuPHY API types definition."""
from enum import Enum
from typing import List, NamedTuple, NewType

import numpy

__all__ = [
    "T_CuPHYTracker",
    "T_CUDADataType",
    "T_CuPHYDataType",
    "T_PuschSetupPhase",
    "T_PuschRunPhase",
    "T_PdschTxHndl",
    "T_PuschRxHndl",
    "T_CellStatPrm",
    "T_PdschDbgPrms",
    "T_PdschTxStatPrms",
    "T_CuPHYTensor",
    "T_PdschCellDynPrm",
    "T_PdschUeGrpPrm",
    "T_PdschUePrm",
    "T_PdschCwPrm",
    "T_CsirsRrcDynPrm",
    "T_PmW",
    "T_PdschCellGrpDynPrm",
    "T_PdschDataIn",
    "T_PdschDataOut",
    "T_PdschTxDynPrms",
    "T_PuschRxStatDbgPrms",
    "T_PuschRxStatPrms",
    "T_PuschRxCellDynPrm",
    "T_PuschRxDmrsPrm",
    "T_PuschRxUeGrpPrm",
    "T_PuschRxUePrm",
    "T_PuschRxCellGrpDynPrm",
    "T_PuschRxDataIn",
    "T_UciOnPuschOutOffsets",
    "T_PuschRxDataOut",
    "T_PuschRxDataInOut",
    "T_PuschRxDynDbgPrms",
    "T_PuschRxDynPrms"
]


T_PdschTxHndl = NewType('T_PdschTxHndl', numpy.uint64)
T_PuschRxHndl = NewType('T_PuschRxHndl', numpy.uint64)

class T_CuPHYTracker(NamedTuple):
    """Implements cuPHY Tracker type.

    Args:
        memoryFootprint List[numpy_uint64]: A single element list with a pointer to a cuphyMemoryFootprint object
    """
    memoryFootprint : List[numpy.uint64]




class T_CUDADataType(Enum):
    """Provides CudaDataType_t.

    TODO: Should use something exposed from cuda instead of recreating here.
    """

    CUDA_R_16F  =  2, 'real as a half'
    CUDA_C_16F  =  6, 'complex as a pair of half numbers'
    CUDA_R_16BF = 14, 'real as a nv_bfloat16'
    CUDA_C_16BF = 15, 'complex as a pair of nv_bfloat16 numbers'
    CUDA_R_32F  =  0, 'real as a float'
    CUDA_C_32F  =  4, 'complex as a pair of float numbers'
    CUDA_R_64F  =  1, 'real as a double'
    CUDA_C_64F  =  5, 'complex as a pair of double numbers'
    CUDA_R_4I   = 16, 'real as a signed 4-bit int'
    CUDA_C_4I   = 17, 'complex as a pair of signed 4-bit int numbers'
    CUDA_R_4U   = 18, 'real as a unsigned 4-bit int'
    CUDA_C_4U   = 19, 'complex as a pair of unsigned 4-bit int numbers'
    CUDA_R_8I   =  3, 'real as a signed 8-bit int'
    CUDA_C_8I   =  7, 'complex as a pair of signed 8-bit int numbers'
    CUDA_R_8U   =  8, 'real as a unsigned 8-bit int'
    CUDA_C_8U   =  9, 'complex as a pair of unsigned 8-bit int numbers'
    CUDA_R_16I  = 20, 'real as a signed 16-bit int'
    CUDA_C_16I  = 21, 'complex as a pair of signed 16-bit int numbers'
    CUDA_R_16U  = 22, 'real as a unsigned 16-bit int'
    CUDA_C_16U  = 23, 'complex as a pair of unsigned 16-bit int numbers'
    CUDA_R_32I  = 10, 'real as a signed 32-bit int'
    CUDA_C_32I  = 11, 'complex as a pair of signed 32-bit int numbers'
    CUDA_R_32U  = 12, 'real as a unsigned 32-bit int'
    CUDA_C_32U  = 13, 'complex as a pair of unsigned 32-bit int numbers'
    CUDA_R_64I  = 24, 'real as a signed 64-bit int'
    CUDA_C_64I  = 25, 'complex as a pair of signed 64-bit int numbers'
    CUDA_R_64U  = 26, 'real as a unsigned 64-bit int'
    CUDA_C_64U  = 27, 'complex as a pair of unsigned 64-bit int numbers'


class T_CuPHYDataType(Enum):
    """Provides T_CuPHYDataType."""
    CUPHY_VOID  = -1,                        'uninitialized type'
    CUPHY_BIT   = 20,                        '1-bit value'
    CUPHY_R_8I  = T_CUDADataType.CUDA_R_8I,  '8-bit signed integer real values'
    CUPHY_C_8I  = T_CUDADataType.CUDA_C_8I,  '8-bit signed integer complex values'
    CUPHY_R_8U  = T_CUDADataType.CUDA_R_8U,  '8-bit unsigned integer real values'
    CUPHY_C_8U  = T_CUDADataType.CUDA_C_8U,  '8-bit unsigned integer complex values'
    CUPHY_R_16I = 21,                        '16-bit signed integer real values'
    CUPHY_C_16I = 22,                        '16-bit signed integer complex values'
    CUPHY_R_16U = 23,                        '16-bit unsigned integer real values'
    CUPHY_C_16U = 24,                        '16-bit unsigned integer complex values'
    CUPHY_R_32I = T_CUDADataType.CUDA_R_32I, '32-bit signed integer real values'
    CUPHY_C_32I = T_CUDADataType.CUDA_C_32I, '32-bit signed integer complex values'
    CUPHY_R_32U = T_CUDADataType.CUDA_R_32U, '32-bit unsigned integer real values'
    CUPHY_C_32U = T_CUDADataType.CUDA_C_32U, '32-bit unsigned integer complex values'
    CUPHY_R_16F = T_CUDADataType.CUDA_R_16F, 'half precision (16-bit) real values'
    CUPHY_C_16F = T_CUDADataType.CUDA_C_16F, 'half precision (16-bit) complex values'
    CUPHY_R_32F = T_CUDADataType.CUDA_R_32F, 'single precision (32-bit) real values'
    CUPHY_C_32F = T_CUDADataType.CUDA_C_32F, 'single precision (32-bit) complex values'
    CUPHY_R_64F = T_CUDADataType.CUDA_R_64F, 'single precision (64-bit) real values'
    CUPHY_C_64F = T_CUDADataType.CUDA_C_64F, 'double precision (64-bit) complex values'


class T_PuschSetupPhase(Enum):
    """Provides T_PuschSetupPhase enumeration."""

    PUSCH_SETUP_PHASE_INVALID = 0, 'invalid setup phase for historical API'
    PUSCH_SETUP_PHASE_1       = 1, 'PUSCH Setup Phase 1 - calculate HARQ buffer sizes'
    PUSCH_SETUP_PHASE_2       = 2, 'PUSCH Setup Phase 2 - perform rest of the setup'
    PUSCH_SETUP_MAX_PHASES    = 3, 'End of PUSCH Setup Phase range'
    
class T_PuschRunPhase(Enum):
    """Provides T_PuschRunPhase enumeration."""

    PUSCH_RUN_PHASE_INVALID = 0, 
    PUSCH_RUN_PHASE_1       = 1, 
    PUSCH_RUN_PHASE_2       = 2, 
    PUSCH_RUN_PHASE_3       = 3,
    PUSCH_RUN_MAX_PHASES    = 4, 

class T_PuschEqCoefAlgoType(Enum):
    """Provides equalizer algorithm enumeration"""

    PUSCH_EQ_ALGO_TYPE_RZF             = 0, 'rZf'
    PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE = 1, 'noiseDiagMmse'
    PUSCH_EQ_ALGO_TYPE_MMSE_IRC        = 2, 'mmseIrc'


class T_CellStatPrm(NamedTuple):
    """Implements cuPHY Cell Static Parameters common to all channels.

    Args:
        phyCellId (numpy.uint16): Physical cell ID.
        nRxAnt (numpy.uint16): Number of receiving antennas.
        nTxAnt (numpy.uint16): Number of transmitting antennas.
        nPrbUlBwp (numpy.uint16): Number of PRBs (Physical Resource Blocks) allocated in
            UL BWP (bandwidth part).
        nPrbDlBwp (numpy.uint16): Number of PRBs allocated in DL BWP.
        mu (numpy.uint8): Numerology [0, 3].
    """
    phyCellId : numpy.uint16
    nRxAnt : numpy.uint16
    nTxAnt : numpy.uint16
    nPrbUlBwp : numpy.uint16
    nPrbDlBwp : numpy.uint16
    mu : numpy.uint8


class T_PdschDbgPrms(NamedTuple):
    """Implements PDSCH channel debug parameters.

    Args:
        cfgFilename (str): Name of HDF5 file that drives the DL pipeline. No file, if empty.
        refCheck (bool): If True, compare the output of each pipeline component with the reference
            output from the cfgFileName file that drives the pipeline.
        cfgIdenticalLdpcEncCfgs (bool): Enable single cuPHY LDPC call for all TBs, if True.
            Will be reset at runtime if LDPC config. params are different across TBs.
    """
    cfgFilename : str
    refCheck : bool
    cfgIdenticalLdpcEncCfgs : bool


class T_PdschTxStatPrms(NamedTuple):
    """Implements PDSCH static parameters type.

    Defines `nMaxCells` quantity of static parameters, where `nMaxCells`
    is defined as the length of the `cellStatPrms` array.

    Args:
        cellStatPrms (List[T_CellStatPrm]): List of cell-specific static parameters with
            `nMaxCells` elements.
        dbg (List[T_PdschDbgPrms]): List of cell-specific debug parameters with `nMaxCells`
            elements.
        read_TB_CRC (bool): If True, TB CRCs are read from input buffers and not computed.
        full_slot_processing (bool): If false, all cells ran on this PdschTx will undergo: TB-CRC +
            CB-CRC/segmentation + LDPC encoding + rate-matching/scrambling. If true, all cells ran
            on this PdschTx will undergo full slot processing:  TB-CRC + CB-CRC/segmentation +
            LDPC encoding + rate-matching/scrambling/layer-mapping + modulation + DMRS.
            NB: This mode is an a priori known characteristic of the cell; a cell will never switch
            between modes.
        stream_priority (int): CUDA stream priority for all internal to PDSCH streams. Should match
            the priority of CUDA stream passed in T_PdschDynPrms during setup.
        nMaxCellsPerSlot (numpy.uint16): Maximum number of cells supported.
            nCells <= nMaxCellsPerSlot and nMaxCellsPerSlot <= PDSCH_MAX_CELLS_PER_CELL_GROUP.
            If 0, cuPHY compile-time constant PDSCH_MAX_CELLS_PER_CELL_GROUP is used.
        nMaxUesPerCellGroup (numpy.uint16): Maximum number of UEs supported in a cell group, i.e.,
            across all the cells. nMaxUesPerCellGroup <= PDSCH_MAX_UES_PER_CELL_GROUP.
            If 0, the compile-time constant PDSCH_MAX_UES_PER_CELL_GROUP is used.
        nMaxCBsPerTB (numpy.uint16): Maximum number of CBs supported per TB; limit valid for any UE
            in that cell. nMaxCBsPerTb <= MAX_N_CBS_PER_TB_SUPPORTED.
            If 0, the compile-time constant MAX_N_CBS_PER_TB_SUPPORTED is used.
        nMaxPrb (numpy.uint16): Maximum value of cuphyCellStatPrm_t.nPrbDlBwp supported by PdschTx
            object. nMaxPrb <= 273. If 0, 273 is used.
        outInfo (List[T_cuPHYTracker]): pointer to cuPHY tracker
    """
    cellStatPrms : List[T_CellStatPrm]
    dbg : List[T_PdschDbgPrms]
    read_TB_CRC : bool
    full_slot_processing : bool
    stream_priority : int
    nMaxCellsPerSlot : numpy.uint16
    nMaxUesPerCellGroup : numpy.uint16
    nMaxCBsPerTB : numpy.uint16
    nMaxPrb : numpy.uint16
    outInfo : List[T_CuPHYTracker]


class T_CuPHYTensor(NamedTuple):
    """Implements cuPHY Tensor.

    Args:
        dimensions (List[int]): Logical dimensions of the tensor.
        strides (List[int]): Physical stride dimensions of the tensor.
        dataType (T_CuPHYDataType): Defines the type of each element.
        pAddr (numpy.uint64): Raw pointer to the tensor buffer.
    """
    dimensions : List[int]
    strides : List[int]
    dataType : T_CuPHYDataType
    pAddr : numpy.uint64


class T_PdschCellDynPrm(NamedTuple):
    """Implements PDSCH dynamic cell parameters type.

    Args:
        nCsiRsPrms (numpy.uint16): Number of CSI-RS params co-scheduled for this cell.
        csiRsPrmsOffset (numpy.uint16): Start index for this cell's nCsiRsPrms elements in the
            pCsiRsPrms array of `T_PdschCellGrpDynPrm`. All elements are allocated continuously.
        cellPrmStatIdx (numpy.uint16): Index to cell-static parameter information, i.e., to the
            `cellStatPrms` array of the `T_PdschStatPrms` struct.
        cellPrmDynIdx (numpy.uint16): Index to cell-dynamic parameter information, i.e., to the
            `cellPrms` array of the `T_PdschCellGrpDynPrm` struct.
        slotNum (numpy.uint16): Slot number. Value: 0 -> 319.
        testModel (numpy.uint8): Testing Mode. Value: 0 -> 1.
    """
    nCsiRsPrms : numpy.uint16
    csiRsPrmsOffset : numpy.uint16
    cellPrmStatIdx : numpy.uint16
    cellPrmDynIdx : numpy.uint16
    slotNum : numpy.uint16
    testModel : numpy.uint8


class T_PdschUeGrpPrm(NamedTuple):
    """Implements PDSCH UE group parameters type.

    Args:
        cellPrmIdx (int): Index of UE group's parent cell dynamic parameters.
        resourceAlloc (numpy.uint8): For specifying resource allocation type [TS38.214, sec 5.1.2.2]
        rbBitmap List[numpy.uint8]: For resource alloc type 0. [TS38.212, sec 7.3.1.2.2].
            Bitmap of RBs, rounded up to multiple of 32. LSB of byte 0 of the bitmap
            represents the RB 0
        rbStart (numpy.uint16): For resource allocation type 1. [TS38.214, sec 5.1.2.2.2].
            The starting resource block within the BWP for this PDSCH. Value: 0 -> 274.
        rbSize (numpy.uint16): For resource allocation type 1. [TS38.214, sec 5.1.2.2.2].
            The number of resource block within for this PDSCH. Value: 1 -> 275.
        dlDmrsSymbPos (numpy.uint16): DMRS symbol positions [TS38.211, sec 7.4.1.1.2 and
            Tables 7.4.1.1.2-3 and 7.4.1.1.2-4]. Bitmap occupying the 14 LSBs with:
            Bit 0: first symbol; and for each bit 0: no DMRS 1: DMRS.
        StartSymbolIndex (numpy.uint8): Start symbol index of PDSCH mapping from the start of the
            slot, S. [TS38.214, Table 5.1.2.1-1]. Value: 0 -> 13.
        NrOfSymbols (numpy.uint8): PDSCH duration in symbols, L. [TS38.214, Table 5.1.2.1-1].
            Value: 1 -> 14.
        uePrmIdxs List[numpy.uint16]: List, of length `nUes`, of indices into the uePrms list of
            `T_PdschCellGrpDynPrm`.
        numDmrsCdmGrpsNoData (numpy.uint8): Number of DM-RS CDM groups without data [TS38.212
            sec 7.3.1.2, TS38.214 Table 4.1-1]. It determines the ratio of PDSCH EPRE to DM-RS EPRE
            Value: 1 -> 3.
        dlDmrsScramblingId (numpy.uint16): UL-DMRS-Scrambling-ID [TS38.211, sec 7.4.1.1.2].
            Value: 0 -> 65535.
    """
    cellPrmIdx : int
    resourceAlloc : numpy.uint8
    rbBitmap : List[numpy.uint8]
    rbStart : numpy.uint16
    rbSize : numpy.uint16
    dlDmrsSymbPos : numpy.uint16
    StartSymbolIndex : numpy.uint8
    NrOfSymbols : numpy.uint8
    uePrmIdxs : List[numpy.uint16]
    numDmrsCdmGrpsNoData : numpy.uint8
    dlDmrsScramblingId : numpy.uint16


class T_PdschUePrm(NamedTuple):
    """Implements PDSCH UE parameters.

    Args:
        ueGrpPrmIdx (int): Index to parent UE group.
        SCID (numpy.uint8): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
            Should match what is sent in DCI 1_1, otherwise set to 0.
            Value : 0 -> 1.
        nrOfLayers (numpy.uint8): Number of layers [TS38.211, sec 7.3.1.3].
            Value: 1 -> 8.
        nDmrsBmsk (numpy.uint16): Set bits in bitmask specify DMRS ports used. Port 0
            corresponds to least significant bit. Used to compute layers.
        BWPStart (numpy.uint16): Bandwidth part start RB index from
            reference CRB [TS38.213 sec 12]. Used only if ref. point is 1.
        refPoint (numpy.uint8): DMRS reference point. Value 0 -> 1.
        beta_dmrs (numpy.float32): Fronthaul DMRS amplitude scaling.
        beta_qam (numpy.float32): Fronthaul QAM amplitude scaling.
        RNTI (numpy.uint16): The RNTI used for identifying the UE when receiving the PDU.
            RNTI == Radio Network Temporary Identifier. Value: 1 -> 65535.
        dataScramblingId (numpy.uint16): dataScramblingIdentityPdsch [TS38.211, sec 7.3.1.1].
            Value: 0 -> 65535.
        cwIdxs (List[int]): List of `nCw` elements containing indices into the cwPrms list of
            `T_PdschCellGrpDynPrm`.
        enablePrcdBf (bool): Enable pre-coding for this UE.
        pmwPrmIdx (int): Index to pre-coding matrix array, i.e., to the pmwPrms list of the
            `T_PdschCellGrpDynPrm`.
    """
    ueGrpPrmIdx : int
    SCID : numpy.uint8
    nrOfLayers : numpy.uint8
    dmrsPortBmsk : numpy.uint16
    BWPStart : numpy.uint16
    refPoint : numpy.uint8
    beta_dmrs : numpy.float32
    beta_qam : numpy.float32
    RNTI : numpy.uint16
    dataScramblingId : numpy.uint16
    cwIdxs : List[int]
    enablePrcdBf : bool
    pmwPrmIdx : int


class T_PdschCwPrm(NamedTuple):
    """Implements PDSCH codeword parameters type.

    Args:
        uePrmIdx (int): Index to parent UE.
        targetCodeRate (numpy.uint16): Target code rate
           Assuming code rate is codeRate = x/1024.0,
           where x contains a single digit after decimal point,
           targetCodeRate = static_cast<uint16_t>(x * 10) = static_cast<uint16_t>(codeRate * 1024 * 10)
        qamModOrder (numpy.uint8): Modulation order.
            Value: 2, 4, 6 or 8.
        rvIndex (numpy.uint8): Redundancy version index [TS38.212, Table 5.4.2.1- 2 and 38.214,
            Table 5.1.2.1-2], should match value sent in DCI.
            Value: 0 -> 3.
        tbStartOffset (numpy.uint32): Starting index (in bytes) of transport block within
            `tbInput` array in `T_PdschDataIn`.
        TBSize (numpy.uint32): Transmit block size (in bytes) [TS38.214 sec 5.1.3.2].
        n_PRB_LBRM (numpy.uint16): Number of PRBs used for LBRM TB size computation.
            Possible values: {32, 66, 107, 135, 162, 217, 273}.
        maxLayers (numpy.uint8): Number of layers used for LBRM TB size computation (at most 4).
        maxQm (numpy.uint8): Modulation order used for LBRM TB size computation. Value: 6 or 8.
    """
    uePrmIdx : int
    targetCodeRate : numpy.uint16
    qamModOrder : numpy.uint8
    rvIndex : numpy.uint8
    tbStartOffset : numpy.uint32
    TBSize : numpy.uint32
    n_PRB_LBRM : numpy.uint16
    maxLayers : numpy.uint8
    maxQm : numpy.uint8


class T_CsirsRrcDynPrm(NamedTuple):
    """TBD.  Will implement PDSCH CSIRS dynamic parameters

    Args:
        tbd : placeholder parameters.
    """
    tbd : int


class T_PmW(NamedTuple):
    """Implement PDSCH precoding matrix

    Pre-coding matrix used only if `T_PdschUePrm_t.enablePrcdBf` is True.
    Layout of the data is such that `T_PdschUePrm_t.nUeLayers` is lower dimension.
    The `nPorts` is the number of columns.
    Memory layout in expected to be in following manner with row-major layout.

    Args:
        w (numpy.ndarray): Pre-coding matrix. Each element of the matrix is of type __half2.
        nPorts (numpy.uint8): Number of ports for this UE.
    """
    w : numpy.ndarray
    nPorts : numpy.uint8


class T_PdschCellGrpDynPrm(NamedTuple):
    """Implements PDSCH dynamic cell group parameters type.

    Args:
        cellPrms (List[T_PdschCellDynPrm]): List of per-cell dynamic parameters with `nCells`
            elements.
        ueGrpPrms (List[T_PdschUeGrpPrm]): List of per-UE-group parameters with `nUeGrps` elements.
        uePrms List[T_PdschUePrm]: List of per-UE parameters with `nUes` elements.
        cwPrms List[T_PdschCwPrm]: List of per-CW parameters with `nCws` elements.
        csiRsPrms List[T_CsirsRrcDynPrm]: List of per-cell CSI-RS parameters with `nCsiRsPrms`
            elements.
        pmwPrms List[T_PmW]: List of pre-coding matrices.
    """
    cellPrms : List[T_PdschCellDynPrm]
    ueGrpPrms : List[T_PdschUeGrpPrm]
    uePrms : List[T_PdschUePrm]
    cwPrms : List[T_PdschCwPrm]
    csiRsPrms : List[T_CsirsRrcDynPrm]
    pmwPrms : List[T_PmW]


class T_PdschDataIn(NamedTuple):
    """Implements PDSCH Data In type.

    Args:
        tbInput List[numpy.ndarray]: A list of transport block input buffers, one buffer per cell,
            indexed by `cellPrmDynIdx`. Each `tbInput` element points to a flat array with all TBs
            for that cell. Currently per-cell TB allocations are contiguous, zero-padded to byte
            boundary. Each element of the flat TB arrays is a numpy.uint8 byte.
    """
    tbInput : List[numpy.ndarray]


class T_PdschDataOut(NamedTuple):
    """Implements PDSCH Data Out type.

    Args:
        dataTx List[T_CuPHYTensor]: Array of tensors with each tensor (indexed by `cellPrmDynIdx`)
            representing the transmit slot buffer of a cell in the cell group.
            Each cell's tensor may have a different geometry.
    """
    dataTx : List[T_CuPHYTensor]


class T_PdschTxDynPrms(NamedTuple):
    """Implements PDSCH dynamic parameters type.

    Args:
        cuStream (int): CUDA stream on which pipeline is launched.
        procModeBmsk (numpy.uint64): Processing modes (e.g., full-slot processing w/ profile 0
            PDSCH_PROC_MODE_FULL_SLOT|PDSCH_PROC_MODE_PROFILE0).
        cellGrpDynPrm (T_PdschCellGrpDynPrm): cell group configuration parameters. Each pipeline
            will process a single cell-group.
        dataIn (T_PdschDataIn): PDSCH data input.
        tbCRCDataIn (T_PdschDataIn): Optional TB CRCs.
        dataOut (T_PdschDataOut): PDSCH data output that will contain `cellGrpDynPrm.nCells`
            tensors.
    """
    cuStream : int  # TODO: Is there a Python cudaStream_t?
    procModeBmsk : numpy.uint64
    cellGrpDynPrm : T_PdschCellGrpDynPrm
    dataIn : T_PdschDataIn
    tbCRCDataIn : T_PdschDataIn
    dataOut : T_PdschDataOut


class T_PuschRxStatDbgPrms(NamedTuple):
    """Implements cuPHY PUSCH Rx Static Debug Parameters.

    Args:
        outFileName (str): Output file capturing pipeline intermediate states. No capture if None.
        descrmOn (numpy.uint8) : Descrambling enable/disable.
        enableApiLogging numpy.uint8: Control the API logging of PUSCH static parameters.
    """
    outFileName : str
    descrmOn : numpy.uint8
    enableApiLogging : numpy.uint8


class T_PuschRxStatPrms(NamedTuple):
    """Implements cuPHY PUSCH Rx Cell Static Parameters.

    Args:
        WFreq (T_CuPHYTensor): Channel estimation filter for wide bandwidth.
        WFreq4 (T_CuPHYTensor): Channel estimation filter for medium bandwidth.
        WFreqSmall (T_CuPHYTensor): Channel estimation filter for small bandwidth.
        ShiftSeq (T_CuPHYTensor): Channel estimation shift sequence for nominal bandwidth.
        UnShiftSeq (T_CuPHYTensor): Channel estimation unshift sequence for nominal bandwidth.
        ShiftSeq4 (T_CuPHYTensor): Channel estimation shift sequence for medium bandwidth.
        UnShiftSeq4 (T_CuPHYTensor): Channel estimation unshift sequence for medium bandwidth.
        enableCfoCorrection (numpy.uint8): Carrier frequency offset estimation/correction flag.
            0 - Disable.
            1 - Enable.
        enablePuschTdi (numpy.uint8): Time domain interpolation flag.
            0 - Disable.
            1 - Enable.
        enableDftSOfdm (numpy.uint8): Global DFT-s-OFDM enabling flag.
            0 - Disable.
            1 - Enable.
        enableTbSizeCheck (numpy.uint8): Global PUSCH tbSizeCheck enabling flag.
            0 - Disable.
            1 - Enable.
        stream_priority (int): CUDA stream priority for internal PUSCH streams-pool.
            Should match the priority of CUDA stream passed in `cuphyCreatePuschRx()`.
        ldpcnIterations (numpy.uint8): Number of LDPC decoder iterations.
        ldpcEarlyTermination (numpy.uint8): LDPC decoder early termination flag.
            0 - Run `ldpcnInterations` always.
            1 - Terminate early on passing CRC.
        ldpcUseHalf (numpy.uint8): LDPC use FP16 flag.
            0 - Use FP32 LLRs.
            1 - Use FP16 LLRs.
        ldpcAlgoIndex (numpy.uint8): LDPC Decoder algorithm index. See cuPHY documentation for
            a list of algorithms.
        ldpcFlags (numpy.uint8): LDPC decoder configuration flags. See cuPHY documentation for
            flags.
        ldpcKernelLaunch (numpy.uint32): LDPC launch configuration flag. See cuPHY documentation
            for kernel launch.
        enableEqIrc (numpy.uint8): Flag for MMSE-IRC equalizer.
            0 - Disabled.
            1 - Enabled.
        eqCoeffAlgo (T_PuschEqCoefAlgoType): PUSCH equalizer algorithm.
            0, rZf
            1, noiseDiagMmse
            2, mmseIrc
        enableRssiMeasurement (numpy.uint8): Flag for RSSI measurement.
            0 - Disabled.
            1 - Enabled.
        enableSinrMeasurement (numpy.uint8): Flag for SINR measurement.
            0 - Disabled.
            1 - Enabled.
        nCells (numpy.uint16): Number of active cells.
        nMaxCells (numpy.uint16): Total # of cell configurations supported by the pipeline during
            its lifetime. Maximum # of cells scheduled in a slot. Out of `nMaxCells`, the
            `nMaxCellsPerSlot` most resource hungry cells are used for resource provisioning
            purposes.
        nMaxCellsPerSlot (numpy.uint16): Must be <= `nMaxCells`.
        cellStatPrms (List[T_CellStatPrm]): Static cell parameters common to all channels.
        nMaxTbs (numpy.uint32): Maximum number of transport blocks that will be supported by
            PuschRx object.
        nMaxCbsPerTb (numpy.uint32) : Maximum number of code blocks per transport block that
            will be supported by PuschRx object.
        nMaxTotCbs (numpy.uint32): Total number of code blocks (sum of # code blocks across all
            transport blocks) that will be supported by PuschRx object.
        nMaxRx (numpy.uint32): Maximum number of Rx antennas that will be supported by PuschRx
            object.
        nMaxPrb (numpy.uint32): Maximum number of PRBs that will be supported by PuschRx object.
        dbg (T_PuschRxStatDbgPrms): Debug parameters.
        outInfo (List[T_cuPHYTracker]): pointer to cuPHY tracker
    """
    WFreq : T_CuPHYTensor
    WFreq4 : T_CuPHYTensor
    WFreqSmall : T_CuPHYTensor
    ShiftSeq : T_CuPHYTensor
    UnShiftSeq : T_CuPHYTensor
    ShiftSeq4 : T_CuPHYTensor
    UnShiftSeq4 : T_CuPHYTensor
    enableCfoCorrection : numpy.uint8
    enablePuschTdi : numpy.uint8
    enableDftSOfdm : numpy.uint8
    enableTbSizeCheck: numpy.uint8
    stream_priority : int
    ldpcnIterations : numpy.uint8
    ldpcEarlyTermination : numpy.uint8
    ldpcUseHalf : numpy.uint8
    ldpcAlgoIndex : numpy.uint8
    ldpcFlags : numpy.uint8
    ldpcKernelLaunch : numpy.uint32
    enableEqIrc : numpy.uint8
    eqCoeffAlgo : T_PuschEqCoefAlgoType
    enableRssiMeasurement : numpy.uint8
    enableSinrMeasurement : numpy.uint8
    nCells : numpy.uint16
    nMaxCells : numpy.uint16
    nMaxCellsPerSlot : numpy.uint16
    cellStatPrms : List[T_CellStatPrm]
    nMaxTbs : numpy.uint32
    nMaxCbsPerTb : numpy.uint32
    nMaxTotCbs : numpy.uint32
    nMaxRx : numpy.uint32
    nMaxPrb : numpy.uint32
    dbg : T_PuschRxStatDbgPrms
    outInfo : List[T_CuPHYTracker]


class T_PuschRxCellDynPrm(NamedTuple):
    """Implements cuPHY PUSCH Rx Cell Dynamic Parameters

    Args:
        cellPrmStatIdx: Index to cell-static parameter information
        cellPrmDynIdx: Index to cell-dynamic parameter information
        slotNum: Slot number
    """
    cellPrmStatIdx : numpy.uint16
    cellPrmDynIdx : numpy.uint16
    slotNum : numpy.uint16


class T_PuschRxDmrsPrm(NamedTuple):
    """Implements cuPHY PUSCH Rx DMRS Parameters.

    Args:
        dmrsAddlnPos (numpy.uint8):
        dmrsMaxLen (numpy.uint8):
        numDmrsCdmGrpsNoData (numpy.uint8): Number of DM-RS CDM groups without data
            [TS38.212 sec 7.3.1.1, TS38.214 Table 4.1-1]. Value: 1 -> 3.
        ulDmrsScramblingId (numpy.uint8): UL-DMRS-Scrambling-ID [TS38.211, sec 6.4.1.1.1].
            Value: 0 -> 65535.
    """
    dmrsAddlnPos : numpy.uint8
    dmrsMaxLen : numpy.uint8
    numDmrsCdmGrpsNoData : numpy.uint8
    ulDmrsScramblingId : numpy.uint8


class T_PuschRxUeGrpPrm(NamedTuple):
    """Implements cuPHY PUSCH Rx co-scheduled UE Group Dynamic Parameters.

    Args:
        cellPrmIdx (int): UE group's parent cell dynamic parameters index.
        dmrsDynPrm (T_PuschRxDmrsPrm): DMRS information.
        rbStart (numpy.uint16): For resource allocation type 0. [TS38.214, sec 6.1.2.2.2].
                The starting resource block within the BWP for this PUSCH. Value: 0 -> 274.
        rbSize (numpy.uint16): For resource allocation type 1. [TS38.214, sec 6.1.2.2.2].
            The number of resource block within for this PUSCH. Value: 1 -> 275.
        StartSymbolIndex (numpy.uint8): Start symbol index of PUSCH mapping from the start of the
            slot, S. [TS38.214, Table 6.1.2.1-1]. Value: 0 -> 13.
        NrOfSymbols (numpy.uint8): PUSCH duration in symbols, L. [TS38.214, Table 6.1.2.1-1].
            Value: 1 -> 14.
        dmrsSymLocBmsk (numpy.uint8): DMRS location bitmask (LSB 14 bits).
            PUSCH symbol locations derived from dmrsSymLocBmsk.
            Bit i is "1" if symbol i is DMRS.
            For example if there are DMRS are symbols 2 and 3, then:
            dmrsSymLocBmsk = 0000 0000 0000 1100.
        rssiSymLocBmsk (numpy.uint8): Symbol location bitmask for RSSI measurement (LSB 14 bits).
            Bit i is "1" if symbol i needs be to measured, 0 disables RSSI calculation.
            For example to measure RSSI on DMRS symbols 2, 6 and 9, use:
            rssiSymLocBmsk = 0000 0010 0100 0100.
        uePrmIdxs List[numpy.uint16]: List of UE indices.
    """
    cellPrmIdx : int
    dmrsDynPrm : T_PuschRxDmrsPrm
    rbStart : numpy.uint16
    rbSize : numpy.uint16
    StartSymbolIndex : numpy.uint8
    NrOfSymbols : numpy.uint8
    dmrsSymLocBmsk : numpy.uint8
    rssiSymLocBmsk : numpy.uint8
    uePrmIdxs : List[numpy.uint16]


class T_PuschRxUePrm(NamedTuple):
    """Implements cuPHY PUSCH Rx UE Dynamic Parameters.

    Args:
        pduBitmap (numpy.uint16):
            Bit 0 indicates if data present.
            Bit 1 indicates if uci present.
            Bit 2 indicates if ptrs present.
            Bit 3 indicates DFT-S transmission.
            Bit 4 indicates if sch data present.
            Bit 5 indicates if CSI-P2 present.
        ueGrpPrmIdx (numpy.uint16): Index to parent UE Group.
        SCID (numpy.uint8): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
            Should match what is sent in DCI 1_1, otherwise set to 0. Value : 0 -> 1.
        dmrsPortBmsk (numpy.uint8): TODO
        mcsTable (numpy.uint8): MCS (Modulation and Coding Scheme) Table Id.
            [TS38.214, sec 6.1.4.1].
            0: notqam256. Table 5.1.3.1-1.
            1: qam256. Table 5.1.3.1-2.
            2: qam64LowSE. Table 5.1.3.1-3.
            3: notqam256-withTransformPrecoding [TS38.214, table 6.1.4.1-1].
            4: qam64LowSE-withTransformPrecoding [TS38.214, table 6.1.4.1-2].
        mcsIndex (numpy.uint8): MCS index within the mcsTableIndex table. [TS38.214, sec 6.1.4.1].
            Value: 0 -> 31.
        TBSize (numpy.uint32): TBSize in bytes provided by L2 based on FAPI 10.04.
        targetCodeRate (numpy.uint16): Target coding rate [TS38.214 sec 6.1.4.1].
            This is the number of information bits per 1024 coded bits expressed in 0.1 bit units.
        qamModOrder (numpy.uint8): QAM modulation [TS38.214 sec 6.1.4.1].
        rvIndex (numpy.uint8): Redundancy version index [TS38.214, sec 6.1.4],
            should match value sent in DCI. Value: 0 -> 3.
        RNTI (numpy.uint16): The RNTI used for identifying the UE when receiving the PDU.
            RNTI == Radio Network Temporary Identifier. Value: 1 -> 65535.
        dataScramblingId (numpy.uint16): dataScramblingIdentityPusch [TS38.211, sec 6.3.1.1].
            Value: 0 -> 65535.
        nrOfLayers (numpy.uint8): Number of layers [TS38.211, sec 6.3.1.3].
            Value: 1 -> 4.
        newDataIndicator (numpy.uint8): Indicates if this new data or a retransmission
            [TS38.212, sec 7.3.1.1].
            Value:
            0: Retransmission.
            1: New data.
        harqProcessId (numpy.uint8): HARQ process number [TS38.212, sec 7.3.1.1].
        i_lbrm (numpy.uint8) : 0 = Do not use LBRM. 1 = Use LBRM per 38.212 5.4.2.1 and 6.2.5.
        n_PRB_LBRM (numpy.uint16): number of PRBs used for LBRM TB size computation.
            Possible values: {32, 66, 107, 135, 162, 217, 273}.
        maxLayers (numpy.uint8): Number of layers used for LBRM TB size computation (at most 4).
        maxQm (numpy.uint8): Modulation order used for LBRM TB size computation.
            Value: 6 or 8.
        enableTfPrcd (numpy.uint8): DFT-s-OFDM enabling per UE
        uciPrms: TODO
    """
    pduBitmap : numpy.uint16
    ueGrpPrmIdx : numpy.uint16
    SCID : numpy.uint8
    dmrsPortBmsk : numpy.uint8
    mcsTable : numpy.uint8
    mcsIndex : numpy.uint8
    TBSize: numpy.uint32
    targetCodeRate : numpy.uint16
    qamModOrder : numpy.uint8
    rvIndex : numpy.uint8
    RNTI : numpy.uint16
    dataScramblingId : numpy.uint16
    nrOfLayers : numpy.uint8
    newDataIndicator : numpy.uint8
    harqProcessId : numpy.uint8
    i_lbrm : numpy.uint8
    maxLayers : numpy.uint8
    maxQm : numpy.uint8
    n_PRB_LBRM : numpy.uint16
    enableTfPrcd: numpy.uint8
    uciPrms : None


class T_PuschRxCellGrpDynPrm(NamedTuple):
    """Implements cuPHY PUSCH Rx Cell Group Dynamic Parameters.

    Args:
        cellPrms (List[T_PuschRxCellDynPrm]): List of cell dynamic parameters, one entry per cell.
        ueGrpPrms (List[T_PuschRxUeGrpPrm]): List of UE group dynamic parameters, one entry per UE
            group.
        uePrms (List[T_PuschRxUePrm]): List of UE dynamic parameters, one entry per UE.
    """
    cellPrms : List[T_PuschRxCellDynPrm]
    ueGrpPrms : List[T_PuschRxUeGrpPrm]
    uePrms : List[T_PuschRxUePrm]


class T_PuschRxDataIn(NamedTuple):
    """Implements PUSCH Data In type.

    Args:
        tDataRx (List[T_CuPHYTensor]): List of tensors with each tensor (indexed by
            `cellPrmDynIdx`) representing the receive slot buffer of a cell in the cell group.
            Each cell's tensor may have a different geometry.
    """
    tDataRx : List[T_CuPHYTensor]


class T_UciOnPuschOutOffsets(NamedTuple):
    """Implements UCI on PUSCH output locations.

    Args:
        harqPayloadByteOffset (numpy.uint32): TODO
        harqCrcFlagOffset (numpy.uint16): TODO
        csi1PayloadByteOffset (numpy.uint32): TODO
        csi1CrcFlagOffset (numpy.uint16): TODO
        numCsi2BitsOffset (numpy.uint16): TODO
        csi2PayloadByteOffset (numpy.uint32): TODO
        csi2CrcFlagOffset (numpy.uint16): TODO
    """
    harqPayloadByteOffset : numpy.uint32
    harqCrcFlagOffset : numpy.uint16
    csi1PayloadByteOffset : numpy.uint32
    csi1CrcFlagOffset : numpy.uint16
    numCsi2BitsOffset : numpy.uint16
    csi2PayloadByteOffset : numpy.uint32
    csi2CrcFlagOffset : numpy.uint16


class T_PuschRxDataOut(NamedTuple):
    """Implements PUSCH Data Out type.

    Args:
        totNumTbs (numpy.ndarray):
        totNumCbs (numpy.ndarray):
        totNumPayloadBytes (numpy.ndarray):
        totNumUciSegs (numpy.ndarray):
        harqBufferSizeInBytes (List[numpy.uint32]): HARQ buffer sizes, returned during setup
            phase 1.
        cbCrcs (numpy.ndarray): Array of CB CRCs.
        tbCrcs (numpy.ndarray): Array of TB CRCs.
        tbPayloads (numpy.ndarray): Array of TB payloads.
        uciPayloads (numpy.ndarray): TODO
        uciCrcFlags (numpy.ndarray): TODO
        pUciDTXs (numpy.ndarray): TODO
        numCsi2Bits (numpy.ndarray): TODO
        startOffsetsCbCrc (numpy.ndarray): nUes offsets providing start offset of UE CB-CRCs within
            `cbCrcs`. The UE ordering is identical to input UE ordering.
        startOffsetsTbCrc (numpy.ndarray): nUes offsets providing start offset of UE TB-CRCs within
            `tbCrcs`. The UE ordering is identical to input UE ordering.
        startOffsetsTbPayload (numpy.ndarray): nUes offsets providing start offset of UE TB-payload
            within `tbPayloads`. The UE ordering is identical to input UE ordering.
        taEsts (numpy.ndarray): Array of nUes estimates in microseconds. UE ordering identical
            to input UE ordering.
        rssi (numpy.ndarray): Array of nUeGrps estimates in dB. Per UE group total power
            (signal + noise + interference) averaged over allocated PRBs, DMRS additional positions
            and summed over Rx antenna.
        rsrp (numpy.ndarray): Array of nUes RSRP estimates in dB. Per UE signal power averaged over
            allocated PRBs, DMRS additional positions, Rx antenna and summed over layers.
        noiseVarPreEq (numpy.ndarray): Array of nUeGrps pre-equalizer noise variance estimates
            in dB.
        noiseVarPostEq (numpy.ndarray): Array of nUes post equalizer noise variance estimates
            in dB.
        sinrPreEq (numpy.ndarray): Array of nUes pre-equalizer SINR estimates in dB.
        sinrPostEq (numpy.ndarray): Array of nUes post-equalizer estimates SINR in dB.
        cfoHz (numpy.ndarray): Array of nUEs carrier frequency offsets in Hz.
        HarqDetectionStatus (numpy.ndarray): Value:
            1 = CRC Pass
            2 = CRC Failure
            3 = DTX
            4 = No DTX (indicates UCI detection).
            Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in
            cuPHY since DTX detection is present.
        CsiP1DetectionStatus (numpy.ndarray): Value:
            1 = CRC Pass
            2 = CRC Failure
            3 = DTX
            4 = No DTX (indicates UCI detection).
            Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in
            cuPHY since DTX detection is present.
        CsiP2DetectionStatus (numpy.ndarray): Value:
            1 = CRC Pass
            2 = CRC Failure
            3 = DTX
            4 = No DTX (indicates UCI detection).
            Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in
            cuPHY since DTX detection is present.
    """
    totNumTbs: numpy.ndarray # length 1 numpy.uint32
    totNumCbs: numpy.ndarray # length 1 numpy.uint32
    totNumPayloadBytes: numpy.ndarray # length 1 numpy.uint32
    totNumUciSegs: numpy.ndarray # length 1 numpy.uint16
    harqBufferSizeInBytes : List[numpy.uint32]
    cbCrcs : numpy.ndarray
    tbCrcs : numpy.ndarray
    tbPayloads : numpy.ndarray
    uciPayloads : numpy.ndarray
    uciCrcFlags : numpy.ndarray
    pUciDTXs : numpy.ndarray
    numCsi2Bits : numpy.ndarray
    startOffsetsCbCrc : numpy.ndarray
    startOffsetsTbCrc : numpy.ndarray
    startOffsetsTbPayload : numpy.ndarray
    uciOnPuschOutOffsets : None  # List[T_UciOnPuschOutOffsets]
    taEsts : numpy.ndarray
    rssi : numpy.ndarray
    rsrp : numpy.ndarray
    noiseVarPreEq : numpy.ndarray
    noiseVarPostEq : numpy.ndarray
    sinrPreEq : numpy.ndarray
    sinrPostEq : numpy.ndarray
    cfoHz : numpy.ndarray
    HarqDetectionStatus : numpy.ndarray
    CsiP1DetectionStatus : numpy.ndarray
    CsiP2DetectionStatus : numpy.ndarray


class T_PuschRxDataInOut(NamedTuple):
    """Implements cuPHY PUSCH Data In/Out.

    Args:
        harqBuffersInOut (List[numpy.ndarray]): Array of In/Out HARQ buffers.
            The In/Out HARQ buffers will be read or written depending on ndi and TB CRC pass result.
            The In/Out HARQ buffers themselves are located in GPU memory.
            The “array of pointers” must be read-able from a GPU kernel (handled at the c++ binding
            side). An allocation from cudaHostAlloc with
            cudaHostAllocPortable | cudaHostAllocMapped is sufficient.
    """
    harqBuffersInOut : List[numpy.ndarray]


class T_PuschRxDynDbgPrms(NamedTuple):
    """Implements PUSCH channel dynamic debug parameters.

    Args:
        enableApiLogging (numpy.uint8): control the API logging of PUSCH dynamic parameters
    """
    enableApiLogging : numpy.uint8


class T_PuschRxDynPrms(NamedTuple):
    """Implements cuPHY PUSCH Rx Pipeline Dynamic Parameters.

    Args:
        cuStream (int): CUDA stream on which pipeline is launched.
        setupPhase (T_PuschSetupPhase): Setup phase.
        procModeBmsk (numpy.uint64): Processing modes bitmask.
        cellGrpDynPrm (T_PuschRxCellGrpDynPrm): Cell group dynamic parameters.
        dataIn (T_PuschRxDataIn): Input data parameters.
        dataOut (T_PuschRxDataOut): Output data parameters.
        dataInOut (T_PuschRxDataInOut): Input/Output data parameters.
            TODO: May need rework for python input.
        cpuCopyOn (numpy.uint8):
        dbg (T_PuschRxDynDbgPrms):
    """
    cuStream : int  # TODO: Is there a python cudaStream_t?
    setupPhase : T_PuschSetupPhase
    procModeBmsk : numpy.uint64
    cellGrpDynPrm : T_PuschRxCellGrpDynPrm
    dataIn : T_PuschRxDataIn
    dataOut : T_PuschRxDataOut
    dataInOut : T_PuschRxDataInOut
    cpuCopyOn : numpy.uint8
    dbg : T_PuschRxDynDbgPrms
