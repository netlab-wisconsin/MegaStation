/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>
#include <queue>
#include <condition_variable>
#include <bitset>
#include <map>

#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "pdsch_tx.hpp"
#include "pycuphy_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace cuphy;
using namespace std::complex_literals;
namespace py = pybind11;

#ifndef AERIAL_PYTHON_PDSCH_CPP
#define AERIAL_PYTHON_PDSCH_CPP

namespace pycuphy {

class PdschPipeline {

    public:

    PdschPipeline():
    dataTxTensor(std::vector<tensor_device>(PDSCH_MAX_CELLS_PER_CELL_GROUP)),
    tbInputPtr(std::vector<uint8_t*> (PDSCH_MAX_CELLS_PER_CELL_GROUP))
    {}

    uint64_t createPdschTx(const py::object& statPrms);
    void setupPdschTx(uint64_t pdschHandle, const py::object& dynPrms);
    void runPdschTx(uint64_t pdschHandle);

    py::array_t<float> getLdpcOutputPerTbPerCell(uint64_t pdschHandle, int cellIdx, int tbIdx, const uint64_t ldpcOutputHostPtr);
    void destroyPdschTx(uint64_t pdschHandle);

    private:

    cuphyPdschStatPrms_t                    pdschStatPrms;
    cuphyTracker_t                          pdschTracker;
    cuphyPdschDynPrms_t                     pdschDynParams;
    cuphyPdschCellGrpDynPrm_t               pdschCellGrpDynPrms;
    cuphyPdschDataIn_t                      pdschDataIn;
    cuphyPdschDataIn_t                      tbCrcDataIn;
    std::vector<tensor_device>              dataTxTensor;
    std::vector<uint8_t*>                   tbInputPtr;
};


// TODO: Remove this or move it to Aerial Sim codebase. Or rename it and make it generic.
inline void print_pdsch_static_asim(const cuphyPdschStatPrms_t* static_params)
{
    if (static_params == nullptr) return;
    int n_cells = static_params->nCells;
    printf("PDSCH Static parameters for %d cells\n", n_cells);

    // Parameters common across all cells
    printf("read_TB_CRC:           %4d\n", static_params->read_TB_CRC);
    printf("full_slot_processing:  %4d\n", static_params->full_slot_processing);
    printf("stream_priority:       %4d\n", static_params->stream_priority);
    printf("nMaxCellsPerSlot:      %4d\n", static_params->nMaxCellsPerSlot);
    printf("nMaxUesPerCellGroup:   %4d\n", static_params->nMaxUesPerCellGroup);
    printf("nMaxCBsPerTB:          %4d\n", static_params->nMaxCBsPerTB);
    printf("nMaxPrb:               %4u\n", static_params->nMaxPrb);

    // Cell specific parameters
    cuphyCellStatPrm_t* cell_static_params = static_params->pCellStatPrms;
    cuphyPdschDbgPrms_t* cell_dbg_params   = static_params->pDbg;
    for (int cell_id = 0; cell_id < n_cells; cell_id++) {
        printf("------------------------------------\n");
        printf("Cell %d\n", cell_id);
        printf("------------------------------------\n");
        printf("phyCellId:      %4d\n", cell_static_params[cell_id].phyCellId);
        printf("nRxAnt:         %4d\n", cell_static_params[cell_id].nRxAnt);
        printf("nTxAnt:         %4d\n", cell_static_params[cell_id].nTxAnt);
        printf("nPrbUlBwp:      %4d\n", cell_static_params[cell_id].nPrbUlBwp);
        printf("nPrbDlBwp:      %4d\n", cell_static_params[cell_id].nPrbDlBwp);
        printf("mu:             %4d\n", cell_static_params[cell_id].mu);
        // Debug fields
        printf("\nDBG:\n");
        printf("pCfgFileName:             %s\n", cell_dbg_params[cell_id].pCfgFileName);
        printf("refCheck:                 %d\n", cell_dbg_params[cell_id].refCheck);
        printf("cfgIdenticalLdpcEncCfgs:  %d\n", cell_dbg_params[cell_id].cfgIdenticalLdpcEncCfgs);
        printf("\n");
    }
}


uint64_t PdschPipeline::createPdschTx(const py::object& statPrms){

    // Set cuphyPdschStatPrms_t first.
    pdschStatPrms.read_TB_CRC          = statPrms.attr("read_TB_CRC").cast<bool>();
    pdschStatPrms.full_slot_processing = statPrms.attr("full_slot_processing").cast<bool>();
    pdschStatPrms.stream_priority      = statPrms.attr("stream_priority").cast<int>();
    pdschStatPrms.nMaxCellsPerSlot     = statPrms.attr("nMaxCellsPerSlot").cast<uint16_t>();
    pdschStatPrms.nMaxUesPerCellGroup  = statPrms.attr("nMaxUesPerCellGroup").cast<uint16_t>();
    pdschStatPrms.nMaxCBsPerTB         = statPrms.attr("nMaxCBsPerTB").cast<uint16_t>();
    pdschStatPrms.nMaxPrb              = statPrms.attr("nMaxPrb").cast<uint16_t>();

    pdschTracker.pMemoryFootprint      = nullptr;
    pdschStatPrms.pOutInfo             = &pdschTracker;

    const py::list cellStatPrm = statPrms.attr("cellStatPrms");
    uint16_t nCells = cellStatPrm.size();

    pdschStatPrms.nCells = nCells;
    pdschStatPrms.pCellStatPrms = new cuphyCellStatPrm_t[nCells];
    pdschStatPrms.pDbg = new cuphyPdschDbgPrms_t[nCells];

    for (int cell_id = 0; cell_id < nCells; cell_id ++ ) {

        const py::object cell_static_params = cellStatPrm[cell_id];

        pdschStatPrms.pCellStatPrms[cell_id].phyCellId    = cell_static_params.attr("phyCellId").cast<uint16_t>();
        pdschStatPrms.pCellStatPrms[cell_id].nRxAnt       = cell_static_params.attr("nRxAnt").cast<uint16_t>();
        pdschStatPrms.pCellStatPrms[cell_id].nTxAnt       = cell_static_params.attr("nTxAnt").cast<uint16_t>();
        pdschStatPrms.pCellStatPrms[cell_id].nPrbUlBwp    = cell_static_params.attr("nPrbUlBwp").cast<uint16_t>();
        pdschStatPrms.pCellStatPrms[cell_id].nPrbDlBwp    = cell_static_params.attr("nPrbDlBwp").cast<uint16_t>();
        pdschStatPrms.pCellStatPrms[cell_id].mu           = cell_static_params.attr("mu").cast<uint8_t>();
        if (pdschStatPrms.pCellStatPrms[cell_id].mu > 1) {
            throw std::runtime_error("Unsupported numerology value!");
        }
    }

    py::list dbg;
    int dbgSize = 0;
    if (!(std::string(py::str(statPrms.attr("dbg"))) == "None")) {
        dbg = statPrms.attr("dbg");
        dbgSize = dbg.size();
    }
    else {
        for (int cell_id = 0; cell_id < nCells; cell_id ++ ) {
            pdschStatPrms.pDbg[cell_id].pCfgFileName = ""; // if nullptr, it will cause pipeline broken by hdf5_filename = static_params->pDbg->pCfgFileName in pdsch_tx.cpp
            pdschStatPrms.pDbg[cell_id].refCheck = 0;
            pdschStatPrms.pDbg[cell_id].cfgIdenticalLdpcEncCfgs = 1;
        }
    }

    for (int cell_id = 0; cell_id < dbgSize; cell_id ++ ){

        const py::object dbgPrms = dbg[cell_id];
        if ((std::string(py::str(dbgPrms.attr("cfgFilename"))) == "None") || (dbgPrms.attr("cfgFilename").cast<std::string>() == ""))
            pdschStatPrms.pDbg[cell_id].pCfgFileName = nullptr;
        else
            pdschStatPrms.pDbg[cell_id].pCfgFileName = dbgPrms.attr("cfgFilename").cast<std::string>().c_str();

        pdschStatPrms.pDbg[cell_id].refCheck = dbgPrms.attr("refCheck").cast<bool>();
        pdschStatPrms.pDbg[cell_id].cfgIdenticalLdpcEncCfgs = dbgPrms.attr("cfgIdenticalLdpcEncCfgs").cast<bool>();
    }

    // Create pipeline.
    std::unique_ptr<cuphyPdschTxHndl_t> pPdschHandle = std::make_unique<cuphyPdschTxHndl_t>();
    cuphyStatus_t status = cuphyCreatePdschTx(pPdschHandle.get(), &pdschStatPrms);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        std::cerr << "Error! cuphyCreatePdschTx(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }

    // Allocate memory for dynamic parameters. Note: Only one cell group here.
    pdschDynParams.pCellGrpDynPrm = &pdschCellGrpDynPrms;

    pdschCellGrpDynPrms.pCellPrms  = new cuphyPdschCellDynPrm_t[nCells];
    pdschCellGrpDynPrms.pUeGrpPrms = new cuphyPdschUeGrpPrm_t[PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP];
    pdschCellGrpDynPrms.pUePrms    = new cuphyPdschUePrm_t[PDSCH_MAX_UES_PER_CELL_GROUP];
    pdschCellGrpDynPrms.pCwPrms    = new cuphyPdschCwPrm_t[PDSCH_MAX_CWS_PER_CELL_GROUP];
    pdschCellGrpDynPrms.pPmwPrms   = new cuphyPmW_t[PDSCH_MAX_UES_PER_CELL_GROUP];


    for (int i = 0; i < PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP; i++){
        pdschCellGrpDynPrms.pUeGrpPrms[i].rbBitmap    = new uint8_t[MAX_RBMASK_BYTE_SIZE];
        pdschCellGrpDynPrms.pUeGrpPrms[i].pUePrmIdxs  = new uint16_t[PDSCH_MAX_UES_PER_CELL_GROUP];
        pdschCellGrpDynPrms.pUeGrpPrms[i].pDmrsDynPrm = new cuphyPdschDmrsPrm_t[1];
    }

    for (int i = 0; i < PDSCH_MAX_UES_PER_CELL_GROUP; i++){
        pdschCellGrpDynPrms.pUePrms[i].pCwIdxs        = new uint16_t[2];
    }

    pdschDynParams.pDataOut           = new cuphyPdschDataOut_t;
    pdschDynParams.pDataOut->pTDataTx = new cuphyTensorPrm_t[PDSCH_MAX_CELLS_PER_CELL_GROUP];
    pdschDynParams.pStatusInfo        = new cuphyPdschStatusOut_t;

    return (uint64_t)(*pPdschHandle);
}


void PdschPipeline::setupPdschTx(uint64_t pdschHandle, const py::object& dynPrms){

    if(0){ // debug
        PdschTx* pipeline_ptr  = static_cast<PdschTx*>((cuphyPdschTxHndl_t)pdschHandle);
        printf("PdschTx* (begining of setupPdschTx): %ld \n \n", (uint64_t)pipeline_ptr);
        const cuphyPdschStatPrms_t* static_params = pipeline_ptr->static_params;
        print_pdsch_static_asim(static_params);
    }

    uint64_t cuStream                   = dynPrms.attr("cuStream").cast<uint64_t>();
    pdschDynParams.cuStream             = (cudaStream_t)cuStream;
    pdschDynParams.procModeBmsk         = dynPrms.attr("procModeBmsk").cast<uint64_t>();

    const py::object cellGrpDynPrm      = dynPrms.attr("cellGrpDynPrm");
    const py::object dataIn             = dynPrms.attr("dataIn");
    const py::object dataOut            = dynPrms.attr("dataOut");

    const py::list cellPrms             = cellGrpDynPrm.attr("cellPrms");
    const py::list ueGrpPrms            = cellGrpDynPrm.attr("ueGrpPrms");
    const py::list uePrms               = cellGrpDynPrm.attr("uePrms");
    const py::list cwPrms               = cellGrpDynPrm.attr("cwPrms");

    py::list csiRsPrms;
    if (!(std::string(py::str(cellGrpDynPrm.attr("csiRsPrms"))) == "None")){
        csiRsPrms = cellGrpDynPrm.attr("csiRsPrms");
    }

    py::list pmwPrms;
    if (!(std::string(py::str(cellGrpDynPrm.attr("pmwPrms"))) == "None")){
        pmwPrms = cellGrpDynPrm.attr("pmwPrms");
    }

    // cell dyn parameters
    uint16_t nCells = cellPrms.size();
    pdschCellGrpDynPrms.nCells = nCells;
    for (int cell_id = 0; cell_id < nCells; cell_id ++ ){
        const py::object cell_dyn_params = cellPrms[cell_id];

        pdschCellGrpDynPrms.pCellPrms[cell_id].nCsiRsPrms      = cell_dyn_params.attr("nCsiRsPrms").cast<uint16_t>();
        pdschCellGrpDynPrms.pCellPrms[cell_id].csiRsPrmsOffset = cell_dyn_params.attr("csiRsPrmsOffset").cast<uint16_t>();
        pdschCellGrpDynPrms.pCellPrms[cell_id].cellPrmStatIdx  = cell_dyn_params.attr("cellPrmStatIdx").cast<uint16_t>();
        pdschCellGrpDynPrms.pCellPrms[cell_id].cellPrmDynIdx   = cell_dyn_params.attr("cellPrmDynIdx").cast<uint16_t>();
        pdschCellGrpDynPrms.pCellPrms[cell_id].slotNum         = cell_dyn_params.attr("slotNum").cast<uint16_t>();
        pdschCellGrpDynPrms.pCellPrms[cell_id].testModel       = cell_dyn_params.attr("testModel").cast<uint8_t>();

        pdschCellGrpDynPrms.pCellPrms[cell_id].pdschStartSym   = 0; //cell_dyn_params.attr("pdschStartSym").cast<uint8_t>();
        pdschCellGrpDynPrms.pCellPrms[cell_id].nPdschSym       = 0; //cell_dyn_params.attr("nPdschSym").cast<uint8_t>();
        pdschCellGrpDynPrms.pCellPrms[cell_id].dmrsSymLocBmsk  = 0; //cell_dyn_params.attr("dmrsSymLocBmsk").cast<uint8_t>();

    }

    // ueGrpPrms
    uint16_t nUeGrps                          = ueGrpPrms.size();
    pdschCellGrpDynPrms.nUeGrps               = nUeGrps;

    for (int ue_group_id = 0; ue_group_id < nUeGrps; ue_group_id++){
        const py::object ue_grp_params = ueGrpPrms[ue_group_id];

        int cellPrmIdx = ue_grp_params.attr("cellPrmIdx").cast<int>();
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].pCellPrm = &pdschCellGrpDynPrms.pCellPrms[cellPrmIdx];
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].resourceAlloc = ue_grp_params.attr("resourceAlloc").cast<uint8_t>();
        const py::list rbBitmap = ue_grp_params.attr("rbBitmap");
        for (int idx = 0; idx < rbBitmap.size(); idx++) {
            pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].rbBitmap[idx] = rbBitmap[idx].cast<uint8_t>();
        }
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].startPrb = ue_grp_params.attr("rbStart").cast<uint16_t>();
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].nPrb = ue_grp_params.attr("rbSize").cast<uint16_t>();
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].pdschStartSym = ue_grp_params.attr("StartSymbolIndex").cast<uint8_t>();
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].nPdschSym = ue_grp_params.attr("NrOfSymbols").cast<uint8_t>();
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].dmrsSymLocBmsk = ue_grp_params.attr("dlDmrsSymbPos").cast<uint16_t>();

        // pUePrmIdxs
        py::list uePrmIdxs                                       = ue_grp_params.attr("uePrmIdxs");
        uint16_t nUes                                            = uePrmIdxs.size();
        pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].nUes         = nUes;
        for (int idx = 0; idx < nUes; idx++){
            pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].pUePrmIdxs[idx] = uePrmIdxs[idx].cast<uint16_t>();
        }

        // dmrs
        cuphyPdschDmrsPrm_t* pDmrsDynPrm = pdschCellGrpDynPrms.pUeGrpPrms[ue_group_id].pDmrsDynPrm;

        pDmrsDynPrm->nDmrsCdmGrpsNoData = ue_grp_params.attr("numDmrsCdmGrpsNoData").cast<uint8_t>();
        pDmrsDynPrm->dmrsScrmId = ue_grp_params.attr("dlDmrsScramblingId").cast<uint8_t>();
    }

    // uePrms
    uint16_t nUes = uePrms.size();
    pdschCellGrpDynPrms.nUes = nUes;
    for (int ue_id = 0; ue_id < nUes; ue_id++){
        const py::object ue_params = uePrms[ue_id];

        int ueGrpPrmIdx                                       = ue_params.attr("ueGrpPrmIdx").cast<int>();
        pdschCellGrpDynPrms.pUePrms[ue_id].pUeGrpPrm          = &pdschCellGrpDynPrms.pUeGrpPrms[ueGrpPrmIdx];
        pdschCellGrpDynPrms.pUePrms[ue_id].scid               = ue_params.attr("SCID").cast<uint8_t>();
        pdschCellGrpDynPrms.pUePrms[ue_id].nUeLayers          = ue_params.attr("nrOfLayers").cast<uint8_t>();
        pdschCellGrpDynPrms.pUePrms[ue_id].dmrsPortBmsk       = ue_params.attr("dmrsPortBmsk").cast<uint16_t>();
        pdschCellGrpDynPrms.pUePrms[ue_id].BWPStart           = ue_params.attr("BWPStart").cast<uint16_t>();
        pdschCellGrpDynPrms.pUePrms[ue_id].refPoint           = ue_params.attr("refPoint").cast<uint8_t>();
        pdschCellGrpDynPrms.pUePrms[ue_id].beta_dmrs          = ue_params.attr("beta_dmrs").cast<float>();
        pdschCellGrpDynPrms.pUePrms[ue_id].beta_qam           = ue_params.attr("beta_qam").cast<float>();
        pdschCellGrpDynPrms.pUePrms[ue_id].rnti               = ue_params.attr("RNTI").cast<uint16_t>();
        pdschCellGrpDynPrms.pUePrms[ue_id].dataScramId        = ue_params.attr("dataScramblingId").cast<uint16_t>();
        pdschCellGrpDynPrms.pUePrms[ue_id].enablePrcdBf       = ue_params.attr("enablePrcdBf").cast<uint8_t>();

        if (pdschCellGrpDynPrms.pUePrms[ue_id].enablePrcdBf)
            pdschCellGrpDynPrms.pUePrms[ue_id].pmwPrmIdx      = ue_params.attr("pmwPrmIdx").cast<uint16_t>();
        else
            pdschCellGrpDynPrms.pUePrms[ue_id].pmwPrmIdx      = 0;
        // cwIdxs
        py::list cwIdxs                                       = ue_params.attr("cwIdxs");
        uint8_t nCw                                           = cwIdxs.size();
        pdschCellGrpDynPrms.pUePrms[ue_id].nCw                = nCw;

        for (int cw_idx = 0; cw_idx < nCw; cw_idx++ ){
            pdschCellGrpDynPrms.pUePrms[ue_id].pCwIdxs[cw_idx] = cwIdxs[cw_idx].cast<uint16_t>();
        }
    }


    // cwPrms
    uint16_t nCws                 = cwPrms.size();
    pdschCellGrpDynPrms.nCws      = nCws;
    for (int cw_idx = 0; cw_idx < nCws; cw_idx++){
        const py::object cw_params = cwPrms[cw_idx];

        int uePrmIdx                                          = cw_params.attr("uePrmIdx").cast<int>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].pUePrm            = &pdschCellGrpDynPrms.pUePrms[uePrmIdx];
        pdschCellGrpDynPrms.pCwPrms[cw_idx].targetCodeRate    = cw_params.attr("targetCodeRate").cast<uint16_t>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].qamModOrder       = cw_params.attr("qamModOrder").cast<uint8_t>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].rv                = cw_params.attr("rvIndex").cast<uint8_t>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].tbStartOffset     = cw_params.attr("tbStartOffset").cast<uint32_t>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].tbSize            = cw_params.attr("TBSize").cast<uint32_t>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].n_PRB_LBRM        = cw_params.attr("n_PRB_LBRM").cast<uint16_t>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].maxLayers         = cw_params.attr("maxLayers").cast<uint8_t>();
        pdschCellGrpDynPrms.pCwPrms[cw_idx].maxQm             = cw_params.attr("maxQm").cast<uint8_t>();
    }

    // CSI-RS prms, API not implemented.
    uint16_t nCsiRsPrms = csiRsPrms.size();
    if (nCsiRsPrms > 0){
        throw std::runtime_error("csi rs API not implemented! \n");
    }
    pdschCellGrpDynPrms.nCsiRsPrms = 0;

    // pmw prms
    uint16_t nPrecodingMatrices              = pmwPrms.size();
    pdschCellGrpDynPrms.nPrecodingMatrices = nPrecodingMatrices;

    if (nPrecodingMatrices > 0){
        for (int pmw_idx = 0; pmw_idx < nPrecodingMatrices; pmw_idx++){
            py::object pmw_prms             = pmwPrms[pmw_idx];

            pdschCellGrpDynPrms.pPmwPrms[pmw_idx].nPorts  = pmw_prms.attr("nPorts").cast<uint8_t>();

            // Not fully solved! Numpy does not support complex32.
            py::array temp = pmw_prms.attr("w");
            py::array_t<std::complex<float>, py::array::f_style | py::array::forcecast> pmw_array = temp;
            py::buffer_info buf = pmw_array.request();
            std::complex<float> *ptr = static_cast<std::complex<float> *>(buf.ptr);
            for (size_t idx = 0; idx < buf.size; idx++){
                pdschCellGrpDynPrms.pPmwPrms[pmw_idx].matrix[idx].x = (__half)ptr[idx].real();
                pdschCellGrpDynPrms.pPmwPrms[pmw_idx].matrix[idx].y = (__half)ptr[idx].imag();
            }
        }
    }

    // data in
    py::list tbInput = dataIn.attr("tbInput");
    if( (uint16_t)tbInput.size() != nCells){
        throw std::runtime_error("Cell number for tbInput does not match cellPrms!");
    }

    for (int cell_idx = 0; cell_idx < nCells; cell_idx ++) {
        py::array temp_array = tbInput[cell_idx];
        py::array_t<uint8_t, py::array::f_style | py::array::forcecast> tbInput_array = temp_array;
        py::buffer_info tbInput_buff = tbInput_array.request();
        tbInputPtr[cell_idx] = static_cast<uint8_t *>(tbInput_buff.ptr);
    }

    pdschDataIn = {tbInputPtr.data(), cuphyPdschDataIn_t::CPU_BUFFER};
    pdschDynParams.pDataIn = &pdschDataIn;

    // Optional TB CRC Data in.
    py::object pyTbCrcDataIn;
    if (!(std::string(py::str(dynPrms.attr("tbCRCDataIn"))) == "None")) {
        pyTbCrcDataIn = dynPrms.attr("tbCRCDataIn");

        py::list tbCrcInput = pyTbCrcDataIn.attr("tbInput");
        if( (uint16_t)tbCrcInput.size() != nCells){
            throw std::runtime_error("Cell number for tbCRCInput does not match cellPrms!");
        }
        std::vector<uint8_t*> tbCrcInputPtr(nCells);

        for (int cell_idx = 0; cell_idx < nCells; cell_idx ++) {
            py::array temp_array = tbCrcInput[cell_idx];
            py::array_t<uint8_t, py::array::f_style | py::array::forcecast> tbCrcInputArray = temp_array;
            py::buffer_info tbCrcInputBuff = tbCrcInputArray.request();
            tbCrcInputPtr[cell_idx] = static_cast<uint8_t *>(tbCrcInputBuff.ptr);
        }

        tbCrcDataIn                      = {tbCrcInputPtr.data(), cuphyPdschDataIn_t::CPU_BUFFER};
        pdschDynParams.pTbCRCDataIn      = &tbCrcDataIn;

    }
    else {
        tbCrcDataIn                       = {nullptr, cuphyPdschDataIn_t::CPU_BUFFER};
        pdschDynParams.pTbCRCDataIn       = &tbCrcDataIn;
    }


    // data out
    py::list dataTx = dataOut.attr("dataTx");

    if( (uint16_t)dataTx.size() != nCells) {
        throw std::runtime_error("Cell number for dataTx does not match cellPrms!");
    }


    for (int cell_id = 0; cell_id < nCells; cell_id++) {
        py::object tensor_desc      = dataTx[cell_id];

        py::list   dimensions       = tensor_desc.attr("dimensions");
        if (dimensions.size() != 3){
            throw std::runtime_error("DataOut tensor is not 3 dimensions!");
        }

        int dim0                    = dimensions[0].cast<int>();
        int dim1                    = dimensions[1].cast<int>();
        int dim2                    = dimensions[2].cast<int>();

        uint64_t pAddr              = tensor_desc.attr("pAddr").cast<uint64_t>();

        std::string dataType        = std::string(py::str(tensor_desc.attr("dataType").attr("_name_")));
        cuphyDataType_t type;

        if (dataType == "CUPHY_C_16F"){
            type = CUPHY_C_16F;
        }
        else if (dataType == "CUPHY_C_32F")
        {
            type = CUPHY_C_32F;
        }
        else{
            throw std::runtime_error("DataOut type is neither CUPHY_C_16F or CUPHY_C_32F!");
        }

        dataTxTensor[cell_id]  =  tensor_device((void*)pAddr, type, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);

        pdschDynParams.pDataOut->pTDataTx[cell_id].desc  = dataTxTensor[cell_id].desc().handle();
        pdschDynParams.pDataOut->pTDataTx[cell_id].pAddr = dataTxTensor[cell_id].addr();
    }

    // Setup pipeline.
    cuphyStatus_t status = cuphySetupPdschTx((cuphyPdschTxHndl_t)pdschHandle, &pdschDynParams, nullptr);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        std::cerr << "Error! cuphySetupPdschTx(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }

}


void PdschPipeline::runPdschTx(uint64_t pdschHandle) {

    cuphyPdschProcMode_t pdschProcMode = static_cast<cuphyPdschProcMode_t>(
        (uint32_t)PDSCH_PROC_MODE_NO_GRAPHS | (uint32_t)PDSCH_INTER_CELL_BATCHING
    );

    cuphyStatus_t status = cuphyRunPdschTx((cuphyPdschTxHndl_t)pdschHandle, pdschProcMode);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        std::cerr << "Error! cuphyRunPdschTx(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }
}


py::array_t<float> PdschPipeline::getLdpcOutputPerTbPerCell(uint64_t pdschHandle, int cellIdx, int tbIdx, const uint64_t ldpcOutputHostPtr) {

    PdschTx* pipelinePtr = static_cast<PdschTx*>((cuphyPdschTxHndl_t)pdschHandle);

    // Outputs.
    int numCbsForTbIdx = 0;
    int numBitsForTbIdx = 0;

    typed_tensor<CUPHY_BIT, pinned_alloc> hostOutTensor = pipelinePtr->getHostOutputLDPCTBPerCell(
        cellIdx, tbIdx, &numCbsForTbIdx, &numBitsForTbIdx, pipelinePtr->dynamic_params->cuStream
    );

    // Convert to float for Numpy and return the Numpy array.
    uint32_t dim0 = hostOutTensor.dimensions()[0];
    uint32_t dim1 = hostOutTensor.dimensions()[1];
    cuphy::tensor_pinned outTensor((void*)ldpcOutputHostPtr, CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
    outTensor.convert(hostOutTensor, pipelinePtr->dynamic_params->cuStream);
    CUDA_CHECK(cudaStreamSynchronize(pipelinePtr->dynamic_params->cuStream));

    return hostToNumpy<float>((float*)ldpcOutputHostPtr, dim0, dim1);
}


void PdschPipeline::destroyPdschTx(uint64_t pdschHandle) {

    // Clean up, release allocated memory. Note: Only one cell group here.
    delete[] pdschStatPrms.pCellStatPrms;
    delete[] pdschStatPrms.pDbg;

    for (int i = 0; i < PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP; i++){
        delete[] pdschCellGrpDynPrms.pUeGrpPrms[i].rbBitmap;
        delete[] pdschCellGrpDynPrms.pUeGrpPrms[i].pUePrmIdxs;
        delete[] pdschCellGrpDynPrms.pUeGrpPrms[i].pDmrsDynPrm;
    }

    for (int i = 0; i < PDSCH_MAX_UES_PER_CELL_GROUP; i++){
        delete[] pdschCellGrpDynPrms.pUePrms[i].pCwIdxs;
    }

    delete [] pdschCellGrpDynPrms.pCellPrms;
    delete [] pdschCellGrpDynPrms.pUeGrpPrms;
    delete [] pdschCellGrpDynPrms.pUePrms;
    delete [] pdschCellGrpDynPrms.pCwPrms;
    delete [] pdschCellGrpDynPrms.pPmwPrms;


    delete [] pdschDynParams.pDataOut->pTDataTx;
    delete pdschDynParams.pStatusInfo;
    delete pdschDynParams.pDataOut;

    cuphyStatus_t status = cuphyDestroyPdschTx((cuphyPdschTxHndl_t)pdschHandle);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        std::cerr << "Error! cuphyDestroyPdschTx(): " << cuphyGetErrorString(status) << std::endl;
        exit(1);
    }
}

}



#endif
