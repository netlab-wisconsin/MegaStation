/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <unistd.h>     /* For SYS_xxx definitions */
#include <syscall.h>    /* For SYS_xxx definitions */
#include <nvToolsExt.h> // NVTX CUDA profiler annotaions
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "psch_rx_tx_cmn.hpp"

#define DEBUG_TRACE(...)                           \
    do                                             \
    {                                              \
        if(m_dbgMsgLevel > 0) printf(__VA_ARGS__); \
    } while(0)

cuPHYTestWorker::cuPHYTestWorker(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<cuphyTestWrkrCmdQ>& cmdQ, std::shared_ptr<cuphyTestWrkrRspQ>& rspQ, int uldl, uint32_t debugMessageLevel) :
    m_name(name),
    m_wrkrId(workerId),
    m_thrdId(0),
    m_gpuId(gpuId),
    m_mpsSubctxSmCount(mpsSubctxSmCount),
    m_shPtrCmdQ(cmdQ),
    m_shPtrRspQ(rspQ),
    m_cpuId(cpuId),
    m_schdPolicy(cpuThrdSchdPolicy),
    m_prio(cpuThrdPrio),
    m_dbgMsgLevel(debugMessageLevel),
    m_fp16Mode(1),
    m_descramblingOn(1),
    m_ref_check_pdsch(false),
    m_ref_check_pdcch(false),
    m_ref_check_csirs(false),
    m_ref_check_pucch(false),
    m_ref_check_prach(false),
    m_ref_check_ssb(false),
    m_ref_check_srs(false),
    m_ref_check_bfc(false),
    m_identical_ldpc_configs(true),
    m_pdsch_group_cells(false), /* Will be updated in pdschTxInit as needed */
    m_pusch_group_cells(false), /* Will be updated in puschRxInit as needed */
    m_pucch_group_cells(false), /* Will be updated in pucchRxInit as needed */
    m_pdsch_proc_mode(PDSCH_PROC_MODE_NO_GRAPHS),
    m_pdcch_proc_mode(0),
    m_csirs_proc_mode(0),
    m_pusch_proc_mode(0),
    m_pucch_proc_mode(0),
    m_prach_proc_mode(0),
    m_srs_proc_mode(0),
    m_longPattern(0),
    m_nStrms(0),
    m_nStrms_pdsch(0),
    m_nItrsPerStrm(0),
    m_nTimingItrs(0),
    m_smCount(0),
    m_nSmIds(0),
    m_uldlMode(uldl),
    m_runBWC(false),
    m_runSRS(false),
    m_nSRSCells(0),
    m_nBWCCells(0),
    m_nPDCCHCells(0),
    m_nPUCCHCells(0),
    m_nPRACHCells(0),
    m_nSSBCells(0),
    m_nSsbSlots(0),
    m_runSRS2(false),
    m_runPDSCH(false),
    m_runPRACH(false),
    m_runPUSCH(false),
    m_runPUCCH(false),
    m_runPDCCH(false),
    m_runCSIRS(false),
    m_runSSB(false),
    m_srsCtx(false),
    m_totSRSStartTime(0),
    m_totSRSRunTime(0),
    m_totSRS2StartTime(0),
    m_totSRS2RunTime(0),
    m_totPUSCHStartTime(0),
    m_totPUSCHRunTime(0),
    m_totPUSCH2StartTime(0),
    m_totPUSCH2RunTime(0),
    m_totPUCCHStartTime(0),
    m_totPUCCHRunTime(0),
    m_totPUCCH2StartTime(0),
    m_totPUCCH2RunTime(0),
    m_totPRACHStartTime(0),
    m_totPRACHRunTime(0)
{
    // Start worker thread
    m_thrd = std::thread(&cuPHYTestWorker::run, this);
}

cuPHYTestWorker::~cuPHYTestWorker()
{
    // Send termination message to worker thread
    auto shPtrPayload = std::make_shared<cuPHYTestExitMsgPayload>();
    shPtrPayload->rsp = true;

    if(m_runSRS)
    {
        cuphyStatus_t statusDestroy = cuphyDestroySrsRx(m_srsRxHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy)
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "cuPHYTestWorker Destructor Error: cuphyDestroySrsRx (SRS1)");
    }
    if(m_runSRS2)
    {
        cuphyStatus_t statusDestroy = cuphyDestroySrsRx(m_srsRxHndl2);
        if(CUPHY_STATUS_SUCCESS != statusDestroy)
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "cuPHYTestWorker Destructor Error: cuphyDestroySrsRx (SRS2)");
    }

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_EXIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
    m_shPtrRspQ->receive(shPtrRsp, m_wrkrId);

    if(m_thrd.joinable())
    {
        try
        {
            m_thrd.join();
            DEBUG_TRACE("MainThread [tid %s]: Joining worker thread\n", getThreadIdStr().c_str());
        }
        catch(const std::exception& e)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {} while joining worker thread id {}", e.what(), m_wrkrId);
        }
    }
    DEBUG_TRACE("MainThread [tid %s]: Destructor completed\n", getThreadIdStr().c_str());
}

void cuPHYTestWorker::readSmIds()
{
    get_sm_ids(m_gpuId, m_smIdsGpu.addr(), m_smCount, m_uqPtrWrkrCuStrm->handle());

    CUDA_CHECK(cudaEventRecord(m_shPtrRdSmIdWaitEvent->handle(), m_uqPtrWrkrCuStrm->handle()));

#if 0
    m_uqPtrWrkrCuStrm->synchronize();
    m_smIds.resize(smCount);
    uint32_t* pSmIds = m_smIdsCpu.addr();
    std::copy(pSmIds, pSmIds + smCount, m_smIds.begin());
    for(int i = 0; i < smCount; ++i)
    {
        DEBUG_TRACE("Worker[%d]: SM id %d\n", m_wrkrId, pSmIds[i]);
    }
#endif
}

void cuPHYTestWorker::createCuCtx()
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: createCuCtx pre context creation (mpsSubctxSmCount %u)\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_mpsSubctxSmCount);

#if CUDART_VERSION < 11040 // min CUDA version for MPS programmatic API
    printf("MPS programmatic API support requires CUDA 11.4 or higher\n");
    exit(EXIT_FAILURE);
#endif

    static std::mutex ctxPartitionCfgMutex;
    {
        std::lock_guard<std::mutex> ctxPartitionCfgMutexLock(ctxPartitionCfgMutex);
        printf("Make request m_wrkrId %d smCountRequested %d\n", m_wrkrId, m_mpsSubctxSmCount);
        m_cuCtx.create(m_gpuId, m_mpsSubctxSmCount, &m_smCount);
        printf("m_wrkrId %d smCountRequested %d smCountApplied %d\n", m_wrkrId, m_mpsSubctxSmCount, m_smCount);
        m_cuCtx.bind();
    }
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: createCuCtx post context creation\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // CUDA_CHECK(cudaSetDevice(m_gpuId));

    int maxThreadsPerBlock = 0;
    CU_CHECK(cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_gpuId));

    int maxThreadsPerMultiProcessor = 0;
    CU_CHECK(cuDeviceGetAttribute(&maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, m_gpuId));

    int maxThreadBlocksPerMultiProcessor = maxThreadsPerMultiProcessor / maxThreadsPerBlock;

    // Now allocate resources
    // printf("smIdBufSize = %d\n", m_smCount*maxThreadBlocksPerMultiProcessor);

    m_uqPtrWrkrCuStrm           = std::make_unique<cuphy::stream>(cudaStreamNonBlocking);
    m_shPtrStopEvent            = std::make_shared<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrSRSStopEvent         = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrSRS2StopEvent        = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrSSBStopEvent         = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_shPtrRdSmIdWaitEvent      = std::make_shared<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrPuschDelayStopEvent  = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrPusch2DelayStopEvent = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrSRSDelayStopEvent    = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrPdschIterStopEvent   = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrBWCStopEvent         = std::make_unique<cuphy::event>(cudaEventDisableTiming);

    m_uqPtrTimeSRSStartEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimeSRSEndEvent      = std::make_unique<cuphy::event>();
    m_uqPtrTimeSRS2StartEvent   = std::make_unique<cuphy::event>();
    m_uqPtrTimeSRS2EndEvent     = std::make_unique<cuphy::event>();
    m_uqPtrTimePRACHStartEvent  = std::make_unique<cuphy::event>();
    m_uqPtrTimePRACHEndEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCHStartEvent  = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCHEndEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCH2StartEvent = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCH2EndEvent   = std::make_unique<cuphy::event>();

    m_uqPtrTimePUSCHStartEvent  = std::make_unique<cuphy::event>();
    m_uqPtrTimePUSCHEndEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimePUSCH2StartEvent = std::make_unique<cuphy::event>();
    m_uqPtrTimePUSCH2EndEvent   = std::make_unique<cuphy::event>();

    //  1 SM id per thread block

    m_nSmIds   = m_smCount * maxThreadBlocksPerMultiProcessor;
    m_smIdsGpu = std::move(cuphy::buffer<uint32_t, cuphy::device_alloc>(m_nSmIds));
    m_smIdsCpu = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSmIds));

    // print setup:
    printf("%s worker %d :", m_name.c_str(), m_wrkrId);
    printf("\n--> assigned %d SMs", m_smCount);
    printf("\n--> Runs %d stream(s) in parallel", m_nStrms);
    if(m_name.compare("PschTxRxTestWorker") == 0)
    {
        printf("\n--> Each stream processes %d PUSCH + PDSCH test-vector(s) in series\n\n", m_nItrsPerStrm);
    }
    if(m_name.compare("PdschTxTestWorker") == 0)
    {
        printf("\n--> Each stream processes %d PDSCH test-vector(s) in series\n\n", m_nItrsPerStrm);
    }
    if(m_name.compare("PuschRxTestWorker") == 0)
    {
        printf("\n--> Each stream processes %d PUSCH test-vector(s) in series\n\n", m_nItrsPerStrm);
    }
}

void cuPHYTestWorker::setThrdProps()
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: setThrdProps\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    //------------------------------------------------------------------
    // Bump up CPU thread prio
    pid_t       pid = (pid_t)syscall(SYS_gettid);
    sched_param schdPrm;
    schdPrm.sched_priority = m_prio;

    // pid_t pid = (pid_t) syscall(SYS_gettid);
    /*
    int schdSetRet = sched_setscheduler(pid, m_schdPolicy, &schdPrm);

    if(0 == schdSetRet)
    {
        DEBUG_TRACE("%s id %d [tid %s]: pid %d policy %d prio %d\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), pid, m_schdPolicy, m_prio);
    }
    else
    {
        DEBUG_TRACE("%s id %d [tid %s]: Failed to set scheduling algo pid %d, prio %d, return code %d: err %s\n",
                    m_name.c_str(),
                    m_wrkrId,
                    getThreadIdStr().c_str(),
                    pid,
                    m_prio,
                    schdSetRet,
                    strerror(errno));
    }
    */

    //------------------------------------------------------------------
    // Set thread affinity to specified CPU
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    CPU_SET(m_cpuId, &cpuSet);
    // DEBUG_TRACE("%s Pipeline[%d]: setting affinity of pipeline %d (pid %d) to CPU Id %d\n", m_name.c_str(), instIdx, pid, cpuId);

    int affinitySetRet = sched_setaffinity(pid, sizeof(cpuSet), &cpuSet);

    if(0 == affinitySetRet)
    {
        DEBUG_TRACE("%s id %d [tid %s]: pid %d set affinity to CPU %d\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), pid, m_cpuId);
    }
    else
    {
        DEBUG_TRACE("%s id %d [tid %s]: failed to set affinity pid %d to CPU %d, return code %d err %s\n",
                    m_name.c_str(),
                    m_wrkrId,
                    getThreadIdStr().c_str(),
                    pid,
                    m_cpuId,
                    affinitySetRet,
                    strerror(errno));
    }
}

//----------------------------------------------------------------------------------------------------------
// Message handlers
void cuPHYTestWorker::emptyHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: emptyHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
}

void cuPHYTestWorker::initHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: initHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    cuPHYTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);

    m_thrdId = std::this_thread::get_id();
    setThrdProps();

    m_uqPtrTimeStartEvent = std::make_unique<cuphy::event>();

    m_pschDlUlSyncEvents.resize(m_nStrms);
    m_timerSingleStreamItr.resize(m_nStrms);
    m_maxNumCbErrors.resize(m_nStrms);
    m_timePdschSlotEndEvents.resize(m_nItrsPerStrm);
    m_timeNextPdschSlotStartEvents.resize(m_nItrsPerStrm + 2);
    m_timeNextPdcchSlotStartEvents.resize(m_nItrsPerStrm + 2);
    m_timeNextCSIRSSlotStartEvents.resize(m_nItrsPerStrm + 2);
    m_timeBWCSlotStartEvents.resize(m_nItrsPerStrm);
    m_timeBWCSlotEndEvents.resize(m_nItrsPerStrm);
    m_timePdcchSlotEndEvents.resize(m_nItrsPerStrm);
    m_timeCSIRSSlotEndEvents.resize(m_nItrsPerStrm);
    m_totRunTimePdschItr.resize(m_nItrsPerStrm);
    m_totPdschSlotStartTime.resize(m_nItrsPerStrm + 1);
    m_totPdcchStartTimes.resize(m_nItrsPerStrm);
    m_totBWCIterStartTime.resize(m_nItrsPerStrm);
    m_totRunTimeBWCItr.resize(m_nItrsPerStrm);
    m_totRunTimePdcchItr.resize(m_nItrsPerStrm);
    m_totRunTimeCSIRSItr.resize(m_nItrsPerStrm);
    m_totCSIRSStartTimes.resize(m_nItrsPerStrm);
    
    // Below are SSB timing vectors, max # of SSB = m_nItrsPerStrm
    m_timeSSBSlotStartEvents.resize(m_nItrsPerStrm);
    m_timeSSBSlotEndEvents.resize(m_nItrsPerStrm);
    m_totSSBStartTime.resize(m_nItrsPerStrm);
    m_totSSBRunTime.resize(m_nItrsPerStrm);

    auto& cuStrmPrioMap = initMsgPayload.cuStrmPrioMap;
    m_cuStrmPrioPusch   = (cuStrmPrioMap.find("PUSCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUSCH") : 0;
    m_cuStrmPrioPusch2  = (cuStrmPrioMap.find("PUSCH2") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUSCH2") : 0;
    m_cuStrmPrioPdsch   = (cuStrmPrioMap.find("PDSCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PDSCH") : 0;
    m_cuStrmPrioSRS     = (cuStrmPrioMap.find("SRS") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("SRS") : 0;
    m_cuStrmPrioPrach   = (cuStrmPrioMap.find("PRACH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PRACH") : 0;
    m_cuStrmPrioPucch   = (cuStrmPrioMap.find("PUCCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUCCH") : 0;
    m_cuStrmPrioPucch2  = (cuStrmPrioMap.find("PUCCH2") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUCCH2") : 0;
    m_cuStrmPrioPdcch   = (cuStrmPrioMap.find("PDCCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PDCCH") : 0;
    m_cuStrmPrioPrach   = (cuStrmPrioMap.find("PRACH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PRACH") : 0;
    m_cuStrmPrioSSB     = (cuStrmPrioMap.find("SSB") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("SSB") : 0;
    m_cuStrmPrioCSIRS   = (cuStrmPrioMap.find("CSIRS") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("CSIRS") : 0;
    printf("CUDA stream prios: PUSCH %d (-u 4, PUSCH2 %d) PDSCH %d SRS %d PRACH %d PDCCH %d PUCCH %d PUCCH2 %d SSB %d CSIRS %d\n", m_cuStrmPrioPusch, m_cuStrmPrioPusch2, m_cuStrmPrioPdsch, m_cuStrmPrioSRS, m_cuStrmPrioPrach, m_cuStrmPrioPdcch, m_cuStrmPrioPucch, m_cuStrmPrioPucch, m_cuStrmPrioSSB, m_cuStrmPrioCSIRS);
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        m_cuStrms.emplace_back(cudaStreamNonBlocking);
        m_stopEvents.emplace_back(cudaEventDisableTiming);
        m_stop2Events.emplace_back(cudaEventDisableTiming);

        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            m_timerSingleStreamItr[strmIdx].emplace_back();
            m_maxNumCbErrors[strmIdx].emplace_back(0);

            m_pschDlUlSyncEvents[strmIdx].emplace_back();
        }
    }
    // extra stop event for PUSCH1 in longPattern == 3 or longPattern == 6
#if 1
    m_shPtrGpuStartSyncFlag = initMsgPayload.shPtrCpuGpuSyncFlag;
    CU_CHECK(cuMemHostGetDevicePointer(&m_ptrGpuStartSyncFlag, m_shPtrGpuStartSyncFlag->addr(), 0));
#else
    static std::mutex syncFlagWriteMutex;
    {
        std::lock_guard<std::mutex> syncFlagWriteMutexLock(syncFlagWriteMutex);

        m_shPtrGpuStartSyncFlag = initMsgPayload.shPtrCpuGpuSyncFlag;
        CU_CHECK(cuMemHostGetDevicePointer(&m_ptrGpuStartSyncFlag, m_shPtrGpuStartSyncFlag->addr(), 0));
        CU_CHECK(cuStreamWriteValue32(reinterpret_cast<CUstream>(m_cuStrms[0].handle()), (CUdeviceptr)m_ptrGpuStartSyncFlag, m_wrkrId, 0));
        m_cuStrms[0].synchronize();
        printf("shPtrGpuStartSyncFlag %d, wrkrId %d\n", (*m_shPtrGpuStartSyncFlag)[0], m_wrkrId);
    }
#endif

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::csirsInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: csirsInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesCSIRS = initMsgPayload.inFileNames;

    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = 0;
    }
    else
    {
        m_runCSIRS = true;

        int nCSIRSObjects = m_pdsch_group_cells ? 1 : m_nCSIRSCells;

        m_csirsTxPipes.resize(nCSIRSObjects);
        m_csirsTxStaticApiDataSets.resize(nCSIRSObjects);
        m_csirsTxDynamicApiDataSets.resize(nCSIRSObjects);

        // CSIRS runs in a single slot in both DDDSU and DDDSUUDDDD patterns

        for(uint32_t strmIdx = 0; strmIdx < nCSIRSObjects; ++strmIdx)
        {
            m_CSIRSStopEvents.emplace_back(cudaEventDisableTiming);
            m_csirsTxStaticApiDataSets[strmIdx].reserve(1);
            m_csirsTxDynamicApiDataSets[strmIdx].reserve(1);
            m_cuStrmsCSIRS.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioCSIRS);
        }

        for(uint32_t i = 0; i < m_nCSIRSCells; ++i)
        {
            if(m_pdsch_group_cells)
            {
                if(i == 0)
                {
                    m_csirsTxStaticApiDataSets[0].emplace_back(inFileNamesCSIRS[i], m_nCSIRSCells);
                    m_csirsTxPipes[0].emplace_back(m_csirsTxStaticApiDataSets[i][0].csirsStatPrms);
                }
                else
                {
                    m_csirsTxStaticApiDataSets[0][0].cumulativeUpdate(inFileNamesCSIRS[i]);
                }
            }
            else
            {
                m_csirsTxStaticApiDataSets[i].emplace_back(inFileNamesCSIRS[i]);
                m_csirsTxPipes[i].emplace_back(m_csirsTxStaticApiDataSets[i][0].csirsStatPrms);
            }
        }
    }
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::csirsSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: csirsSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesCSIRS = initMsgPayload.inFileNames;

    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = 0;
        return;
    }
    m_runCSIRS = true;

    int nCSIRSObjects = m_pdsch_group_cells ? 1 : m_nCSIRSCells;

    m_csirsTxDynamicApiDataSets.clear();
    m_csirsTxDynamicApiDataSets.resize(nCSIRSObjects);
    for(uint32_t i = 0; i < m_nCSIRSCells; ++i)
    {
        if(m_pdsch_group_cells)
        {
            if(i == 0)
            {
                m_csirsTxDynamicApiDataSets[i].emplace_back(inFileNamesCSIRS[i], m_csirsTxStaticApiDataSets[i][0].csirsStatPrms.nMaxCellsPerSlot, m_cuStrmsCSIRS[i].handle(), m_csirs_proc_mode);
            }
            else
            {
                // Nothing to cumulative update for the static parameters.
                // Update the dyanmic parameters
                m_csirsTxDynamicApiDataSets[0][0].cumulativeUpdate(inFileNamesCSIRS[i], m_cuStrmsCSIRS[0].handle());
            }
        }
        else
        {
            m_csirsTxDynamicApiDataSets[i].emplace_back(inFileNamesCSIRS[i], m_csirsTxStaticApiDataSets[i][0].csirsStatPrms.nMaxCellsPerSlot, m_cuStrmsCSIRS[i].handle(), m_csirs_proc_mode);
        }
    }
    for(int i = 0; i < nCSIRSObjects; i++)
    {
        m_csirsTxPipes[i][0].setup(m_csirsTxDynamicApiDataSets[i][0].csirs_dyn_params);
    }

    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::ssbInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: ssbInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSSB = initMsgPayload.inFileNames;

    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = 0;
    }
    else
    {
        m_runSSB = true;

        int nSSBObjects = m_pdsch_group_cells ? 1 : m_nSSBCells;

        m_ssbTxPipes.resize(nSSBObjects);
        m_ssbTxStaticApiDataSets.resize(nSSBObjects);
        m_ssbTxDynamicApiDataSets.resize(nSSBObjects);

        // SSB runs in m_nSsbSlots slot in both DDDSU and DDDSUUDDDD patterns starting from slot 4

        for(uint32_t strmIdx = 0; strmIdx < nSSBObjects; ++strmIdx)
        {
            m_ssbTxStaticApiDataSets[strmIdx].reserve(1);
            m_ssbTxDynamicApiDataSets[strmIdx].reserve(1);
            m_cuStrmsSSB.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioSSB);
        }

        for(uint32_t i = 0; i < m_nSSBCells; ++i)
        {
            if(m_pdsch_group_cells)
            {
                if(i == 0)
                {
                    m_ssbTxStaticApiDataSets[i].emplace_back(m_nSSBCells);
                    m_ssbTxPipes[i].emplace_back(m_ssbTxStaticApiDataSets[i][0].ssbStatPrms);
                }
                else
                {
                    // Nothing to cumulative update for the static parameters.
                    // Update the dyanmic parameters
                }
            }
            else
            {
                m_ssbTxStaticApiDataSets[i].emplace_back();
                m_ssbTxPipes[i].emplace_back(m_ssbTxStaticApiDataSets[i][0].ssbStatPrms);
            }
        }
    }
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::ssbSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: ssbSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSSB = initMsgPayload.inFileNames;

    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = 0;
    }
    else
    {
        m_runSSB = true;

        int nSSBObjects = m_pdsch_group_cells ? 1 : m_nSSBCells;

        m_ssbTxDynamicApiDataSets.clear();
        m_ssbTxDynamicApiDataSets.resize(nSSBObjects);
        for(uint32_t i = 0; i < m_nSSBCells; ++i)
        {
            if(m_pdsch_group_cells)
            {
                if(i == 0)
                {
                    m_ssbTxDynamicApiDataSets[i].emplace_back(inFileNamesSSB[i], m_ssbTxStaticApiDataSets[i][0].ssbStatPrms.nMaxCellsPerSlot, m_cuStrmsSSB[i].handle(), m_ssb_proc_mode);
                }
                else
                {
                    // Nothing to cumulative update for the static parameters.
                    // Update the dyanmic parameters
                    m_ssbTxDynamicApiDataSets[0][0].cumulativeUpdate(inFileNamesSSB[i], m_cuStrmsSSB[0].handle());
                }
            }
            else
            {
                m_ssbTxDynamicApiDataSets[i].emplace_back(inFileNamesSSB[i], m_ssbTxStaticApiDataSets[i][0].ssbStatPrms.nMaxCellsPerSlot, m_cuStrmsSSB[i].handle(), m_ssb_proc_mode);
            }
        }
        for(int i = 0; i < nSSBObjects; i++)
        {
            m_ssbTxPipes[i][0].setup(m_ssbTxDynamicApiDataSets[i][0].ssb_dyn_params);
        }
    }
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pucchRxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pucchRxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPUCCH = initMsgPayload.inFileNames;
    //std::vector<std::vector<std::string>> pucchStreamFiles;
    std::vector<std::string> tempInput;

    if(m_pucch_group_cells)
    {
        if(m_uldlMode == 5)
        {
            m_nStrms_pucch = 2;
        }
        else
            m_nStrms_pucch = 1;
        tempInput.resize(inFileNamesPUCCH.size() / m_nStrms_pucch);
    }
    else
    {
        m_nStrms_pucch = inFileNamesPUCCH.size();
        tempInput.resize(1);
    }

    m_pucchRxPipes.resize(m_nStrms_pucch);
    m_pucchStaticDatasetVec.resize(m_nStrms_pucch);
    size_t MAX_N_F234_UCI = CUPHY_PUCCH_F2_MAX_UCI + CUPHY_PUCCH_F3_MAX_UCI;

    std::string outputFilename = std::string();

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pucch; ++strmIdx)
    {
        if(m_longPattern && strmIdx >= m_nStrms_pucch / 2)
        {
            m_cuStrmsPucch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPucch2);
        }
        else
        {
            m_cuStrmsPucch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPucch);
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));

        m_PUCCHStopEvents.emplace_back(cudaEventDisableTiming);
        m_PUCCHStop2Events.emplace_back(cudaEventDisableTiming);

        for(uint32_t i = 0; i < m_nItrsPerStrm; ++i)
        {
            //uint32_t cellIdx = strmIdx + i * m_nPUCCHCells;

            if(m_pucch_group_cells)
            {
                tempInput.assign(inFileNamesPUCCH.begin() + i * (inFileNamesPUCCH.size() / m_nStrms_pucch), inFileNamesPUCCH.begin() + (i + 1) * (inFileNamesPUCCH.size() / m_nStrms_pucch));
            }
            else
            {
                tempInput.assign(1, inFileNamesPUCCH[strmIdx]);
            }

            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));

            m_pucchStaticDatasetVec[strmIdx].emplace_back(tempInput, m_cuStrmsPucch[strmIdx].handle(), outputFilename); // empty output filename for now

            m_pucchRxPipes[strmIdx].emplace_back(m_pucchStaticDatasetVec[strmIdx][i].pucchStatPrms, m_cuStrmsPucch[strmIdx].handle());
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
    }

    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::prachInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: prachInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPRACH = initMsgPayload.inFileNames;

    m_prachRxPipes.resize(m_nStrms_prach);
    m_prachDatasetVec.resize(m_nStrms_prach);

    // loop over streams (i.e., pipeline objects)
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_prach; ++strmIdx)
    {
        m_cuStrmsPrach.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPrach);
        m_PRACHStopEvents.emplace_back(cudaEventDisableTiming);

        m_prachDatasetVec[strmIdx].reserve(m_nItrsPerStrm);

        // loop over stream iterations; each iteration here is work
        // submitted on a given pipeline object in separate setup/run calls
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            // loop over all cells that need to be processed per pipeline in a single iteration.
            // This number should be 1, unless the m_prach_group_cells is true.
            for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_prach; ++cellPerStrmIdx)
            {
                uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_prach * m_nCellsPerStrm_prach + cellPerStrmIdx;

                if(cellPerStrmIdx == 0)
                {
                    m_prachDatasetVec[strmIdx].emplace_back(inFileNamesPRACH[cellIdx], m_cuStrmsPrach[strmIdx].handle(), m_prach_proc_mode, m_ref_check_prach);
                }
                else
                {
                    m_prachDatasetVec[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPRACH[cellIdx], m_cuStrmsPrach[strmIdx].handle());
                }
            }

            m_prachDatasetVec[strmIdx][itrIdx].finalize(m_cuStrmsPrach[strmIdx].handle());

            // intialize pipeline
            cuphyPrachStatPrms_t& prachStatPrms = m_prachDatasetVec[strmIdx][itrIdx].prachStatPrms;
            m_prachRxPipes[strmIdx].emplace_back(prachStatPrms);
        }
    }

    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pucchRxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pucchRxSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPUCCH = initMsgPayload.inFileNames;
    std::vector<std::string>       tempInput(1);

    if(m_pucch_group_cells)
    {
        tempInput.resize(inFileNamesPUCCH.size() / m_nStrms_pucch);
    }
    else
        tempInput.resize(1);

    m_pucchDynDatasetVec.clear();
    m_pucchEvalDatasetVec.clear();
    m_pucchDynDatasetVec.resize(m_nStrms_pucch);
    m_pucchEvalDatasetVec.resize(m_nStrms_pucch);

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pucch; ++strmIdx)
    {
        for(uint32_t i = 0; i < m_nItrsPerStrm; ++i)
        {
            uint32_t cellIdx = strmIdx + i * m_nStrms_pucch;

            // Load datasets
            if(m_pucch_group_cells)
            {
                tempInput.assign(inFileNamesPUCCH.begin() + i * (inFileNamesPUCCH.size() / m_nStrms_pucch), inFileNamesPUCCH.begin() + (i + 1) * (inFileNamesPUCCH.size() / m_nStrms_pucch));
            }
            else
            {
                tempInput.assign(1, inFileNamesPUCCH[strmIdx]);
            }
            //CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));

            m_pucchDynDatasetVec[strmIdx].emplace_back(tempInput, m_cuStrmsPucch[strmIdx].handle(), m_pucch_proc_mode);
            m_pucchEvalDatasetVec[strmIdx].emplace_back(tempInput, m_cuStrmsPucch[strmIdx].handle());
            cuphyPucchDynPrms_t& pucchDynPrm                       = m_pucchDynDatasetVec[strmIdx][i].pucchDynPrm;
            m_pucchDynDatasetVec[strmIdx][i].pucchDynPrm.cuStream  = m_cuStrmsPucch[strmIdx].handle(); // save stream in dynamic parameters
            m_pucchDynDatasetVec[strmIdx][i].pucchDynPrm.cpuCopyOn = m_ref_check_pucch;                // option to copy uci output to CPU immediately after run
            // Setup pucch receiver object

            cuphyPucchBatchPrmHndl_t const batchPrmHndl = nullptr; // batchPrms currently un-used

            m_pucchRxPipes[strmIdx][i].setup(pucchDynPrm, batchPrmHndl);
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
    }
    // send response
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::puschRxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: puschRxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    //cuPHYTestInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);

    cuPHYTestPuschRxInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<cuPHYTestPuschRxInitMsgPayload>(shPtrPayload);
    std::vector<std::string>              inFileNamesPuschRx = initMsgPayload.inFileNames;
    std::vector<std::string>              tempInput;

    if(m_pusch_group_cells)
    {
        if(m_uldlMode == 5)
        {
            m_nStrms = 2;
        }
        else
            m_nStrms = 1;
        tempInput.resize(inFileNamesPuschRx.size() / m_nStrms);
    }
    else
    {
        m_nStrms = inFileNamesPuschRx.size();
        tempInput.resize(1);
    }
    // resize
    m_puschRxStaticApiDataSets.resize(m_nStrms);
    m_puschRxPipes.resize(m_nStrms);
    m_maxNumCbErrors.resize(m_nStrms);

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        m_puschRxStaticApiDataSets[strmIdx].reserve(m_nItrsPerStrm);
        // DDDSUUDDDD

        if(m_longPattern && strmIdx >= m_nStrms / 2)
        {
            m_cuStrmsPusch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPusch2);
        }
        else
        {
            m_cuStrmsPusch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPusch);
        }

        // loop over stream iterations
        for(uint32_t i = 0; i < m_nItrsPerStrm; ++i)
        {
            // uint32_t cellIdx = strmIdx + i * m_nStrms;

            // load static
            if(m_pusch_group_cells)
            {
                tempInput.assign(inFileNamesPuschRx.begin() + i * (inFileNamesPuschRx.size() / m_nStrms), inFileNamesPuschRx.begin() + (i + 1) * (inFileNamesPuschRx.size() / m_nStrms));
                m_puschRxStaticApiDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle(), std::string(), 1, 0, initMsgPayload.enableLdpcThroughputMode, &initMsgPayload.puschPrms,
                                                                 static_cast<cuphyPuschLdpcKernelLaunch_t>(m_ldpc_kernel_launch_mode));
            }
            else
            {
                tempInput.assign(1, inFileNamesPuschRx[strmIdx]);
                m_puschRxStaticApiDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle(), std::string(), 1, 0, initMsgPayload.enableLdpcThroughputMode, &initMsgPayload.puschPrms,
                                                                 static_cast<cuphyPuschLdpcKernelLaunch_t>(m_ldpc_kernel_launch_mode));
            }

            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));

            // intialize pipeline

            m_puschRxPipes[strmIdx].emplace_back(m_puschRxStaticApiDataSets[strmIdx][i].puschStatPrms, m_cuStrmsPusch[strmIdx].handle());

            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));

            // intialize error counter
            m_maxNumCbErrors[strmIdx].emplace_back(0);
        }
    }

    // send response
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::srsInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: srsInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSRS = initMsgPayload.inFileNames;

    uint32_t nSRSStrms = 1;
    uint32_t nCells1   = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;

    std::vector<std::string> inFileNamesSRS1(inFileNamesSRS.begin(), inFileNamesSRS.begin() + nCells1);
    std::vector<std::string> inFileNamesSRS2(inFileNamesSRS.begin() + nCells1, inFileNamesSRS.end());

    m_cuStrmsSRS.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioSRS);
    m_SRSStopEvents.emplace_back(cudaEventDisableTiming);

    m_srsStaticApiDatasetVec.resize(1);
    m_srsStaticApiDatasetVec[0].emplace_back(inFileNamesSRS1, m_cuStrmsSRS[0].handle());

    if(m_runSRS2)
    {
        m_srsStaticApiDatasetVec2.resize(1);
        m_srsStaticApiDatasetVec2[0].emplace_back(inFileNamesSRS2, m_cuStrmsSRS[0].handle());
    }

    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsSRS[0].handle()));

    cuphyStatus_t statusCreate = cuphyCreateSrsRx(&m_srsRxHndl, &m_srsStaticApiDatasetVec[0][0].srsStatPrms, m_cuStrmsSRS[0].handle());
    if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

    if(m_runSRS2)
    {
        statusCreate = cuphyCreateSrsRx(&m_srsRxHndl2, &m_srsStaticApiDatasetVec2[0][0].srsStatPrms, m_cuStrmsSRS[0].handle());
        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);
    }

    // send response
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pdcchTxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdcchTxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPdcchTx = initMsgPayload.inFileNames;

    if(inFileNamesPdcchTx.size() == 0)
    {
        m_runPDCCH = false;
    }
    else
    {
        m_runPDCCH = true;
        // resize
        m_pdcchTxStaticApiDataSets.resize(m_nStrms_pdcch);
        m_pdcchTxPipes.resize(m_nStrms_pdcch);
        //printf("# streams %d for PDCCH (general is %d) with cells per stream %d\n", m_nStrms_pdcch, m_nStrms, m_nCellsPerStrm_pdcch);

        // loop over streams (i.e., pipeline objects)
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdcch; ++strmIdx)
        {
            m_pdcchTxStaticApiDataSets[strmIdx].reserve(m_nItrsPerStrm); //NOTE: pdcch datasets need to be reserved.
                                                                         // loop over stream iterations; each iteration here is work submitted on a given pipeline object in separate setup/run calls
                                                                         // FIXME Do we only read parameters for 1st "pattern"?
            m_cuStrmsPdcch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPdcch);

            m_PDCCHStopEvents.emplace_back(cudaEventDisableTiming);
            // loop over stream iterations
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                if(strmIdx == 0)
                    m_pdcchInterSlotStartEventVec.emplace_back(cudaEventDisableTiming);

                // loop over all cells that need to be processed per pipeline in a single iteration.
                // This number should be 1, unless the m_pdcch_group_cells is true.
                // FIXME Assuming that when m_nCellsPerStrm_pdcch != 1, m_nStrms_pdcch is 1. Confirm this holds for all uldl use cases.
                for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdcch; ++cellPerStrmIdx)
                {
                    uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdcch * m_nCellsPerStrm_pdcch + cellPerStrmIdx;
                    /*printf("PDCCH init handler, strmIdx %d, itrIdx %d and group_cells %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       m_pdcch_group_cells,
                       cellIdx,
                       inFileNamesPdcchTx[cellIdx].c_str());*/

                    if(cellPerStrmIdx == 0)
                    {
                        m_pdcchTxStaticApiDataSets[strmIdx].emplace_back(m_nCellsPerStrm_pdcch); // Providing no argument would default to 1 max. cell
                    }
                    /*else
                    {
                        ; //cumulatively update m_pdcchTxStatApiDataSets[strmIdx][itrIdx] Nothing to update for now.
                    }*/
                }

                // intialize pipeline
                m_pdcchTxPipes[strmIdx].emplace_back(m_pdcchTxStaticApiDataSets[strmIdx][itrIdx].pdcchStatPrms);
                //FIXME shall I catch a exception and exit(1)?
            }
        }
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::pdschTxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPdschTxInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<cuPHYTestPdschTxInitMsgPayload>(shPtrPayload);
    std::vector<std::string>              inFileNamesPdschTx = initMsgPayload.inFileNames;
    if(inFileNamesPdschTx.size() == 0)
    {
        m_runPDSCH = false;
    }
    else
    {
        m_runPDSCH = true;
        // resize
        m_pdschTxStaticApiDataSets.resize(m_nStrms_pdsch);
        m_pdschTxPipes.resize(m_nStrms_pdsch);
        //printf("# streams %d for PDSCH (general is %d) with cells per stream %d\n", m_nStrms_pdsch, m_nStrms, m_nCellsPerStrm_pdsch);
        // loop over streams (i.e., pipeline objects)

        m_GPUtime_d = std::move(cuphy::buffer<uint64_t, cuphy::device_alloc>(1));

        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
        {
            m_pdschTxStaticApiDataSets[strmIdx].reserve(m_nItrsPerStrm); //NOTE: pdsch datasets need to be reserved.
                                                                         // loop over stream iterations; each iteration here is work submitted on a given pipeline object in separate setup/run calls
                                                                         // FIXME Do we only read parameters for 1st "pattern"?
            m_cuStrmsPdsch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPdsch);

            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                if(strmIdx == 0)
                    m_pdschInterSlotStartEventVec.emplace_back(cudaEventDisableTiming);

                // loop over all cells that need to be processed per pipeline in a single iteration.
                // This number should be 1, unless the m_pdsch_group_cells is true.
                // FIXME Assuming that when m_nCellsPerStrm_pdsch != 1, m_nStrms_pdsch is 1. Confirm this holds for all uldl use cases.
                for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdsch; ++cellPerStrmIdx)
                {
                    uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdsch * m_nCellsPerStrm_pdsch + cellPerStrmIdx;
                    /*printf("PDSCH init handler, strmIdx %d, itrIdx %d and group_cells %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       m_pdsch_group_cells,
                       cellIdx,
                       inFileNamesPdschTx[cellIdx].c_str());*/
                    // load static
                    std::string outFileName = std::string();
                    if(cellPerStrmIdx == 0)
                    {
                        m_pdschTxStaticApiDataSets[strmIdx].emplace_back(inFileNamesPdschTx[cellIdx], outFileName, m_ref_check_pdsch, m_identical_ldpc_configs, m_cuStrmPrioPdsch, initMsgPayload.pdschPrms.maxNCbsPerTb, initMsgPayload.pdschPrms.maxNTbs, initMsgPayload.pdschPrms.maxNPrbs);
                    }
                    else
                    {
                        //Cumulatively update the static parameters for a given {pipeline, iteration}
                        //This else clause is exercised when multiple cells are grouped in a cellg group for a single pipeline object (PdschTx)
                        m_pdschTxStaticApiDataSets[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPdschTx[cellIdx], outFileName, m_ref_check_pdsch, m_identical_ldpc_configs);
                    }
                }
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdsch[strmIdx].handle()));

                // DBG print static parameters
                //m_pdschTxStaticApiDataSets[strmIdx][itrIdx].print();

                // intialize pipeline
                m_pdschTxPipes[strmIdx].emplace_back(m_pdschTxStaticApiDataSets[strmIdx][itrIdx].pdschStatPrms);
            }
            // extra event for last slot in loncPattern == 3 and longPattern == 6
            m_pdschInterSlotStartEventVec.emplace_back(cudaEventDisableTiming);
        }
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pdcchTxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: PDCCHSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPDCCH = initMsgPayload.inFileNames;

    if(inFileNamesPDCCH.size() == 0)
        return;
    // resize

    m_runPDCCH = true;

    // reset
    m_pdcchTxDynamicApiDataSets.clear();
    m_pdcchTxDynamicApiDataSets.resize(m_nPDCCHCells);

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdcch; ++strmIdx)
    {
        m_pdcchTxDynamicApiDataSets[strmIdx].reserve(m_nItrsPerStrm);

        // loop over stream iterations
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdcch; ++cellPerStrmIdx)
            {
                uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdcch * m_nCellsPerStrm_pdcch + cellPerStrmIdx;
                /*printf("PDCCH setup handler, strmIdx %d, itrIdx %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       cellIdx,
                       inFileNamesPdcchTx[cellIdx].c_str());*/

                //uint32_t cellIdx = strmIdx + itrIdx * m_nPDCCHCells;
                //printf("strmIdx %d, itrIdx %d, cellIdx %d, m PDCCH cells %d\n", strmIdx, itrIdx, cellIdx, m_nPDCCHCells);

                if(cellPerStrmIdx == 0)
                {
                    m_pdcchTxDynamicApiDataSets[strmIdx].emplace_back(inFileNamesPDCCH[cellIdx], m_nCellsPerStrm_pdcch, m_cuStrmsPdcch[strmIdx].handle(), m_pdcch_proc_mode);
                }
                else
                {
                    //Cumulatively update the dynamic parameters for a given pipeline, iteration
                    m_pdcchTxDynamicApiDataSets[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPDCCH[cellIdx], m_cuStrmsPdcch[strmIdx].handle());
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdcch[strmIdx].handle())); //FIXME needed?

            // setup pipeline
            //FIXME exit on exception?
            m_pdcchTxPipes[strmIdx][itrIdx].setup(m_pdcchTxDynamicApiDataSets[strmIdx][itrIdx].pdcch_dyn_params);
            //printf("strmIdx %d itrIdx %d\n", strmIdx, itrIdx);
            //m_pdcchTxPipes[strmIdx][itrIdx].handle()->pipeline->printPdcchConfig(m_pdcchTxDynamicApiDataSets[strmIdx][itrIdx].pdcch_dyn_params);
            //FIXME reset output buffer? TODO
            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdcch[strmIdx].handle()));
        }
    }
    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::bfcInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: BFCInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesBFC = initMsgPayload.inFileNames;

    m_bwcStaticApiDatasetVec.clear();
    m_bwcStaticApiDatasetVec.resize(m_nItrsPerStrm);
    m_bwcPipelineVec.clear();
    m_bwcPipelineVec.resize(m_nItrsPerStrm);

    for(int32_t iSlot = 0; iSlot < m_nItrsPerStrm; ++iSlot)
    {
        std::vector<std::string> inFileNamesBFCSlot(inFileNamesBFC.begin() + iSlot * m_nBWCCells, inFileNamesBFC.begin() + (iSlot + 1) * m_nBWCCells);
        m_bwcStaticApiDatasetVec[iSlot].emplace_back(inFileNamesBFCSlot, m_cuStrmsPdsch[0].handle());
        m_bwcPipelineVec[iSlot].emplace_back(m_bwcStaticApiDatasetVec[iSlot][0].bfwStatPrms, m_cuStrmsPdsch[0].handle());
    }
    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdsch[0].handle()));

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_BFC_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::bfcSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: BFCSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesBFC = initMsgPayload.inFileNames;
    // resize

    uint64_t procModeBmsk = m_pdsch_proc_mode;

    m_bwcDynamicApiDatasetVec.clear();
    m_bwcEvalDatasetVec.clear();
    m_bwcDynamicApiDatasetVec.resize(m_nItrsPerStrm);
    m_bwcEvalDatasetVec.resize(m_nItrsPerStrm);

    for(int32_t iSlot = 0; iSlot < m_nItrsPerStrm; ++iSlot)
    {
        //HERE
        std::vector<std::string> inFileNamesBFCSlot(inFileNamesBFC.begin() + iSlot * m_nBWCCells, inFileNamesBFC.begin() + (iSlot + 1) * m_nBWCCells);

        m_bwcDynamicApiDatasetVec[iSlot].emplace_back(inFileNamesBFCSlot, m_cuStrmsPdsch[0].handle(), procModeBmsk);
        m_bwcEvalDatasetVec[iSlot].emplace_back(inFileNamesBFCSlot, m_cuStrmsPdsch[0].handle());
        m_bwcPipelineVec[iSlot][0].setup(m_bwcDynamicApiDatasetVec[iSlot][0].bfwDynPrms);
    }

    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdsch[0].handle()));

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_BFC_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::srsSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: srsSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSRS = initMsgPayload.inFileNames;

    uint32_t nSRSStrms = 1;
    uint32_t nCells1   = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;

    std::vector<std::string> inFileNamesSRS1(inFileNamesSRS.begin(), inFileNamesSRS.begin() + nCells1);
    std::vector<std::string> inFileNamesSRS2(inFileNamesSRS.begin() + nCells1, inFileNamesSRS.end());

    m_srsDynamicApiDatasetVec.clear();
    m_srsDynamicApiDatasetVec.resize(1);
    m_srsEvalDatasetVec.clear();
    m_srsEvalDatasetVec.resize(1);

    m_srsDynamicApiDatasetVec[0].emplace_back(inFileNamesSRS1, m_cuStrmsSRS[0].handle(), m_srs_proc_mode);
    m_srsEvalDatasetVec[0].emplace_back(inFileNamesSRS1, m_cuStrmsSRS[0].handle());

    if(m_runSRS2)
    {
        m_srsEvalDatasetVec2.clear();
        m_srsEvalDatasetVec2.resize(1);
        m_srsDynamicApiDatasetVec2.clear();
        m_srsDynamicApiDatasetVec2.resize(1);
        m_srsDynamicApiDatasetVec2[0].emplace_back(inFileNamesSRS2, m_cuStrmsSRS[0].handle());
        m_srsEvalDatasetVec2[0].emplace_back(inFileNamesSRS2, m_cuStrmsSRS[0].handle());
    }

    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsSRS[0].handle()));
    cuphySrsBatchPrmHndl_t const batchPrmHndl = nullptr; // batchPrms currently un-used

    if(m_ref_check_srs)
    {
        m_srsDynamicApiDatasetVec[0][0].srsDynPrm.cpuCopyOn = 1;
    }
    else
    {
        m_srsDynamicApiDatasetVec[0][0].srsDynPrm.cpuCopyOn = 0;
    }
    cuphyStatus_t statusSetup = cuphySetupSrsRx(m_srsRxHndl, &m_srsDynamicApiDatasetVec[0][0].srsDynPrm, batchPrmHndl);
    if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
    if(m_runSRS2)
    {
        if(m_ref_check_srs)
        {
            m_srsDynamicApiDatasetVec2[0][0].srsDynPrm.cpuCopyOn = 1;
        }
        else
        {
            m_srsDynamicApiDatasetVec2[0][0].srsDynPrm.cpuCopyOn = 0;
        }
        statusSetup = cuphySetupSrsRx(m_srsRxHndl2, &m_srsDynamicApiDatasetVec2[0][0].srsDynPrm, batchPrmHndl);
        if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::prachSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: PRACHSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<cuPHYTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPRACH = initMsgPayload.inFileNames;

    // resize

    if(inFileNamesPRACH.size() == 0)
        return;

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_prach; ++strmIdx)
    {
        // loop over stream iterations
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            cuphyPrachDynPrms_t& prachDynPrms = m_prachDatasetVec[strmIdx][itrIdx].prachDynPrms;
            m_prachRxPipes[strmIdx][itrIdx].setup(prachDynPrms);
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPrach[strmIdx].handle()));
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::deinitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: deinitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    cuPHYTestDeinitMsgPayload const& deinitMsgPayload = *std::static_pointer_cast<cuPHYTestDeinitMsgPayload>(shPtrPayload);

    // clean-up PRACH
    // We need an explcit clean-up here and not rely on implicit clean-up in destructor.
    // Destructor is called in main thread with CUDA primary context whereas PRACH pipeline and associated CUDA resources 
    // are created with CUDA sub-context in worker thread. We need to ensure CUDA resources are freed-up in same context  
    // in which they were created. Bug: http://nvbugs/3612084
    if(m_runPRACH)
    {
        m_prachRxPipes.clear();
        m_prachDatasetVec.clear();
    }

    if(deinitMsgPayload.rsp)
    {
        // Send deinit done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_DEINIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::exitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: exitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    cuPHYTestExitMsgPayload const& exitMsgPayload = *std::static_pointer_cast<cuPHYTestExitMsgPayload>(shPtrPayload);
    if(exitMsgPayload.rsp)
    {
        // Send deinit done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_EXIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::puschRxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: puschRxSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPuschRxSetupMsgPayload&      puschRxSetupMsgPayload = *std::static_pointer_cast<cuPHYTestPuschRxSetupMsgPayload>(shPtrPayload);
    std::vector<std::string>              inFileNamesPuschRx     = puschRxSetupMsgPayload.inFileNamesPuschRx;
    bool                                  rsp                    = puschRxSetupMsgPayload.rsp;
    std::vector<std::string>              tempInput;

    if(m_pusch_group_cells)
    {
        tempInput.resize(inFileNamesPuschRx.size() / m_nStrms);
    }
    else
        tempInput.resize(1);

    if(m_runPUSCH)
    {
        // reset
        m_puschRxDynamicApiDataSets.clear();
        m_puschRxDynamicApiDataSets.resize(m_nStrms);
        m_puschRxEvalDataSets.clear();
        m_puschRxEvalDataSets.resize(m_nStrms);

        cuphyPuschBatchPrmHndl_t batchPrmHndl = nullptr;
        // loop over streams
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            m_puschRxDynamicApiDataSets[strmIdx].reserve(m_nItrsPerStrm);
            m_puschRxEvalDataSets[strmIdx].reserve(m_nItrsPerStrm);

            // Loop over iterations
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                uint32_t cellIdx = strmIdx + itrIdx * m_nStrms;

                // Load datasets
                if(m_pusch_group_cells)
                {
                    tempInput.assign(inFileNamesPuschRx.begin() + itrIdx * (inFileNamesPuschRx.size() / m_nStrms), inFileNamesPuschRx.begin() + (itrIdx + 1) * (inFileNamesPuschRx.size() / m_nStrms));
                    m_puschRxDynamicApiDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle(), m_pusch_proc_mode, false, m_fp16Mode);
                    m_puschRxEvalDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle());
                }
                else
                {
                    std::vector<std::string> v;
                    v.push_back(inFileNamesPuschRx[cellIdx]);
                    m_puschRxDynamicApiDataSets[strmIdx].emplace_back(v, m_cuStrmsPusch[strmIdx].handle(), m_pusch_proc_mode, false, m_fp16Mode);
                    m_puschRxEvalDataSets[strmIdx].emplace_back(v, m_cuStrmsPusch[strmIdx].handle());
                }
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));

                // initialize pipeline - phase 1
                m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_1;
                m_puschRxPipes[strmIdx][itrIdx].setup(m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm, batchPrmHndl);

                // Allocate HARQ buffers based on the calculated requirements from setupPhase 1
                m_puschRxDynamicApiDataSets[strmIdx][itrIdx].EasyAllocHarqBuffers(m_cuStrmsPusch[strmIdx].handle());

                // initialize pipeline - phase 2
                m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_2;
                m_puschRxPipes[strmIdx][itrIdx].setup(m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm, batchPrmHndl);
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));
            }
        }
    }
    if(puschRxSetupMsgPayload.rsp)
    {
        // Send setup done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pdschTxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPdschTxSetupMsgPayload& pdschTxSetupMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxSetupMsgPayload>(shPtrPayload);
    std::vector<std::string>         inFileNamesPdschTx     = pdschTxSetupMsgPayload.inFileNamesPdschTx;
    bool                             rsp                    = pdschTxSetupMsgPayload.rsp;

    if(m_runPDSCH)
    {
        // reset
        m_pdschTxDynamicApiDataSets.clear();
        m_pdschTxDynamicApiDataSets.resize(m_nStrms_pdsch);

#ifdef ENABLE_F01_STREAM_PRIO
        std::vector<cuphy::stream>& cuStrmsPdsch = m_cuStrmsPdsch;
#else
        std::vector<cuphy::stream>& cuStrmsPdsch = (4 != m_uldlMode) ? m_cuStrmsPdsch : m_cuStrms;
#endif // ENABLE_F01_STREAM_PRIO

        cuphyPdschBatchPrmHndl_t batchPrmHndl = nullptr;
        // loop over streams
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
        {
            m_pdschTxDynamicApiDataSets[strmIdx].reserve(m_nItrsPerStrm); // Note: pdsch datasets need to be resereved to prevent segFault.
            // Loop over iterations
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                cuphy::pdsch_tx* pdschTxPipe = &m_pdschTxPipes[strmIdx][itrIdx];

                // loop over all cells that need to be processed per pipeline in a single iteration.
                // This number should be 1, unless the m_pdsch_group_cells is true.
                // FIXME Assuming that when m_nCellsPerStrm_pdsch != 1, m_nStrms_pdsch is 1. Confirm this holds for all uldl use cases.
                for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdsch; ++cellPerStrmIdx)
                {
                    uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdsch * m_nCellsPerStrm_pdsch + cellPerStrmIdx;
                    /*printf("PDSCH setup handler, strmIdx %d, itrIdx %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       cellIdx,
                       inFileNamesPdschTx[cellIdx].c_str());*/

                    // Load datasets
                    //if(cellPerStrmIdx == 0)
                    if(cellPerStrmIdx == 0)
                    {
                        m_pdschTxDynamicApiDataSets[strmIdx].emplace_back(inFileNamesPdschTx[cellIdx], m_nCellsPerStrm_pdsch, cuStrmsPdsch[strmIdx].handle(), m_pdsch_proc_mode, m_pdschTxStaticApiDataSets[strmIdx][itrIdx].pdschStatPrms);
                    }
                    else
                    {
                        //Cumulatively update the dynamic parameters for a given pipeline, iteration
                        m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPdschTx[cellIdx], cuStrmsPdsch[strmIdx].handle(), m_pdsch_proc_mode);
                    }

                    // Load datasets
                    // Update filename for ref_check
                    // Workaround as the filename used for ref. checks is stored in the static parameters. TODO change this in the future
                    updateFileNameMultipleCells(pdschTxPipe->handle(), cellPerStrmIdx, inFileNamesPdschTx[cellIdx].c_str());
                }
                CUDA_CHECK(cudaStreamSynchronize(cuStrmsPdsch[strmIdx].handle()));

                // DBG print dynamic parameters
                //m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].print();

                // setup pipeline
                pdschTxPipe->setup(m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].pdsch_dyn_params, batchPrmHndl);

                // reset output buffer
                m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].resetOutputTensors(cuStrmsPdsch[strmIdx].handle());
                CUDA_CHECK(cudaStreamSynchronize(cuStrmsPdsch[strmIdx].handle()));
            }
        }
    }
    if(pdschTxSetupMsgPayload.rsp)
    {
        // Send setup done response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::runPUCCH(const cudaEvent_t& startEvent)
{
#if USE_NVTX
    nvtxRangePush("PUCCH");
#endif
    // wait for start event

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), startEvent, 0));

    for(uint32_t strmIdx = 1; strmIdx < m_nStrms_pucch; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[strmIdx].handle(), startEvent, 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHStartEvent->handle(), m_cuStrmsPucch[0].handle()));

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pucch; ++strmIdx)
        {
            // run
            uint64_t procModeBmsk = 0; // procModeBmsk currently un-used
            m_pucchRxPipes[strmIdx][itrIdx].run(procModeBmsk);

            if(m_ref_check_pucch)
            {
                cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
                int pucchErrors = m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

                if(pucchErrors != 0)
                {
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "PUCCH reference checks: {} errors", pucchErrors);
                    throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                }
                else
                {
                    printf("PUCCH REFERENCE CHECK: PASSED!\n");
                }
            }

            // synch
            if(strmIdx != 0)
            {
                m_PUCCHStopEvents[strmIdx].record(m_cuStrmsPucch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), m_PUCCHStopEvents[strmIdx].handle(), 0));
            }
        }
    }
    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHEndEvent->handle(), m_cuStrmsPucch[0].handle()));
    m_PUCCHStopEvents[0].record(m_cuStrmsPucch[0].handle());

#if USE_NVTX
    nvtxRangePop();
#endif
}

void cuPHYTestWorker::runPUCCH_U5(const cudaEvent_t& startEvent, const cudaEvent_t& startEvent2)
{
    uint32_t pucch2SyncStrmId = m_nStrms_pucch / 2;
    uint64_t procModeBmsk     = 0;

#if USE_NVTX
    nvtxRangePush("PUCCH1");
#endif
    // delay of PUCCH in B1
    if(m_longPattern == 5)
    {
        gpu_us_delay(516, 0, m_cuStrmsPucch[0].handle(), 1);
    }
    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), startEvent, 0));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHStartEvent->handle(), m_cuStrmsPucch[0].handle()));
    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[pucch2SyncStrmId].handle(), startEvent, 0));

    // PUCCH1
    for(uint32_t strmIdx = 1; strmIdx < pucch2SyncStrmId; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[strmIdx].handle(), startEvent, 0));
    }

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations

    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = 0; strmIdx < pucch2SyncStrmId; ++strmIdx)
        {
            // run
            m_pucchRxPipes[strmIdx][itrIdx].run(procModeBmsk);

            // cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
            //  m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

            if(m_ref_check_pucch)
            {
                cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
                int pucchErrors = m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

                if(pucchErrors != 0)
                {
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "PUCCH reference checks: {} errors", pucchErrors);
                    throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                }
                else
                {
                    printf("PUCCH REFERENCE CHECK: PASSED!\n");
                }
            }

            // synch
            if(strmIdx != 0)
            {
                m_PUCCHStopEvents[strmIdx].record(m_cuStrmsPucch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), m_PUCCHStopEvents[strmIdx].handle(), 0));
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHEndEvent->handle(), m_cuStrmsPucch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_PUCCHStopEvents[0].handle(), m_cuStrmsPucch[0].handle()));

#if USE_NVTX
    nvtxRangePop();
#endif
    // PUCCH2
    // wait for delay

#if USE_NVTX
    nvtxRangePush("PUCCH2");
#endif
    for(uint32_t strmIdx = pucch2SyncStrmId; strmIdx < m_nStrms_pucch; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[strmIdx].handle(), startEvent2, 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCH2StartEvent->handle(), m_cuStrmsPucch[pucch2SyncStrmId].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = pucch2SyncStrmId; strmIdx < m_nStrms_pucch; ++strmIdx)
        {
            // run
            m_pucchRxPipes[strmIdx][itrIdx].run(procModeBmsk);

            // cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
            // m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

            // synch
            if(strmIdx != pucch2SyncStrmId)
            {
                m_PUCCHStop2Events[strmIdx - pucch2SyncStrmId].record(m_cuStrmsPucch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[pucch2SyncStrmId].handle(), m_PUCCHStop2Events[strmIdx - pucch2SyncStrmId].handle(), 0));
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCH2EndEvent->handle(), m_cuStrmsPucch[pucch2SyncStrmId].handle()));
    CUDA_CHECK(cudaEventRecord(m_PUCCHStop2Events[0].handle(), m_cuStrmsPucch[pucch2SyncStrmId].handle()));
    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), m_PUCCHStop2Events[0].handle(), 0));
#if USE_NVTX
    nvtxRangePop();
#endif
}
void cuPHYTestWorker::runPRACH(const cudaEvent_t& startEvent)
{
#if USE_NVTX
    nvtxRangePush("PRACH");
#endif
    // wait for start event

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPrach[0].handle(), startEvent, 0));
    for(uint32_t strmIdx = 1; strmIdx < m_nStrms_prach; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPrach[strmIdx].handle(), startEvent, 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePRACHStartEvent->handle(), m_cuStrmsPrach[0].handle()));
    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_prach; ++strmIdx)
        {
            // run
            m_prachRxPipes[strmIdx][itrIdx].run();

            // synch
            if(strmIdx != 0)
            {
                m_PRACHStopEvents[strmIdx].record(m_cuStrmsPrach[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPrach[0].handle(), m_PRACHStopEvents[strmIdx].handle(), 0));
            }
        }

        if(m_ref_check_prach)
        {
            int errors = 0;
            for(int i = 0; i < m_nStrms_prach; i++)
            {
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPrach[i].handle()));
                errors += m_prachDatasetVec[i][itrIdx].evaluateOutput();
            }

            if(errors != 0)
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "PRACH reference checks: {} errors", errors);
                throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
            }
            else
            {
                NVLOGI_FMT(NVLOG_PRACH, "PRACH REFERENCE CHECK: PASSED");
            }
        }
    }
    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePRACHEndEvent->handle(), m_cuStrmsPrach[0].handle()));
    m_PRACHStopEvents[0].record(m_cuStrmsPrach[0].handle());

#if USE_NVTX
    nvtxRangePop();
#endif
}

void cuPHYTestWorker::runSRS1(const cudaEvent_t& startEvent)
{
#if USE_NVTX
    nvtxRangePush("SRS1");
#endif

    uint32_t nSRSStrms = 1;

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSRS[0].handle(), startEvent, 0));
    
    // SRS always start after 501
    gpu_us_delay(501, m_gpuId, m_cuStrmsSRS[0].handle(), 1);
    
    CUDA_CHECK(cudaEventRecord(m_uqPtrSRSDelayStopEvent->handle(), m_cuStrmsSRS[0].handle()));


    // wait for start event
    for(uint32_t strmIdx = 0; strmIdx < nSRSStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSRS[strmIdx].handle(), m_uqPtrSRSDelayStopEvent->handle(), 0));
    }
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRSStartEvent->handle(), m_cuStrmsSRS[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = 0; strmIdx < nSRSStrms; ++strmIdx)
        {
            // start timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPusch[strmIdx].handle());

            // run
            uint64_t procModeBmsk = 0; // procModeBmsk currently un-used

            cuphyStatus_t statusRun = cuphyRunSrsRx(m_srsRxHndl, procModeBmsk);
            if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

            // end timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_end(m_cuStrmsPusch[strmIdx].handle());

            // synch
            if(strmIdx != 0)
            {
                m_SRSStopEvents[strmIdx].record(m_cuStrmsSRS[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSRS[0].handle(), m_SRSStopEvents[strmIdx].handle(), 0));
            }
            if(m_ref_check_srs)
            {
                cudaStreamSynchronize(m_cuStrmsSRS[strmIdx].handle());
                m_srsEvalDatasetVec[strmIdx][itrIdx].evalSrsRx(m_srsDynamicApiDatasetVec[strmIdx][itrIdx].srsDynPrm, m_srsDynamicApiDatasetVec[strmIdx][itrIdx].tSrsChEstVec, m_srsDynamicApiDatasetVec[strmIdx][itrIdx].dataOut.pRbSnrBuffer, m_srsDynamicApiDatasetVec[strmIdx][itrIdx].dataOut.pSrsReports, m_cuStrmsSRS[strmIdx].handle());
            }
        }
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrSRSStopEvent->handle(), m_cuStrmsSRS[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRSEndEvent->handle(), m_cuStrmsSRS[0].handle()));

#if USE_NVTX
    nvtxRangePop();
#endif
}

void cuPHYTestWorker::runPUSCH_U5(const cudaEvent_t& startEvent)
{
    uint32_t pusch2SyncStrmId = m_nStrms / 2;
    uint64_t procModeBmsk     = m_pusch_proc_mode;
    uint32_t delay1Ms;
    uint32_t delay2Ms;

#if USE_NVTX
    nvtxRangePush("PUSCH1");
#endif

    switch(m_longPattern)
    {
    case 1:
        delay1Ms = 0;
        delay2Ms = 500;
        break;
    case 2:
        delay1Ms = 0;
        delay2Ms = 500;
        break;
    case 3:
        delay1Ms = 500;
        delay2Ms = 1000;
        break;
    case 4:
        delay1Ms = 0;
        delay2Ms = 0;
        break;
    case 5:
        delay1Ms = 516;
        delay2Ms = 0;
        break;
    case 6:
        delay1Ms = 1016;
        delay2Ms = 0;
        break;
    default:
        delay1Ms = 0;
        delay2Ms = 0;
    }

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[pusch2SyncStrmId].handle(), startEvent, 0));
    if(delay1Ms)
        gpu_us_delay(delay1Ms, 0, m_cuStrmsPusch[0].handle(), 1);
    if(delay2Ms)
        gpu_us_delay(delay2Ms, 0, m_cuStrmsPusch[pusch2SyncStrmId].handle(), 1);
    CUDA_CHECK(cudaEventRecord(m_uqPtrPuschDelayStopEvent->handle(), m_cuStrmsPusch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_uqPtrPusch2DelayStopEvent->handle(), m_cuStrmsPusch[pusch2SyncStrmId].handle()));
    // PUSCH1
    // wait for delay
    for(uint32_t strmIdx = 0; strmIdx < pusch2SyncStrmId; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrPuschDelayStopEvent->handle(), 0));
        // Uncomment line below to force PUSCH to wait for SRS to be finished
        //    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrSRSStopEvent->handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHStartEvent->handle(), m_cuStrmsPusch[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = 0; strmIdx < pusch2SyncStrmId; ++strmIdx)
        {
            // start timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPusch[strmIdx].handle());
            // run
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_PHASE_3);

            // end timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_end(m_cuStrmsPusch[strmIdx].handle());

            // synch
            if(strmIdx != 0)
            {
                m_stopEvents[strmIdx].record(m_cuStrmsPusch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHEndEvent->handle(), m_cuStrmsPusch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_stopEvents[0].handle(), m_cuStrmsPusch[0].handle()));

#if USE_NVTX
    nvtxRangePop();
#endif
    // PUSCH2
    // wait for delay

#if USE_NVTX
    nvtxRangePush("PUSCH2");
#endif
    for(uint32_t strmIdx = pusch2SyncStrmId; strmIdx < m_nStrms; ++strmIdx)
    {
        if(m_longPattern < 4)
        {
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrPusch2DelayStopEvent->handle(), 0));
        }
        else
        {
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_stopEvents[0].handle(), 0));
        }
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCH2StartEvent->handle(), m_cuStrmsPusch[pusch2SyncStrmId].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = pusch2SyncStrmId; strmIdx < m_nStrms; ++strmIdx)
        {
            // start timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPusch[strmIdx].handle());
            // run
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_PHASE_3);

            // end timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_end(m_cuStrmsPusch[strmIdx].handle());

            // synch
            if(strmIdx != pusch2SyncStrmId)
            {
                m_stop2Events[strmIdx - pusch2SyncStrmId].record(m_cuStrmsPusch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[pusch2SyncStrmId].handle(), m_stop2Events[strmIdx - pusch2SyncStrmId].handle(), 0));
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCH2EndEvent->handle(), m_cuStrmsPusch[pusch2SyncStrmId].handle()));
    CUDA_CHECK(cudaEventRecord(m_stop2Events[0].handle(), m_cuStrmsPusch[pusch2SyncStrmId].handle()));
    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[0].handle(), m_stop2Events[0].handle(), 0));
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPusch[0].handle()));
#if USE_NVTX
    nvtxRangePop();
#endif
}
void cuPHYTestWorker::runPUSCH(const cudaEvent_t& startEvent)
{
    uint64_t procModeBmsk = m_pusch_proc_mode;
#if USE_NVTX
    nvtxRangePush("PUSCH");
#endif
    gpu_us_delay(500, 0, m_cuStrmsPusch[0].handle(), 1);
    CUDA_CHECK(cudaEventRecord(m_uqPtrPuschDelayStopEvent->handle(), m_cuStrmsPusch[0].handle()));

    // wait for delay
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrPuschDelayStopEvent->handle(), 0));
        // Uncomment line below to force PUSCH to wait for SRS to be finished
        //    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrSRSStopEvent->handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHStartEvent->handle(), m_cuStrmsPusch[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            // start timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPusch[strmIdx].handle());

            // run
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_PHASE_3);
            // end timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_end(m_cuStrmsPusch[strmIdx].handle());

            // synch
            if(strmIdx != 0)
            {
                m_stopEvents[strmIdx].record(m_cuStrmsPusch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_stopEvents[strmIdx].handle(), 0));
            }
        }
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHEndEvent->handle(), m_cuStrmsPusch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPusch[0].handle()));
#if USE_NVTX
    nvtxRangePop();
#endif
}

void cuPHYTestWorker::runSRS2()
{
    //Run second SRS

    uint32_t nSRS2Strms = 1;
    // wait for start event

#if USE_NVTX
    nvtxRangePush("SRS2");
#endif
    for(uint32_t strmIdx = 0; strmIdx < nSRS2Strms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSRS[strmIdx].handle(), m_shPtrStopEvent->handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRS2StartEvent->handle(), m_cuStrmsSRS[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently
    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        for(uint32_t strmIdx = 0; strmIdx < nSRS2Strms; ++strmIdx)
        {
            // start timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPusch[strmIdx].handle());

            // run

            uint64_t procModeBmsk = 0; // procModeBmsk currently un-used

            cuphyStatus_t statusRun = cuphyRunSrsRx(m_srsRxHndl2, procModeBmsk);
            if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

            // end timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_end(m_cuStrmsPusch[strmIdx].handle());

            // synch
            if(strmIdx != 0)
            {
                m_SRSStopEvents[strmIdx].record(m_cuStrmsSRS[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSRS[0].handle(), m_SRSStopEvents[strmIdx].handle(), 0));
            }
            if(m_ref_check_srs)
            {
                cudaStreamSynchronize(m_cuStrmsSRS[strmIdx].handle());
                m_srsEvalDatasetVec2[strmIdx][itrIdx].evalSrsRx(m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].srsDynPrm, m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].tSrsChEstVec, m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].dataOut.pRbSnrBuffer, m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].dataOut.pSrsReports, m_cuStrmsSRS[strmIdx].handle());
            }
        }
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrSRS2StopEvent->handle(), m_cuStrmsSRS[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRS2EndEvent->handle(), m_cuStrmsSRS[0].handle()));
#if USE_NVTX
    nvtxRangePop();
#endif
}

void cuPHYTestWorker::puschRxRunHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: puschRxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPuschRxRunMsgPayload& puschRxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPuschRxRunMsgPayload>(shPtrPayload);

    cuphy::stream* syncStrm = nullptr;

    if(m_runPUSCH)
        syncStrm = &m_cuStrmsPusch[0];
    else if(m_runPRACH)
        syncStrm = &m_cuStrmsPrach[0];
    else if(m_runSRS)
        syncStrm = &m_cuStrmsSRS[0];
    else if(m_runPUCCH)
        syncStrm = &m_cuStrmsPucch[0];

    if(m_runPUSCH || m_runPRACH || m_runSRS || m_runPUCCH)
    {
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), puschRxRunMsgPayload.startEvent, 0));

        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), syncStrm->handle()));

        if(m_runSRS)
            runSRS1(puschRxRunMsgPayload.startEvent);

        if(m_runPUSCH)
        {
            if(m_longPattern)
            {
                runPUSCH_U5(puschRxRunMsgPayload.startEvent);
            }
            else
                runPUSCH(puschRxRunMsgPayload.startEvent);
        }

        if(m_runPUCCH)
        {
            if(m_runPUSCH)
            {
                if(m_longPattern)
                    runPUCCH_U5(getPuschStartEvent() ? getPuschStartEvent() : puschRxRunMsgPayload.startEvent, getPusch2StartEvent());
                else

                    runPUCCH(getPuschStartEvent());
            }
            else
            {
                if(m_longPattern)
                    runPUCCH_U5(puschRxRunMsgPayload.pucchStartEvent ? puschRxRunMsgPayload.pucchStartEvent : puschRxRunMsgPayload.startEvent, puschRxRunMsgPayload.prachStartEvent ? puschRxRunMsgPayload.prachStartEvent : puschRxRunMsgPayload.startEvent);
                else
                    runPUCCH(puschRxRunMsgPayload.pucchStartEvent ? puschRxRunMsgPayload.pucchStartEvent : puschRxRunMsgPayload.startEvent);
            }
        }

        if(m_runPRACH)
        {
            if(m_runPUSCH)
            {
                runPRACH(getPusch2StartEvent());
            }
            else
            {
                runPRACH(puschRxRunMsgPayload.prachStartEvent ? puschRxRunMsgPayload.prachStartEvent : puschRxRunMsgPayload.startEvent);
            }
        }
    }
    if(m_runSRS2)
        runSRS2();

    if(m_runPRACH)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_PRACHStopEvents[0].handle(), 0));
    if(m_runPUCCH)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_PUCCHStopEvents[0].handle(), 0));
    if(m_runSRS)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_uqPtrSRSStopEvent->handle(), 0));
    if(m_runSRS2)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_uqPtrSRS2StopEvent->handle(), 0));

    if(syncStrm) // if any stream is available, otherwise no need to record event for measuring timing
    {
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), syncStrm->handle()));
    }

    // send CPU response message
    if(puschRxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPuschRxRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pschTxRxRunHandlerNoStrmPrio(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pschTxRxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPschTxRxRunMsgPayload& pschTxRxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPschTxRxRunMsgPayload>(shPtrPayload);

    // wait for start event
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrms[strmIdx].handle(), pschTxRxRunMsgPayload.startEvent, 0));
    }
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrms[0].handle()));

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently
    //printf("m_nStrms %d, m_nItrsPerStrm %d, PDSCH strms %d\n", m_nStrms, m_nItrsPerStrm, m_nStrms_pdsch);

    // Loop over stremas
    // FIXME: use default stream to record end of PDSCH
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            // start timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrms[strmIdx].handle());

            // run
            if(strmIdx < m_nStrms_pdsch)
            {
                m_pdschTxPipes[strmIdx][itrIdx].run(m_pdsch_proc_mode);
            }
            m_stop2Events[strmIdx].record(m_cuStrms[strmIdx].handle());
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_PHASE_3);

            // end timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_end(m_cuStrms[strmIdx].handle());
        }

        // synch
        if(strmIdx != 0)
        {
            m_stopEvents[strmIdx].record(m_cuStrms[strmIdx].handle());
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrms[0].handle(), m_stopEvents[strmIdx].handle(), 0));
        }

        CUDA_CHECK(cudaStreamWaitEvent(0, m_stop2Events[strmIdx].handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_timePdschSlotEndEvents[0].handle(), 0));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHEndEvent->handle(), m_cuStrms[0].handle()));
    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrms[0].handle()));

    // send CPU response message
    if(pschTxRxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPschRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::runPDCCHItr(const cudaEvent_t& pdcchSlotStartEvent, uint32_t itrIdx)
{
    //printf("runPDCCHItr m_nPDCCHCells %d vs. m_nStrms_pdcch %d\n", m_nPDCCHCells, m_nStrms_pdcch);
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdcch; ++strmIdx)
    {
        // hold all the streams (i.e cells) until startEvent for the slot
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[strmIdx].handle(), pdcchSlotStartEvent, 0));

        if(strmIdx == 0) // if PDCCH is run standalone record start event here
        {
            CUDA_CHECK(cudaEventRecord(m_timeNextPdcchSlotStartEvents[itrIdx].handle(), m_cuStrmsPdcch[0].handle()));
        }
        // run iteration and time
        // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPdsch[strmIdx].handle());

        m_pdcchTxPipes[strmIdx][itrIdx].run(0 /* unused proc. bitmask */);

        if(m_ref_check_pdcch)
        {
            // Providing no argument to refCheck (default is verbose=false) results in no prints during ref. checks
            int err = m_pdcchTxDynamicApiDataSets[strmIdx][itrIdx].refCheck(true); //FIXME potentially move somewhere where it won't affect perf. Can call w/o arg.
            if(err != 0)
            {
                // NVLOG will log number of mismatch in pdcchDynApiDataSET::refCheck
                throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
            }
            else
            {
                printf("PDCCH REFERENCE CHECK: PASSED!\n");
            }
        }

        // record end
        if(strmIdx != 0)
        {
            m_PDCCHStopEvents[strmIdx].record(m_cuStrmsPdcch[strmIdx].handle());
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[0].handle(), m_PDCCHStopEvents[strmIdx].handle(), 0));
        }
    }
    CUDA_CHECK(cudaEventRecord(m_timePdcchSlotEndEvents[itrIdx].handle(), m_cuStrmsPdcch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_PDCCHStopEvents[0].handle(), m_cuStrmsPdcch[0].handle()));

    // CSIRS
    if(m_runCSIRS)
    {
        int nCSIRSObjects = m_pdsch_group_cells ? 1 : m_nCSIRSCells;
        for(uint32_t strmIdx = 0; strmIdx < nCSIRSObjects; ++strmIdx)
        {
            // hold all the streams (i.e cells) until startEvent for the slot
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsCSIRS[strmIdx].handle(), m_PDCCHStopEvents[0].handle(), 0));

            if(strmIdx == 0) // if PDCCH is run standalone record start event here
            {
                CUDA_CHECK(cudaEventRecord(m_timeNextCSIRSSlotStartEvents[itrIdx].handle(), m_cuStrmsPdcch[0].handle()));
            }
            // run iteration and time
            // RUN CSIRS
            m_csirsTxPipes[strmIdx][0].run();
            // record end

            if(m_ref_check_csirs)
            {
                // Providing no argument to refCheck (default is verbose=false) results in no prints during ref. checks
                int err = m_csirsTxDynamicApiDataSets[strmIdx][0].refCheck(/*true*/); //FIXME potentially move somewhere where it won't affect perf.
                if(err != 0)
                {
                    // NVLOG will log number of mismatch in csirsDynApiDataset::refCheck
                    throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                }
                else
                {
                    printf("CSIRS REFERENCE CHECK: PASSED!\n");
                }
            }

            if(strmIdx != 0)
            {
                m_CSIRSStopEvents[strmIdx].record(m_cuStrmsCSIRS[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsCSIRS[0].handle(), m_CSIRSStopEvents[strmIdx].handle(), 0));
            }
        }

        CUDA_CHECK(cudaEventRecord(m_timeCSIRSSlotEndEvents[itrIdx].handle(), m_cuStrmsCSIRS[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_CSIRSStopEvents[0].handle(), m_cuStrmsCSIRS[0].handle()));
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[0].handle(), m_CSIRSStopEvents[0].handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_pdcchInterSlotStartEventVec[itrIdx].handle(), m_cuStrmsPdcch[0].handle()));
}

void cuPHYTestWorker::runPDSCH_U5_3_6(std::shared_ptr<void>& shPtrPayload)
{
    // unpack message
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);
    uint64_t                       procModeBmsk         = m_pdsch_proc_mode;

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    //printf("m_runPDSCH %d, m_runBWC %d, m_nBWCCells %d, m_nStrms_pdsch %d\n", m_runPDSCH, m_runBWC, m_nBWCCells, m_nStrms_pdsch);

    if(m_runPDSCH)
    {
#if USE_NVTX
        nvtxRangePush("PDSCH+BWC");
#endif
        // find time slot duration in us
        auto      mu                 = m_pdschTxStaticApiDataSets[0].data()->pdschStatPrms.pCellStatPrms->mu;
        const int time_slot_duration = 1000 / (1 << mu);

        // DDDSUUDDDD, run one BWC first
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdsch[0].handle()));
        get_gpu_time(m_GPUtime_d.addr(), m_cuStrmsPdsch[0].handle()); // record start of time slot
        gpu_us_delay(375, 0, m_cuStrmsPdsch[0].handle(), 1);
        CUDA_CHECK(cudaEventRecord(m_timeNextPdschSlotStartEvents[0].handle(), m_cuStrmsPdsch[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[0].handle(), m_cuStrmsPdsch[0].handle()));
        for(uint32_t strmIdx = 0; strmIdx < 1; ++strmIdx)
        {
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_pdschInterSlotStartEventVec[0].handle(), 0));
            CUDA_CHECK(cudaEventRecord(m_timeBWCSlotStartEvents[0].handle(), m_cuStrmsPdsch[0].handle()));
            if(m_runBWC)
            {
                m_bwcPipelineVec[0][0].run(m_pdsch_proc_mode);
            }

            // record end
            if(strmIdx != 0)
            {
                m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
            }
        }

#if USE_NVTX
        nvtxRangePop();
#endif
        CUDA_CHECK(cudaEventRecord(m_timeBWCSlotEndEvents[0].handle(), m_cuStrmsPdsch[0].handle()));
        constexpr uint64_t NS_PER_US                     = 1000UL;
        uint64_t           gpu_slot_start_time_offset_ns = static_cast<uint64_t>(0 + 1) * (time_slot_duration * NS_PER_US);
        gpu_ns_delay_until(m_GPUtime_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
        get_gpu_time(m_GPUtime_d.addr(), m_cuStrmsPdsch[0].handle()); // record start of time slot
        CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[0].handle(), m_cuStrmsPdsch[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_timeNextPdschSlotStartEvents[1].handle(), m_cuStrmsPdsch[0].handle()));

        if(m_runBWC && m_ref_check_bfc)
        {
            float refCheckSnrThd = 30.0f;
            cudaStreamSynchronize(m_cuStrmsPdsch[0].handle());
            m_bwcEvalDatasetVec[0][0].bfwEvalCoefs(m_bwcDynamicApiDatasetVec[0][0], m_cuStrmsPdsch[0].handle(), refCheckSnrThd, true);
        }

        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
            {
                // hold all the streams (i.e cells) until startEvent for the slot
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_pdschInterSlotStartEventVec[itrIdx].handle(), 0));

                // run iteration and time
                // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPdsch[strmIdx].handle());

                m_pdschTxPipes[strmIdx][itrIdx].run(procModeBmsk);

                // record end
                if(strmIdx != 0)
                {
                    m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
                }
            }
            constexpr uint64_t NS_PER_US                     = 1000UL;
            uint64_t           gpu_slot_start_time_offset_ns = static_cast<uint64_t>(itrIdx + 1) * (time_slot_duration * NS_PER_US);

            CUDA_CHECK(cudaEventRecord(m_timePdschSlotEndEvents[itrIdx].handle(), m_cuStrmsPdsch[0].handle()));

            // wait for PDCCH
            if(m_runPDCCH)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_PDCCHStopEvents[0].handle(), 0));
            }
            else if(pdschTxRunMsgPayload.pdcchStopEventVec)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), (*pdschTxRunMsgPayload.pdcchStopEventVec)[itrIdx].handle(), 0));
            }

            CUDA_CHECK(cudaEventRecord(m_uqPtrPdschIterStopEvent->handle(), m_cuStrmsPdsch[0].handle()));

            if(itrIdx != m_nItrsPerStrm - 1)
            {
                if(m_runBWC)
                {
                    uint32_t bwcLongPatternOffset = 1;
                    CUDA_CHECK(cudaEventRecord(m_timeBWCSlotStartEvents[itrIdx + bwcLongPatternOffset].handle(), m_cuStrmsPdsch[0].handle()));

                    for(uint32_t strmIdx = 0; strmIdx < 1; ++strmIdx)
                    {
                        // wait for PDSCH cells to be processed
                        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_uqPtrPdschIterStopEvent->handle(), 0));
                        m_bwcPipelineVec[itrIdx + bwcLongPatternOffset][0].run(m_pdsch_proc_mode);

                        // record end
                        if(strmIdx != 0)
                        {
                            if(m_stopEvents.size() <= strmIdx)
                            { // Temp. printf to showcase segfault cause when --G is present
                                NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT,  "Error! strmIdx {} but vector size {}", strmIdx, m_stopEvents.size());
                                exit(1);
                            }
                            m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
                        }
                    }
                    CUDA_CHECK(cudaEventRecord(m_timeBWCSlotEndEvents[itrIdx + bwcLongPatternOffset].handle(), m_cuStrmsPdsch[0].handle()));
                    CUDA_CHECK(cudaEventRecord(m_uqPtrBWCStopEvent->handle(), m_cuStrmsPdsch[0].handle()));

                    if(m_ref_check_bfc)
                    {
                        float refCheckSnrThd = 30.0f;
                        cudaStreamSynchronize(m_cuStrmsPdsch[0].handle());
                        m_bwcEvalDatasetVec[itrIdx + bwcLongPatternOffset][0].bfwEvalCoefs(m_bwcDynamicApiDatasetVec[itrIdx + bwcLongPatternOffset][0], m_cuStrmsPdsch[0].handle(), refCheckSnrThd, true);
                    }
                }
            }
            gpu_ns_delay_until(m_GPUtime_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
            CUDA_CHECK(cudaEventRecord(m_timeNextPdschSlotStartEvents[itrIdx + 2].handle(), m_cuStrmsPdsch[0].handle()));

            if(m_runPDCCH)
            {
#if USE_NVTX
                nvtxRangePush("PDCCH");
#endif
                runPDCCHItr(m_pdschInterSlotStartEventVec[itrIdx].handle(), itrIdx);
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_PDCCHStopEvents[0].handle(), 0));
#if USE_NVTX
                nvtxRangePop();
#endif
            }

            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_uqPtrBWCStopEvent->handle(), 0));
            CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
        }

        // send GPU response message
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPdsch[0].handle()));
    }
}

void cuPHYTestWorker::runPDCCH(std::shared_ptr<void>& shPtrPayload)
{
    // unpack message
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);

#if USE_NVTX
    nvtxRangePush("PDCCH");
#endif

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdcch[0].handle()));
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        // hold all the streams (i.e cells) until startEvent for the slot
        if(itrIdx != 0)
        {
            if(pdschTxRunMsgPayload.pdschInterSlotEventVec) //PDCCH running in dedicated context alongside PDSCH
            {
                runPDCCHItr((*pdschTxRunMsgPayload.pdschInterSlotEventVec)[itrIdx].handle(), itrIdx);
            }
            else // PDCCH running standalone
            {
                runPDCCHItr(m_pdcchInterSlotStartEventVec[itrIdx - 1].handle(), itrIdx);
            }
        }
        else
        {
            if(pdschTxRunMsgPayload.pdschInterSlotEventVec) //PDCCH running in dedicated context alongside PDSCH
                runPDCCHItr((*pdschTxRunMsgPayload.pdschInterSlotEventVec)[0].handle(), itrIdx);
            else
                runPDCCHItr(pdschTxRunMsgPayload.startEvent, itrIdx);
        }
    }

    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPdcch[0].handle()));

#if USE_NVTX
    nvtxRangePop();
#endif
}

void cuPHYTestWorker::pdschTxRunHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    /*printf("pdschTxRunHandle with m_runPDSCH %d, m_run_PDCCH %d, m_long_pattern %d\n",
        m_runPDSCH, m_runPDCCH, m_longPattern);*/
    if(m_runPDSCH)
    {
        // if m_runPDCCH is set, the PDCCH will be run too in the runPDSCH_* calls below
        if(m_longPattern == 3 or m_longPattern == 6)
        {
            runPDSCH_U5_3_6(shPtrPayload);
        }
        else
            runPDSCH_U3_U5_1_2_4_5(shPtrPayload);
    }
    else if(m_runPDCCH || m_runCSIRS) // Only exercised when there's no PDSCH
    {
        runPDCCH(shPtrPayload);
    }
    else if(m_runSSB)
    {
        runSSB(shPtrPayload);
    }

    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);
    // send CPU response message
    if(pdschTxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPdschTxRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::runSSB(std::shared_ptr<void>& shPtrPayload)
{
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);

    if(m_runSSB)
    {
        int nSSBObjects = m_pdsch_group_cells ? 1 : m_nSSBCells;
#if USE_NVTX
        nvtxRangePush("SSB");
#endif
        // common starting point
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSSB[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsSSB[0].handle()));
        for(int ssbSlotIdx = 0; ssbSlotIdx < m_nSsbSlots; ssbSlotIdx ++)
        {
            for(int i = 0; i < nSSBObjects; i++)
            // if u 5, start at 4, 5, 6, 7 th slot
            {
                if(i != 0) CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSSB[i].handle(), pdschTxRunMsgPayload.startEvent, 0));

                // if u 5, start at 4th slot
                if(pdschTxRunMsgPayload.pdschInterSlotEventVec != nullptr)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSSB[i].handle(), (*pdschTxRunMsgPayload.pdschInterSlotEventVec)[4 + ssbSlotIdx].handle(), 0)); // 4 5 6 7 th slot
                }
            }

            CUDA_CHECK(cudaEventRecord(m_timeSSBSlotStartEvents[ssbSlotIdx].handle(), m_cuStrmsSSB[0].handle()));

            for(int i = 0; i < nSSBObjects; i++)
            {
                m_ssbTxPipes[i][0].run(0 /*ssb_proc_mode*/);

                if(m_ref_check_ssb)
                {
                    // Providing no argument to refCheck (default is verbose=false) results in no prints during ref. checks
                    int err = m_ssbTxDynamicApiDataSets[i][0].refCheck(/*true*/); //FIXME potentially move somewhere where it won't affect perf.
                    if(err != 0)
                    {
                        // NVLOG will log number of mismatch in ssbDynApiDataset::refCheck
                        throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                    }
                    else
                    {
                        NVLOGI_FMT(NVLOG_SSB, "SSB REFERENCE CHECK: PASSED");
                    }
                }
                // record end
                if(i != 0)
                {
                    m_stopEvents[i].record(m_cuStrmsSSB[i].handle());
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSSB[0].handle(), m_stopEvents[i].handle(), 0));
                }
            }

            CUDA_CHECK(cudaEventRecord(m_timeSSBSlotEndEvents[ssbSlotIdx].handle(), m_cuStrmsSSB[0].handle()));
           }
        // send GPU response message
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsSSB[0].handle()));
#if USE_NVTX
        nvtxRangePop();
#endif
    }
}
void cuPHYTestWorker::runPDSCH_U3_U5_1_2_4_5(std::shared_ptr<void>& shPtrPayload)
{
    // unpack message
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);
    uint64_t                       procModeBmsk         = m_pdsch_proc_mode;

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    //printf("m_runPDSCH %d, m_runBWC %d, m_nBWCCells %d, m_nStrms_pdsch %d\n", m_runPDSCH, m_runBWC, m_nBWCCells, m_nStrms_pdsch);

    if(m_runPDSCH)
    {
#if USE_NVTX
        nvtxRangePush("PDSCH+BWC"); //FIXME I don't think that includes BWC..
#endif
        // find time slot duration in us
        auto      mu                 = m_pdschTxStaticApiDataSets[0].data()->pdschStatPrms.pCellStatPrms->mu;
        const int time_slot_duration = 1000 / (1 << mu);

        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdsch[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[0].handle(), m_cuStrmsPdsch[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_timeNextPdschSlotStartEvents[0].handle(), m_cuStrmsPdsch[0].handle()));
        get_gpu_time(m_GPUtime_d.addr(), m_cuStrmsPdsch[0].handle()); // record start of time slot

        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            // DDDSUUDDDD, run one BWC first

            for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
            {
                // hold all the streams (i.e cells) until startEvent for the slot
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_pdschInterSlotStartEventVec[itrIdx].handle(), 0));

                // run iteration and time
                // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrmsPdsch[strmIdx].handle());

                m_pdschTxPipes[strmIdx][itrIdx].run(procModeBmsk);

                // record end
                if(strmIdx != 0)
                {
                    m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
                }
            }
            constexpr uint64_t NS_PER_US                     = 1000UL;
            uint64_t           gpu_slot_start_time_offset_ns = static_cast<uint64_t>(itrIdx + 1) * (time_slot_duration * NS_PER_US);

            CUDA_CHECK(cudaEventRecord(m_timePdschSlotEndEvents[itrIdx].handle(), m_cuStrmsPdsch[0].handle()));

            if(m_runPDCCH)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_PDCCHStopEvents[0].handle(), 0));
            }
            else if(pdschTxRunMsgPayload.pdcchStopEventVec)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), (*pdschTxRunMsgPayload.pdcchStopEventVec)[itrIdx].handle(), 0));
            }

            if((itrIdx == m_nItrsPerStrm - 1) && (m_longPattern || m_runBWC))
                gpu_ns_delay_until(m_GPUtime_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());

            if((itrIdx == m_nItrsPerStrm - 1) && (!m_longPattern) && m_runBWC)
            {
                CUDA_CHECK(cudaEventRecord(m_timeNextPdschSlotStartEvents[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
                gpu_us_delay(375, 0, m_cuStrmsPdsch[0].handle(), 1);
            }
            CUDA_CHECK(cudaEventRecord(m_uqPtrPdschIterStopEvent->handle(), m_cuStrmsPdsch[0].handle()));

            // only differnece beteeen -u3 and -u5 longPattern 1 2 4 5
            if((!m_longPattern))
            {
                if(m_runBWC)
                {
                    CUDA_CHECK(cudaEventRecord(m_timeBWCSlotStartEvents[itrIdx].handle(), m_cuStrmsPdsch[0].handle()));
                    for(uint32_t strmIdx = 0; strmIdx < 1; ++strmIdx)
                    {
                        // wait for PDSCH cells to be processed
                        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_uqPtrPdschIterStopEvent->handle(), 0));

                        m_bwcPipelineVec[itrIdx][0].run(m_pdsch_proc_mode);

                        // record end
                        if(strmIdx != 0)
                        {
                            if(m_stopEvents.size() <= strmIdx)
                            { // Temp. printf to showcase segfault cause when --G is present
                                NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT,  "Error! strmIdx {} but vector size {}", strmIdx, m_stopEvents.size());
                                exit(1);
                            }
                            m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
                        }
                    }

                    CUDA_CHECK(cudaEventRecord(m_uqPtrBWCStopEvent->handle(), m_cuStrmsPdsch[0].handle()));
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_uqPtrBWCStopEvent->handle(), 0));

                    CUDA_CHECK(cudaEventRecord(m_timeBWCSlotEndEvents[itrIdx].handle(), m_cuStrmsPdsch[0].handle()));

                    if(m_ref_check_bfc)
                    {
                        float refCheckSnrThd = 30.0f;
                        cudaStreamSynchronize(m_cuStrmsPdsch[0].handle());
                        m_bwcEvalDatasetVec[itrIdx][0].bfwEvalCoefs(m_bwcDynamicApiDatasetVec[itrIdx][0], m_cuStrmsPdsch[0].handle(), refCheckSnrThd, true);
                    }
                }
            }

            // use delay kernel to wait up to the start of the next time slot
            if(itrIdx != m_nItrsPerStrm - 1)
            {
                gpu_ns_delay_until(m_GPUtime_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
                CUDA_CHECK(cudaEventRecord(m_timeNextPdschSlotStartEvents[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
            }

            if(m_runPDCCH || m_runCSIRS)
            {
#if USE_NVTX
                nvtxRangePush("PDCCH/CSIRS");
#endif
                runPDCCHItr(m_pdschInterSlotStartEventVec[itrIdx].handle(), itrIdx);
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_PDCCHStopEvents[0].handle(), 0));
#if USE_NVTX
                nvtxRangePop();
#endif
            }

            CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
        }

        // send GPU response message
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPdsch[0].handle()));
#if USE_NVTX
        nvtxRangePop();
#endif
    }
}

void cuPHYTestWorker::pschTxRxRunHandlerStrmPrio(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pschTxRxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPschTxRxRunMsgPayload& pschTxRxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPschTxRxRunMsgPayload>(shPtrPayload);

    // wait for start event
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), pschTxRxRunMsgPayload.startEvent, 0));
    }
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdsch[0].handle()));

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently
    //printf("m_nStrms %d, m_nItrsPerStrm %d, PDSCH strms %d\n", m_nStrms, m_nItrsPerStrm, m_nStrms_pdsch);

    // Loop over stremas
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            // start timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_begin(m_cuStrms[strmIdx].handle());

            // run
            if(strmIdx < m_nStrms_pdsch)
            {
                m_pdschTxPipes[strmIdx][itrIdx].run(m_pdsch_proc_mode);
            }
            CUDA_CHECK(cudaEventRecord(m_pschDlUlSyncEvents[strmIdx][itrIdx].handle(), m_cuStrmsPdsch[strmIdx].handle()));

            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_pschDlUlSyncEvents[strmIdx][itrIdx].handle()));
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_PHASE_3);

            // end timer
            // m_timerSingleStreamItr[strmIdx][itrIdx].record_end(m_cuStrms[strmIdx].handle());
        }

        // synch
        if(strmIdx != 0)
        {
            m_stopEvents[strmIdx].record(m_cuStrmsPusch[strmIdx].handle());
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
        }
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPusch[0].handle()));

    // send CPU response message
    if(pschTxRxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPschRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pschTxRxRunHandler(std::shared_ptr<void>& shPtrPayload)
{
#ifdef ENABLE_F01_STREAM_PRIO
    pschTxRxRunHandlerStrmPrio(shPtrPayload);
#else
    pschTxRxRunHandlerNoStrmPrio(shPtrPayload);
#endif // ENABLE_F01_STREAM_PRIO
}

void cuPHYTestWorker::evalHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: evalHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    cuPHYTestEvalMsgPayload& evalMsgPayload = *std::static_pointer_cast<cuPHYTestEvalMsgPayload>(shPtrPayload);

    if(m_runPUSCH)
    {
        // update codeblock errors
        if(evalMsgPayload.cbErrors)
        {
            for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
            {
                for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
                {
#ifndef ENABLE_F01_STREAM_PRIO
                    if(m_uldlMode == 4)
                        m_cuStrms[strmIdx].synchronize();
#endif
                    m_cuStrmsPusch[strmIdx].synchronize();
                    m_puschRxPipes[strmIdx][itrIdx].writeDbgSynch(m_cuStrmsPusch[strmIdx].handle());
                    m_cuStrmsPusch[strmIdx].synchronize();
                    uint32_t numCbErrors = m_puschRxEvalDataSets[strmIdx][itrIdx].computeNumCbErrors(m_puschRxDynamicApiDataSets[strmIdx][itrIdx]);

                    m_maxNumCbErrors[strmIdx][itrIdx] = std::max(m_maxNumCbErrors[strmIdx][itrIdx], numCbErrors);
                }
            }
        }
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimePUSCHEndEvent->handle());
        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPUSCHRunTime += elapsedTimeMs;

        if(m_uldlMode != 4)
        {
            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_uqPtrTimePUSCHStartEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUSCHStartTime += elapsedTimeMs;
        }

        if(m_longPattern)
        {
            float       elapsedTimeMs = 0.0f;
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                                 m_uqPtrTimeStartEvent->handle(),
                                                 m_uqPtrTimePUSCH2EndEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUSCH2RunTime += elapsedTimeMs;

            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_uqPtrTimePUSCH2StartEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUSCH2StartTime += elapsedTimeMs;
        }
    }

    if(m_runPRACH)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimePRACHEndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPRACHRunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimePRACHStartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPRACHStartTime += elapsedTimeMs;
    }

    if(m_runPUCCH)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimePUCCHEndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPUCCHRunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimePUCCHStartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPUCCHStartTime += elapsedTimeMs;

        if(m_longPattern)
        {
            float       elapsedTimeMs = 0.0f;
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                                 m_uqPtrTimeStartEvent->handle(),
                                                 m_uqPtrTimePUCCH2EndEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUCCH2RunTime += elapsedTimeMs;

            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_uqPtrTimePUCCH2StartEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUCCH2StartTime += elapsedTimeMs;
        }
    }

    if(m_runSRS)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimeSRSEndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRSRunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimeSRSStartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRSStartTime += elapsedTimeMs;
    }

    if(m_runSRS2)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimeSRS2EndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRS2RunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimeSRS2StartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRS2StartTime += elapsedTimeMs;
    }

    if(m_runSSB)
    {
        float       elapsedTimeMs = 0.0f;
        for(int ssbSlotIdx = 0; ssbSlotIdx < m_nSsbSlots; ssbSlotIdx ++) // calculate SSB time among slots
        {
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_timeSSBSlotEndEvents[ssbSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totSSBRunTime[ssbSlotIdx] += elapsedTimeMs;

            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                    m_uqPtrTimeStartEvent->handle(),
                                    m_timeSSBSlotStartEvents[ssbSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totSSBStartTime[ssbSlotIdx] += elapsedTimeMs;
        }
    }

    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e;

        if(m_runPDSCH)
        {
            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timePdschSlotEndEvents[itrIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totRunTimePdschItr[itrIdx] += elapsedTimeMs;
            if(m_uldlMode != 4)
            {
                e = cudaEventElapsedTime(&elapsedTimeMs,
                                         m_uqPtrTimeStartEvent->handle(),
                                         m_timeNextPdschSlotStartEvents[itrIdx].handle());

                if(cudaSuccess != e) throw cuphy::cuda_exception(e);

                m_totPdschSlotStartTime[itrIdx] += elapsedTimeMs;
            }
        }

        if(m_runCSIRS)
        {
            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timeCSIRSSlotEndEvents[itrIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);
            m_totRunTimeCSIRSItr[itrIdx] += elapsedTimeMs;

            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timeNextCSIRSSlotStartEvents[itrIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totCSIRSStartTimes[itrIdx] += elapsedTimeMs;
        }

        if(m_runPDCCH)
        {
            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timePdcchSlotEndEvents[itrIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);
            m_totRunTimePdcchItr[itrIdx] += elapsedTimeMs;

            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timeNextPdcchSlotStartEvents[itrIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPdcchStartTimes[itrIdx] += elapsedTimeMs;
        }

        if(m_runBWC)
        {
            elapsedTimeMs = 0.0f;

            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timeBWCSlotStartEvents[itrIdx].handle());
            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totBWCIterStartTime[itrIdx] += elapsedTimeMs;

            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timeBWCSlotEndEvents[itrIdx].handle());
            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totRunTimeBWCItr[itrIdx] += elapsedTimeMs;
        }
    }

    // Handle last BWC slot
    if(m_runBWC)
    {
        float elapsedTimeMs = 0.0;
        ;
        cudaError_t e = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_timeNextPdschSlotStartEvents[m_nItrsPerStrm].handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);
        m_totPdschSlotStartTime[m_nItrsPerStrm] += elapsedTimeMs;
    }

    // send response
    if(evalMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_EVAL, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::printHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: printHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // std::string pipeline_str = (m_runPipelineInParallel) ? "parallel":"serial";
    printf("\n%s worker # %d\n", m_name.c_str(), m_wrkrId);
    // printf("\n--> Runs %d streams in parallel", m_nStrms);
    // printf("\n--> Each stream processes %d slot-cell(s) in series\n\n", m_nItrsPerStrm);

    cuPHYTestPrintMsgPayload& printMsgPayload = *std::static_pointer_cast<cuPHYTestPrintMsgPayload>(shPtrPayload);

    uint32_t nSRSStrms  = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;
    uint32_t nSRS2Strms = (m_nSRSCells) / 2;
    uint32_t maxNStrms  = std::max(std::max(m_nStrms, m_nPRACHCells), nSRSStrms);
    float    avgLatency;

    // print max Cb errors
    if(printMsgPayload.cbErrors)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                printf("--> strm # %d, itr # %d : max number of Cb Errors  :  %d out of %d \n", strmIdx, itrIdx, m_maxNumCbErrors[strmIdx][itrIdx], m_puschRxEvalDataSets[strmIdx][itrIdx].nCbs);
            }
        }
    }

    // send response
    if(printMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PRINT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::resetEvalHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: resetEvalHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    cuPHYTestResetEvalMsgPayload& resetEvalPayload = *std::static_pointer_cast<cuPHYTestResetEvalMsgPayload>(shPtrPayload);

    // reset max Cb error buffers
    if(resetEvalPayload.cbErrors)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                m_maxNumCbErrors[strmIdx][itrIdx] = 0;
            }
        }
    }
    m_totSRSStartTime    = 0;
    m_totSRSRunTime      = 0;
    m_totSRS2StartTime   = 0;
    m_totSRS2RunTime     = 0;
    m_totPRACHStartTime  = 0;
    m_totPRACHRunTime    = 0;
    m_totPUSCHRunTime    = 0;
    m_totPUSCHStartTime  = 0;
    m_totPUSCH2StartTime = 0;
    m_totPUSCH2RunTime   = 0;
    m_totPUCCHStartTime  = 0;
    m_totPUCCHRunTime    = 0;
    m_totPUCCH2StartTime = 0;
    m_totPUCCH2RunTime   = 0;

    // reset timers
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
        m_totRunTimePdschItr[itrIdx]    = 0;
        m_totPdschSlotStartTime[itrIdx] = 0;
        m_totBWCIterStartTime[itrIdx]   = 0;
        m_totRunTimeBWCItr[itrIdx]      = 0;
        m_totPdcchStartTimes[itrIdx]    = 0;
        m_totRunTimePdcchItr[itrIdx]    = 0;
        m_totCSIRSStartTimes[itrIdx]    = 0;
        m_totRunTimeCSIRSItr[itrIdx]    = 0;
        m_totSSBStartTime[itrIdx]       = 0;
        m_totSSBRunTime[itrIdx]         = 0;
    }
    m_totPdschSlotStartTime[m_nItrsPerStrm] = 0;
    // There can be an extra slot for PDSCH/BWC in some modes
    for(uint32_t itrIdx = 0; itrIdx < m_totPdschSlotStartTime.size(); ++itrIdx)
    {
        m_totPdschSlotStartTime[itrIdx] = 0;
    }

    uint32_t nSRSStrms = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;

    // send response
    if(resetEvalPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_RESET_EVAL, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pdschTxCleanHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxCleanHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    cuPHYTestPdschTxCleanMsgPayload& pdschTxCleanMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxCleanMsgPayload>(shPtrPayload);

    if(m_runPDSCH)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
        {
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                cuphy::pdsch_tx& pdschTxPipe = m_pdschTxPipes[strmIdx][itrIdx];

                PdschTx*                         pipeline_ptr = static_cast<PdschTx*>(pdschTxPipe.handle());
                const cuphyPdschCellGrpDynPrm_t* cell_group   = pipeline_ptr->dynamic_params->pCellGrpDynPrm;

                for(int ue_group_id = 0; ue_group_id < cell_group->nUeGrps; ue_group_id++)
                {
                    delete[] cell_group->pUeGrpPrms[ue_group_id].pUePrmIdxs;
                    delete[] cell_group->pUeGrpPrms[ue_group_id].pDmrsDynPrm;
                }

                for(int ue_id = 0; ue_id < cell_group->nUes; ue_id++)
                {
                    delete[] cell_group->pUePrms[ue_id].pCwIdxs;
                }
            }
        }
    }
    if(pdschTxCleanMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::setWaitValHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: setWaitValHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    cuPHYTestSetWaitValCmdMsgPayload& setWaitValCmdMsgPayload = *std::static_pointer_cast<cuPHYTestSetWaitValCmdMsgPayload>(shPtrPayload);

    // Wait on device value for all streams
    for(auto& cuStrm : m_cuStrms)
    {
        // CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), reinterpret_cast<CUdeviceptr>(m_shPtrGpuStartSyncFlag->addr()), setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
        CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), m_ptrGpuStartSyncFlag, setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
    }
    // printf("m_wrkrId %d m_shPtrGpuStartSyncFlag %d syncFlagVal %d\n", m_wrkrId, (*m_shPtrGpuStartSyncFlag)[0], setWaitValCmdMsgPayload.syncFlagVal);
    for(auto& cuStrm : m_cuStrmsPusch)
    {
        // CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), reinterpret_cast<CUdeviceptr>(m_shPtrGpuStartSyncFlag->addr()), setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
        CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), m_ptrGpuStartSyncFlag, setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
    }
    for(auto& cuStrm : m_cuStrmsPdsch)
    {
        // CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), reinterpret_cast<CUdeviceptr>(m_shPtrGpuStartSyncFlag->addr()), setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
        CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), m_ptrGpuStartSyncFlag, setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
    }

    if(setWaitValCmdMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<cuPHYTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SET_WAIT_VAL, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pschRdSmIdHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pschRdSmIdHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    readSmIds();

    cuPHYTestReadSmIdsCmdMsgPayload& readSmIdsCmdMsgPayload = *std::static_pointer_cast<cuPHYTestReadSmIdsCmdMsgPayload>(shPtrPayload);
    if(readSmIdsCmdMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload            = std::make_shared<cuPHYTestReadSmIdsRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrWaitEvent = m_shPtrRdSmIdWaitEvent;

        auto shPtrRsp = std::make_shared<cuPHYTestWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_READ_SM_IDS, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::msgProcess(std::shared_ptr<cuPHYTestWrkrCmdMsg>& shPtrMsg)
{
    using msgHandler_t = void (cuPHYTestWorker::*)(std::shared_ptr<void> & shPtrPayload);

    static constexpr std::array<std::pair<cuPHYTestWrkrCmdMsgType, msgHandler_t>, N_CUPHY_TEST_WRKR_CMD_MSGS> MSG_HANDLER_LUT{
        {{CUPHY_TEST_WRKR_CMD_MSG_INIT, &cuPHYTestWorker::initHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT, &cuPHYTestWorker::puschRxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT, &cuPHYTestWorker::pdschTxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP, &cuPHYTestWorker::puschRxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP, &cuPHYTestWorker::pdschTxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN, &cuPHYTestWorker::puschRxRunHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN, &cuPHYTestWorker::pdschTxRunHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN, &cuPHYTestWorker::pschTxRxRunHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_EVAL, &cuPHYTestWorker::evalHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PRINT, &cuPHYTestWorker::printHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_RESET_EVAL, &cuPHYTestWorker::resetEvalHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN, &cuPHYTestWorker::pdschTxCleanHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SET_WAIT_VAL, &cuPHYTestWorker::setWaitValHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_READ_SM_IDS, &cuPHYTestWorker::pschRdSmIdHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_DEINIT, &cuPHYTestWorker::deinitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_EXIT, &cuPHYTestWorker::exitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_BFC_INIT, &cuPHYTestWorker::bfcInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_BFC_SETUP, &cuPHYTestWorker::bfcSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT, &cuPHYTestWorker::srsInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP, &cuPHYTestWorker::srsSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT, &cuPHYTestWorker::prachInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP, &cuPHYTestWorker::prachSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT, &cuPHYTestWorker::pucchRxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP, &cuPHYTestWorker::pucchRxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT, &cuPHYTestWorker::pdcchTxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP, &cuPHYTestWorker::pdcchTxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT, &cuPHYTestWorker::ssbInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP, &cuPHYTestWorker::ssbSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT, &cuPHYTestWorker::csirsInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP, &cuPHYTestWorker::csirsSetupHandler}}};
    if(shPtrMsg->type < N_CUPHY_TEST_WRKR_CMD_MSGS)
    {
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: msgProcess msgType %d %s\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), shPtrMsg->type, CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

        // If assert fails, ensure the MSG_HANDLER_LUT table matches the enumerated message types in cuPHYTestWrkrCmdMsgType
        assert(shPtrMsg->type == MSG_HANDLER_LUT[shPtrMsg->type].first);
        (this->*(MSG_HANDLER_LUT[shPtrMsg->type].second))(shPtrMsg->payload);
    }
    else
    {
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: msgProcess - Message type not supported\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    }
}

void cuPHYTestWorker::run()
{
    createCuCtx();
    DEBUG_TRACE("%s [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Enter run loop for worker %d\n", m_name.c_str(), getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_wrkrId);
    for(;;)
    {
        std::shared_ptr<cuPHYTestWrkrCmdMsg> shPtrCmd;
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: run loop for worker\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
        // DEBUG_TRACE("Thread %s: Begin message receive\n", m_name.c_str());
        m_shPtrCmdQ->receive(shPtrCmd, m_wrkrId);
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Received message: %s\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrCmd->type]);

        // DEBUG_TRACE("Thread %s: Begin message processing: %s\n", m_name.c_str(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type];
        msgProcess(shPtrCmd);
        if(CUPHY_TEST_WRKR_CMD_MSG_EXIT == shPtrCmd->type) break;

        // DEBUG_TRACE("Thread %s: Processed message: %s\n", m_name.c_str(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    }
    DEBUG_TRACE("%s [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Exit run loop for worker %d\n", m_name.c_str(), getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_wrkrId);
}

// //----------------------------------------------------------------------------------------------------------
// // APIs invoked by orchestration thread

void cuPHYTestWorker::init(uint32_t nStrms, uint32_t nItrsPerStrm, uint32_t nTimingItrs, std::map<std::string, int>& cuStrmPrioMap, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrCpuGpuSyncFlag, uint32_t nPDCCHCells, uint32_t nCSIRSCells, uint32_t nBWCCells, uint32_t nSRSCells, uint32_t nPRACHCells, uint32_t nPUCCHCells, uint32_t nSSBCells, bool srsSplit, bool waitRsp, uint32_t longPattern, bool srsCtx)
{
    // set run configuration
    m_longPattern  = longPattern;
    m_nStrms       = nStrms;
    m_nStrms_pdsch = nStrms; // Will be overwritten to 1 in pdschTxInit iff pdsch_group_cells is 1.
    m_nItrsPerStrm = nItrsPerStrm;
    m_nTimingItrs  = nTimingItrs;
    m_runSRS2      = srsSplit;
    m_nSRSCells    = nSRSCells;
    m_nPRACHCells  = nPRACHCells;
    m_nBWCCells    = nBWCCells;
    m_nPDCCHCells  = nPDCCHCells;
    m_nPUCCHCells  = nPUCCHCells;
    m_nSSBCells    = nSSBCells;
    m_nCSIRSCells  = nCSIRSCells;
    m_srsCtx       = srsCtx;

    // send init message
    auto shPtrPayload                 = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp                 = true;
    shPtrPayload->shPtrCpuGpuSyncFlag = shPtrCpuGpuSyncFlag;
    shPtrPayload->cuStrmPrioMap       = cuStrmPrioMap;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);

    // response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::pdcchTxInit(std::vector<std::string> inFileNamesPdcchTx, bool group_cells, uint32_t cells_per_stream, bool ref_check, uint64_t pdcch_proc_mode, bool waitRsp)
{
    // set pdcch configuration
    m_ref_check_pdcch     = ref_check;
    m_pdcch_group_cells   = group_cells;
    m_nCellsPerStrm_pdcch = cells_per_stream;
    m_pdcch_proc_mode     = pdcch_proc_mode;
    m_nStrms_pdcch        = (group_cells) ? 1 : m_nPDCCHCells; // NB: should not use m_nStrms if no grouping

    // Send initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPdcchTx;

    if(inFileNamesPdcchTx.size() == 0)
    {
        m_runPDCCH = false;
        return;
    }
    m_runPDCCH    = true;
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::puschRxInit(std::vector<std::string> inFileNamesPuschRx, uint32_t fp16Mode, int puschRxDescramblingOn, bool printCbErrors, uint64_t pusch_proc_mode, bool enableLdpcThroughputMode, bool groupCells, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode, bool waitRsp)
{
    // pusch configuration
    if(inFileNamesPuschRx.size() == 0)
    {
        m_runPUSCH = false;
        return;
    }

    m_fp16Mode                = fp16Mode;
    m_descramblingOn          = puschRxDescramblingOn;
    m_printCbErrors           = printCbErrors;
    m_pusch_proc_mode         = pusch_proc_mode;
    m_ldpc_kernel_launch_mode = ldpcLaunchMode;
    m_pusch_group_cells       = groupCells;
    // Send initialization message
    auto shPtrPayload                      = std::make_shared<cuPHYTestPuschRxInitMsgPayload>();
    shPtrPayload->rsp                      = true;
    shPtrPayload->enableLdpcThroughputMode = enableLdpcThroughputMode;
    shPtrPayload->inFileNames              = inFileNamesPuschRx;
    shPtrPayload->puschPrms                = puschPrms;

    m_runPUSCH = true;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::pdschTxInit(std::vector<std::string> inFileNamesPdschTx, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, bool group_cells, uint32_t cells_per_stream, maxPDSCHPrms pdschPrms, bool waitRsp)
{
    if(inFileNamesPdschTx.size() == 0)
    {
        m_runPDSCH = false;
        return;
    }
    // set pdsch configuration
    m_ref_check_pdsch        = ref_check_pdsch;
    m_identical_ldpc_configs = identical_ldpc_configs;
    m_pdsch_proc_mode        = pdsch_proc_mode;
    m_pdsch_group_cells      = group_cells;
    m_nCellsPerStrm_pdsch    = cells_per_stream;
    m_nStrms_pdsch           = (group_cells) ? 1 : m_nStrms;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestPdschTxInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPdschTx;
    shPtrPayload->pdschPrms   = pdschPrms;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::csirsInit(std::vector<std::string> inFileNamesCSIRS, bool ref_check_csirs, bool group_cells, uint64_t csirs_proc_mode, bool waitRsp)
{
    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = false;
        return;
    }
    m_runCSIRS = true;

    m_ref_check_csirs   = ref_check_csirs;
    m_pdsch_group_cells = group_cells;
    m_csirs_proc_mode   = csirs_proc_mode;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesCSIRS;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT, m_wrkrId, shPtrPayload);

    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::ssbInit(std::vector<std::string> inFileNamesSSB, bool ref_check_ssb, bool group_cells, uint32_t nSsbSlots, uint64_t ssb_proc_mode, bool waitRsp)
{
    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = false;
        return;
    }
    m_runSSB = true;

    m_pdsch_group_cells = group_cells;
    m_ref_check_ssb     = ref_check_ssb;
    m_ssb_proc_mode     = ssb_proc_mode;
    m_nSsbSlots         = nSsbSlots;

    // reset SSB timer
    for(int ssbSlotIdx = 0; ssbSlotIdx < m_nSsbSlots; ssbSlotIdx ++)
    {
        m_totSSBStartTime[ssbSlotIdx]    = 0;
        m_totSSBRunTime[ssbSlotIdx]      = 0;
    }

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSSB;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT, m_wrkrId, shPtrPayload);

    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::prachInit(std::vector<std::string> inFileNamesPRACH, uint64_t proc_mode, bool ref_check_prach, bool group_cells, uint32_t cells_per_stream, bool waitRsp)
{
    if(inFileNamesPRACH.size() == 0)
    {
        m_runPRACH = false;
        return;
    }
    m_runPRACH = true;
    m_prach_proc_mode = proc_mode;

    m_prach_group_cells   = group_cells;
    m_ref_check_prach     = ref_check_prach;
    m_nCellsPerStrm_prach = cells_per_stream;
    m_nStrms_prach        = (group_cells) ? 1 : m_nPRACHCells; // NB: should not use m_nStrms if no grouping

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPRACH;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::pucchRxInit(std::vector<std::string> inFileNamesPUCCH, bool ref_check_pucch, bool groupCells, uint64_t pucch_proc_mode, bool waitRsp)
{
    if(inFileNamesPUCCH.size() == 0)
    {
        m_runPUCCH = false;
        return;
    }
    m_runPUCCH        = true;
    m_pucch_proc_mode = pucch_proc_mode;

    m_pucch_group_cells   = groupCells;
    m_ref_check_pucch     = ref_check_pucch;
    m_nCellsPerStrm_pucch = groupCells ? inFileNamesPUCCH.size() : 1;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPUCCH;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::srsInit(std::vector<std::string> inFileNamesSRS, bool ref_check_srs, uint64_t srs_proc_mode, bool splitSRS, bool waitRsp)
{
    if(inFileNamesSRS.size() == 0)
    {
        m_runSRS  = false;
        m_runSRS2 = false;
        return;
    }
    m_srs_proc_mode = srs_proc_mode;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSRS;
    m_runSRS                  = true;
    m_ref_check_srs           = ref_check_srs;

    if(splitSRS)
        m_runSRS2 = true;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::csirsSetup(std::vector<std::string> inFileNamesCSIRS, bool waitRsp)
{
    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = false;
        return;
    }
    m_runCSIRS = true;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesCSIRS;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::ssbSetup(std::vector<std::string> inFileNamesSSB, bool waitRsp)
{
    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = false;
        return;
    }
    m_runSSB = true;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSSB;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::pucchRxSetup(std::vector<std::string> inFileNamesPUCCH, bool waitRsp)
{
    if(inFileNamesPUCCH.size() == 0)
    {
        m_runPUCCH = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPUCCH;
    m_runPUCCH                = true;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::pdcchTxSetup(std::vector<std::string> inFileNamesPDCCH, bool waitRsp)
{
    if(inFileNamesPDCCH.size() == 0)
    {
        m_runPDCCH = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPDCCH;
    m_runPDCCH                = true;

    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::prachSetup(std::vector<std::string> inFileNamesPRACH, bool waitRsp)
{
    if(inFileNamesPRACH.size() == 0)
    {
        m_runPRACH = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPRACH;
    m_runPRACH                = true;
    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::bfcInit(std::vector<std::string> inFileNamesBFC, bool ref_check_bfc, bool waitRsp)
{
    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesBFC;
    if(inFileNamesBFC.size() == 0)
    {
        m_runBWC = false;
        return;
    }
    m_runBWC        = true;
    m_ref_check_bfc = ref_check_bfc;
    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_BFC_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_BFC_INIT, m_wrkrId);
    }
}
void cuPHYTestWorker::bfcSetup(std::vector<std::string> inFileNamesBFC, bool waitRsp)
{
    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesBFC;
    if(inFileNamesBFC.size() == 0)
    {
        m_runBWC = false;
        return;
    }
    m_runBWC = true;
    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_BFC_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_BFC_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::deinit(bool waitRsp)
{
    auto shPtrPayload = std::make_shared<cuPHYTestDeinitMsgPayload>();
    shPtrPayload->rsp = true;

    // Cleanup
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_DEINIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_DEINIT, m_wrkrId);
    }
}

void cuPHYTestWorker::puschRxSetup(std::vector<std::string> inFileNamesPuschRx, bool waitRsp)
{
    if(inFileNamesPuschRx.size() == 0)
    {
        m_runPUSCH = false;
        return;
    }

    if(inFileNamesPuschRx.size() != 0)
    {
        // send message
        auto shPtrPayload                = std::make_shared<cuPHYTestPuschRxSetupMsgPayload>();
        shPtrPayload->rsp                = true;
        shPtrPayload->inFileNamesPuschRx = inFileNamesPuschRx;

        auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP, m_wrkrId, shPtrPayload);
        DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

        m_shPtrCmdQ->send(shPtrMsg);
    }
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::srsSetup(std::vector<std::string> inFileNamesSRS, bool waitRsp)
{
    if(inFileNamesSRS.size() == 0)
    {
        m_runSRS = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSRS;
    // send initalization message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP, m_wrkrId);
    }
}
void cuPHYTestWorker::pdschTxSetup(std::vector<std::string> inFileNamesPdschTx, bool waitRsp)
{
    if(inFileNamesPdschTx.size() == 0)
    {
        m_runPDSCH = false;
        return;
    }
    // send message
    auto shPtrPayload                = std::make_shared<cuPHYTestPdschTxSetupMsgPayload>();
    shPtrPayload->rsp                = true;
    shPtrPayload->inFileNamesPdschTx = inFileNamesPdschTx;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::puschRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, cudaEvent_t prachStartEvent, cudaEvent_t pucchStartEvent, bool waitRsp)
{
    auto shPtrPayload             = std::make_shared<cuPHYTestPuschRxRunMsgPayload>();
    shPtrPayload->rsp             = true;
    shPtrPayload->startEvent      = startEvent;
    shPtrPayload->prachStartEvent = prachStartEvent;
    shPtrPayload->pucchStartEvent = pucchStartEvent;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        getPuschRxRunRsp(shPtrRsp);
        cuPHYTestPuschRxRunRspMsgPayload& puschRxRunRspMsgPayload = *std::static_pointer_cast<cuPHYTestPuschRxRunRspMsgPayload>(shPtrRsp->payload);
        shPtrStopEvent                                            = puschRxRunRspMsgPayload.shPtrStopEvent;
    }
}

void cuPHYTestWorker::pdschTxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp, std::vector<cuphy::event>* pdcchStopEventVec, std::vector<cuphy::event>* pdschInterSlotEventVec)
{
    auto shPtrPayload                    = std::make_shared<cuPHYTestPdschTxRunMsgPayload>();
    shPtrPayload->rsp                    = true;
    shPtrPayload->startEvent             = startEvent;
    shPtrPayload->pdcchStopEventVec      = pdcchStopEventVec;
    shPtrPayload->pdschInterSlotEventVec = pdschInterSlotEventVec;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        getPdschTxRunRsp(shPtrRsp);

        cuPHYTestPdschTxRunRspMsgPayload& pdschTxRunRspMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunRspMsgPayload>(shPtrRsp->payload);
        shPtrStopEvent                                            = pdschTxRunRspMsgPayload.shPtrStopEvent;
    }
}

void cuPHYTestWorker::pschTxRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp)
{
    auto shPtrPayload        = std::make_shared<cuPHYTestPschTxRxRunMsgPayload>();
    shPtrPayload->rsp        = true;
    shPtrPayload->startEvent = startEvent;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        getPschTxRxRunRsp(shPtrRsp);

        cuPHYTestPschRunRspMsgPayload& pschRunRspMsgPayload = *std::static_pointer_cast<cuPHYTestPschRunRspMsgPayload>(shPtrRsp->payload);
        shPtrStopEvent                                      = pschRunRspMsgPayload.shPtrStopEvent;
    }
}

void cuPHYTestWorker::getPuschRxRunRsp(std::shared_ptr<cuPHYTestWrkrRspMsg>& shPtrRsp)
{
    m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN, m_wrkrId);
}

void cuPHYTestWorker::getPdschTxRunRsp(std::shared_ptr<cuPHYTestWrkrRspMsg>& shPtrRsp)
{
    m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN, m_wrkrId);
}

void cuPHYTestWorker::getPschTxRxRunRsp(std::shared_ptr<cuPHYTestWrkrRspMsg>& shPtrRsp)
{
    m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN, m_wrkrId);
}

void cuPHYTestWorker::eval(bool cbErrors, bool isPschTxRx, bool waitRsp)
{
    // pack message
    auto shPtrPayload        = std::make_shared<cuPHYTestEvalMsgPayload>();
    shPtrPayload->rsp        = waitRsp;
    shPtrPayload->cbErrors   = cbErrors;
    shPtrPayload->isPschTxRx = isPschTxRx;

    // send message to worker
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_EVAL, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_EVAL, m_wrkrId);
    }
}

void cuPHYTestWorker::print(bool cbErrors, bool isPschTxRx, bool waitRsp)
{
    // pack message
    auto shPtrPayload        = std::make_shared<cuPHYTestPrintMsgPayload>();
    shPtrPayload->rsp        = waitRsp;
    shPtrPayload->cbErrors   = cbErrors;
    shPtrPayload->isPschTxRx = isPschTxRx;

    // send message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PRINT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PRINT, m_wrkrId);
    }
}

void cuPHYTestWorker::resetEvalBuffers(bool cbErrors, bool waitRsp)
{
    // pack message
    auto shPtrPayload      = std::make_shared<cuPHYTestResetEvalMsgPayload>();
    shPtrPayload->rsp      = waitRsp;
    shPtrPayload->cbErrors = cbErrors;

    // send message
    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_RESET_EVAL, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_RESET_EVAL, m_wrkrId);
    }
}

void cuPHYTestWorker::pdschTxClean(bool waitRsp)
{
    auto shPtrPayload = std::make_shared<cuPHYTestPdschTxCleanMsgPayload>();
    shPtrPayload->rsp = waitRsp;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN, m_wrkrId);
    }
}

void cuPHYTestWorker::setWaitVal(uint32_t syncFlagVal, bool waitRsp)
{
    auto shPtrPayload         = std::make_shared<cuPHYTestSetWaitValCmdMsgPayload>();
    shPtrPayload->rsp         = waitRsp;
    shPtrPayload->syncFlagVal = syncFlagVal;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SET_WAIT_VAL, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SET_WAIT_VAL, m_wrkrId);
    }
}

void cuPHYTestWorker::readSmIds(std::shared_ptr<cuphy::event>& shPtrRdSmIdWaitEvent, bool waitRsp)
{
    auto shPtrPayload = std::make_shared<cuPHYTestReadSmIdsCmdMsgPayload>();
    shPtrPayload->rsp = waitRsp;

    auto shPtrMsg = std::make_shared<cuPHYTestWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_READ_SM_IDS, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), CUPHY_TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<cuPHYTestWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_READ_SM_IDS, m_wrkrId);

        cuPHYTestReadSmIdsRspMsgPayload& readSmIdsRspMsgPayload = *std::static_pointer_cast<cuPHYTestReadSmIdsRspMsgPayload>(shPtrRsp->payload);
        shPtrRdSmIdWaitEvent                                    = readSmIdsRspMsgPayload.shPtrWaitEvent;
    }
}

std::vector<float> cuPHYTestWorker::getBWCIterStartTimes()
{
    return m_totBWCIterStartTime;
}
std::vector<float> cuPHYTestWorker::getBWCIterTimes()
{
    return m_totRunTimeBWCItr;
}
std::vector<float> cuPHYTestWorker::getPdschIterTimes()
{
    return m_totRunTimePdschItr;
}
std::vector<float> cuPHYTestWorker::getPdschSlotStartTimes()
{
    return m_totPdschSlotStartTime;
}
std::vector<float> cuPHYTestWorker::getCSIRSStartTimes()
{
    return m_totCSIRSStartTimes;
}
std::vector<float> cuPHYTestWorker::getCSIRSIterTimes()
{
    return m_totRunTimeCSIRSItr;
}
std::vector<float> cuPHYTestWorker::getPdcchStartTimes()
{
    return m_totPdcchStartTimes;
}
std::vector<float> cuPHYTestWorker::getPdcchIterTimes()
{
    return m_totRunTimePdcchItr;
}
std::vector<float> cuPHYTestWorker::getTotSSBStartTime()
{
    return m_totSSBStartTime;
}
std::vector<float> cuPHYTestWorker::getTotSSBRunTime()
{
    return m_totSSBRunTime;
}
float cuPHYTestWorker::getTotSRSStartTime()
{
    return m_totSRSStartTime;
}
float cuPHYTestWorker::getTotSRSRunTime()
{
    return m_totSRSRunTime;
}
float cuPHYTestWorker::getTotSRS2StartTime()
{
    return m_totSRS2StartTime;
}
float cuPHYTestWorker::getTotSRS2RunTime()
{
    return m_totSRS2RunTime;
}
float cuPHYTestWorker::getTotPrachStartTime()
{
    return m_totPRACHStartTime;
}
float cuPHYTestWorker::getTotPrachRunTime()
{
    return m_totPRACHRunTime;
}
float cuPHYTestWorker::getTotPuschStartTime()
{
    return m_totPUSCHStartTime;
}
float cuPHYTestWorker::getTotPuschRunTime()
{
    return m_totPUSCHRunTime;
}
float cuPHYTestWorker::getTotPusch2StartTime()
{
    return m_totPUSCH2StartTime;
}

float cuPHYTestWorker::getTotPusch2RunTime()
{
    return m_totPUSCH2RunTime;
}
float cuPHYTestWorker::getTotPucchStartTime()
{
    return m_totPUCCHStartTime;
}
float cuPHYTestWorker::getTotPucchRunTime()
{
    return m_totPUCCHRunTime;
}
float cuPHYTestWorker::getTotPucch2StartTime()
{
    return m_totPUCCH2StartTime;
}
float cuPHYTestWorker::getTotPucch2RunTime()
{
    return m_totPUCCH2RunTime;
}
cudaEvent_t cuPHYTestWorker::getPuschStartEvent()
{
    if(m_longPattern == 2 || m_longPattern == 5)
        return nullptr;
    else
        return m_uqPtrPuschDelayStopEvent->handle();
}
cudaEvent_t cuPHYTestWorker::getPusch2StartEvent()
{
    if(m_longPattern)
    {
        if(m_longPattern < 4)
        {
            return m_uqPtrPusch2DelayStopEvent->handle();
        }
        else
            return m_stopEvents[0].handle();
    }
    else
    {
        return m_uqPtrPuschDelayStopEvent->handle();
    }
}
std::vector<cuphy::event>* cuPHYTestWorker::getPdcchStopEventVecPtr()
{
    return &m_pdcchInterSlotStartEventVec;
}
std::vector<cuphy::event>* cuPHYTestWorker::getPdschInterSlotEventVecPtr()
{
    return &m_pdschInterSlotStartEventVec;
}
