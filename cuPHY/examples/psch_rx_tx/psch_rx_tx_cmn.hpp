/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(PSCH_RX_TX_CMN_HPP_INCLUDED_)
#define PSCH_RX_TX_CMN_HPP_INCLUDED_

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
#include "util.hpp"
#include "datasets.hpp"
#include "pdsch_tx.hpp"
#include "pdcch_tx.hpp"
#include "pusch_rx_test.hpp"
#include "pucch_rx.hpp"

// #define ENABLE_F01_STREAM_PRIO

//----------------------------------------------------------------------------------------------------------
// cuPHYTestWrkrCmdMsgType - Available message types from main test orchestration thread to the worker thread
//----------------------------------------------------------------------------------------------------------
enum cuPHYTestWrkrCmdMsgType
{
    CUPHY_TEST_WRKR_CMD_MSG_INIT         = 0,
    CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT   = 1,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT   = 2,
    CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP  = 3,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP  = 4,
    CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN    = 5,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN    = 6,
    CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN     = 7,
    CUPHY_TEST_WRKR_CMD_MSG_EVAL         = 8,
    CUPHY_TEST_WRKR_CMD_MSG_PRINT        = 9,
    CUPHY_TEST_WRKR_CMD_MSG_RESET_EVAL   = 10,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN  = 11,
    CUPHY_TEST_WRKR_CMD_MSG_SET_WAIT_VAL = 12,
    CUPHY_TEST_WRKR_CMD_MSG_READ_SM_IDS  = 13,
    CUPHY_TEST_WRKR_CMD_MSG_DEINIT       = 14,
    CUPHY_TEST_WRKR_CMD_MSG_EXIT         = 15,
    CUPHY_TEST_WRKR_CMD_MSG_BFC_INIT     = 16,
    CUPHY_TEST_WRKR_CMD_MSG_BFC_SETUP    = 17,
    CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT     = 18,
    CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP    = 19,
    CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT   = 20,
    CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP  = 21,
    CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT   = 22,
    CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP  = 23,
    CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT   = 24,
    CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP  = 25,
    CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT     = 26,
    CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP    = 27,
    CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT   = 28,
    CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP  = 29,
    N_CUPHY_TEST_WRKR_CMD_MSGS           = 30,
    CUPHY_TEST_WRKR_CMD_MSG_INVALID      = N_CUPHY_TEST_WRKR_CMD_MSGS
};

static constexpr std::array<const char*, N_CUPHY_TEST_WRKR_CMD_MSGS> CUPHY_TEST_WRKR_CMD_MSG_TO_STR{
    "CUPHY_TEST_WRKR_CMD_MSG_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN",
    "CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN",
    "CUPHY_TEST_WRKR_CMD_MSG_EVAL",
    "CUPHY_TEST_WRKR_CMD_MSG_PRINT",
    "CUPHY_TEST_WRKR_CMD_MSG_RESET_EVAL",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN",
    "CUPHY_TEST_WRKR_CMD_MSG_SET_WAIT_VAL",
    "CUPHY_TEST_WRKR_CMD_MSG_READ_SM_IDS",
    "CUPHY_TEST_WRKR_CMD_MSG_DEINIT",
    "CUPHY_TEST_WRKR_CMD_MSG_EXIT",
    "CUPHY_TEST_WRKR_CMD_MSG_BFC_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_BFC_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP"};

enum cuPHYTestWrkrRspMsgType
{
    CUPHY_TEST_WRKR_RSP_MSG_INIT         = 0,
    CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT   = 1,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT   = 2,
    CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP  = 3,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP  = 4,
    CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN    = 5,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN    = 6,
    CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN     = 7,
    CUPHY_TEST_WRKR_RSP_MSG_EVAL         = 8,
    CUPHY_TEST_WRKR_RSP_MSG_PRINT        = 9,
    CUPHY_TEST_WRKR_RSP_MSG_RESET_EVAL   = 10,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN  = 11,
    CUPHY_TEST_WRKR_RSP_MSG_SET_WAIT_VAL = 12,
    CUPHY_TEST_WRKR_RSP_MSG_READ_SM_IDS  = 13,
    CUPHY_TEST_WRKR_RSP_MSG_DEINIT       = 14,
    CUPHY_TEST_WRKR_RSP_MSG_EXIT         = 15,
    CUPHY_TEST_WRKR_RSP_MSG_BFC_INIT     = 16,
    CUPHY_TEST_WRKR_RSP_MSG_BFC_SETUP    = 17,
    CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT     = 18,
    CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP    = 19,
    CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT   = 20,
    CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP  = 21,
    CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT   = 22,
    CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP  = 23,
    CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT   = 24,
    CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP  = 25,
    CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT     = 26,
    CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP    = 27,
    CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT   = 28,
    CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP  = 29,
    N_CUPHY_TEST_WRKR_RSP_MSGS           = 30,
    CUPHY_TEST_WRKR_RSP_MSG_INVALID      = N_CUPHY_TEST_WRKR_RSP_MSGS
};

static constexpr std::array<const char*, N_CUPHY_TEST_WRKR_RSP_MSGS> CUPHY_TEST_WRKR_RSP_MSG_TO_STR =
    {
        "CUPHY_TEST_WRKR_RSP_MSG_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN",
        "CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN",
        "CUPHY_TEST_WRKR_RSP_MSG_EVAL",
        "CUPHY_TEST_WRKR_RSP_MSG_PRINT",
        "CUPHY_TEST_WRKR_RSP_MSG_RESET_EVAL",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN",
        "CUPHY_TEST_WRKR_RSP_MSG_SET_WAIT_VAL",
        "CUPHY_TEST_WRKR_RSP_MSG_READ_SM_IDS",
        "CUPHY_TEST_WRKR_RSP_MSG_DEINIT",
        "CUPHY_TEST_WRKR_RSP_MSG_EXIT",
        "CUPHY_TEST_WRKR_RSP_MSG_BFC_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_BFC_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP"};

// N_MAX_RX: maximum number of threads the message is broadcasted to
static constexpr uint32_t CUPHY_TEST_N_MAX_MSG_RX = 64;
template <typename MSG_T, uint32_t N_MAX_MSG_RX = CUPHY_TEST_N_MAX_MSG_RX>
struct cuPHYTestMsg
{
    cuPHYTestMsg(MSG_T inType, int32_t inRxId, std::shared_ptr<void> inPayload = std::shared_ptr<void>()) :
        type(inType),
        payload(inPayload),
        rxIdBset(std::bitset<N_MAX_MSG_RX>().set(inRxId)){};
    cuPHYTestMsg(MSG_T inType, std::bitset<N_MAX_MSG_RX> inRxIdBset = 0, std::shared_ptr<void> inPayload = std::shared_ptr<void>()) :
        type(inType),
        payload(inPayload),
        rxIdBset(inRxIdBset){};
    cuPHYTestMsg()                    = delete;
    cuPHYTestMsg(cuPHYTestMsg const&) = delete;
    cuPHYTestMsg& operator=(cuPHYTestMsg const&) = delete;
    ~cuPHYTestMsg()                              = default;

    MSG_T                     type;
    std::bitset<N_MAX_MSG_RX> rxIdBset = 0; // if same message is broadcasted to multiple threads
    std::shared_ptr<void>     payload;
};

using cuPHYTestWrkrCmdMsg = cuPHYTestMsg<cuPHYTestWrkrCmdMsgType>;
using cuPHYTestWrkrRspMsg = cuPHYTestMsg<cuPHYTestWrkrRspMsgType>;

inline std::string getThreadIdStr(std::thread::id thrdId = std::this_thread::get_id())
{
    static std::mutex m_ostreamMutex;

    std::lock_guard<std::mutex> ostreamMutexLock(m_ostreamMutex);
    std::ostringstream          oss;
    oss << thrdId;
    return oss.str();
}

// Address of the context
inline uint64_t getCurrCuCtxId()
{
    CUcontext cuCtx;
    CU_CHECK(cuCtxGetCurrent(&cuCtx));
    return reinterpret_cast<uint64_t>(cuCtx);
}

template <typename MSG_TYPE, typename MSG>
class cuphyTestMsgQ {
public:
    cuphyTestMsgQ(std::string name = "") :
        m_name(name){};
    ~cuphyTestMsgQ() = default;

    void send(std::shared_ptr<MSG>& shPtrMsg) // reference to shared_ptr optional (used for performance)
    {
        {
            std::lock_guard<std::mutex> mutexLockGuard(m_mutex);
            m_queue.push(shPtrMsg);
        }
        // printf("%s tid %s: send message %d with rxIdBset 0x%llx\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong());
        m_cv.notify_all();
    }

    void receive(std::shared_ptr<MSG>& shPtrMsg, int32_t rxId = -1) // reference to shared_ptr needed for functionality
    {
        std::unique_lock<std::mutex> mutexLock(m_mutex);

        // while(m_cuphyTestMsgQueue.empty()) {m_msgCv.wait(msgMutexLock);}
        m_cv.wait(mutexLock, [this, &shPtrMsg, &rxId] {
            bool msgAvail = false;
            if(!m_queue.empty())
            {
                shPtrMsg = m_queue.front();
                // printf("%s tid %s: received-p1 message %d with rxIdBset 0x%llx rxId %d msgAvail %u bitVal %u\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong(), rxId, (rxId >= 0) ? shPtrMsg->rxIdBset[rxId] : true, static_cast<bool>(shPtrMsg->rxIdBset[rxId]));

                if(rxId >= 0)
                {
                    msgAvail = shPtrMsg->rxIdBset[rxId];
                    shPtrMsg->rxIdBset.reset(rxId); // ensure bit clear occurs under mutex protection
                }
                else
                {
                    msgAvail = true;
                }
            }
            return msgAvail;
        });

        // printf("%s tid %s: received-p2 message %d with rxIdBset 0x%llx\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong());

        if(rxId >= 0)
        {
            if(shPtrMsg->rxIdBset.none()) m_queue.pop();
        }
        else
        {
            m_queue.pop();
        }
    }

    void receive(std::shared_ptr<MSG>& shPtrMsg, MSG_TYPE msgType, int32_t rxId = -1) // reference to shared_ptr needed for functionality
    {
        std::unique_lock<std::mutex> mutexLock(m_mutex);

        // while(m_cuphyTestMsgQueue.empty()) {m_msgCv.wait(msgMutexLock);}
        m_cv.wait(mutexLock, [this, &shPtrMsg, &rxId, &msgType] {
            bool msgAvail = false;
            if(!m_queue.empty())
            {
                shPtrMsg = m_queue.front();
                // printf("%s tid %s: received-p1 message %d with rxIdBset 0x%llx rxId %d msgAvail %u bitVal %u\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong(), rxId, (rxId >= 0) ? shPtrMsg->rxIdBset[rxId] : true, static_cast<bool>(shPtrMsg->rxIdBset[rxId]));

                if(msgType == shPtrMsg->type)
                {
                    if(rxId >= 0)
                    {
                        msgAvail = shPtrMsg->rxIdBset[rxId];
                        shPtrMsg->rxIdBset.reset(rxId); // ensure bit clear occurs under mutex protection
                    }
                    else
                    {
                        msgAvail = true;
                    }
                }
            }
            return msgAvail;
        });

        // printf("%s tid %s: received-p2 message %d with rxIdBset 0x%llx\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong());

        if(rxId >= 0)
        {
            if(shPtrMsg->rxIdBset.none()) m_queue.pop();
        }
        else
        {
            m_queue.pop();
        }
    }

    bool isEmpty() const { return m_queue.empty(); };

private:
    std::string                      m_name;
    std::queue<std::shared_ptr<MSG>> m_queue;
    std::mutex                       m_mutex;
    std::condition_variable          m_cv;
};

using cuphyTestWrkrCmdQ   = cuphyTestMsgQ<cuPHYTestWrkrCmdMsgType, cuPHYTestWrkrCmdMsg>;
using cuphyTestWrkrRspQ   = cuphyTestMsgQ<cuPHYTestWrkrRspMsgType, cuPHYTestWrkrRspMsg>;
using cuphySharedWrkrCmdQ = std::shared_ptr<cuphyTestMsgQ<cuPHYTestWrkrCmdMsgType, cuPHYTestMsg<cuPHYTestWrkrCmdMsgType>>>;
using cuphySharedWrkrRspQ = std::shared_ptr<cuphyTestMsgQ<cuPHYTestWrkrRspMsgType, cuPHYTestMsg<cuPHYTestWrkrRspMsgType>>>;

//----------------------------------------------------------------------------------------------------------
//  cuPHYTestMsg Message payloads

struct cuPHYTestInitMsgPayload
{
    bool                                                          rsp;
    std::vector<std::string>                                      inFileNames;
    std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>> shPtrCpuGpuSyncFlag;
    std::map<std::string, int>                                    cuStrmPrioMap;
};

struct cuPHYTestPuschRxInitMsgPayload : cuPHYTestInitMsgPayload
{
    bool         enableLdpcThroughputMode;
    maxPUSCHPrms puschPrms;
};
struct cuPHYTestPdschTxInitMsgPayload : cuPHYTestInitMsgPayload
{
    maxPDSCHPrms pdschPrms;
};

// CUPHY_TEST_WRKR_CMD_MSG_DEINIT payload
struct cuPHYTestDeinitMsgPayload
{
    bool rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_EXIT payload
struct cuPHYTestExitMsgPayload
{
    bool rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_READ_SM_IDS
struct cuPHYTestReadSmIdsCmdMsgPayload
{
    bool rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP payload
struct cuPHYTestPuschRxSetupMsgPayload
{
    std::vector<std::string> inFileNamesPuschRx;
    bool                     rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP payload
struct cuPHYTestPdschTxSetupMsgPayload
{
    std::vector<std::string> inFileNamesPdschTx;
    bool                     rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN payload
struct cuPHYTestPuschRxRunMsgPayload
{
    bool        rsp;
    cudaEvent_t startEvent;
    cudaEvent_t prachStartEvent;
    cudaEvent_t pucchStartEvent;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN payload
struct cuPHYTestPdschTxRunMsgPayload
{
    bool                       rsp;
    cudaEvent_t                startEvent;
    std::vector<cuphy::event>* pdcchStopEventVec;
    std::vector<cuphy::event>* pdschInterSlotEventVec;
};

// CUPHY_TEST_WRKR_CMD_MSG_BFC_RUN payload
struct cuPHYTestBFCRunMsgPayload
{
    bool        rsp;
    cudaEvent_t startEvent;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN payload
struct cuPHYTestPschTxRxRunMsgPayload
{
    bool        rsp;
    cudaEvent_t startEvent;
};

// CUPHY_TEST_WRKR_CMD_MSG_EVAL payload
struct cuPHYTestEvalMsgPayload
{
    bool rsp;
    bool cbErrors;
    bool isPschTxRx;
};

// CUPHY_TEST_WRKR_CMD_MSG_PRINT payload
struct cuPHYTestPrintMsgPayload
{
    bool rsp;
    bool cbErrors;
    bool isPschTxRx;
};

// CUPHY_TEST_WRKR_CMD_MSG_RESET_EVAL payload
struct cuPHYTestResetEvalMsgPayload
{
    bool rsp;
    bool cbErrors;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN payload
struct cuPHYTestPdschTxCleanMsgPayload
{
    bool rsp;
};

// CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN payload
struct cuPHYTestPuschRxRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN payload
struct cuPHYTestPdschTxRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN payload
struct cuPHYTestPschRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUPHY_TEST_WRKR_RSP_MSG_SRS_RUN payload
struct cuPHYTestSRSRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUPHY_TEST_WRKR_RSP_MSG_SET_WAIT_VAL payload
struct cuPHYTestSetWaitValCmdMsgPayload
{
    bool     rsp;
    uint32_t workerId;
    uint32_t syncFlagVal;
};

// CUPHY_TEST_WRKR_RSP_MSG_READ_SM_IDS payload
struct cuPHYTestReadSmIdsRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrWaitEvent;
};

struct cuPHYTestRspMsgPayload
{
    uint32_t workerId;
};

//----------------------------------------------------------------------------------------------------------
// cuPHYTestWorker - A worker class which uses resources (CPU thread, CUDA sub-context, HW CPU/GPU Ids) to
// setup/run one or more UL/DL pipelines. Worker accepts commands via message based interface.
//----------------------------------------------------------------------------------------------------------
class cuPHYTestWorker {
public:
    cuPHYTestWorker(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsActiveThrdPct, std::shared_ptr<cuphyTestWrkrCmdQ>& cmdQ, std::shared_ptr<cuphyTestWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel);

    cuPHYTestWorker(cuPHYTestWorker const&) 
    { 
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Error: Copy construction not supported");
    };
    cuPHYTestWorker& operator=(cuPHYTestWorker const&) = delete;
    ~cuPHYTestWorker();

    void init(uint32_t nStrms, uint32_t nItrsPerStrm, uint32_t nTimingItrs, std::map<std::string, int>& cuStrmPrioMap, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrCpuGpuSyncFlag, uint32_t nPDCCHCells = 0, uint32_t nCSIRSCells = 0, uint32_t nBWCCells = 0, uint32_t nSRSCells = 0, uint32_t nPRACHCells = 0, uint32_t nPUCCHCells = 0, uint32_t nSSBCells = 0, bool srsSplit = false, bool waitRsp = true, uint32_t longPattern = 0, bool srsCtx = false);
    void bfcInit(std::vector<std::string> inFileNamesBFC, bool ref_check_bfc, bool waitRsp = true);
    void bfcSetup(std::vector<std::string> inFileNamesBFC, bool waitRsp = true);
    void prachInit(std::vector<std::string> inFileNamesPRACH, uint64_t proc_mode, bool ref_check_prach, bool group_cells, uint32_t cells_per_stream, bool waitRsp = true);
    void prachSetup(std::vector<std::string> inFileNamesPRACH, bool waitRsp = true);
    void pdcchTxSetup(std::vector<std::string> inFileNamesPDCCH, bool waitRsp = true);
    void srsSetup(std::vector<std::string> inFileNamesBFC, bool waitRsp = true);
    void csirsInit(std::vector<std::string> inFileNamesCSIRS, bool ref_check_csirs, bool group_cells, uint64_t csirs_proc_mode, bool waitRsp = true);
    void csirsSetup(std::vector<std::string> inFileNamesCSIRS, bool waitRsp = true);
    void ssbInit(std::vector<std::string> inFileNamesSSB, bool ref_check_ssb, bool group_cells, uint32_t nSsbSlots, uint64_t ssb_proc_mode, bool waitRsp = true);
    void ssbSetup(std::vector<std::string> inFileNamesSSB, bool waitRsp = true);
    void srsInit(std::vector<std::string> inFileNamesSRS, bool ref_check_srs, uint64_t srs_proc_mode, bool splitSRS = false, bool waitRsp = true);
    void pdcchTxInit(std::vector<std::string> inFileNamesPDCCH, bool group_cells, uint32_t cells_per_stream, bool ref_check_pdcch, uint64_t pdcch_proc_mode, bool waitRsp = true);
    void pucchRxInit(std::vector<std::string> inFileNamesPUCCH, bool ref_check_pucch, bool groupCells, uint64_t pucch_proc_mode, bool waitRsp = true);
    void pucchRxSetup(std::vector<std::string> inFileNamesPUCCH, bool waitRsp = true);
    void puschRxInit(std::vector<std::string> inFileNamesPuschRx, uint32_t fp16Mode, int puschRxDescramblingOn, bool printCbErrors, uint64_t pusch_proc_mode, bool enableLdpcThroughputMode, bool groupCells, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode, bool waitRsp = true);
    void pdschTxInit(std::vector<std::string> inFileNamesPdschTx, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, bool group_cells, uint32_t cells_per_stream, maxPDSCHPrms pdschPrms, bool waitRsp = true);

    void deinit(bool waitRsp = true);
    void puschRxSetup(std::vector<std::string> inFileNamesPuschRx, bool waitRsp = true);
    void pdschTxSetup(std::vector<std::string> inFileNamesPdschTx, bool waitRsp = true);
    void puschRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, cudaEvent_t prachStartEvent = nullptr, cudaEvent_t pucchStartEvent = nullptr, bool waitRsp = true);
    void pdschTxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp = true, std::vector<cuphy::event>* pdcchStopEventVec = nullptr, std::vector<cuphy::event>* pdschInterSlotEventVec = nullptr);
    void pschTxRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp = true);
    void eval(bool cbErrors = false, bool isPschTxRx = false, bool waitRsp = true);
    void print(bool cbErrors = false, bool isPschTxRx = false, bool waitRsp = true);
    void resetEvalBuffers(bool cbErrors = false, bool waitRsp = true);
    void pdschTxClean(bool waitRsp = true);
    void setWaitVal(uint32_t syncFlagVal, bool waitRsp = true);

    void runSSB(std::shared_ptr<void>& shPtrPayload);
    void runPRACH(const cudaEvent_t& startEvent);
    void runPUCCH(const cudaEvent_t& startEvent);
    void runPUCCH_U5(const cudaEvent_t& startEvent, const cudaEvent_t& startEvent2);
    void runPDSCH_U5_3_6(std::shared_ptr<void>& shPtrPayload);
    void runPDSCH_U3_U5_1_2_4_5(std::shared_ptr<void>& shPtrPayload);
    void runSRS1(const cudaEvent_t& startEvent);
    void runPUSCH(const cudaEvent_t& startEvent);
    void runPUSCH_U5(const cudaEvent_t& startEvent);
    void runSRS2();
    void runPDCCH(std::shared_ptr<void>& shPtrPayload);
    void runPDCCHItr(const cudaEvent_t& pdcchSlotStartEvent, uint32_t itrIdx);

    void                       readSmIds(std::shared_ptr<cuphy::event>& shPtrRdSmIdWaitEvent, bool waitRsp = true);
    void                       getPuschRxRunRsp(std::shared_ptr<cuPHYTestWrkrRspMsg>& shPtrRsp);
    void                       getPdschTxRunRsp(std::shared_ptr<cuPHYTestWrkrRspMsg>& shPtrRsp);
    void                       getPschTxRxRunRsp(std::shared_ptr<cuPHYTestWrkrRspMsg>& shPtrRsp);
    std::vector<float>         getBWCIterStartTimes();
    std::vector<float>         getBWCIterTimes();
    std::vector<float>         getPdcchStartTimes();
    std::vector<float>         getPdcchIterTimes();
    std::vector<float>         getCSIRSStartTimes();
    std::vector<float>         getCSIRSIterTimes();
    std::vector<float>         getPdschIterTimes();
    std::vector<float>         getPdschSlotStartTimes();
    std::vector<float>         getTotSSBStartTime();
    std::vector<float>         getTotSSBRunTime();
    float                      getTotSRSStartTime();
    float                      getTotSRSRunTime();
    float                      getTotSRS2StartTime();
    float                      getTotSRS2RunTime();
    float                      getTotPrachStartTime();
    float                      getTotPrachRunTime();
    float                      getTotPuschStartTime();
    float                      getTotPuschRunTime();
    float                      getTotPusch2StartTime();
    float                      getTotPusch2RunTime();
    float                      getTotPucchStartTime();
    float                      getTotPucchRunTime();
    float                      getTotPucch2StartTime();
    float                      getTotPucch2RunTime();
    std::vector<cuphy::event>* getPdschInterSlotEventVecPtr();
    std::vector<cuphy::event>* getPdcchStopEventVecPtr();
    cudaEvent_t                getPuschStartEvent();
    cudaEvent_t                getPusch2StartEvent();
    inline uint64_t            getCuCtxId() { return reinterpret_cast<uint64_t>(m_cuCtx.handle()); }
    uint32_t*                  getSmIdsGpu(uint32_t& nSmIds)
    {
        nSmIds = m_nSmIds;
        return m_smIdsGpu.addr();
    };
    uint32_t* getSmIdsCpu(uint32_t& nSmIds)
    {
        nSmIds = m_nSmIds;
        return m_smIdsCpu.addr();
    };

private:
    void msgProcess(std::shared_ptr<cuPHYTestWrkrCmdMsg>& shPtrMsg); // reference to shared_ptr optional (used for performance)
    void initHandler(std::shared_ptr<void>& shPtrPayload);
    void puschRxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void pdcchTxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void csirsInitHandler(std::shared_ptr<void>& shPtrPayload);
    void ssbInitHandler(std::shared_ptr<void>& shPtrPayload);
    void bfcInitHandler(std::shared_ptr<void>& shPtrPayload);
    void bfcSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void pdcchTxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void prachInitHandler(std::shared_ptr<void>& shPtrPayload);
    void srsInitHandler(std::shared_ptr<void>& shPtrPayload);
    void pucchRxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void deinitHandler(std::shared_ptr<void>& shPtrPayload);
    void exitHandler(std::shared_ptr<void>& shPtrPayload);
    void puschRxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void srsSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void pucchRxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void ssbSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void csirsSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void prachSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void puschRxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pdcchTxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pschTxRxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pschTxRxRunHandlerNoStrmPrio(std::shared_ptr<void>& shPtrPayload);
    void pschTxRxRunHandlerStrmPrio(std::shared_ptr<void>& shPtrPayload);
    void evalHandler(std::shared_ptr<void>& shPtrPayload);
    void printHandler(std::shared_ptr<void>& shPtrPayload);
    void resetEvalHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxCleanHandler(std::shared_ptr<void>& shPtrPayload);
    void emptyHandler(std::shared_ptr<void>& shPtrPayload);
    void pschRdSmIdHandler(std::shared_ptr<void>& shPtrPayload);
    void setWaitValHandler(std::shared_ptr<void>& shPtrPayload);

    void run();
    void createCuCtx();
    void readSmIds();
    void setThrdProps();

    // worker identity
    std::string     m_name;
    std::thread::id m_thrdId;
    int32_t         m_wrkrId;

    // GPU side info
    int                m_gpuId;
    int32_t            m_mpsSubctxSmCount; // Fraction of GPU SMs allocated to the CUDA sub-context
    cuphy::cudaContext m_cuCtx;            // CUDA sub-context

    // CPU side info
    int         m_cpuId;
    std::thread m_thrd;
    int         m_schdPolicy;
    int         m_prio;

    // Communication
    std::shared_ptr<cuphyTestWrkrCmdQ> m_shPtrCmdQ;
    std::shared_ptr<cuphyTestWrkrRspQ> m_shPtrRspQ;

    int m_uldlMode;

    // Run parameters (specify how workload divided for each slot pattern)
    uint32_t m_longPattern;  // 0: DDDSU, >= 1: DDDSUUDDDD patterns
    uint32_t m_nPRACHCells;  // number of PRACH cells
    uint32_t m_nSRSCells;    // number of SRS cells
    uint32_t m_nBWCCells;    // number of BWC cells
    uint32_t m_nPDCCHCells;  // number of PDCCH cells
    uint32_t m_nCSIRSCells;  // number of CSIRS cells
    uint32_t m_nPUCCHCells;  // number of PUCCCH  cells
    uint32_t m_nSSBCells;    // number of SSB  cells
    uint32_t m_nStrms;       // number of parallel workers
    uint32_t m_nItrsPerStrm; // number of iterations per parallel worker. Note: m_nCells = m_nStrms * m_nItrsPerStrm
    uint32_t m_nCellsPerStrm_pdsch; //
    uint32_t m_nCellsPerStrm_pdcch; //
    uint32_t m_nCellsPerStrm_pucch; // number of iterations per parallel worker. Note: m_nPUCCHCells = m_nStrms_pucch * m_nCellsPerStrm_pucch
    uint32_t m_nStrms_pdsch;        // number of parallel workers for pdsch for uldl==4
    uint32_t m_nStrms_pdcch;        // number of parallel workers for pdcch for uldl==4
    uint32_t m_nStrms_pucch;        // number of parallel workers for pucch for uldl==4 //ToDO?? is it better to change the name to m_nCellGroups_pucch?
    uint32_t m_nCellsPerStrm_prach; //
    uint32_t m_nStrms_prach;        // number of parallel workers for prach
    uint32_t m_nSsbSlots;           // number SSBs per pattern, starting from slot 4

    // Stream PRIOs
    uint32_t m_cuStrmPrioPusch;
    uint32_t m_cuStrmPrioPusch2;
    uint32_t m_cuStrmPrioPdsch;
    uint32_t m_cuStrmPrioPdcch;
    uint32_t m_cuStrmPrioPrach;
    uint32_t m_cuStrmPrioPucch;
    uint32_t m_cuStrmPrioPucch2;
    uint32_t m_cuStrmPrioSRS;
    uint32_t m_cuStrmPrioSSB;
    uint32_t m_cuStrmPrioCSIRS;

    // streams. Dim: m_nStrms
    std::vector<cuphy::stream> m_cuStrms;
    std::vector<cuphy::stream> m_cuStrmsPusch;
    std::vector<cuphy::stream> m_cuStrmsPusch2;
    std::vector<cuphy::stream> m_cuStrmsPdsch;
    std::vector<cuphy::stream> m_cuStrmsSRS;
    std::vector<cuphy::stream> m_cuStrmsSSB;
    std::vector<cuphy::stream> m_cuStrmsPrach;
    std::vector<cuphy::stream> m_cuStrmsPdcch;
    std::vector<cuphy::stream> m_cuStrmsCSIRS;
    std::vector<cuphy::stream> m_cuStrmsPucch;

    // Pipeline handles. Dim: m_nStrms x m_nItrsPerStrm
    std::vector<std::vector<cuphy::pusch_rx>> m_puschRxPipes;
    std::vector<std::vector<cuphy::pdsch_tx>> m_pdschTxPipes;
    std::vector<std::vector<cuphy::pdcch_tx>> m_pdcchTxPipes;
    std::vector<std::vector<cuphy::pucch_rx>> m_pucchRxPipes;

    std::shared_ptr<cuphy::event>  m_shPtrStopEvent;
    std::shared_ptr<cuphy::event>  m_shPtrRdSmIdWaitEvent;
    std::unique_ptr<cuphy::stream> m_uqPtrWrkrCuStrm;
    std::unique_ptr<cuphy::event>  m_uqPtrPdschIterStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPuschDelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPusch2DelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSRSDelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSRSStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSRS2StopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSSBStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPRACHStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrBWCStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPUCCHStopEvent;

    std::vector<cuphy::event> m_stopEvents;       // Dim: m_nStrms
    std::vector<cuphy::event> m_stop2Events;      // Dim: m_nStrms
    std::vector<cuphy::event> m_SRSStopEvents;    // Dim: m_nStrms
    std::vector<cuphy::event> m_PDCCHStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_CSIRSStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_PRACHStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_PUCCHStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_PUCCHStop2Events; // Dim: m_nStrms
    std::vector<cuphy::event> m_pdschInterSlotStartEventVec;
    std::vector<cuphy::event> m_pdcchInterSlotStartEventVec;

    std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>> m_shPtrGpuStartSyncFlag;
    CUdeviceptr                                                   m_ptrGpuStartSyncFlag;

    std::unique_ptr<cuphy::event> m_uqPtrTimeStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRSStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRSEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRS2StartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRS2EndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePRACHStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePRACHEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCHStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCHEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCH2StartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCH2EndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCHStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCHEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCH2StartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCH2EndEvent;

    // change to multiple SSB slots
    std::vector<cuphy::event>              m_timeSSBSlotStartEvents;
    std::vector<cuphy::event>              m_timeSSBSlotEndEvents;

    std::vector<std::vector<cuphy::event>> m_pschDlUlSyncEvents; // Dim: m_nStrms x nItrsPerStrm
    std::vector<cuphy::event>              m_timePdschSlotEndEvents;
    std::vector<cuphy::event>              m_timeNextPdschSlotStartEvents;
    std::vector<cuphy::event>              m_timeNextPdcchSlotStartEvents;
    std::vector<cuphy::event>              m_timeNextCSIRSSlotStartEvents;
    std::vector<cuphy::event>              m_timeBWCSlotStartEvents;
    std::vector<cuphy::event>              m_timeBWCSlotEndEvents;
    std::vector<cuphy::event>              m_timePdcchSlotEndEvents;
    std::vector<cuphy::event>              m_timeCSIRSSlotEndEvents;

    // Datasets. Dim: m_nStrms x m_nItrsPerStrm
    std::vector<std::vector<StaticApiDataset>>      m_puschRxStaticApiDataSets;
    std::vector<std::vector<DynApiDataset>>         m_puschRxDynamicApiDataSets;
    std::vector<std::vector<EvalDataset>>           m_puschRxEvalDataSets;
    std::vector<std::vector<pdschStaticApiDataset>> m_pdschTxStaticApiDataSets;
    std::vector<std::vector<pdschDynApiDataset>>    m_pdschTxDynamicApiDataSets;
    std::vector<std::vector<pdcchStaticApiDataset>> m_pdcchTxStaticApiDataSets;
    std::vector<std::vector<pdcchDynApiDataset>>    m_pdcchTxDynamicApiDataSets;

    // PUCCH
    std::vector<std::vector<pucchStaticApiDataset>>    m_pucchStaticDatasetVec;
    std::vector<std::vector<pucchDynApiDataset>>       m_pucchDynDatasetVec;
    std::vector<std::vector<EvalPucchDataset>>         m_pucchEvalDatasetVec;
    std::vector<std::vector<cuphyPucchBatchPrmHndl_t>> m_pucchBatchPrmHndlVec; // not used for the moment

    //SRS
    std::vector<std::vector<srsStaticApiDataset>> m_srsStaticApiDatasetVec;
    std::vector<std::vector<srsDynApiDataset>>    m_srsDynamicApiDatasetVec;
    std::vector<std::vector<srsEvalDataset>>      m_srsEvalDatasetVec;

    std::vector<std::vector<srsStaticApiDataset>> m_srsStaticApiDatasetVec2;
    std::vector<std::vector<srsDynApiDataset>>    m_srsDynamicApiDatasetVec2;
    std::vector<std::vector<srsEvalDataset>>      m_srsEvalDatasetVec2;

    cuphySrsRxHndl_t m_srsRxHndl;
    cuphySrsRxHndl_t m_srsRxHndl2;

    std::vector<std::vector<uint32_t>>                                    m_nSRSCellsVec;
    std::vector<std::vector<cuphySrsChEstDynPrms_t>>                      m_srsDynPrmsVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tDataRxVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tFreqInterpCoefsVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tSRSHEstVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tSRSDbgVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>>> m_statDescrBufCpuVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>>> m_dynDescrBufCpuVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::device_alloc>>> m_statDescrBufGpuVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::device_alloc>>> m_dynDescrBufGpuVec;
    std::vector<std::vector<cuphySrsChEstHndl_t>>                         m_srsChEstHndlVec;

    // BFC parameters

    std::vector<std::vector<bfwStaticApiDataset>> m_bwcStaticApiDatasetVec;
    std::vector<std::vector<bfwDynApiDataset>>    m_bwcDynamicApiDatasetVec;
    std::vector<std::vector<bfwEvalDataset>>      m_bwcEvalDatasetVec;
    std::vector<std::vector<cuphy::bfw_tx>>       m_bwcPipelineVec;

    // PRACH
    std::vector<std::vector<PrachApiDataset>> m_prachDatasetVec;
    std::vector<std::vector<cuphy::prach_rx>> m_prachRxPipes;

    //SSB
    std::vector<std::vector<cuphy::ssb_tx>>       m_ssbTxPipes;
    std::vector<std::vector<ssbStaticApiDataset>> m_ssbTxStaticApiDataSets;
    std::vector<std::vector<ssbDynApiDataset>>    m_ssbTxDynamicApiDataSets;

    //CSIRS
    std::vector<std::vector<cuphy::csirs_tx>>       m_csirsTxPipes;
    std::vector<std::vector<csirsStaticApiDataset>> m_csirsTxStaticApiDataSets;
    std::vector<std::vector<csirsDynApiDataset>>    m_csirsTxDynamicApiDataSets;

    // timing objects.
    std::vector<float>                           m_totRunTimePdschItr;    // Dim: nItrsPerStrm
    std::vector<float>                           m_totPdschSlotStartTime; // Dim: nItrsPerStrm
    std::vector<float>                           m_totBWCIterStartTime;      // Dim: nItrsPerStrm
    std::vector<float>                           m_totRunTimeBWCItr;      // Dim: nItrsPerStrm
    std::vector<float>                           m_totPdcchStartTimes;    // Dim: nItrsPerStrm
    std::vector<float>                           m_totRunTimePdcchItr;    // Dim: nItrsPerStrm
    std::vector<float>                           m_totCSIRSStartTimes;    // Dim: nItrsPerStrm
    std::vector<float>                           m_totRunTimeCSIRSItr;    // Dim: nItrsPerStrm
    std::vector<std::vector<cuphy::event_timer>> m_timerSingleStreamItr;  // Dim: nItrsPerStrm
    std::vector<float>                           m_totSSBStartTime; // Dim: m_nSsbSlots
    std::vector<float>                           m_totSSBRunTime;   // Dim: m_nSsbSlots
    float                                        m_totSRSStartTime;
    float                                        m_totSRSRunTime;
    float                                        m_totSRS2StartTime;
    float                                        m_totSRS2RunTime;
    float                                        m_totPRACHStartTime;
    float                                        m_totPRACHRunTime;
    float                                        m_totPUSCHStartTime;
    float                                        m_totPUSCHRunTime;
    float                                        m_totPUSCH2RunTime;
    float                                        m_totPUSCH2StartTime;
    float                                        m_totPUCCHRunTime;
    float                                        m_totPUCCHStartTime;
    float                                        m_totPUCCH2RunTime;
    float                                        m_totPUCCH2StartTime;

    // variable delay memory pointer
    cuphy::buffer<uint64_t, cuphy::device_alloc> m_GPUtime_d;

    // evaluation objects
    std::vector<std::vector<uint32_t>> m_maxNumCbErrors; // Dim: m_nStrms x m_nItrsPerStrm
    uint32_t                           m_nTimingItrs;
    bool                               m_printCbErrors;

    // pusch/pdsch configurations
    uint32_t                                     m_dbgMsgLevel;
    uint32_t                                     m_descramblingOn;
    bool                                         m_ref_check_pdsch;
    bool                                         m_ref_check_pdcch;
    bool                                         m_ref_check_csirs;
    bool                                         m_ref_check_pucch;
    bool                                         m_ref_check_prach;
    bool                                         m_ref_check_ssb;
    bool                                         m_ref_check_srs;
    bool                                         m_ref_check_bfc;
    bool                                         m_identical_ldpc_configs;
    cuphyPdschProcMode_t                         m_pdsch_proc_mode;
    uint64_t                                     m_pdcch_proc_mode;
    uint64_t                                     m_csirs_proc_mode;
    uint64_t                                     m_ssb_proc_mode;
    uint64_t                                     m_pusch_proc_mode;
    uint64_t                                     m_pucch_proc_mode;
    uint64_t                                     m_prach_proc_mode;
    uint64_t                                     m_srs_proc_mode;
    uint32_t                                     m_ldpc_kernel_launch_mode;
    uint32_t                                     m_fp16Mode;
    int32_t                                      m_smCount;
    uint32_t                                     m_nSmIds;
    bool                                         m_runBWC;
    bool                                         m_runPDSCH;
    bool                                         m_runSRS;
    bool                                         m_runSRS2;
    bool                                         m_runPRACH;
    bool                                         m_runPUSCH;
    bool                                         m_runPDCCH;
    bool                                         m_runCSIRS;
    bool                                         m_runPUCCH;
    bool                                         m_runSSB;
    bool                                         m_pdsch_group_cells;
    bool                                         m_pdcch_group_cells;
    bool                                         m_pusch_group_cells;
    bool                                         m_pucch_group_cells;
    bool                                         m_prach_group_cells;
    bool                                         m_srsCtx;  // if true, srs runs on a separate context with a different fixed delay
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> m_smIdsCpu;
    cuphy::buffer<uint32_t, cuphy::device_alloc> m_smIdsGpu;
};

#endif // !defined(PSCH_RX_TX_CMN_HPP_INCLUDED_)
