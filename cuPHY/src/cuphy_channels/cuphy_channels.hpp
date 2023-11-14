/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CUPHY_CHANNELS_HPP_INCLUDED_)
#define CUPHY_CHANNELS_HPP_INCLUDED_

#include <iostream>
#include "cuphy_api.h"
#include "cuphy.hpp"

namespace cuphy
{

////////////////////////////////////////////////////////////////////////
// pusch_rx_deleter
struct pusch_rx_deleter
{
    typedef cuphyPuschRxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyPuschRx(p);
    }

};

////////////////////////////////////////////////////////////////////////
// unique_pusch_rx_ptr
using unique_pusch_rx_ptr = std::unique_ptr<cuphyPuschRx, pusch_rx_deleter>;

////////////////////////////////////////////////////////////////////////
// pusch_rx
class pusch_rx
{
public:
    //------------------------------------------------------------------
    // pusch_rx()
    pusch_rx(const cuphyPuschStatPrms_t& staticParams,
             cudaStream_t                strm = 0)
    {
        cuphyPuschRxHndl_t rx = nullptr;
        cuphyStatus_t      s  = cuphyCreatePuschRx(&rx,
                                                   &staticParams,
                                                   strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
        rx_ptr_.reset(rx);
    }
    pusch_rx() : rx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    cuphyStatus_t setup(cuphyPuschDynPrms_t& dynamicParams, const cuphyPuschBatchPrmHndl_t batchPrmHndl)
    {     
        cuphyStatus_t s = cuphySetupPuschRx(handle(),
                                            &dynamicParams,
                                            batchPrmHndl);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            if (dynamicParams.pStatusOut->status == cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB) 
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB error in cuphySetupPuschRx(): {}. Triggered by TB {} in cell group and cellPrmStatIdx {}", cuphyGetErrorString(s), dynamicParams.pStatusOut->ueIdx, dynamicParams.pStatusOut->cellPrmStatIdx);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            } 
            else if (dynamicParams.pStatusOut->status == cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH) 
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH error in cuphySetupPuschRx(): {}. Triggered by TB {} in cell group and cellPrmStatIdx {}", cuphyGetErrorString(s), dynamicParams.pStatusOut->ueIdx, dynamicParams.pStatusOut->cellPrmStatIdx);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            } 
            else 
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Error! cuphySetupPuschRx(): {}", cuphyGetErrorString(s));
                throw cuphy::cuphy_exception(s);
            }
            
        }
        return CUPHY_STATUS_SUCCESS;
    }
    //------------------------------------------------------------------
    // run()
    void run(cuphyPuschRunPhase_t runPhase)
    {
        cuphyStatus_t s = cuphyRunPuschRx(handle(), runPhase);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // writeDbgSynch()
    void writeDbgSynch(cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyWriteDbgBufSynch(handle(),
                                                strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }

    pusch_rx(pusch_rx&& rx) : rx_ptr_(std::move(rx.rx_ptr_)) {}
    pusch_rx& operator=(pusch_rx&& rx) { rx_ptr_ = std::move(rx.rx_ptr_); return *this;}

    cuphyPuschRxHndl_t handle() { return rx_ptr_.get(); }

    pusch_rx& operator=(const pusch_rx&) = delete;
    pusch_rx(const pusch_rx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_pusch_rx_ptr rx_ptr_;
};


////////////////////////////////////////////////////////////////////////
// pdsch_tx_deleter
struct pdsch_tx_deleter
{
    typedef cuphyPdschTxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyPdschTx(p);
    }

};

////////////////////////////////////////////////////////////////////////
// unique_pdsch_tx_ptr
using unique_pdsch_tx_ptr = std::unique_ptr<cuphyPdschTx, pdsch_tx_deleter>;

////////////////////////////////////////////////////////////////////////
// pdsch_tx
class pdsch_tx
{
public:
    //------------------------------------------------------------------
    // pdsch_tx()
    pdsch_tx(const cuphyPdschStatPrms_t& staticParams)
    {
        cuphyPdschTxHndl_t tx = nullptr;
        cuphyStatus_t      s  = cuphyCreatePdschTx(&tx,
                                                   &staticParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
        tx_ptr_.reset(tx);
    }
    pdsch_tx() : tx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    void setup(cuphyPdschDynPrms_t &         dynamicParams,
               const cuphyPdschBatchPrmHndl_t batchPrmHndl)
    {
        cuphyStatus_t s = cuphySetupPdschTx(handle(),
                                            &dynamicParams,
                                            batchPrmHndl);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // run()
    void run(uint64_t      procModeBmsk)
    {
        cuphyStatus_t s = cuphyRunPdschTx(handle(),
                                         procModeBmsk);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }

    pdsch_tx(pdsch_tx&& tx) : tx_ptr_(std::move(tx.tx_ptr_)) {}
    pdsch_tx& operator=(pdsch_tx&& tx) { tx_ptr_ = std::move(tx.tx_ptr_); return *this;}

    cuphyPdschTxHndl_t handle() { return tx_ptr_.get(); }

    pdsch_tx& operator=(const pdsch_tx&) = delete;
    pdsch_tx(const pdsch_tx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_pdsch_tx_ptr tx_ptr_;
};


////////////////////////////////////////////////////////////////////////
// pdcch_tx_deleter
struct pdcch_tx_deleter
{
    typedef cuphyPdcchTxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyPdcchTx(p);
    }
};

////////////////////////////////////////////////////////////////////////
// unique_pdcch_tx_ptr
using unique_pdcch_tx_ptr = std::unique_ptr<cuphyPdcchTx, pdcch_tx_deleter>;

////////////////////////////////////////////////////////////////////////
// pdcch_tx
class pdcch_tx
{
public:
    //------------------------------------------------------------------
    // pdcch_tx()
    pdcch_tx(const cuphyPdcchStatPrms_t& staticParams)
    {
        cuphyPdcchTxHndl_t tx = nullptr;
        cuphyStatus_t      s  = cuphyCreatePdcchTx(&tx,
                                                   &staticParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! cuphyCreatePdcchTx()");
            throw cuphy::cuphy_exception(s);
        }
        tx_ptr_.reset(tx);
    }
    pdcch_tx() : tx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    void setup(cuphyPdcchDynPrms_t &         dynamicParams)
    {
        cuphyStatus_t s = cuphySetupPdcchTx(handle(),
                                            &dynamicParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error! cuphySetupPdcchTx()");
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // run()
    void run(uint64_t      procModeBmsk)
    {
        cuphyStatus_t s = cuphyRunPdcchTx(handle(),
                                          procModeBmsk);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error! cuphyRunPdcchTx()");
            throw cuphy::cuphy_exception(s);
        }
    }

    pdcch_tx(pdcch_tx&& tx) : tx_ptr_(std::move(tx.tx_ptr_)) {}
    pdcch_tx& operator=(pdcch_tx&& tx) { tx_ptr_ = std::move(tx.tx_ptr_); return *this;}

    cuphyPdcchTxHndl_t handle() { return tx_ptr_.get(); }

    pdcch_tx& operator=(const pdcch_tx&) = delete;
    pdcch_tx(const pdcch_tx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_pdcch_tx_ptr tx_ptr_;
};

////////////////////////////////////////////////////////////////////////
// ssb_tx_deleter
struct ssb_tx_deleter
{
    typedef cuphySsbTxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroySsbTx(p);
    }
};

////////////////////////////////////////////////////////////////////////
// unique_ssb_tx_ptr
using unique_ssb_tx_ptr = std::unique_ptr<cuphySsbTx, ssb_tx_deleter>;

////////////////////////////////////////////////////////////////////////
// ssb_tx
class ssb_tx
{
public:
    //------------------------------------------------------------------
    // ssb_tx()
    ssb_tx(const cuphySsbStatPrms_t& staticParams)
    {
        cuphySsbTxHndl_t tx = nullptr;
        cuphyStatus_t      s  = cuphyCreateSsbTx(&tx,
                                                  &staticParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error! cuphyCreateSsbTx()");
            throw cuphy::cuphy_exception(s);
        }
        tx_ptr_.reset(tx);
    }
    ssb_tx() : tx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    void setup(cuphySsbDynPrms_t &         dynamicParams)
    {
        cuphyStatus_t s = cuphySetupSsbTx(handle(),
                                           &dynamicParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error! cuphySetupSsbTx()");
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // run()
    void run(uint64_t      procModeBmsk)
    {
        cuphyStatus_t s = cuphyRunSsbTx(handle(),
                                         procModeBmsk);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error! cuphyRunSsbTx()");
            throw cuphy::cuphy_exception(s);
        }
    }

    ssb_tx(ssb_tx&& tx) : tx_ptr_(std::move(tx.tx_ptr_)) {}
    ssb_tx& operator=(ssb_tx&& tx) { tx_ptr_ = std::move(tx.tx_ptr_); return *this;}

    cuphySsbTxHndl_t handle() { return tx_ptr_.get(); }

    ssb_tx& operator=(const ssb_tx&) = delete;
    ssb_tx(const ssb_tx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_ssb_tx_ptr tx_ptr_;
};


////////////////////////////////////////////////////////////////////////
// pucch_rx_deleter
struct pucch_rx_deleter
{
    typedef cuphyPucchRxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyPucchRx(p);
    }
};

////////////////////////////////////////////////////////////////////////
// unique_pdsch_tx_ptr
using unique_pucch_rx_ptr = std::unique_ptr<cuphyPucchRx, pucch_rx_deleter>;

////////////////////////////////////////////////////////////////////////
// pucch_rx
class pucch_rx
{
public:
    //------------------------------------------------------------------
    // pucch_rx()
    pucch_rx(const cuphyPucchStatPrms_t& staticParams,
             cudaStream_t                strm = 0)
    {
        cuphyPucchRxHndl_t rx = nullptr;
        cuphyStatus_t      s  = cuphyCreatePucchRx(&rx,
                                                   &staticParams,
                                                   strm);

        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
        rx_ptr_.reset(rx);
    }
    pucch_rx() : rx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    void setup(cuphyPucchDynPrms_t&           dynamicParams,
               const cuphyPucchBatchPrmHndl_t batchPrmHndl)
    {
        cuphyStatus_t s = cuphySetupPucchRx(handle(),
                                            &dynamicParams,
                                            batchPrmHndl);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // run()
    void run(uint64_t procModeBmsk)
    {
        cuphyStatus_t s = cuphyRunPucchRx(handle(),
                                          procModeBmsk);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // writeDbgSynch()
    void writeDbgSynch(cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyWriteDbgBufSynchPucch(handle(),
                                                     strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }

    pucch_rx(pucch_rx&& rx) : rx_ptr_(std::move(rx.rx_ptr_)) {}
    pucch_rx& operator=(pucch_rx&& rx) { rx_ptr_ = std::move(rx.rx_ptr_); return *this;}

    cuphyPucchRxHndl_t handle() { return rx_ptr_.get(); }

    pucch_rx& operator=(const pucch_rx&) = delete;
    pucch_rx(const pucch_rx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_pucch_rx_ptr rx_ptr_;
};

////////////////////////////////////////////////////////////////////////
// bfw_tx_deleter
struct bfw_tx_deleter
{
    typedef cuphyBfwTxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyBfwTx(p);
    }
};

// unique_bfw_tx_ptr
using unique_bfw_tx_ptr = std::unique_ptr<cuphyBfwTx, bfw_tx_deleter>;

////////////////////////////////////////////////////////////////////////
// bfw_tx
class bfw_tx
{
public:
    //------------------------------------------------------------------
    // bfw_tx()
    bfw_tx(const cuphyBfwStatPrms_t& staticParams,
           cudaStream_t              cuStrm = 0)
    {
        cuphyBfwTxHndl_t tx = nullptr;
        cuphyStatus_t     s = cuphyCreateBfwTx(&tx,
                                               &staticParams,
                                               cuStrm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Error! cuphyCreateBfwTx()");
            throw cuphy::cuphy_exception(s);
        }
        tx_ptr_.reset(tx);
    }
    bfw_tx() : tx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    void setup(cuphyBfwDynPrms_t &dynamicParams)
    {
        cuphyStatus_t s = cuphySetupBfwTx(handle(),
                                          &dynamicParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Error! cuphySetupBfwTx()");
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // run()
    void run(uint64_t procModeBmsk)
    {
        cuphyStatus_t s = cuphyRunBfwTx(handle(),
                                        procModeBmsk);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Error! cuphyRunBfwTx()");
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // writeDbgSynch()
    void writeDbgSynch(cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyWriteDbgBufSynchBfw(handle(),
                                                     strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
    }


    bfw_tx(bfw_tx&& tx) noexcept : tx_ptr_(std::move(tx.tx_ptr_)) {}
    bfw_tx& operator=(bfw_tx&& tx) { tx_ptr_ = std::move(tx.tx_ptr_); return *this;}

    cuphyBfwTxHndl_t handle() { return tx_ptr_.get(); }

    bfw_tx& operator=(const bfw_tx&) = delete;
    bfw_tx(const bfw_tx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_bfw_tx_ptr tx_ptr_;
};

////////////////////////////////////////////////////////////////////////
// prach_rx_deleter
struct prach_rx_deleter
{
    typedef cuphyPrachRxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyPrachRx(p);
    }

};

////////////////////////////////////////////////////////////////////////
// unique_prach_rx_ptr
using unique_prach_rx_ptr = std::unique_ptr<cuphyPrachRx, prach_rx_deleter>;

////////////////////////////////////////////////////////////////////////
// prach_rx
class prach_rx
{
public:
    //------------------------------------------------------------------
    // prach_rx()
    prach_rx(const cuphyPrachStatPrms_t& staticParams)
    {
        cuphyPrachRxHndl_t rx = nullptr;
        cuphyStatus_t      s  = cuphyCreatePrachRx(&rx,
                                                   &staticParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "Error! cuphyCreatePrachRx()");
            throw cuphy::cuphy_exception(s);
        }
        rx_ptr_.reset(rx);
    }
    prach_rx() : rx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    void setup(cuphyPrachDynPrms_t &         dynamicParams)
    {
        cuphyStatus_t s = cuphySetupPrachRx(handle(),
                                            &dynamicParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "Error! cuphySetupPrachRx()");
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    // run()
    void run()
    {
        cuphyStatus_t s = cuphyRunPrachRx(handle());
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "Error! cuphySetupPrachRx()");
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------

    prach_rx(prach_rx&& rx) : rx_ptr_(std::move(rx.rx_ptr_)) {}
    prach_rx& operator=(prach_rx&& rx) { rx_ptr_ = std::move(rx.rx_ptr_); return *this;}

    cuphyPrachRxHndl_t handle() { return rx_ptr_.get(); }

    prach_rx& operator=(const prach_rx&) = delete;
    prach_rx(const prach_rx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_prach_rx_ptr rx_ptr_;
};


// csirs_tx_deleter
struct csirs_tx_deleter
{
    typedef cuphyCsirsTxHndl_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyCsirsTx(p);
    }
};

////////////////////////////////////////////////////////////////////////
// unique_csirs_tx_ptr
using unique_csirs_tx_ptr = std::unique_ptr<cuphyCsirsTx, csirs_tx_deleter>;

////////////////////////////////////////////////////////////////////////
// csirs_tx
class csirs_tx
{
public:
    //------------------------------------------------------------------
    // csirs_tx()
    csirs_tx(const cuphyCsirsStatPrms_t& staticParams)
    {
        cuphyCsirsTxHndl_t tx = nullptr;
        cuphyStatus_t      s  = cuphyCreateCsirsTx(&tx,
                                                   &staticParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Error! cuphyCreateCsirsTx()");
            throw cuphy::cuphy_exception(s);
        }
        tx_ptr_.reset(tx);
    }
    csirs_tx() : tx_ptr_(){}
    //------------------------------------------------------------------
    // setup()
    void setup(cuphyCsirsDynPrms_t &         dynamicParams)
    {
        cuphyStatus_t s = cuphySetupCsirsTx(handle(),
                                            &dynamicParams);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Error! cuphySetupCsirsTx()");
            throw cuphy::cuphy_exception(s);
        }
    }
    //------------------------------------------------------------------
    void run()
    {
        cuphyStatus_t s = cuphyRunCsirsTx(handle());
        if(CUPHY_STATUS_SUCCESS != s)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Error! cuphyRunCsirsTx()");
            throw cuphy::cuphy_exception(s);
        }
    }

    csirs_tx(csirs_tx&& tx) : tx_ptr_(std::move(tx.tx_ptr_)) {}
    csirs_tx& operator=(csirs_tx&& tx) { tx_ptr_ = std::move(tx.tx_ptr_); return *this;}

    cuphyCsirsTxHndl_t handle() { return tx_ptr_.get(); }

    csirs_tx& operator=(const csirs_tx&) = delete;
    csirs_tx(const csirs_tx&) = delete;
private:
    //------------------------------------------------------------------
    // Data
    unique_csirs_tx_ptr tx_ptr_;
};


} // namespace cuphy

#endif // !defined(CUPHY_CHANNELS_HPP_INCLUDED_)
