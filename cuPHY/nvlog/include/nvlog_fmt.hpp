#pragma once
#ifndef NVLOG_FMT_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
//#define FMTLOG_HEADER_ONLY
#ifdef NVIPC_FMTLOG_ENABLE
#include "fmtlog.h"
#endif
#pragma GCC diagnostic pop

#ifdef NVIPC_FMTLOG_ENABLE
#include <string_view>
#endif
#include <unistd.h>
#include "memtrace.h"
#include "exit_handler.hpp"
    
    
extern exit_handler& pExitHandler;
    
    
inline void EXIT_L1(int val)                                                    
{                                                                       
    if(val==EXIT_FAILURE)                                               
    {                                                                   
        pExitHandler.test_trigger_exit();                                   
    }                                                                   
    else                                                                
    {                                                                   
        exit(EXIT_SUCCESS);                                             
    }                                                                   
}                                                                       

#ifdef NVIPC_FMTLOG_ENABLE
#define NVLOG_PRINTF(log_level, component_id, format_printf, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
    if (log_level >= g_nvlog_component_levels[nvlog_get_component_id(component_id)]) { \
        try { \
            MemtraceDisableScope md; \
            FMTLOG_ONCE_PRINTF(log_level, "[%s] " format_printf, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
        } \
        catch (...) { \
            printf("Caught exception in NVLOG_PRINTF with log_level %d format string %s at %s:%d\n", log_level, format_printf, __FILE__,__LINE__); \
            ::EXIT_L1(EXIT_FAILURE); \
        } \
    } \
} while (0)

#define NVLOGE_PRINTF(log_level, event, component_id, format_printf, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
    static_assert(std::is_same<decltype(event),aerial_event_code_t>::value,"Event is not of type aerial_event_code_t"); \
    if (log_level >= g_nvlog_component_levels[nvlog_get_component_id(component_id)]) { \
        try { \
            MemtraceDisableScope md; \
            FMTLOG_ONCE_PRINTF(log_level, "[%s][%s] " format_printf, #event, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
        } \
        catch (...) { \
            printf("Caught exception in NVLOG_PRINTF with log_level %d format string %s at %s:%d\n", log_level, format_printf, __FILE__, __LINE__); \
            ::EXIT_L1(EXIT_FAILURE); \
        } \
    } \
} while (0)

#define NVLOGV_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::DBG,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGD_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::DBG,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGI_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::INF,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGW_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::WRN,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGC_FMT(component_id, format_fmt, ...) NVLOG_FMT(fmtlog::WRN,component_id, format_fmt, ##__VA_ARGS__)
#define NVLOGE_NO_FMT(component_id, event_level, format_fmt, ...) NVLOG_FMT_EVT(fmtlog::ERR,component_id, event_level, format_fmt, ##__VA_ARGS__)

#define NVLOGI_FMT_EVT(component_id, event_level, format_fmt, ...) NVLOG_FMT_EVT(fmtlog::INF,component_id, event_level, format_fmt, ##__VA_ARGS__)
#define NVLOGE_FMT(component_id, event_level, format_fmt, ...) NVLOG_FMT_EVT(fmtlog::ERR,component_id, event_level, format_fmt, ##__VA_ARGS__)

#define NVLOGF_FMT(component_id, event_level, format_fmt, ...) do { \
    MemtraceDisableScope md; \
    NVLOG_FMT_EVT(fmtlog::ERR,component_id, event_level, format_fmt, ##__VA_ARGS__); \
    usleep(100000); \
    ::pExitHandler.test_trigger_exit(); \
} while (0)

#define NVLOG_FMT(log_level, component_id, format_fmt, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
    if (log_level >= g_nvlog_component_levels[nvlog_get_component_id(component_id)]) { \
        try { \
            if (1) {/* Optimized method, string formatting in background thread */ \
                FMTLOG(log_level, "[{}] " format_fmt, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
            } else { /* Slow method, formatting in foreground thread, useful for catching exceptions */ \
                std::string s = fmt::format(format_fmt, ##__VA_ARGS__); \
                FMTLOG(log_level, "[{}] {}", nvlog_get_component_name(component_id), s); \
            } \
        } \
        catch (...) { \
            printf("Caught exception in NVLOG_FMT with log_level %d format string %s at %s:%d\n", log_level, format_fmt, __FILE__, __LINE__); \
        } \
    } \
} while (0)

#define NVLOG_FMT_EVT(log_level, component_id, event_level, format_fmt, ...) do { \
    static_assert(nvlog_component_is_valid(component_id),"Component ID (TAG) doesn't exist"); \
	static_assert(std::is_same<decltype(event_level),aerial_event_code_t>::value,"Event is not of type aerial_event_code_t"); \
	try { \
		FMTLOG(log_level, "[{}] [{}] " format_fmt, #event_level, nvlog_get_component_name(component_id), ##__VA_ARGS__); \
	} \
	catch (...) { \
		printf("Caught exception in NVLOG_FMT with log_level %d format string %s at %s:%d\n", log_level, format_fmt, __FILE__, __LINE__); \
    } \
} while (0)

struct nvlog_component_ids
{
    int id;
    const char* name;
};

constexpr int nvlog_max_id = 1024;

inline constexpr nvlog_component_ids g_nvlog_component_ids[] {
    //Reserve number 0 for no tag print
    {0, ""},

    // nvlog
    {10, "NVLOG"},
    {11, "NVLOG.TEST"},
    {12, "NVLOG.ITAG"},
    {13, "NVLOG.STAG"},
    {14, "NVLOG.STAT"},
    {15, "NVLOG.OBSERVER"},
    {16, "NVLOG.CPP"},
    {17, "NVLOG.SHM"},
    {18, "NVLOG.UTILS"},
    {19, "NVLOG.C"},
    {20, "NVLOG.EXIT_HANDLER"},

    // nvipc
    {30, "NVIPC"},
    {31, "NVIPC:YAML"},
    {32, "NVIPC.SHM_UTILS"},
    {33, "NVIPC.CUDAPOOL"},
    {34, "NVIPC.CUDAUTILS"},
    {35, "NVIPC.TESTCUDA"},
    {36, "NVIPC.SHM"},
    {37, "NVIPC.QUEUE"},
    {38, "NVIPC.IPC"},
    {39, "NVIPC.FD_SHARE"},
    {40, "NVIPC.DEBUG"},
    {41, "NVIPC.EFD"},
    {42, "NVIPC.EPOLL"},
    {43, "NVIPC.MEMPOOL"},
    {44, "NVIPC.RING"},
    {45, "NVIPC.SEM"},
    {46, "NVIPC.SHMLOG"},
    {47, "NVIPC.CONF"},
    {48, "NVIPC.DOCA"},
    {49, "NVIPC.DOCA_UTILS"},
    {50, "NVIPC.DPDK"},
    {51, "NVIPC.DPDK_UTILS"},
    {52, "NVIPC.GPUDATAUTILS"},
    {53, "NVIPC.GPUDATAPOOL"},
    {54, "NVIPC.TIMING"},
    {55, "NVIPC.DUMP"},
    {56, "NVIPC.UDP"},
    {57, "INIT"},
    {58, "TEST"},
    {59, "PHY"},
    {60, "MAC"},
    {61, "NVIPC.PCAP"},

    // cuPHYController
    {100, "CTL"},
    {101, "CTL.SCF"},
    {102, "CTL.ALTRAN"},
    {103, "CTL.DRV"},
    {104, "CTL.YAML"},
    {105, "CTL.STARTUP_TIMES"},

    // cuPHYDriver
    {200, "DRV"},
    {201, "DRV.SA"},
    {202, "DRV.TIME"},
    {203, "DRV.CTX"},
    {204, "DRV.API"},
    {205, "DRV.FH"},
    {206, "DRV.GEN_CUDA"},
    {207, "DRV.GPUDEV"},
    {208, "DRV.PHYCH"},
    {209, "DRV.TASK"},
    {210, "DRV.WORKER"},
    {211, "DRV.DLBUF"},
    {212, "DRV.CSIRS"},
    {213, "DRV.PBCH"},
    {214, "DRV.PDCCH_DL"},
    {215, "DRV.PDSCH"},
    {216, "DRV.MAP_DL"},
    {217, "DRV.FUNC_DL"},
    {218, "DRV.HARQ_POOL"},
    {219, "DRV.ORDER_CUDA"},
    {220, "DRV.ORDER_ENTITY"},
    {221, "DRV.PRACH"},
    {222, "DRV.PUCCH"},
    {223, "DRV.PUSCH"},
    {224, "DRV.MAP_UL"},
    {225, "DRV.FUNC_UL"},
    {226, "DRV.ULBUF"},
    {227, "DRV.MPS"},
    {228, "DRV.METRICS"},
    {229, "DRV.MEMFOOT"},
    {230, "DRV.CELL"},
    {231, "DRV.EXCP"},
    {232, "DRV.CV_MEM_BNK"},
    {233, "DRV.DLBFW"},
    {234, "DRV.ULBFW"},
    {235, "DRV.CUPHY_PTI"},
    {236, "DRV.SYMBOL_TIMINGS"},
    {237, "DRV.PACKET_TIMINGS"},
    {238, "DRV.UL_PACKET_SUMMARY"},
    {239, "DRV.SRS"},

    // cuphyl2adapter
    {300, "L2A"},
    {301, "L2A.MAC"},
    {302, "L2A.MACFACT"},
    {303, "L2A.PROXY"},
    {304, "L2A.EPOLL"},
    {305, "L2A.TRANSPORT"},
    {306, "L2A.MODULE"},
    {307, "L2A.TICK"},
    {308, "L2A.UEMD"},
    {309, "L2A.PARAM"},
    {310, "L2A.SIM"},
    {311, "L2A.PROCESSING_TIMES"},
    {312, "L2A.TICK_TIMES"},

    // scfl2adapter
    {330, "SCF"},
    {331, "SCF.MAC"},
    {332, "SCF.DISPATCH"},
    {333, "SCF.PHY"},
    {334, "SCF.SLOTCMD"},
    {335, "SCF.L2SA"},
    {336, "SCF.DUMMYMAC"},
    {337, "SCF.CALLBACK"},
    {338, "SCF.TICK_TEST"},
    {339, "SCF.UL_FAPI_VALIDATE"},

    // testMAC
    {400, "MAC"},
    {401, "MAC.LP"},
    {402, "MAC.FAPI"},
    {403, "MAC.UTILS"},
    {404, "MAC.SCF"},
    {405, "MAC.ALTRAN"},
    {406, "MAC.CFG"},
    {407, "MAC.PROC"},
    {408, "MAC.VALD"},
    {409, "MAC.PROCESSING_TIMES"},

    // ru-emulator
    {500, "RU"},
    {501, "RU.EMULATOR"},
    {502, "RU.PARSER"},
    {503, "RU.LATE_PACKETS"},
    {504, "RU.SYMBOL_TIMINGS"},
    {505, "RU.TX_TIMINGS"},
    {506, "RU.TX_TIMINGS_SUM"},

    // aerial-fh-driver
    {600, "FH"},
    {601, "FH.FLOW"},
    {602, "FH.FH"},
    {603, "FH.GPU_MP"},
    {604, "FH.LIB"},
    {605, "FH.MEMREG"},
    {606, "FH.METRICS"},
    {607, "FH.NIC"},
    {608, "FH.PDUMP"},
    {609, "FH.PEER"},
    {610, "FH.QUEUE"},
    {611, "FH.RING"},
    {612, "FH.TIME"},
    {613, "FH.GPU_COMM"},
    {614, "FH.STREAMRX"},
    {615, "FH.GPU"},
    {616, "FH.RMAX"},
    {617, "FH.GPU_COMM_CUDA"},
    {618, "FH.DOCA"},
    {619, "FH.NIC"},
    {620, "FH.STATS"},

    // fh_generator
    {650, "FHGEN"},
    {651, "FHGEN.GEN"},
    {652, "FHGEN.WORKER"},
    {653, "FHGEN.YAML"},
    {654, "FHGEN.ORAN_SLOT_ITER"},

    // compression_decompression
    {700, "COMP"},

    // cuphyoam
    {800, "OAM"},

    // cuphy
    // cuPHY channels
    {900, "CUPHY"},
    {901, "CUPHY.SSB_TX"},
    {902, "CUPHY.PDCCH_TX"},
    {904, "CUPHY.PDSCH_TX"},
    {905, "CUPHY.CSIRS_TX"},
    {906, "CUPHY.PRACH_RX"},
    {907, "CUPHY.PUCCH_RX"},
    {908, "CUPHY.PUSCH_RX"},
    {909, "CUPHY.BFW"},
    {910, "CUPHY.SRS_RX"},

     // cuPHY components and common utilities
    {931, "CUPHY.UTILS"}, // do not change
    {932, "CUPHY.MEMFOOT"}, // do not change
    {933, "CUPHY.PTI"},
};

#define NVLOG_FMTLOG_NUM_TAGS (sizeof(g_nvlog_component_ids) / sizeof(nvlog_component_ids))
inline int g_nvlog_component_levels[NVLOG_FMTLOG_NUM_TAGS];

constexpr bool nvlog_component_is_valid(int id)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (id == c.id)
        {
            return true;
        }
    }
    return false;
};

constexpr bool nvlog_component_is_valid(std::string_view name)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (name == c.name)
        {
            return true;
        }
    }
    return false;
};


constexpr const char * nvlog_get_component_name(int id)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (id == c.id)
        {
            return c.name;
        }
    }
    return "UNKNOWN";
};

constexpr const char * nvlog_get_component_name(std::string_view name)
{
    for (auto &c : g_nvlog_component_ids)
    {
        if (name == c.name)
        {
            return c.name;
        }
    }
    return "UNKNOWN";
};

constexpr int nvlog_get_component_id(int id)
{
    for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)
    {
        auto &c = g_nvlog_component_ids[i];
        if (id == c.id)
        {
            return i;
        }
    }
    return 0;
};

constexpr int nvlog_get_component_id(std::string_view name)
{
    for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)
    {
        auto &c = g_nvlog_component_ids[i];
        if (name == c.name)
        {
            return i;
        }
    }
    return 0;
};

#else
#define NVLOGV_FMT(component_id, format_fmt, ...) 
#define NVLOGD_FMT(component_id, format_fmt, ...) 
#define NVLOGI_FMT(component_id, format_fmt, ...) 
#define NVLOGW_FMT(component_id, format_fmt, ...) 
#define NVLOGC_FMT(component_id, format_fmt, ...) 
#define NVLOGE_NO_FMT(component_id, event_level, format_fmt, ...) 

#define NVLOGI_FMT_EVT(component_id, event_level, format_fmt, ...) 
#define NVLOGE_FMT(component_id, event_level, format_fmt, ...) 

#define NVLOGF_FMT(component_id, event_level, format_fmt, ...) do { \
    usleep(100000); \
    ::pExitHandler.test_trigger_exit(); \
} while (0)

#endif

#endif // NVLOG_FMT_HPP
