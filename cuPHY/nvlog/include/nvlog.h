/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _NVLOG_H_
#define _NVLOG_H_

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include "aerial_event_code.h"

#ifdef __cplusplus /* For both C and C++ */
extern "C" {
#endif

// Default nvlog configuration file
#define NVLOG_DEFAULT_CONFIG_FILE "cuPHY/nvlog/config/nvlog_config.yaml"

// Module tag base numbers
#define NVLOG_TAG_BASE_RESERVED 0            // reserved
#define NVLOG_TAG_BASE_NVLOG 10              // nvlog
#define NVLOG_TAG_BASE_NVIPC 30              // nvIPC
#define NVLOG_TAG_BASE_CUPHY_CONTROLLER 100  // cuphycontroller
#define NVLOG_TAG_BASE_CUPHY_DRIVER 200      // cuphydriver
#define NVLOG_TAG_BASE_L2_ADAPTER 300        // cuphyl2adapter
#define NVLOG_TAG_BASE_SCF_L2_ADAPTER 330    // scfl2adapter
#define NVLOG_TAG_BASE_ALTRAN_L2_ADAPTER 360 // altranl2adapter
#define NVLOG_TAG_BASE_TEST_MAC 400          // testMAC
#define NVLOG_TAG_BASE_RU_EMULATOR 500       // ru-emulator
#define NVLOG_TAG_BASE_FH_DRIVER 600         // aerial-fh-driver
#define NVLOG_TAG_BASE_FH_GENERATOR 650      // fh_generator
#define NVLOG_TAG_BASE_COMPRESSION 700       // compression_decompression
#define NVLOG_TAG_BASE_CUPHY_OAM 800         // cuphyoam
#define NVLOG_TAG_BASE_CUPHY 900             // cuPHY

// Log levels
#define NVLOG_NONE 0 // Set config.shm_log_level or config.console_log_level to NVLOG_NONE can disable all SHM or console log
#define NVLOG_FATAL 1
#define NVLOG_ERROR 2
#define NVLOG_CONSOLE 3
#define NVLOG_WARN 4
#define NVLOG_INFO 5
#define NVLOG_DEBUG 6
#define NVLOG_VERBOSE 7

#define NVLOG_DEFAULT_TAG_NUM 1024
#define NVLOG_NAME_MAX_LEN 32                  // Log name string length should be less than 32

void nvlog_c_print(int level, int itag, const char* format, ...);
void nvlog_e_c_print(int level, int itag, const char* event, const char* format, ...);


#define NVLOG_C(level, itag, format, ...) nvlog_c_print(level, itag, format, ##__VA_ARGS__)
#define NVLOG_E_C(level, itag, event, format, ...) nvlog_e_c_print(level, itag, #event, format, ##__VA_ARGS__)
#define NVLOG(level, itag, format, ...) nvlog_c_print(level, itag, format, ##__VA_ARGS__)

#ifndef __cplusplus // Disable nvlog for C
#define NVLOGV(tag, format, ...) NVLOG_C(NVLOG_VERBOSE, tag, format, ##__VA_ARGS__)
#define NVLOGD(tag, format, ...) NVLOG_C(NVLOG_DEBUG,   tag, format, ##__VA_ARGS__)
#define NVLOGI(tag, format, ...) NVLOG_C(NVLOG_INFO,    tag, format, ##__VA_ARGS__)
#define NVLOGW(tag, format, ...) NVLOG_C(NVLOG_WARN,    tag, format, ##__VA_ARGS__)
#define NVLOGE(tag, event, format, ...) NVLOG_E_C(NVLOG_ERROR, tag, event, format, ##__VA_ARGS__)
#define NVLOGC(tag, format, ...) NVLOG_C(NVLOG_WARN,    tag, format, ##__VA_ARGS__)
#define NVLOGF(tag, event, format, ...) do { \
    NVLOG_E_C(NVLOG_FATAL, tag, event, format, ##__VA_ARGS__); \
    usleep(100000); \
    exit(1); \
} while (0)

#define NVLOGE_NO(tag, event, format, ...) NVLOG_E_C(NVLOG_ERROR, tag, event, format, ##__VA_ARGS__)
#endif

#ifdef NVIPC_FMTLOG_ENABLE
int fmt_log_level_validate(int level, int itag, const char** stag);
void nvlog_vprint_fmt(int level, const char* stag, const char* format, va_list va);
void nvlog_e_vprint_fmt(int level, const char* event, const char* stag, const char* format, va_list va);
#endif
void nvlog_c_init(const char *file);
void nvlog_c_close();


// Copy at most (dest_size - 1) bytes and make sure it is terminated by '\0'.
static inline char* nvlog_safe_strncpy(char* dest, const char* src, size_t dest_size)
{
    if(dest == NULL)
    {
        return dest;
    }

    if(src == NULL)
    {
        *dest = '\0'; // Set destination to empty string
        return dest;
    }

    char* ret_dest          = strncpy(dest, src, dest_size - 1); // Reserve 1 byte for '\0'
    *(dest + dest_size - 1) = '\0';                              // Safely terminate the string with '\0'
    return ret_dest;
}

// Get monotonic time stamp
static inline int nvlog_gettime(struct timespec* ts)
{
    return clock_gettime(CLOCK_MONOTONIC, ts);
}

// Get real-time time stamp
static inline int nvlog_gettime_rt(struct timespec* ts)
{
    return clock_gettime(CLOCK_REALTIME, ts);
}

static inline int64_t nvlog_timespec_interval(struct timespec* t1, struct timespec* t2)
{
    return (t2->tv_sec - t1->tv_sec) * 1000000000LL + t2->tv_nsec - t1->tv_nsec;
}

static inline void nvlog_timespec_add(struct timespec* ts, int64_t ns)
{
    ns += ts->tv_nsec;
    ts->tv_sec += ns / 1000000000L;
    ts->tv_nsec = ns % 1000000000L;
}

static inline int64_t nvlog_get_interval(struct timespec* start)
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return (now.tv_sec - start->tv_sec) * 1000000000LL + now.tv_nsec - start->tv_nsec;
}

// struct timeval
static inline int64_t nvlog_timeval_interval(struct timeval* t1, struct timeval* t2)
{
    return (t2->tv_sec - t1->tv_sec) * 1000000LL + t2->tv_usec - t1->tv_usec;
}

#if defined(__cplusplus) /* For both C and C++ */
} /* extern "C" */
#endif

#if defined(__cplusplus)
#include "nvlog.hpp"
#endif

#endif /* _NVLOG_H_ */
