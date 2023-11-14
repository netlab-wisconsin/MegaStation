/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _STAT_LOG_H_
#define _STAT_LOG_H_

#include <time.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define STAT_NAME_MAX_LEN (32)

// Decide the statistic period by timer or counter
typedef enum
{
    STAT_MODE_NONE    = 0,
    STAT_MODE_COUNTER = 1,
    STAT_MODE_TIMER   = 2,
    STAT_MODE_MAX     = 3
} print_mode_t;

typedef struct stat_log_t stat_log_t;
struct stat_log_t
{
    int (*init)(stat_log_t* stat);

    int (*add)(stat_log_t* stat, int64_t value);

    int (*time_interval)(stat_log_t* stat);

    int (*set_clock_source)(stat_log_t* stat, clockid_t clk_src);

    int (*set_log_level)(stat_log_t* stat, int log_level);

    int (*set_limit)(stat_log_t* stat, int64_t min, int64_t max);

    int (*print)(stat_log_t* stat);

    int (*close)(stat_log_t* stat);
};

stat_log_t* stat_log_open(const char* name, int mode, int64_t period);

int assert_time_order(struct timespec ts_old, struct timespec ts_new);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _STAT_LOG_H_ */
