
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>
#include <sys/time.h>
#include <sys/types.h>
#include <fcntl.h>
#include <semaphore.h>
#include <errno.h>

#include "aerial_event_code.h"
#include "nvlog.h"

#define TAG (NVLOG_TAG_BASE_NVLOG + 9) // "NVLOG.C"
#define TAG_LOG_COLLECT (NVLOG_TAG_BASE_NVLOG + 9) // "NVLOG.C"

#ifdef NVIPC_FMTLOG_ENABLE
void nvlog_c_print(int level, int itag, const char* format, ...)
{
    const char* stag = NULL;
    if(fmt_log_level_validate(level,itag,&stag))
    {
        if(stag != NULL)
        {
            va_list va;
            va_start(va, format);
            nvlog_vprint_fmt(level, stag, format, va);
            va_end(va);
        }
    }
}
void nvlog_e_c_print(int level, int itag, const char *event, const char* format, ...)
{
    const char* stag = NULL;
    if(fmt_log_level_validate(level,itag,&stag))
    {
        if(stag != NULL)
        {
            va_list va;
            va_start(va, format);
            nvlog_e_vprint_fmt(level, event, stag, format, va);
            va_end(va);
        }
    }
}
#else
char str[8][3] ={"N","F","E","C","W","I","D","V"};

static int getIndex(int level)
{
    int index = 0;
    switch(level)
    {
        case NVLOG_FATAL:
        {
            index = 1;
            break;
        }
        case NVLOG_ERROR:
        {
            index = 2;
            break;
        }
        case NVLOG_CONSOLE:
        {
            index = 3;
            break;
        }
        case NVLOG_WARN:
        {
            index = 4;
            break;
        }
        case NVLOG_INFO:
        {
            index = 5;
            break;
        }
        case NVLOG_DEBUG:
        {
            index = 6;
            break;
        }
        case NVLOG_VERBOSE:
        {
            index = 7;
            break;
        }
        case NVLOG_NONE:
        default:
        {
            printf("invalid log level %d, setting to WRN level\n", level);
            index = 0;
            break;
        }
    }
    return index;
}

void nvlog_c_print(int level, int itag, const char* format, ...)
{
    if((level == NVLOG_NONE)||(level == NVLOG_VERBOSE)||(level == NVLOG_DEBUG)||(level == NVLOG_INFO)||(level == NVLOG_CONSOLE))
    {
        return;
    }
    va_list args;
    va_start(args, format);
    int index = getIndex(level);
    printf("[%s]: ",str[index]);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}
void nvlog_e_c_print(int level, int itag, const char *event, const char* format, ...)
{
    if((level == NVLOG_NONE)||(level == NVLOG_VERBOSE)||(level == NVLOG_DEBUG)||(level == NVLOG_INFO)||(level == NVLOG_CONSOLE))
    {
        return;
    }
    va_list args;
    va_start(args, format);
    int index = getIndex(level);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void nvlog_c_init(const char *file)
{
    return;
}

void nvlog_c_close()
{
    return;
}
#endif