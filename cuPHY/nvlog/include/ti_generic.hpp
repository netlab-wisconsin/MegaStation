/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "nvlog_fmt.hpp"

//ti_generic - A generic set of task instrumentation macros
//
//Example usage:
//
//TI_GENERIC_INIT("instrumentation 1",10)
//TI_GENERIC_ADD("subtask 1")
//... code
//TI_GENERIC_ADD("subtask 2")
//... code
//TI_GENERIC_ADD("end task")
//TI_GENERIC_DURATION_NVLOGI(TAG)

#define MAX_SUBTASK_CHARS 32

#define TI_GENERIC_INIT(task_name,max_subtasks) \
std::stringstream ti_os; \
char ti_task_name[MAX_SUBTASK_CHARS]; \
int ti_max_subtasks = max_subtasks; \
strcpy(ti_task_name,task_name); \
std::chrono::nanoseconds ti_times[max_subtasks]; \
char ti_subtask_names[max_subtasks][MAX_SUBTASK_CHARS]; \
int ti_subtask_count = 0;

//Creates a subtask marker
#define TI_GENERIC_ADD(subtask_name) \
if(ti_subtask_count < ti_max_subtasks) { \
    strcpy(ti_subtask_names[ti_subtask_count],subtask_name); \
    ti_times[ti_subtask_count] = std::chrono::system_clock::now().time_since_epoch(); \
    ti_subtask_count += 1; \
}

//Prints percentage contribution between each subtask marker
#define TI_GENERIC_PERCENTAGE_NVLOG(LOG_LEVEL,TAG) \
char ti_subtask_results1[4096]; \
int ti_offset1=0; \
double ti_total1=(ti_times[ti_subtask_count-1]-ti_times[0]).count()/1e3; \
for(int ii=0; ii<ti_subtask_count-1; ii++) { \
    double ti_percentage = 100.0*((ti_times[ii+1]-ti_times[ii]).count()/1e3)/ti_total1; \
    ti_offset1 += sprintf(&ti_subtask_results1[ti_offset1], "%s:%.1f,", ti_subtask_names[ii], ti_percentage); \
} \
ti_offset1 += sprintf(&ti_subtask_results1[ti_offset1], " (total: %.1fus),", ti_total1); \
NVLOG_FMT(LOG_LEVEL,TAG,"{{TI PERCENTAGE}} <{}> {}\n", ti_task_name, ti_subtask_results1);

#define TI_GENERIC_PERCENTAGE_NVLOGV(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGD(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGI(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGW(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_PERCENTAGE_NVLOGC(component_id) TI_GENERIC_PERCENTAGE_NVLOG(fmtlog::WRN,component_id)


//Prints duration between each subtask marker
#define TI_GENERIC_DURATION_NVLOG(LOG_LEVEL,TAG) \
char ti_subtask_results2[4096]; \
int ti_offset2=0; \
for(int ii=0; ii<ti_subtask_count-1; ii++) { \
    ti_offset2 += sprintf(&ti_subtask_results2[ti_offset2], "%s:%.3f,", ti_subtask_names[ii], (ti_times[ii+1]-ti_times[ii]).count()/1e3); \
} \
NVLOG_FMT(LOG_LEVEL,TAG,"{{TI DURATION}} <{}> {}\n", ti_task_name, ti_subtask_results2);

#define TI_GENERIC_DURATION_NVLOGV(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_DURATION_NVLOGD(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_DURATION_NVLOGI(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_DURATION_NVLOGW(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_DURATION_NVLOGC(component_id) TI_GENERIC_DURATION_NVLOG(fmtlog::WRN,component_id)


//Prints timestamp of each subtask marker
#define TI_GENERIC_TIMESTAMP_NVLOG(LOG_LEVEL,TAG) \
char ti_subtask_results3[4096]; \
int ti_offset3=0; \
for(int ii=0; ii<ti_subtask_count; ii++) { \
    ti_offset3 += sprintf(&ti_subtask_results3[ti_offset3], "%s:%lu,", ti_subtask_names[ii], ti_times[ii].count()); \
} \
NVLOG_FMT(LOG_LEVEL,TAG,"{{TI TIMESTAMPS}} <{}> {}\n", ti_task_name, ti_subtask_results3);

#define TI_GENERIC_TIMESTAMP_NVLOGV(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGD(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGI(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGW(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_TIMESTAMP_NVLOGC(component_id) TI_GENERIC_TIMESTAMP_NVLOG(fmtlog::WRN,component_id)


//Prints all forms of TI_GENERIC log messages
#define TI_GENERIC_ALL_NVLOG(LOG_LEVEL,TAG) TI_GENERIC_PERCENTAGE_NVLOG(LOG_LEVEL,TAG); TI_GENERIC_DURATION_NVLOG(LOG_LEVEL,TAG); TI_GENERIC_TIMESTAMP_NVLOG(LOG_LEVEL,TAG);

#define TI_GENERIC_ALL_NVLOGV(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_ALL_NVLOGD(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::DBG,component_id)
#define TI_GENERIC_ALL_NVLOGI(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::INF,component_id)
#define TI_GENERIC_ALL_NVLOGW(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::WRN,component_id)
#define TI_GENERIC_ALL_NVLOGC(component_id) TI_GENERIC_ALL_NVLOG(fmtlog::WRN,component_id)