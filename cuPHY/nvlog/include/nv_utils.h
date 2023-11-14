/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _NV_UTILS_H_
#define _NV_UTILS_H_

#ifdef __cplusplus /* For both C and C++ */
extern "C" {
#endif

// Convert process relative path to absolute path
int nv_get_absolute_path(char* absolute_path, const char* relative_path);

// For CPU core binding and priority setting
int nv_set_sched_fifo_priority(int priority);
int nv_assign_thread_cpu_core(int cpu_id);

#if defined(__cplusplus) /* For both C and C++ */
} /* extern "C" */
#endif

#endif /* _NV_UTILS_H_ */
