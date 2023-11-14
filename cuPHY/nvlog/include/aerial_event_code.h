/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _AERIAL_EVENT_CODE_H_
#define _AERIAL_EVENT_CODE_H_

#ifdef __cplusplus /* For both C and C++ */
extern "C" {
#endif

typedef enum
{
    AERIAL_SUCCESS             = 0,
    AERIAL_INVALID_PARAM_EVENT = 1,
    AERIAL_INTERNAL_EVENT      = 2,
    AERIAL_CUDA_API_EVENT      = 3,
    AERIAL_DPDK_API_EVENT      = 4,
    AERIAL_THREAD_API_EVENT    = 5,
    AERIAL_CLOCK_API_EVENT     = 6,
    AERIAL_NVIPC_API_EVENT     = 7,
    AERIAL_ORAN_FH_EVENT       = 8,
    AERIAL_CUPHYDRV_API_EVENT  = 9,
    AERIAL_INPUT_OUTPUT_EVENT  = 10,
    AERIAL_MEMORY_EVENT        = 11,
    AERIAL_YAML_PARSER_EVENT   = 12,
    AERIAL_NVLOG_EVENT         = 13,
    AERIAL_CONFIG_EVENT        = 14,
    AERIAL_FAPI_EVENT          = 15,
    AERIAL_NO_SUPPORT_EVENT    = 16,
    AERIAL_SYSTEM_API_EVENT    = 17,
    AERIAL_L2ADAPTER_EVENT     = 18,
    AERIAL_RU_EMULATOR_EVENT   = 19,
    AERIAL_CUDA_KERNEL_EVENT   = 20,
    AERIAL_CUPHY_API_EVENT     = 21,
    AERIAL_DOCA_API_EVENT      = 22,
    AERIAL_CUPHY_EVENT         = 23,
} aerial_event_code_t;

char* nvlog_strerror(int code);

#if defined(__cplusplus) /* For both C and C++ */
} /* extern "C" */
#endif

#endif /* _AERIAL_EVENT_CODE_H_ */
