/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

typedef struct
{
    void*   addr;
    int32_t stride_elements;
    int32_t num_codewords;
} cuphyTransportBlockLLRDesc_t;

typedef struct
{
    uint32_t* addr;
    int32_t   stride_words;
    int32_t   num_codewords;
} cuphyTransportBlockDataDesc_t;

typedef union
{
    float       f32;
    __half2_raw f16x2;
} cuphyLDPCNormalization_t;

typedef struct
{
    cuphyDataType_t          llr_type;         // Type of LLR input data (CUPHY_R_16F or CUPHY_R_32F)
    int16_t                  num_parity_nodes; // Number of parity nodes
    int16_t                  Z;                // Lifting size
    int16_t                  max_iterations;   // Maximum number of iterations
    int16_t                  Kb;               // Number of "information" variable nodes
    cuphyLDPCNormalization_t norm;             // Normalization (for normalized min-sum)
    uint32_t                 flags;            // Flags
    int16_t                  BG;               // Base graph (1 or 2)
    int16_t                  algo;             // Algorithm (0 for automatic choice)
    void*                    workspace;        // Workspace area
} cuphyLDPCDecodeConfigDesc_t;

typedef struct
{
    cuphyLDPCDecodeConfigDesc_t   config;                                    // Common decoder configuration
    int32_t                       num_tbs;                                   // Number of valid TB descriptors
    cuphyTransportBlockLLRDesc_t  llr_input[CUPHY_LDPC_DECODE_DESC_MAX_TB];  // Input LLR buffers
    cuphyTransportBlockDataDesc_t tb_output[CUPHY_LDPC_DECODE_DESC_MAX_TB];  // Output bit/data buffers
    cuphyTransportBlockLLRDesc_t  llr_output[CUPHY_LDPC_DECODE_DESC_MAX_TB]; // Output LLR buffers (optional)
} cuphyLDPCDecodeDesc_t;