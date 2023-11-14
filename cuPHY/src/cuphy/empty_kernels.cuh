/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef CUPHY_EMPTY_KERNELS_CUH
#define CUPHY_EMPTY_KERNELS_CUH

#include "empty_kernels.hpp"

// empty kernels used in CUDA graph
__global__ void graphs_empty_kernel();
__global__ void graphs_empty_kernel_1_ptr_arg(void* ptr);
__global__ void graphs_empty_kernel_2_ptr_arg(void* ptr1, void* ptr2);
__global__ void graphs_empty_kernel_3_ptr_arg(void* ptr1, void* ptr2, void* ptr3);
__global__ void graphs_empty_kernel_4_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4);
__global__ void graphs_empty_kernel_5_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5);
__global__ void graphs_empty_kernel_6_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6);
__global__ void graphs_empty_kernel_7_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6, void* ptr7);
__global__ void graphs_empty_kernel_8_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6, void* ptr7, void* ptr8);
__global__ void graphs_empty_kernel_9_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6, void* ptr7, void* ptr8, void* ptr9);

__global__ void graphs_empty_kernel_1_grid_constant_arg_48B(const __grid_constant__ testDescr_sz<48> desc);
__global__ void graphs_empty_kernel_1_grid_constant_arg_32B(const __grid_constant__ testDescr_sz<32> desc);

#endif //CUPHY_EMPTY_KERNELS_CUH
