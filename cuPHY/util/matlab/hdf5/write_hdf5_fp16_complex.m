% Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

% ----------------------------------------------------------------------
h5File  = H5F.create('fp16_cmplx_example_single.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
A = single(rand(3) + (i*rand(3)));
hdf5_write_nv3(h5File, 'A', A, 'fp16');
H5F.close(h5File);
