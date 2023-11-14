% Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

h5File  = H5F.create('vlen_example.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

%
% Initialize variable-length data.  wdata{1} is a countdown of
% length LEN0, wdata{2} is a Fibonacci sequence of length LEN1.
%
wdata{1} = int32(10:-1:1);
wdata{2} = int32([ 1 4 9 16 25 36 49 64 81 100 121]);

hdf5_write_nv_vlen(h5File, 'wdata', wdata);

H5F.close(h5File);
