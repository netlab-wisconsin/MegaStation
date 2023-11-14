% Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

h5File  = H5F.create('struct_with_array_example.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

% Create an array of structures, using cell arrays to provide a value
% for each element of the array.
A = struct('serial_no', {1153; 1184; 1027; 1313}, ...
           'temperature', {53.23; 55.12; 130.55; 1252.89}, ...
           'pressure', {24.57; 22.95; 31.23; 84.11}, ...
           'array', {[int32(0), int32(0), int32(1)]; [int32(0), int32(1), int32(0)]; [int32(0), int32(1), int32(1)]; [int32(1), int32(2), int32(3)]});

hdf5_write_nv(h5File, 'A', A);

H5F.close(h5File);
