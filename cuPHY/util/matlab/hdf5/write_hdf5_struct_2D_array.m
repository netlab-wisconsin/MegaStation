% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

h5File  = H5F.create('struct_array_2D_example.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
clear A;
for m = [1:3]
  for n = [1:4]
    A(m,n).m = uint32(m-1);
    A(m,n).n = uint32(n-1);
  end
end
hdf5_write_nv(h5File, 'A', A);

H5F.close(h5File);
