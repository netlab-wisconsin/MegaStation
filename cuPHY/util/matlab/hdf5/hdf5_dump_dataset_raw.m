% Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

function hdf5_dump_dataset_raw(infile, dsetname, outfile)
  % HDF5_DUMP_DATASET_RAW Write a single dataset from an HDF5 file to
  % a binary file. This function uses a raw byte offset in the file
  % to read from, and provides no conversion.
  %
  % Example usage:
  %   hdf5_dump_dataset_raw('TV_cuphy_pusch-TC1_snrdb40.00_iter1_MIMO4x4_PRB272_DataSyms9_qam256.h5', 'DataRx');
  %   hdf5_dump_dataset_raw('TV_cuphy_pusch-TC1_snrdb40.00_iter1_MIMO4x4_PRB272_DataSyms9_qam256.h5', 'DataRx', 'out.bin');

  %---------------------------------------------------------------------
  % Determine the properties of the dataset
  h5FileIn      = H5F.open(infile, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
  dset          = H5D.open(h5FileIn, dsetname);
  dsetOffset    = H5D.get_offset(dset);
  dsetSize      = H5D.get_storage_size(dset);
  dspace        = H5D.get_space(dset);
  dsetType      = H5D.get_type(dset);
  dsetTypeClass = H5T.get_class(dsetType);
  [numdims,h5_dims,h5_maxdims] = H5S.get_simple_extent_dims(dspace);
  % HDF5 dimensions are reversed...
  h5_dims = fliplr(h5_dims);
  fprintf('Dataset %s: size = %d, offset = %d\n', dsetname, dsetSize, dsetOffset);

  if nargin < 3
    outfile = get_filename(infile, dsetname, h5_dims, get_type_desc(dsetType));
  end

  H5T.close(dsetType);
  H5S.close(dspace);
  H5D.close(dset);
  H5F.close(h5FileIn);

  %---------------------------------------------------------------------
  fileInID   = fopen(infile, 'r');
  fseek(fileInID, dsetOffset, 0);
  inData = fread(fileInID, dsetSize, 'uint8=>uint8');
  fclose(fileInID);

  %---------------------------------------------------------------------
  % Open the output file
  fprintf('Writing %d bytes to output file %s\n', dsetSize, outfile);
  fileOutID = fopen(outfile, 'w');
  fwrite(fileOutID, inData, 'uint8');
  fclose(fileOutID);

end

function str = get_type_desc(dsetType)
  class_id = H5T.get_class(dsetType);
  sz       = H5T.get_size(dsetType);
  switch(class_id)
    case H5ML.get_constant_value('H5T_INTEGER')
        sign_type = H5T.get_sign(type_id);
        switch(sign_type)
          case H5ML.get_constant_value('H5T_SGN_NONE')
            sign_str = 'u';
          case H5ML.get_constant_value('H5T_SGN_2');
            sign_str = '';
        end
        str = sprintf('%sint%d', sign_str, sz*8);
    case H5ML.get_constant_value('H5T_FLOAT')
        %fprintf('Floating point\n');
        switch(sz)
          case 2
            str = 'fp16';
          case 4
            str = 'single';
          case 8
            str = 'double';
        end
    case H5ML.get_constant_value('H5T_COMPOUND')
        nmembers = H5T.get_nmembers(dsetType);
        if (2 == nmembers) & (strcmp('re', H5T.get_member_name(dsetType, 0))) & (strcmp('im', H5T.get_member_name(dsetType, 1)))
          scalarType = H5T.get_member_type(dsetType, 0);
          scalarName = get_type_desc(scalarType);
          str = sprintf('%s_complex', scalarName);
          H5T.close(scalarType);
        else
          str = 'compound';
        end
  end
end

function fname = get_filename(infile, dsetname, sz, typestr)
    [filepath, name, ext] = fileparts(infile);
    %nd = ndims(var);
    %sz = size(var);
    %-------------------------------------------------------------------
    % Try to get rid of singleton dimensions
    if sz(1) == 1
        sz(1)=[];
    end
    if sz(end) == 1
        sz(end) = [];
    end
    %-------------------------------------------------------------------
    % Create a string with the dimensions
    szstr = sprintf('%d', sz(1));
    sz(1) = [];
    for s  = sz
        szstr = strcat(szstr, '_', sprintf('%d', s));
    end
    %-------------------------------------------------------------------
    fname = strcat(name, '_', dsetname, '_', szstr, '_', typestr, '.bin');
    fname = fullfile(filepath, fname);
end
