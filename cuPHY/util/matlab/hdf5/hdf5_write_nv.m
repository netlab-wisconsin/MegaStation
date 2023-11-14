% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

function hdf5_write_nv(hdf5loc, name, A, varargin)
  % HDF5_WRITE_NV Write a variable to an HDF5 file
  % HDF5_WRITE_NV(loc, name, A) writes variable A to the location loc
  % in an HDF5 file.
  % Location can be an HDF5 file handle or a HDF5 group handle.
  % Optional string modifiers can be appended. Currently, the only
  % supported modifier is 'fp16', which will result in writing
  % half precision floating point values. (Note that as of this
  % writing, the IEEE fp16 data type is not supported natively by
  % HDF5. However, HDF5 does support the creation of custom floating
  % point types, and also provides conversion functions.)
  %
  % The 'fp16' modifier is ignored when writing MATLAB struct datatypes.
  % Writing struct fields as fp16 values is not currenty supported.
  %
  % Files written are standard HDF5 files, and contents can be viewed
  % with standard HDF5 utilities (e.g. h5dump). However, certain
  % conventions are used for compatibility with the NVIDIA cuPHY library:
  %
  % - Complex arrays are written as HDF5 arrays of a COMPOUND type, where
  %   the compound type has fields 're' and 'im'. (HDF5 does not have a
  %   native concept for complex values.)
  % - Array dimension ordering is reversed, as MATLAB uses column-major
  %   ordering, whereas HDF5 uses row-major
  %
  % Example usage:
  %   h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
  %   A = rand(12, 4);
  %   b = single(randn(3, 9));
  %   hdf5_write_nv(h5File, 'A', A);
  %   hdf5_write_nv(h5File, 'b', b);
  %   H5F.close(h5File);

  % ------------------------------------------------------------------
  % Determine the element type
  H5SrcTypeString = get_hdf5_type(A);
  if isempty(H5SrcTypeString)
      error('Unexpected element class (%s)', class(A));
  end
  % ------------------------------------------------------------------
  if strcmp('H5T_COMPOUND', H5SrcTypeString)
      % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      % Determine the size and type of each MATLAB struct field. Note
      % that in MATLAB, fields of different structs in an array do not
      % necessarily have the same type.
      compoundSizes = [];
      elemTypes = {};
      names = fieldnames(A);
      for iName = 1:length(names)
          elemTypes{end + 1} = get_hdf5_type(A(1).(names{iName}));
          if strcmp(elemTypes{end},'H5T_COMPOUND')
              error('Exported structures cannot have nested structure fields.');
          end
          if isempty(elemTypes{end})
              error('Exported structures must only contain supported primitive types.');
          end
          compoundSizes(end + 1) = H5T.get_size(elemTypes{end});
      end
      % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      % Create a new HDF5 type to represent the MATLAB struct
      compoundOffsets= [0 cumsum(compoundSizes(1:end-1))];
      compoundType = H5T.create('H5T_COMPOUND', sum(compoundSizes));
      for iName = 1:length(names)
          H5T.insert(compoundType, names{iName}, compoundOffsets(iName), elemTypes{iName});
      end

      % Create a dataspace. MATLAB is column major, whereas HDF5 is
      % row major, so we do a fliplr on the dimensions
      % Setting maximum size to [] indicates that the maximum size is
      % current size.
      [arrayRank, arrayDims] = get_squeeze_dims(A);
      arrayDataspace = H5S.create_simple(arrayRank, fliplr(arrayDims), []);

      % Copy data to a local struct with matching fields. (Convert array of
      % structures to structure of arrays for the H5D.write function call
      % below.)
      % Extract the value from each struct using the [A.fieldname] notation,
      % which flattens the retrieved values. (Therefore, we resize.)
      % It doesn't seem like MATLAB does the right thing here with variable
      % classes that differ across different elements:
      % >> B(1).value = 1; B(2).value = 'hello'; [B.value]
      % ans = 'hello'
      for iName = 1:length(names)
          Awrite.(names{iName}) = reshape([A.(names{iName})], size(A));
      end
      %Awrite

      % Create a dataset and write it to the file
      arrayDataset = H5D.create(hdf5loc, name, compoundType, arrayDataspace, 'H5P_DEFAULT');
      %H5D.write(arrayDataset, compoundType, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', A);
      H5D.write(arrayDataset, compoundType, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', Awrite);

      % Cleanup
      H5D.close(arrayDataset);
      H5S.close(arrayDataspace);
      H5T.close(compoundType);
  else
      srcType       = H5T.copy(H5SrcTypeString);
      useFP16       = 0;
      if nargin > 3
          useFP16 = strcmp('fp16', varargin{1});
      end
      % Note: isreal() will return false for class types
      if isreal(A)
          % Check for the fp16 option
          if useFP16
              fileType = create_fp16_type();
              % Seems like MATLAB will only let us write to a low precision
              % float when the source is a double.
              memType  = H5T.copy('H5T_NATIVE_DOUBLE');
              Awrite    = double(A);
          else
              fileType = H5T.copy(srcType);
              memType  = H5T.copy(srcType);
              Awrite = A;
          end
      else
          % Create HDF5 complex data types
          compoundSizes   = [H5T.get_size(srcType) H5T.get_size(srcType)];
          compoundOffsets = [0 cumsum(compoundSizes(1:end-1))];
          memType         = H5T.create('H5T_COMPOUND', sum(compoundSizes));
          H5T.insert(memType, 're', compoundOffsets(1), srcType);
          H5T.insert(memType, 'im', compoundOffsets(2), srcType);

          % Check for the fp16 option
          if useFP16
              dstScalarType   = create_fp16_type();
              compoundSizes   = [H5T.get_size(dstScalarType) H5T.get_size(dstScalarType)];
              compoundOffsets = [0 cumsum(compoundSizes(1:end-1))];
              fileType        = H5T.create('H5T_COMPOUND', sum(compoundSizes));
              H5T.insert(fileType, 're', compoundOffsets(1), dstScalarType);
              H5T.insert(fileType, 'im', compoundOffsets(2), dstScalarType);
              H5T.close(dstScalarType);
          else
              fileType = H5T.copy(memType);
          end

          % Copy real and imaginary data to a local struct with two fields
          Awrite.re = real(A);
          Awrite.im = imag(A);
      end
  
      % Create a dataspace. MATLAB is column major, whereas HDF5 is
      % row major, so we do a fliplr on the dimensions
      % Setting maximum size to [] indicates that the maximum size is
      % current size.
      arrayDataspace = H5S.create_simple(ndims(A), fliplr(size(A)), []);

      % Create a dataset and write it to the file
      arrayDataset = H5D.create(hdf5loc, name, fileType, arrayDataspace, 'H5P_DEFAULT');
      H5D.write(arrayDataset, memType, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', Awrite);
      %H5D.write(arrayDataset, fileType, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', Awrite);

      % Cleanup
      H5D.close(arrayDataset);
      H5S.close(arrayDataspace);
      H5T.close(fileType);
      H5T.close(srcType);
      H5T.close(memType);
  end
end

function H5TypeString = get_hdf5_type(varA)
  % Determine the element type
  switch(class(varA))
    case 'single'
      H5TypeString = 'H5T_NATIVE_FLOAT';
    case 'double'
      H5TypeString = 'H5T_NATIVE_DOUBLE';
    case 'uint8'
      H5TypeString = 'H5T_NATIVE_UINT8';
    case 'uint16'
      H5TypeString = 'H5T_NATIVE_UINT16';
    case 'uint32'
      H5TypeString = 'H5T_NATIVE_UINT32';
    case 'uint64'
      H5TypeString = 'H5T_NATIVE_UINT64';
    case 'int8'
      H5TypeString = 'H5T_NATIVE_INT8';
    case 'int16'
      H5TypeString = 'H5T_NATIVE_INT16';
    case 'int32'
      H5TypeString = 'H5T_NATIVE_INT32';
    case 'int64'
      H5TypeString = 'H5T_NATIVE_INT64';
    case 'struct'
      H5TypeString = 'H5T_COMPOUND';
    otherwise
      H5TypeString = '';
  end
end

% Return a tensor rank and array of dimensions for situations in which
% we want to collapse singleton dimensions (e.g. MATLAB squeeze()).
% MATLAB seems to always have at least 2 dimensions. For compatibility
% with already-generated files, we may not always want to remove
% singleton dimensions.
function [varRank, varDims] = get_squeeze_dims(varA)
  if numel(varA) == 1
    varRank = 1;
    varDims = [1];
  else
    varDims = size(varA);
    varDims(varDims == 1) = [];
    varRank = length(varDims);
  end
end

function fp16_type = create_fp16_type()
    % Create a floating point type based on fp32
    fp16_type = H5T.copy('H5T_NATIVE_FLOAT');
    %fprintf('FP32 precision:     %d\n', H5T.get_precision(fp16_type));
    %fprintf('FP32 offset:        %d\n', H5T.get_offset(fp16_type));
    %fprintf('FP32 exp bias:      %d\n', H5T.get_ebias(fp16_type));
    %[spos, epos, esize, mpos, msize] = H5T.get_fields(fp16_type);
    %fprintf('FP32 sign pos:      %d\n', spos);
    %fprintf('FP32 exp pos:       %d\n', epos);
    %fprintf('FP32 exp size:      %d\n', esize);
    %fprintf('FP32 mantissa pos:  %d\n', mpos);
    %fprintf('FP32 mantissa size: %d\n', msize);

    % https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    % sign_pos = 15
    % exp_pos  = 10
    % exp_size = 5
    % mantissa_pos = 0
    % mantissa_size = 10

    % Order is important - we can't set the size before adjusting fields
    H5T.set_fields(fp16_type, 15, 10, 5, 0, 10);
    H5T.set_precision(fp16_type, 16);
    H5T.set_ebias(fp16_type, 15);
    H5T.set_size(fp16_type, 2);

    %fprintf('FP16 size:          %d\n', H5T.get_size(fp16_type));
    %fprintf('FP16 precision:     %d\n', H5T.get_precision(fp16_type));
    %fprintf('FP16 offset:        %d\n', H5T.get_offset(fp16_type));
    %fprintf('FP16 exp bias:      %d\n', H5T.get_ebias(fp16_type));
    %[spos, epos, esize, mpos, msize] = H5T.get_fields(fp16_type);
    %fprintf('FP16 sign pos:      %d\n', spos);
    %fprintf('FP16 exp pos:       %d\n', epos);
    %fprintf('FP16 exp size:      %d\n', esize);
    %fprintf('FP16 mantissa pos:  %d\n', mpos);
    %fprintf('FP16 mantissa size: %d\n', msize);
end
