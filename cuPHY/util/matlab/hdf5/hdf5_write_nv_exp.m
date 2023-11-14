% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

function hdf5_write_nv_exp(hdf5loc, name, A, varargin)
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
  H5SrcTypeString = get_hdf5_type_str(A);
  if isempty(H5SrcTypeString)
      error('Unexpected element class (%s)', class(A));
  end
  % ------------------------------------------------------------------
  % COMPOUND DATA TYPES
  if strcmp('H5T_COMPOUND', H5SrcTypeString)
      % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      % Determine the type that we want for each field in the MATLAB
      % struct. The choices are: a.) simple "atomic" types, b.) fixed
      % size arrays of atomic types, and c.) variable length arrays
      % of atomic types.
      % When the field has a single element for all structures in the
      % array of structures, we will choose the atomic type.
      % When the field has the same dimensions for all structures in
      % the array of structures, we will choose the fixed length array
      % type.
      % Otherwise, we will use a variable length array.
      names         = fieldnames(A);
      fieldDescs    = struct('name', {}, 'dims', {}, 'mode', {});
      for iName = 1:length(names)
          fieldTypeStr = get_hdf5_type_str(A(1).(names{iName}));
          if strcmp(fieldTypeStr,'H5T_COMPOUND')
              error('Exported structures cannot have nested structure fields.');
          end
          if isempty(fieldTypeStr)
              error('Exported structures must only contain supported primitive types.');
          end
          udims = struct_field_has_uniform_dims(A, names{iName});
          if isempty(udims)
            % variable length
            mode = 'VARIABLE_LENGTH';
          elseif prod(udims) == 1
            mode = 'SCALAR';
          else
            mode = 'FIXED_LENGTH';
          end
          fieldDescs(iName) = struct('name', names{iName}, 'dims', udims, 'mode', mode);
          % Create an HDF5 type for the structure field. We will treat
          % fields that are MATLAB matrices as fixed size HDF5 arrays.
          % As such, the dimensions of this field should be the same
          % for all structs in the array. For differing sizes, the
          % input values should be in a cell array, and HDF5's variable
          % length type can be used.
          typeID = get_hdf5_type(A(1).(names{iName}), mode);
          elemTypeIDs(iName  ) = typeID;
          compoundSizes(iName) = H5T.get_size(typeID);
      end
      % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      % Create a new HDF5 type to represent the MATLAB struct
      compoundOffsets = [0 cumsum(compoundSizes(1:end-1))];
      compoundType    = H5T.create('H5T_COMPOUND', sum(compoundSizes));
      for iName = 1:length(names)
          H5T.insert(compoundType, names{iName}, compoundOffsets(iName), elemTypeIDs(iName));
          H5T.close(elemTypeIDs(iName));
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
      % It doesn't seem like MATLAB does the right thing here with variable
      % classes that differ across different elements:
      % >> B(1).value = 1; B(2).value = 'hello'; [B.value]
      % ans = 'hello'
      for iName = 1:length(fieldDescs)
          fd = fieldDescs(iName);
          switch(fd.mode)
              case 'SCALAR'
                  % Extract the value from each struct using the [A.fieldname] notation,
                  % which flattens the retrieved values. (Resize after that.)
                  Awrite.(fd.name) = reshape([A.(fd.name)], size(A));
              case 'VARIABLE_LENGTH'
                  % Extract the value from each struct using the {A.fieldname} notation,
                  % and reshape.
                  Awrite.(fd.name) = reshape({A.(fd.name)}, size(A));
              case 'FIXED_LENGTH'
                  Awrite.(fd.name) = reshape([A.(fd.name)], [fd.dims size(A)]);
              otherwise
                  error('Unexpected field mode: %s', fieldDescs(iName).mode);
          end
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
  % ------------------------------------------------------------------
  % VARIABLE LENGTH DATA TYPES
  elseif strcmp('H5T_VLEN', H5SrcTypeString)
      % Determine the underlying type by looking at the first element
      % in the cell array. This assumes that all elements are the same
      % type...
      H5ScalarTypeString = get_hdf5_type_str(A{1});
      filetype = H5T.vlen_create(H5ScalarTypeString);
      %memtype = H5T.vlen_create ('H5T_NATIVE_INT');
      [arrayRank, arrayDims] = get_squeeze_dims(A);
      arrayDataspace = H5S.create_simple(arrayRank, fliplr(arrayDims), []);
      dset = H5D.create(hdf5loc, name, filetype, arrayDataspace, 'H5P_DEFAULT');
      %H5D.write (dset, memtype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', A);
      H5D.write (dset, filetype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', A);
      H5D.close(dset);
      H5S.close(arrayDataspace);
      H5T.close(filetype);
      %H5T.close(memtype);
  % ------------------------------------------------------------------
  % MULTIDIMENSIONAL ARRAYS OF PRIMITIVE TYPES
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
              Awrite   = double(A);
          else
              fileType = H5T.copy(srcType);
              memType  = H5T.copy(srcType);
              Awrite   = A;
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

% ######################################################################
% get_hdf5_type_str()
function H5TypeString = get_hdf5_type_str(varA)
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
    case 'cell'
      H5TypeString = 'H5T_VLEN';
    otherwise
      H5TypeString = '';
  end
end

% ######################################################################
% struct_field_has_uniform_dims()
% Examines a field in all elements of a struct array to determine whether
% or not they all have the same size. Returns the uniform dims if all
% elements have the same size, and returns the empty array otherwise.
function udims = struct_field_has_uniform_dims(sA, fieldname)
    udims = size(sA(1).(fieldname));
    for i=2:numel(sA)
        dimCheck = size(sA(i).(fieldname));
        if (numel(dimCheck) ~= numel(udims)) || any(dimCheck ~= udims)
            udims = [];
            break;
        end
    end
end

% ######################################################################
% get_hdf5_type_from_string()
function H5TypeID = get_hdf5_type_from_string(varA, str, variableMode)
    switch(variableMode)
      case 'SCALAR'
        % Basic type
        H5TypeID = H5T.copy(str);
      case 'FIXED_LENGTH'
        % Array of type, using given matrix for dimensions
        [varRank, varDims] = get_squeeze_dims(varA);
        varDims = fliplr(varDims);
        H5TypeID = H5T.array_create(str, varRank, varDims, []);
      case 'VARIABLE_LENGTH'
        % Variable length array of type
        H5TypeID = H5T.vlen_create(str);
      otherwise
        error('Unexpected value for variable mode: %s', variableMode);
    end
end
  
% ######################################################################
% get_hdf5_type()
function H5TypeID = get_hdf5_type(varA, variableMode)
  H5TypeID = -1;
  % Determine the underlying type from the MATLAB class
  switch(class(varA))
    case 'single'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_FLOAT',  variableMode);
    case 'double'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_DOUBLE', variableMode);
    case 'uint8'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_UINT8',  variableMode);
    case 'uint16'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_UINT16', variableMode);
    case 'uint32'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_UINT32', variableMode);
    case 'uint64'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_UINT64', variableMode);
    case 'int8'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_INT8',   variableMode);
    case 'int16'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_INT16',  variableMode);
    case 'int32'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_INT32',  variableMode);
    case 'int64'
        H5TypeID = get_hdf5_type_from_string(varA, 'H5T_NATIVE_INT64',  variableMode);
    otherwise
        error('Could not create HDF5 type ID for class %s', class(varA));
  end
end

% ######################################################################
% get_squeeze_dims()
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

% ######################################################################
% create_fp16_type()
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
