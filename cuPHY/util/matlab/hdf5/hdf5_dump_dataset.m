% Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

function hdf5_dump_dataset(infile, dsetname, outfile)
  % HDF5_DUMP_DATASET Write a single set from an HDF5 file to
  % a binary file.
  %
  % Example usage:
  %   hdf5_dump_dataset('TV_cuphy_pusch-TC1_snrdb40.00_iter1_MIMO4x4_PRB272_DataSyms9_qam256.h5', 'DataRx', 'out.bin');

  %---------------------------------------------------------------------
  % Load the input HDF5 file
  A = hdf5_load_nv(infile);
  outarray = A.(dsetname);
  if nargin < 3
      outfile = get_filename(infile, dsetname, outarray);
  end
  %---------------------------------------------------------------------
  % Open the output file
  fprintf('Output file: %s\n', outfile);
  fileID = fopen(outfile, 'w');
  %---------------------------------------------------------------------
  if isstruct(outarray)
      error('struct dump not implemented');
  elseif isreal(outarray)
      write_real(fileID, outarray);
  else
      write_complex(fileID, outarray);
  end
  %---------------------------------------------------------------------
  fclose(fileID);
end

function write_real(fileID, a)
    fprintf('Writing %d values of type %s\n', numel(a), class(a));
    fwrite(fileID, a(:), class(a));
end

function write_complex(fileID, a)
    % Create a single array of interleaved values
    a_real = [real(a(:))         zeros(numel(a), 1)]';
    a_imag = [zeros(numel(a), 1) imag(a(:))        ]';
    a_interleaved = a_real(:) + a_imag(:);
    fprintf('Writing %d values of type %s (2 values per complex element)\n', numel(a_interleaved), class(a_interleaved));
    fwrite(fileID, a_interleaved, class(a));
end

function fname = get_filename(infile, dsetname, var)
    [filepath, name, ext] = fileparts(infile);
    nd = ndims(var);
    sz = size(var);
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
    % Create a string with the type
    typestr = class(var);
    if ~isreal(var)
        typestr = strcat(typestr, '_complex');
    end
    fname = strcat(name, '_', dsetname, '_', szstr, '_', typestr, '.bin');
    fname = fullfile(filepath, fname);
end
