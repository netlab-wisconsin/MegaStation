% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

function generate_hdf5_tests(output_dir)
  if nargin == 0
    output_dir = '.';
  end

  types = {'single', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32'};
  is_complex = [true false];
  complex_str = {'_complex', ''};
  datasetName = 'A';
  Nrows = 10;
  Ncols = 12;

  for idxType = 1:length(types)
    for idxComplex = 1:length(is_complex)
      fname = fullfile(output_dir, sprintf('test_%s%s.h5', types{idxType}, complex_str{idxComplex}));
      fprintf("%s\n", fname);
      % Create an empty file
      file = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
      % Generate a matrix
      A = gen_matrix(Nrows, Ncols, types{idxType}, is_complex(idxComplex));
      % Write the matrix to the file
      hdf5_write_nv(file, 'A', A);
      H5F.close(file);
    end
  end
end

function A = gen_matrix(Nrows, Ncols, typestr, cmplx)
  % Create an identifiable matrix with the appropriate type
  % using the cast() function
  if cmplx
    %A = cast([1:Nrows]' * ones(1, Ncols) + j * ones(Nrows, 1) * [1:Ncols], typestr);
    A = cast([0:(Nrows-1)]' * ones(1, Ncols) + j * ones(Nrows, 1) * [0:(Ncols-1)], typestr);
  else
    %A = cast([1:Nrows]' * ones(1, Ncols), typestr);
    %A = cast([0:(Nrows-1)]' * ones(1, Ncols), typestr);
    A = cast(repmat([1:Ncols] * Nrows, Nrows, 1) + [0:(Nrows-1)]' * ones(1, Ncols), 'single');
    % For Nrows = 10, Ncols = 12, A = :
    % 10    20    30    40    50    60    70    80    90   100   110   120
    % 11    21    31    41    51    61    71    81    91   101   111   121
    % 12    22    32    42    52    62    72    82    92   102   112   122
    % 13    23    33    43    53    63    73    83    93   103   113   123
    % 14    24    34    44    54    64    74    84    94   104   114   124
    % 15    25    35    45    55    65    75    85    95   105   115   125
    % 16    26    36    46    56    66    76    86    96   106   116   126
    % 17    27    37    47    57    67    77    87    97   107   117   127
    % 18    28    38    48    58    68    78    88    98   108   118   128
    % 19    29    39    49    59    69    79    89    99   109   119   129
  end
end
