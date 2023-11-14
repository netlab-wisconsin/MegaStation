% Copyright 1993-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NOTICE TO LICENSEE:
%
% This source code and/or documentation ("Licensed Deliverables") are
% subject to NVIDIA intellectual property rights under U.S. and
% international Copyright laws.
%
% These Licensed Deliverables contained herein is PROPRIETARY and
% CONFIDENTIAL to NVIDIA and is being provided under the terms and
% conditions of a form of NVIDIA software license agreement by and
% between NVIDIA and Licensee ("License Agreement") or electronically
% accepted by Licensee.  Notwithstanding any terms or conditions to
% the contrary in the License Agreement, reproduction or disclosure
% of the Licensed Deliverables to any third party without the express
% written consent of NVIDIA is prohibited.
%
% NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
% LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
% SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
% PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
% NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
% DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
% NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
% NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
% LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
% SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
% DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
% WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
% ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
% OF THESE LICENSED DELIVERABLES.
%
% U.S. Government End Users.  These Licensed Deliverables are a
% "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
% 1995), consisting of "commercial computer software" and "commercial
% computer software documentation" as such terms are used in 48
% C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
% only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
% 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
% U.S. Government End Users acquire the Licensed Deliverables with
% only those rights set forth herein.
%
% Any use of the Licensed Deliverables in individual and commercial
% software must include, in the user documentation and internal
% comments to the code, the above Disclaimer and U.S. Government End
% Users Notice.

function out = table_symbol_demodulate(in, mod, QAM_noise_var)
  % Implements symbol demodulation (otherwise known as soft demapping)
  % using generated tables, with results intended to be identical to
  % the MATLAB nrSymbolDemodulate() function.
  %---------------------------------------------------------------------
  % PAM noise is 1/2 QAM noise, assuming noise power is equally
  % distributed between the in-phase and quadrature components
  PAM_noise_var = QAM_noise_var / 2;
  %---------------------------------------------------------------------
  % Load a table that matches the input modulation
  switch(mod)
    case 'BPSK'
      % Reuse the QPSK table here!
      T = readtable('QAM4_LLR.txt');
      A = 1 / sqrt(2);
      QAM_bits = uint32(1);
    case 'QPSK'
      T = readtable('QAM4_LLR.txt');
      A = 1 / sqrt(2);
      QAM_bits = uint32(2);
    case '16QAM'
      T = readtable('QAM16_LLR.txt');
      A = 1 / sqrt(10);
      QAM_bits = uint32(4);
    case '64QAM'
      T = readtable('QAM64_LLR.txt');
      A = 1 / sqrt(42);
      QAM_bits = uint32(6);
    case '256QAM'
      T = readtable('QAM256_LLR.txt');
      A = 1 / sqrt(170);
      QAM_bits = uint32(8);
    otherwise
      error('Invalid modulation: %s', mod)
  end
  num_symbols = size(in, 1);
  out = zeros(num_symbols * QAM_bits, 1);
  %---------------------------------------------------------------------
  PAM_bits = max(QAM_bits / 2, 1);
  %---------------------------------------------------------------------
  in_phase   = reshape(real(in), 1, length(in));
  quadrature = reshape(imag(in), 1, length(in));
  LLR_mat    = zeros(QAM_bits, length(in));
  for ii = 1:PAM_bits
    LLR_mat(ii * 2 - 1,:) = interp1(T.Zr, T.(sprintf('bit%d', ii-1)), in_phase, 'linear');
    if QAM_bits > 1
      LLR_mat(ii * 2 - 0,:) = interp1(T.Zr, T.(sprintf('bit%d', ii-1)), quadrature, 'linear');
    end
  end
  out = PAM_noise_var * LLR_mat(:);
end
