% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

%bg_values = [1 2];              % Base graph numbers
bg_values = [2];              % Base graph numbers
%Z_values = [64 96 128 160 192 224 256 288 320 352 384];
%Z_values = [384];
Z_values = [64];
%SNR_values = [10];
SNR_values = [3.5];
%C = [2000];                    % Number of code words
C = [1];                    % Number of code words

% When puncture_values is 1, LLR values for the first 2*Z_c values
% will be zero. (Punctured values are not transmitted, and thus the
% receiver has no information about them.)
%puncture_values = [0];
puncture_values = [1];

% If noise_same == 1, all codewords will use the same noise. Otherwise,
% noise values will be different. If early termination is enabled, then
% the number of iterations required to decode may vary.
%noise_same = 1;
noise_same = 0;

for bgn = bg_values
    for Z_c = Z_values
        if 1 == bgn
            Kb = 22;
            K = 22 * Z_c;
            F = 0;
        else
            K = 10 * Z_c;              
            if Z_c > 64
                Kb = 10;
            elseif Z_c > 56
                Kb = 9;
            elseif Z_c >= 20
                Kb = 8;
            else
                Kb = 6;
            end
            F = (10 - Kb) * Z_c;
        end

        % Generate a random set of bits
        s = [round(rand(Kb * Z_c,1)); zeros(F,1)];

        % Open source encoder:
        % https://github.com/xiaoshaoning/5g-ldpc
        [encoded_bits, H, Z_c, encoded_bits_original] = ldpc_encode(s, bgn);

        NCW_MAX = max(C(:));
        rxcodedcbs = double(1-2*encoded_bits_original);    % Convert to soft bits
        N = length(rxcodedcbs);
        noise = randn(N, NCW_MAX);
        if noise_same
            noise_same_str = '';
        else
            noise_same_str = '_m';   
        end
        rxcodedcbs = repmat(rxcodedcbs, 1, NCW_MAX);

        fprintf('BG = %d, Z = %d, K = %d, N = %d, R = %.2f\n', bgn, Z_c, K, N, K / N);

        for SNR = SNR_values
            % Use non-rate corrected SNR for now
            sigma2 = 10^(-SNR/10)
            scaled_noise = (sqrt(sigma2) * noise);
            rx_plus_noise = rxcodedcbs + scaled_noise;
            % Set filler bits to Inf (to decode to zero)
            rx_plus_noise(K-F+1:K,:) = Inf;
            for NCW = C
                if noise_same
                    % Repeat the first set once for each codeword
                    rx_coded_multi = repmat(rx_plus_noise(:,1), 1, NCW);
                else
                    % Extract the requested number of codewords
                    rx_coded_multi = rx_plus_noise(:,1:NCW);
                end
                for punc = puncture_values
                    rx_coded_multi_store = rx_coded_multi;
                    if punc
                        fname = sprintf('ldpc_BG%i_K%i_SNR%g_%i_p%s.h5', bgn, K, SNR, NCW, noise_same_str);
                        fprintf('Puncturing first %d LLR values\n', 2*Z_c);
                        rx_coded_multi_store(1:(2*Z_c), :) = 0;
                    else
                        fname = sprintf('ldpc_BG%i_K%i_SNR%g_%i%s.h5', bgn, K, SNR, NCW, noise_same_str);
                    end
                    fprintf('Creating file: %s\n', fname);
                    h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
                    fprintf('    writing sourceData dataset\n');
                    hdf5_write_nv(h5File, 'sourceData',   uint8(s));
                    fprintf('    writing inputCodeWord dataset\n');
                    hdf5_write_nv(h5File, 'inputCodeWord',   uint8(encoded_bits_original));
                    fprintf('    writing inputLLR dataset\n');
                    hdf5_write_nv(h5File, 'inputLLR',        single(rx_coded_multi_store));
                    H5F.close(h5File);
                end % puncture_values
            end % NCW
        end % SNR
    end % Z_values
end % bgn
