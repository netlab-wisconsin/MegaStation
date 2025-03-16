% This file prepares data for baseband simulation.

%%%%%% CONFIG BEG %%%%%%
function data_generator(ue, bs, ofdm_ca, ofdm_da, sc_group, num_pilots, num_uplinks,...
                        num_downlinks, modulation_order, code_rate, base_graph, pilot_spacing) %#codegen
arguments
    ue = uint64(8);
    bs = uint64(16);
    ofdm_ca = uint64(2048);
    ofdm_da = uint64(1200);
    sc_group = uint64(8);
    num_pilots = uint64(1);
    num_uplinks = uint64(13);
    num_downlinks = uint64(13);
    modulation_order = 6;
    code_rate = 1 / 3;
    base_graph = 1;
    pilot_spacing = uint64(16);
end

noise_level = 0.03;
ofdm_start = (ofdm_ca - ofdm_da) / 2;
uncoded_uplink = ofdm_da * modulation_order * code_rate;
[encoded_uplink, decoded_uplink] = get_ldpc_config(uncoded_uplink, base_graph, code_rate);
uncoded_downlink = (ofdm_da - ofdm_da / pilot_spacing) * modulation_order * code_rate;
[~, decoded_downlink] = get_ldpc_config(uncoded_downlink, base_graph, code_rate);
%%%%%% CONFIG END %%%%%%

%%%%%% CSI GENERATION BEG %%%%%%
assert(mod(ofdm_ca, sc_group) == 0, "ofdm_ca % sc_group != 0");
assert(mod(ofdm_da, sc_group) == 0, "ofdm_da % sc_group != 0");

sqrt2_norm = 1 / single(sqrt(2));

num_scgroups = ofdm_ca / sc_group;
csi = complex(single(zeros(ue, bs, num_scgroups)));
for iter = 1:num_scgroups
    csi_freq = -1 + 2 * rand(ue, bs, "like", single(1i));
    csi_freq = csi_freq * sqrt2_norm;
    csi(:,:,iter) = csi_freq;
end
%%%%%% CSI GENERATION END %%%%%%

tx_symbols = complex(single(zeros(ofdm_ca, bs, num_pilots + num_uplinks)));
mac_symbols = randi([0,1], decoded_downlink, ue, num_downlinks);

file_tx = fopen('ant_data.data', 'w');
file_mac = fopen('mac_data.data', 'w');

%%%%%% PILOTS GENERATION BEG %%%%%%
zachu_seq = gen_zadoffchu(ofdm_da);
shifts = single((0:ofdm_da-1)') * 1j;
zachu_seq = zachu_seq .* exp(shifts * pi / 4);
pilots = complex(single(zeros(ofdm_ca, sc_group)));
for iter = 1:sc_group
    mask = single(zeros(sc_group,1));
    mask(iter) = 1;
    mask = repmat(mask, ofdm_da / sc_group, 1);
    pilots(ofdm_start+1:ofdm_start+ofdm_da, iter) = zachu_seq .* mask;
end
assert(idivide(ue, sc_group, "ceil") == num_pilots, "ue // sc_group != num_pilots");
for iter = 1:num_pilots
    pilot_iter = complex(single(zeros(ofdm_ca, bs)));
    for jter = 1:ofdm_ca
        pilot_iter(jter,:) = pilots(jter,:) *...
            csi((iter-1)*sc_group+1:iter*sc_group,:,idivide(jter-1, sc_group, "floor") + 1);
    end
    noise = randn(ofdm_ca, bs, "like", single(1i)) * noise_level * sqrt2_norm;
    tx_symbols(:,:,iter) = pilot_iter + noise;
end
%%%%%% PILOTS GENERATION END %%%%%%

smap = [47, 46, 42, 43, 59, 58, 62, 63,...
        45, 44, 40, 41, 57, 56, 60, 61,...
        37, 36, 32, 33, 49, 48, 52, 53,...
        39, 38, 34, 35, 51, 50, 54, 55,...
         7,  6,  2,  3, 19, 18, 22, 23,...
         5,  4,  0,  1, 17, 16, 20, 21,...
        13, 12,  8,  9, 25, 24, 28, 29,...
        15, 14, 10, 11, 27, 26, 30, 31];

% smap = [61, 29, 21, 53, 55, 23, 31, 63,...
%         45, 13,  5, 37, 39,  7, 15, 47,...
%         41,  9,  1, 33, 35,  3, 11, 43,...
%         57, 25, 17, 49, 51, 19, 27, 59,...
%         56, 24, 16, 48, 50, 18, 26, 58,...
%         40,  8,  0, 32, 34,  2, 10, 42,...
%         44, 12,  4, 36, 38,  6, 14, 46,...
%         60, 28, 20, 52, 54, 22, 30, 62];

%%%%%% UPLINKS GENERATION BEG %%%%%%
uplinks_bits = randi([0,1], decoded_uplink, ue, num_uplinks);
uplinks_encoded = zeros(encoded_uplink, ue, num_uplinks);
after_mod = encoded_uplink / modulation_order;
uplinks_modulated = complex((zeros(ofdm_da, ue, num_uplinks)));
for iter = 1:num_uplinks
    uplinks_bits(:,:,iter) = wlanScramble(uplinks_bits(:,:,iter), 93);
    uplinks_encoded(:,:,iter) = nrLDPCEncode(uplinks_bits(:,:,iter), base_graph);
end
uplinks_modulated(1:after_mod,:,:) = qammod(uplinks_encoded, 2^modulation_order, smap,...
                                            InputType='bit', UnitAveragePower=true);
uplinks = complex(single(zeros(ofdm_ca, ue, num_uplinks)));
uplinks(ofdm_start+1:ofdm_start+ofdm_da,:,:) = single(uplinks_modulated);
for iter = 1:num_uplinks
    uplink_iter = complex(single(zeros(ofdm_ca, bs)));
    for jter = 1:ofdm_ca
        uplink_iter(jter,:) = uplinks(jter,:,iter) * csi(:,:,idivide(jter-1, sc_group, "floor") + 1);
    end
    noise = randn(ofdm_ca, bs, "like", single(1i)) * noise_level * sqrt2_norm;
    tx_symbols(:,:,num_pilots+iter) = uplink_iter + noise;
end
%%%%%% UPLINKS GENERATION END %%%%%%

%%%%%% IFFT TX SYMBOLS BEG %%%%%%
tx_symbols = circshift(tx_symbols, ofdm_ca / 2);
tx_symbols = ifft(tx_symbols);
%%%%%% IFFT TX SYMBOLS END %%%%%%

%%%%%% WRITE DATA BEG %%%%%%
fwrite(file_tx, tx_symbols, 'single');
fwrite(file_mac, bit2int(mac_symbols,8,false), 'uint8');

fclose(file_tx);
fclose(file_mac);
%%%%%% WRITE DATA END %%%%%%
end

%%%%%% FUNCTIONS %%%%%%
function [zchu_seq] = gen_zadoffchu(seq_len)
    prime_vec = [
    2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,...
    41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,...
    97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,...
    157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,...
    227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,...
    283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,...
    367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,...
    439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,...
    509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,...
    599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,...
    661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,...
    751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,...
    829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,...
    919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,...
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,...
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,...
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,...
    1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,...
    1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,...
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,...
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,...
    1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,...
    1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783,...
    1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,...
    1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,...
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039];

    iter = 0;
    N = prime_vec(end);
    while iter + 1 < length(prime_vec)
        iter = iter + 1;
        if prime_vec(iter) <= seq_len && prime_vec(iter + 1) > seq_len
            N = prime_vec(iter);
            break;
        end
    end
    Rh = N * 2 / 31;
    R = floor(Rh + 0.5);
    zchu_seq = zadoffChuSeq(R, N);
    zchu_seq = single([zchu_seq; zchu_seq(1:seq_len-N)]);
end

function [inodes] = num_input_cols(bg)
    if bg == 1
        inodes = 22;
    else
        inodes = 10;
    end
end

function [dbits] = num_decoded_bits(bg, zc)
    dbits = num_input_cols(bg) * zc;
end

function [zc] = find_zc(bg, ubits)
    cb_per_symbol = 1;
    zc_vec = ...
     [2,   4,   8,   16, 32, 64,  128, 256, 3,   6,   12,  24, 48, ...
      96,  192, 384, 5,  10, 20,  40,  80,  160, 320, 7,   14, 28, ...
      56,  112, 224, 9,  18, 36,  72,  144, 288, 11,  22,  44, 88, ...
      176, 352, 13,  26, 52, 104, 208, 15,  30,  60,  120, 240];
    zc_vec = sort(zc_vec);
    zc = zc_vec(end);

    iter = 0;
    while iter + 1 < length(zc_vec)
        iter = iter + 1;
        if num_decoded_bits(bg, zc_vec(iter)) * cb_per_symbol <= ubits &&...
           num_decoded_bits(bg, zc_vec(iter+1)) * cb_per_symbol > ubits
            zc = zc_vec(iter);
        end
    end

    zc = uint64(zc);
end

function [ebits, dbits] = get_ldpc_config(ubits, bg, crate)
    info_nodes = uint64(num_input_cols(bg));
    zc = find_zc(bg, ubits);

    ebits = info_nodes / crate * zc;
    dbits = info_nodes * zc;
end