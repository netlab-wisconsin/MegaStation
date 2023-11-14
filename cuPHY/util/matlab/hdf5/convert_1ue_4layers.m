% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

pusch_rx = hdf5_load_nv('pusch_rx_MIMO4x8_PRB272_DataSyms9_1dmrs_1ue_4layers.h5');

config = struct('NumTransportBlocks',          uint32(1),      ...
                'NumLayers',                   uint32(4),      ...
                'InputLayerSize',              uint32(176256), ...
                'NumFillerBits',               uint32(112),    ...
                'TransportBlockSize',          uint32(573574), ...
                'CodeBlocksPerTransportBlock', uint32(69),     ...
                'ScramblingEnabled',           uint32(0),      ...
                'DmrsConfig',                  uint32(2));

h5File = H5F.create('pusch_rx_MIMO4x8_PRB272_SYM9_DMRS1_UE1_L4_SCR0.h5');
hdf5_write_nv(h5File, 'DataRx', pusch_rx.DataRx);
hdf5_write_nv(h5File, 'Data_sym_loc', pusch_rx.Data_sym_loc);
hdf5_write_nv(h5File, 'DescrShiftSeq', pusch_rx.DescrShiftSeq);
hdf5_write_nv(h5File, 'Noise_pwr', pusch_rx.Noise_pwr);
hdf5_write_nv(h5File, 'RxxInv', pusch_rx.RxxInv);
hdf5_write_nv(h5File, 'UnShiftSeq', pusch_rx.UnShiftSeq);
hdf5_write_nv(h5File, 'WFreq', pusch_rx.WFreq);
hdf5_write_nv(h5File, 'config', config);
H5F.close(h5File);
