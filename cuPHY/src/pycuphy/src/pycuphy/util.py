# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Python bindings for cuPHY - Common utilities."""
from typing import Any
from typing import List
from typing import Tuple

from cuda import cudart  # type: ignore
import numpy as np

__all__ = [
    "dmrs_fapi_to_cuphy",
    "dmrs_api_convert",
    "bit_list_to_uint16",
    "check_cuda_errors",
    "get_mcs",
    "tb_size",
    "random_tb"
]


def dmrs_fapi_to_cuphy(dmrs_symb_pos: np.uint16) -> np.uint32:
    """Convert the SCF FAPI PUSCH DMRS ports entry to cuPHY format.

    FAPI follows the following format: A bitmap occupying the 11 LSBs, with:
        bit 0: antenna port 1000
        bit 11: antenna port 1011
        and for each bit
            0: DMRS port not used
            1: DMRS port used
    In other words, it is a bitmap showing which ports are used. The cuPHY interface
    on the other hand wants a 32-bit integer where the DMRS port index for each layer is encoded
    with 4 bits, up to 8 layers.

    Args:
        dmrs (np.uint16): The SCF FAPI DMRS port bitmap.

    Returns:
        np.uint32: The cuPHY format DMRS port indicator.
    """
    bit_array = np.array([int(k) for k in format(dmrs_symb_pos, "015b")[14:0:-1]])
    port_idx = np.nonzero(bit_array)[0]
    port_idx = port_idx.astype(np.uint32)
    cuphy_dmrs_port = np.uint32(0)
    for layer_port_idx in port_idx:
        cuphy_dmrs_port += ((layer_port_idx & 0x0F) << 28 - (layer_port_idx * 4))
    return cuphy_dmrs_port


def dmrs_api_convert(x: int, layer: int) -> int:
    """Convert DMRS port DL interface to UL interface."""
    dmrs_port_bmsk = 0
    ports = hex(x)[2:].zfill(8)
    for i in range(layer):
        dmrs_port_bmsk += 2 ** int(ports[i])
    return dmrs_port_bmsk


def bit_list_to_uint16(x: list) -> np.uint16:
    """Convert a list of bits to an np.uint16.

    Args:
        x (list): A bit array. The first bit corresponds to the LSB.

    Returns:
        np.uint16: The corresponding integer value.
    """
    k = 0
    pow_two = 1
    for b in x:
        k = k + int(b) * pow_two
        pow_two = pow_two * 2
    return np.uint16(k)


def check_cuda_errors(result: cudart.cudaError_t) -> Any:
    """Check CUDA errors."""
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def get_mcs(mcs: int, table_idx: int = 2) -> Tuple[int, float]:
    """Get modulation order and code rate based on MCS index.

    Args:
        mcs (int): MCS index pointing to the table indicated by `table_idx`.
        table_idx (int): Index of the MCS table in TS 38.214 section 5.1.3.1.
            Currently only `table_idx==2` is supported.

    Returns:
        int: Modulation order.
        float: Code rate * 1024.
    """
    assert table_idx == 2

    mcs_table = {
        0: [2, 120.],
        1: [2, 193.],
        2: [2, 308.],
        3: [2, 449.],
        4: [2, 602.],
        5: [4, 378.],
        6: [4, 434.],
        7: [4, 490.],
        8: [4, 553.],
        9: [4, 616.],
        10: [4, 658.],
        11: [6, 466.],
        12: [6, 517.],
        13: [6, 567.],
        14: [6, 616.],
        15: [6, 666.],
        16: [6, 719.],
        17: [6, 772.],
        18: [6, 822.],
        19: [6, 873.],
        20: [8, 682.5],
        21: [8, 711.],
        22: [8, 754.],
        23: [8, 797.],
        24: [8, 841.],
        25: [8, 885.],
        26: [8, 916.5],
        27: [8, 948.],
    }

    mod_order, code_rate = mcs_table[mcs]
    return int(mod_order), code_rate


def tb_size(
        mod_order: int,
        code_rate: float,
        dmrs: List[int],
        num_prb: int,
        num_symbols: int,
        num_layers: int) -> int:
    """Get transport block size based on given parameters.

    Determine transport block size as per TS 38.214 section 5.1.3.2.

    Args:
        mod_order (int): Modulation order.
        code_rate (float): Code rate * 1024 as in section 5.1.3.1 of TS 38.214.
        dmrs (List[int]): List of binary numbers indicating which symbols contain DMRS.
        num_prb (int): Number of PRBs.
        num_symbols (int): Number of symbols.
        num_layers (int): Number of layers.

    Returns:
        int: Transport block size in bits.
    """
    r = code_rate / 1024
    n_sc = 12
    n_re = num_prb * (num_symbols - np.sum(dmrs)) * n_sc  # Overhead parameter N_oh = 0.
    n_info = n_re * r * mod_order * num_layers

    if n_info <= 3824:

        n = np.max([3, np.floor(np.log2(n_info)) - 6])

        n_info_prime = np.max([24, np.power(2, n) * np.floor(n_info / np.power(2, n))])

        tbs_select = [
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
            136,
            144,
            152,
            160,
            168,
            176,
            184,
            192,
            208,
            224,
            240,
            256,
            272,
            288,
            304,
            320,
            336,
            352,
            368,
            384,
            408,
            432,
            456,
            480,
            504,
            528,
            552,
            576,
            608,
            640,
            672,
            704,
            736,
            768,
            808,
            848,
            888,
            928,
            984,
            1032,
            1064,
            1128,
            1160,
            1192,
            1224,
            1256,
            1288,
            1320,
            1352,
            1416,
            1480,
            1544,
            1608,
            1672,
            1736,
            1800,
            1864,
            1928,
            2024,
            2088,
            2152,
            2216,
            2280,
            2408,
            2472,
            2536,
            2600,
            2664,
            2728,
            2792,
            2856,
            2976,
            3104,
            3240,
            3368,
            3496,
            3624,
            3752,
            3824,
        ]

        for tbs_item in tbs_select:
            if tbs_item >= n_info_prime:
                tbs = tbs_item
                break

    else:

        n = np.floor(np.log2(n_info - 24)) - 5

        n_info_prime = np.max(
            [3840, np.power(2, n) * np.round((n_info - 24) / np.power(2, n))]
        )

        if r < 0.25:
            C = np.ceil((n_info + 24) / 3816)

            tbs = int(8 * C * np.ceil((n_info_prime + 24) / 8 / C))

        else:
            if n_info_prime > 8424:

                C = np.ceil((n_info_prime + 24) / 8424)

                tbs = int(8 * C * np.ceil((n_info_prime + 24) / 8 / C))

            else:
                tbs = int(8 * np.ceil((n_info_prime + 24) / 8))

        tbs -= 24

    return tbs


def random_tb(
        mod_order: int,
        code_rate: float,
        dmrs: List[int],
        num_prb: int,
        num_symbols: int,
        num_layers: int) -> np.ndarray:
    """Generate a random transport block.

    Generates random transport block bytes according to given parameters. The transport
    block size is first determined as per TS 38.214 section 5.1.3.2.

    Args:
        mod_order (int): Modulation order.
        code_rate (float): Code rate * 1024 as in section 5.1.3.1 of TS 38.214.
        dmrs (List[int]): List of binary numbers indicating which symbols contain DMRS.
        num_prb (int): Number of PRBs.
        num_symbols (int): Number of symbols.
        num_layers (int): Number of layers.

    Returns:
        np.ndarray: Random transport block payload bytes.
    """
    tbs = tb_size(
        mod_order=mod_order,
        code_rate=code_rate,
        dmrs=dmrs,
        num_prb=num_prb,
        num_symbols=num_symbols,
        num_layers=num_layers
    )
    payload_bytes = np.random.randint(0, 255, size = tbs // 8, dtype=np.uint8)

    return payload_bytes
