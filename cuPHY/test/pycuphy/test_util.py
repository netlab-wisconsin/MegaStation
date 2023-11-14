# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Tests for pycuphy/util.py."""
import pytest

import pycuphy


@pytest.mark.parametrize("dmrs_fapi,dmrs_cuphy_ref", [
    (15, 19070976),
    (240, 17767)
])
def test_dmrs_fapi_to_cuphy(dmrs_fapi, dmrs_cuphy_ref):
    """Test dmrs_fapi_to_cuphy()."""
    dmrs_cuphy = pycuphy.dmrs_fapi_to_cuphy(dmrs_fapi)
    assert dmrs_cuphy_ref == dmrs_cuphy


def test_bit_list_to_uint16():
    """Test bit_list_to_uint16()."""
    array = [0,] * 14
    array[2] = 1
    symb_pos = pycuphy.bit_list_to_uint16(array)
    assert symb_pos == 4
