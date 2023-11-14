# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def parse_power(lines):

    results = []

    for line in lines:

        lst = line.split()

        if len(lst) == 8 and lst[-1] == "MiB":
            result = []
            result.append(lst[0])
            result.append(lst[2])
            result.append(lst[4])
            result.append(lst[6])

            results.append(result)

    return results
