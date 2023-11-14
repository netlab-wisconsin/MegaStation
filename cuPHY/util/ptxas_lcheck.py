#!/usr/bin/env python

# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Run like this:
# make -j 20 2>&1 | ../util/ptxas_lcheck.py

import fileinput
import re
import subprocess

re_func_str  = r'ptxas info    : Function properties for (.*)'
re_local_str = r'    (\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads'

func_name = ''

kernels_with_lmem = []

for line in fileinput.input():
    line = line.rstrip()
    if not func_name:
        m = re.search(re_func_str, line)
        if m:
            func_name = m.group(1)
            #print('function: %s' % func_name)
        else:
            func_name = ''
    else:
        m = re.search(re_local_str, line)
        if m:
            if (int(m.group(1)) != 0) or (int(m.group(2)) != 0) or (int(m.group(3)) != 0):
                #print(line)
                cmd = 'c++filt %s' % func_name
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                func_name_demangled = proc.stdout.readline().decode('ascii')
                kernels_with_lmem.append((func_name_demangled.rstrip(), line))
        else:
            func_name = ''
    print(line)

kernels_with_lmem_sorted = sorted(kernels_with_lmem, key=lambda t:t[0])

print('Kernels with local memory usage:')
for k in kernels_with_lmem_sorted:
    print(k[0])
    print(k[1])
        
