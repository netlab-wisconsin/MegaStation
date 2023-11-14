#!/bin/python3
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vector',  nargs='?', help='Test Vector to log')
parser.add_argument('-t', '--test',    nargs='?', help='Test type (e.g. Sanitizer) Run')
parser.add_argument('-r', '--results', nargs='?', help='Results (e.g. number of errors)')
parser.add_argument('-l', '--log',     default='testSummary.csv', help='File to record logs')

args=parser.parse_args()

if(args.test != None and args.results != None and args.vector != None):
    try:
        testSummary = pandas.read_csv(args.log,index_col=0)
    except:
        testSummary = pandas.DataFrame()

    # Just use the filename if a path was provided
    testvector = args.vector.split('/')[-1]
    testSummary.loc[testvector,args.test] = args.results

    testSummary.to_csv(args.log)
