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

# Python script to generate "clustered" schedules that group rows
# together when they don't have overlapping variable nodes.
#
# Uses the NetworkX library to solve an equivalent "maximum match"
# graph problem.
#
# TODO: Come up with a weighting scheme. BG1 schedules result in
# local memory spills via increased register pressure when compared
# to manual schedules that begin with grouping rows at the "bottom"
# of the graph.

import networkx as nx

def write_pairs(f, bg, num_nodes, match_dict, use_barrier):
    f.write('///////////////////////////////////////////////////////////////////////\n')
    f.write('// row_schedule specialization for BG%d with %d parity nodes\n' % (bg, num_nodes))
    f.write('template <>\n')
    f.write('template <class TRowProcessor, class TIsFirst, int M, int N>\n')
    f.write('__device__\n')
    f.write('void row_schedule<%d, %2d>::process_rows(TRowProcessor&            proc,\n' % (bg, num_nodes))
    f.write('                                       const TIsFirst&,\n')
    f.write('                                       const LDPC_kernel_params& params,\n')
    f.write('                                       int                       (&app_addr)[M],\n')
    f.write('                                       word_t                    (&app)[N])\n')
    f.write('{\n')

    sync_count = 1
    for idx in range(num_nodes):
        if idx in match_dict:
            p = match_dict[idx]
            #print('%d in matches (%d, %d)' % (idx, p[0], p[1]))
            # only write once for each element (it may be in a pair)
            if (idx < p[0]) or (idx < p[1]):
                f.write('    proc.template process_row<%2d, TIsFirst::value>(params, app_addr, app);\n' % p[0])
                f.write('    proc.template process_row<%2d, TIsFirst::value>(params, app_addr, app); __syncthreads(); // (%d)\n' % (p[1], sync_count))
                #f.write('        //---------- (%d)\n' % sync_count)
                sync_count = sync_count + 1
        else:
            #print('%d not in matches' % idx)
            if idx == 4 and use_barrier:
                # Special case: Inserting a barrier after row 4 seems to prevent local memory spills
                f.write('    proc.template process_row<%2d, TIsFirst::value>(params, app_addr, app); __syncthreads(); LDPC2_REG_BARRIER(%d); // (%d)\n' % (idx, idx, sync_count))
            else:
                f.write('    proc.template process_row<%2d, TIsFirst::value>(params, app_addr, app); __syncthreads(); // (%d)\n' % (idx, sync_count))
                #f.write('        //---------- (%d)\n' % sync_count)
            sync_count = sync_count + 1
    f.write('}\n')

bg = 2

infile = 'bg%d_non_intersecting_mod.txt' % bg
use_barrier = (bg == 1)
min_nodes = 4

#***********************************************************************
# Read input data: an adjacency list of non-intersecting rows
G = nx.read_adjlist(infile)
print('%d nodes, %d edges' % (G.number_of_nodes(), G.number_of_edges()))

num_nodes = G.number_of_nodes()

#***********************************************************************
# Open the generated output header file
f = open('ldpc2_cluster_schedule_gen_bg%d.cuh' % bg, 'w')

for n in range(num_nodes, min_nodes, -1):
    if n != num_nodes:
        print('Removing node %d' % n)
        G.remove_node('%d' % n)
    #-------------------------------------------------------------------
    mate = nx.max_weight_matching(G)
    print('maximum matching: %d edges (%d paired nodes)' % (len(mate), 2 * len(mate)))
    #print(mate)
    #print(type(mate))
    matches = {}
    for p in mate:
        print(p)
        matches[int(p[0])] = (int(p[0]), int(p[1]))
        matches[int(p[1])] = (int(p[0]), int(p[1]))
    #print(matches)
    write_pairs(f, bg, n, matches, use_barrier)

f.close()
