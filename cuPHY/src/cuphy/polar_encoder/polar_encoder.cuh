/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */



#if !defined(CUPHY_POLAR_ENCODER_CUH_INCLUDED_)
#define CUPHY_POLAR_ENCODER_CUH_INCLUDED_

#include <cooperative_groups.h>

using namespace cooperative_groups;

namespace polar_encoder
{

enum class SortDir_t : uint8_t
{
    ASCENDING  = 0,
    DESCENDING = 1
};

template <typename T_VAL>
__device__ __forceinline__ void sortComparator(SortDir_t dir, T_VAL& valA, T_VAL& valB)
{
    if(static_cast<SortDir_t>(valA > valB) == dir)
    {
        return;
    }

    // swap(valA, valB)
    T_VAL val = valA;
    valA      = valB;
    valB      = val;
}

// Reference: CUDA sample bitonicSortShared
// sortDir      - sort direction (ascending vs descending)
// nSortEntries - number of elements to be sorted, needs to be a power of 2. Also need blockDim.x*blockDim.y >= nItems/2
// pShEntries     - Pointer to entries to be sorted inplace in shared memory
template <typename Tentry=int16_t>
__device__ inline void bitonicSort(SortDir_t sortDir, uint32_t nSortEntries, Tentry* pShEntries) {

    thread_block const& thisThrdBlk = this_thread_block();
    uint32_t            tIdx        = thisThrdBlk.thread_rank();

    // Each thread implements a comparator which compares 2 elements in the input sequence. Thus a max of
    // (nSortEntries/2) threads are needed in the thread block. All comparisons occur in parllel.
    // Multiple such parallel comparisons are composed to form a comparator stage and multiple comparator stages
    // form the comparator network.
    // There are log2(nSortEntries) comparator stages with stage-n perfomring n*(nSortEntries/2) parallel comparisons
    // Consequently nSortEntries is constrained to be a power of 2

    bool thrdEnable = (tIdx < nSortEntries / 2);

    // Within each comparator stage, the first parallel comparison set starts with the widest possible stride
    // for that stage (i.e. stride = size/2) with subsequent stages narrowing strides down to stride = 1

    //---------------------------------------------------------------------------
    // Push the data through the sorting network
    // for loop below implements a comparator network potentially made of several comparator stages
    for(uint32_t size = 2; size < nSortEntries; size <<= 1)
    {
        // Bitonic sort divides input sequence into 2 halves: threads sorting the Lower half sort in
        // non-increasing order and those sorting the Upper half sort in non-decreasing order

        // Based on the value of size, the sort direction is held steady for the first (size/2) elements after
        // which the direction is flipped.

        // Bitonic merge
        SortDir_t dir = static_cast<SortDir_t>(((tIdx & (size / 2)) != 0));

        //---------------------------------------------------------------------------
        // for loop below implements a single comparator stage where (nSortEntries/2) parallel comparisons are
        // performed. Stride specifies the distance between the two elements in the sequence being compared
        for(uint32_t stride = size / 2; stride > 0; stride >>= 1)
        {
            if(thrdEnable)
            {
                // Since stride is a power of 2, (stride - 1) acts as a bitmask to modulo the thread index to range [0, stride-1]
                // Effectively 2 * tIdx - (tIdx & (stride - 1)) generates indices so that the input indices skip by stride for contiguous thread indices

                // idx + stride generates indices of the second input to the comparator
                uint32_t idx       = 2 * tIdx - (tIdx & (stride - 1));
                uint32_t entry1Idx = idx + 0;
                uint32_t entry2Idx = idx + stride;

                sortComparator<Tentry>(dir,
                                       pShEntries[entry1Idx],
                                       pShEntries[entry2Idx]);
            }
            thisThrdBlk.sync();
        }
    }

    // The sort direction for the last comparator stage is the same as the sort direction
    {
        for(uint32_t stride = nSortEntries / 2; stride > 0; stride >>= 1)
        {
            if(thrdEnable)
            {
                uint32_t idx       = 2 * tIdx - (tIdx & (stride - 1));
                uint32_t entry1Idx = idx + 0;
                uint32_t entry2Idx = idx + stride;

                sortComparator<Tentry>(sortDir,
                                       pShEntries[entry1Idx],
                                       pShEntries[entry2Idx]);
            }
            thisThrdBlk.sync();
        }
    }
}

} // namespace polar_encoder

#endif // !defined(CUPHY_POLAR_ENCODER_CUH_INCLUDED_)
