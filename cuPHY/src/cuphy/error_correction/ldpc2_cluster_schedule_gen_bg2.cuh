/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 42 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 42>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row<30, TIsFirst::value>();
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>();
    proc.template process_row<29, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>();
    proc.template process_row<37, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<36, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>();
    proc.template process_row<40, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<39, TIsFirst::value>();
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<35, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<21, TIsFirst::value>();
    proc.template process_row<38, TIsFirst::value>(); __syncthreads(); // (21)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<24, TIsFirst::value>(); __syncthreads(); // (22)
    proc.template process_row<34, TIsFirst::value>();
    proc.template process_row<41, TIsFirst::value>(); __syncthreads(); // (23)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 41 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 41>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row<30, TIsFirst::value>();
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>();
    proc.template process_row<29, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>();
    proc.template process_row<37, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>();
    proc.template process_row<40, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<35, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<39, TIsFirst::value>();
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<21, TIsFirst::value>();
    proc.template process_row<38, TIsFirst::value>(); __syncthreads(); // (21)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<34, TIsFirst::value>(); __syncthreads(); // (22)
    proc.template process_row<36, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (23)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 40 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 40>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row<30, TIsFirst::value>();
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>();
    proc.template process_row<29, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>();
    proc.template process_row<37, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<35, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<39, TIsFirst::value>();
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<21, TIsFirst::value>();
    proc.template process_row<38, TIsFirst::value>(); __syncthreads(); // (21)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<34, TIsFirst::value>(); __syncthreads(); // (22)
    proc.template process_row<36, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (23)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 39 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 39>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row<29, TIsFirst::value>();
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>();
    proc.template process_row<37, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<38, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<34, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (21)
    proc.template process_row<36, TIsFirst::value>();
    proc.template process_row<35, TIsFirst::value>(); __syncthreads(); // (22)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 38 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 38>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row<29, TIsFirst::value>();
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>();
    proc.template process_row<37, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<35, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<34, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (21)
    proc.template process_row<36, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (22)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 37 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 37>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row<29, TIsFirst::value>();
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<35, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<34, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (21)
    proc.template process_row<36, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (22)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 36 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 36>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row<29, TIsFirst::value>();
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<35, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<34, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (21)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 35 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 35>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row<29, TIsFirst::value>();
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<34, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<27, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (20)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (21)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 34 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 34>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row<29, TIsFirst::value>();
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<27, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>();
    proc.template process_row<33, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<21, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<23, TIsFirst::value>(); __syncthreads(); // (20)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 33 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 33>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row<29, TIsFirst::value>();
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<27, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<21, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<32, TIsFirst::value>();
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<23, TIsFirst::value>(); __syncthreads(); // (20)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 32 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 32>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>();
    proc.template process_row<31, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<27, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<21, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<29, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<23, TIsFirst::value>(); __syncthreads(); // (20)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 31 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 31>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>();
    proc.template process_row<30, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<27, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<21, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<29, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<23, TIsFirst::value>(); __syncthreads(); // (20)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 30 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 30>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<29, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<27, TIsFirst::value>(); __syncthreads(); // (20)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 29 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 29>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<28, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (19)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<27, TIsFirst::value>(); __syncthreads(); // (20)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 28 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 28>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<27, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 27 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 27>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row<26, TIsFirst::value>();
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 26 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 26>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<25, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<18, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 25 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 25>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<24, TIsFirst::value>();
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 24 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 24>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<23, TIsFirst::value>();
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<15, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 23 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 23>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<15, TIsFirst::value>();
    proc.template process_row<22, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 22 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 22>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>();
    proc.template process_row<21, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 21 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 21>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
    proc.template process_row<20, TIsFirst::value>(); __syncthreads(); // (19)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 20 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 20>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
    proc.template process_row<19, TIsFirst::value>(); __syncthreads(); // (18)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 19 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 19>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<18, TIsFirst::value>();
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 18 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 18>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<17, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<12, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (16)
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (17)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 17 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 17>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
    proc.template process_row<16, TIsFirst::value>(); __syncthreads(); // (16)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 16 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 16>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
    proc.template process_row<15, TIsFirst::value>(); __syncthreads(); // (15)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 15 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 15>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
    proc.template process_row<14, TIsFirst::value>(); __syncthreads(); // (14)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 14 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 14>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
    proc.template process_row<13, TIsFirst::value>(); __syncthreads(); // (13)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 13 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 13>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
    proc.template process_row<12, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (12)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 12 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 12>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>();
    proc.template process_row<11, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 11 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 11>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
    proc.template process_row<10, TIsFirst::value>(); __syncthreads(); // (11)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 10 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2, 10>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
    proc.template process_row< 9, TIsFirst::value>(); __syncthreads(); // (10)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 9 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2,  9>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
    proc.template process_row< 8, TIsFirst::value>(); __syncthreads(); // (9)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 8 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2,  8>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
    proc.template process_row< 7, TIsFirst::value>(); __syncthreads(); // (8)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 7 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2,  7>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
    proc.template process_row< 6, TIsFirst::value>(); __syncthreads(); // (7)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 6 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2,  6>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
    proc.template process_row< 5, TIsFirst::value>(); __syncthreads(); // (6)
}
///////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG2 with 5 parity nodes
template <>
template <class TRowProcessor, class TIsFirst>
__device__
void row_schedule<2,  5>::process_rows(TRowProcessor&  proc,
                                       const TIsFirst& )
{
    proc.template process_row< 0, TIsFirst::value>(); __syncthreads(); // (1)
    proc.template process_row< 1, TIsFirst::value>(); __syncthreads(); // (2)
    proc.template process_row< 2, TIsFirst::value>(); __syncthreads(); // (3)
    proc.template process_row< 3, TIsFirst::value>(); __syncthreads(); // (4)
    proc.template process_row< 4, TIsFirst::value>(); __syncthreads(); // (5)
}
