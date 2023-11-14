/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <gtest/gtest.h>
#include "hdf5hpp.hpp"

using namespace hdf5hpp;

////////////////////////////////////////////////////////////////////////
// StructValueAsTest
// Reads values stored in an HDF5 structure using C++ HDF5 wrappers.
// Compare values read to the expected value, when retrieved as the
// actual stored type (without conversion)
TEST(HDF5Host, StructValueAsTest)
{
    hdf5_file             f       = hdf5_file::open("struct_example.h5");
    hdf5hpp::hdf5_dataset dset_A  = f.open_dataset("A");
    EXPECT_EQ(dset_A[0]["one_as_double"].as<double>(),          1.0);
    EXPECT_EQ(dset_A[0]["two_as_single"].as<float>(),           2.0f);
    EXPECT_EQ(dset_A[0]["three_as_uint8"].as<unsigned char>(),  3);
    EXPECT_EQ(dset_A[0]["four_as_uint16"].as<unsigned short>(), 4);
    EXPECT_EQ(dset_A[0]["five_as_uint32"].as<unsigned int>(),   5);
    EXPECT_EQ(dset_A[0]["six_as_int8"].as<signed char>(),       6);
    EXPECT_EQ(dset_A[0]["seven_as_int16"].as<short>(),          7);
    EXPECT_EQ(dset_A[0]["eight_as_int32"].as<int>(),            8);
    EXPECT_EQ(dset_A[0]["nine_as_int64"].as<long int>(),        9);
}

////////////////////////////////////////////////////////////////////////
// StructValueAsConvertTest
// Reads values stored in an HDF5 structure using C++ HDF5 wrappers.
// Compare values read to the expected value, when retrieved as the
// a different type than the stored type (i.e. with conversion)
TEST(HDF5Host, StructValueAsConvertTest)
{
    hdf5_file             f       = hdf5_file::open("struct_example.h5");
    hdf5hpp::hdf5_dataset dset_A  = f.open_dataset("A");
    EXPECT_EQ(dset_A[0]["one_as_double"].as<float>(),           1.0);
    EXPECT_EQ(dset_A[0]["two_as_single"].as<double>(),          2.0f);
    EXPECT_EQ(dset_A[0]["three_as_uint8"].as<unsigned int>(),   3);
    EXPECT_EQ(dset_A[0]["four_as_uint16"].as<unsigned char>(),  4);
    EXPECT_EQ(dset_A[0]["five_as_uint32"].as<unsigned short>(), 5);
    EXPECT_EQ(dset_A[0]["six_as_int8"].as<unsigned char>(),     6);
    EXPECT_EQ(dset_A[0]["seven_as_int16"].as<unsigned int>(),   7);
    EXPECT_EQ(dset_A[0]["eight_as_int32"].as<signed char>(),    8);
    EXPECT_EQ(dset_A[0]["nine_as_int64"].as<unsigned int>(),    9);
}

////////////////////////////////////////////////////////////////////////
// ArrayOfStructTest
// Reads values stored in an HDF5 "array of structures" (e.g. a dataset
// with a compound type), using the C++ wrappers.
// Compare values read to the expected value, which for this case
// is a 'value' field with a value equal to the index.
TEST(HDF5Host, ArrayOfStructTest)
{
    hdf5_file        f      = hdf5_file::open("struct_array_example.h5");
    hdf5_dataset     dset_A = f.open_dataset("A");
    EXPECT_EQ(dset_A.get_dataspace().get_rank(), 1);
    std::vector<hsize_t> dims = dset_A.get_dataspace().get_dimensions();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(dims[0] > 0);
    for(hsize_t i = 0; i < dims[0]; ++i)
    {
        EXPECT_EQ(dset_A[i]["value"].as<unsigned int>(), static_cast<unsigned int>(i));
    }
}

////////////////////////////////////////////////////////////////////////
// ArrayOfStructOutOfRangeTest
// Reads values stored in an HDF5 "array of structures" (e.g. a dataset
// with a compound type), using the C++ wrappers.
// Tests whether an exception is thrown when the array is accessed
// with an index outside of the dataset bounds. Verifies that an
// exception was thrown.
TEST(HDF5Host, ArrayOfStructOutOfRangeTest)
{
    hdf5_file            f       = hdf5_file::open("struct_array_example.h5");
    hdf5_dataset         dset_A  = f.open_dataset("A");
    EXPECT_EQ(dset_A.get_dataspace().get_rank(), 1);
    std::vector<hsize_t> dims    = dset_A.get_dataspace().get_dimensions();
    bool                 failure = false;
    EXPECT_EQ(dims.size(), 1);
    try
    {
        // The last valid element should be dims[0] - 1, so we expect
        // an exception to be thrown here...
        unsigned int u = dset_A[dims[0]]["value"].as<unsigned int>();
    }
    catch(...)
    {
        failure = true;
    }
    EXPECT_TRUE(failure);
}

////////////////////////////////////////////////////////////////////////
// StructWithArrayTest
// Reads values stored in an HDF5 compound data type member that are an
// HDF5 "array" type, using the C++ wrappers.
TEST(HDF5Host, StructWithArrayTest)
{
    typedef std::array<int, 3> array3i;
    hdf5_file            f       = hdf5_file::open("struct_with_array_example.h5");
    hdf5_dataset         dset_A  = f.open_dataset("A");
    array3i              a       = dset_A[3]["array"].as<array3i>();
    for(size_t i = 0; i < 3; ++i)
    {
        // Output of 'h5dump struct_with_array_example.h5':
        // (3): {
        //       1313,
        //       1252.89,
        //       84.11,
        //       [ 1, 2, 3 ]
        //    }
        EXPECT_EQ(a[i], i + 1);
    }
}

////////////////////////////////////////////////////////////////////////
// StructWithArrayReadAsVectorTest
// Reads values stored in an HDF5 compound data type member that are an
// HDF5 "array" type, using the C++ wrappers.
TEST(HDF5Host, StructWithArrayReadAsVectorTest)
{
    typedef std::vector<int> veci;
    hdf5_file            f       = hdf5_file::open("struct_with_array_example.h5");
    hdf5_dataset         dset_A  = f.open_dataset("A");
    veci                 a       = dset_A[3]["array"].as<veci>();
    for(size_t i = 0; i < 3; ++i)
    {
        // Output of 'h5dump struct_with_array_example.h5':
        // (3): {
        //       1313,
        //       1252.89,
        //       84.11,
        //       [ 1, 2, 3 ]
        //    }
        EXPECT_EQ(a[i], i + 1);
    }
}

////////////////////////////////////////////////////////////////////////
// StructWithVariableLengthArrayTest
// Reads values stored in an HDF5 compound data type member that are an
// HDF5 "variable length array" type, using the C++ wrappers.
TEST(HDF5Host, StructWithVariableLengthArrayTest)
{
    typedef std::vector<int>   veci;
    typedef std::array<int, 4> array4i;
    hdf5_file            f        = hdf5_file::open("struct_with_vlen_example.h5");
    hdf5_dataset         dset_A   = f.open_dataset("A");
    veci                 a        = dset_A[3]["vlen"].as<veci>();
    array4i              expected = {1, 2, 3, 4};
    for(size_t i = 0; i < a.size(); ++i)
    {
        // Output of 'h5dump struct_with_vlen_example.h5':
        // (3): {
        //       1313,
        //       1252.89,
        //       84.11,
        //       ( 1, 2, 3, 4)
        //    }
        EXPECT_EQ(a[i], expected[i]);
    }
}

////////////////////////////////////////////////////////////////////////
// StructWithVariableLengthArrayReadAsArrayTest
// Reads values stored in an HDF5 compound data type member that are an
// HDF5 "variable length array" type, using the C++ wrappers.
TEST(HDF5Host, StructWithVariableLengthArrayReadAsArrayTest)
{
    typedef std::array<int, 3> array3i;
    hdf5_file            f       = hdf5_file::open("struct_with_vlen_example.h5");
    hdf5_dataset         dset_A  = f.open_dataset("A");
    array3i              a       = dset_A[0]["vlen"].as<array3i>();
    for(size_t i = 0; i < 3; ++i)
    {
        // Output of 'h5dump struct_with_vlen_example.h5':
        // (0): {
        //       1153,
        //       53.23,
        //       24.57,
        //       (1, 2, 3)
        //    },
        EXPECT_EQ(a[i], i + 1);
    }
}

////////////////////////////////////////////////////////////////////////
// StructWithScalarReadAsVectorTest
// Reads values stored in an HDF5 compound data type member that are an
// HDF5 "scalar" type into a destination vector, using the C++ wrappers.
// Assuming that the correct behavior is to return a vector with size 1.
TEST(HDF5Host, StructWithScalarReadAsVectorTest)
{
    typedef std::vector<int> veci;
    hdf5_file            f       = hdf5_file::open("struct_example.h5");
    hdf5_dataset         dset_A  = f.open_dataset("A");
    veci                 a       = dset_A[0]["eight_as_int32"].as<veci>();
    EXPECT_EQ(a.size(), 1);
    EXPECT_EQ(a[0], 8);
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
