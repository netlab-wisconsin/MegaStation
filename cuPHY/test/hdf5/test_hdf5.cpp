/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <gtest/gtest.h>
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"


using namespace hdf5hpp;
using namespace cuphy;

////////////////////////////////////////////////////////////////////////
// StructValueAsTest
// Reads values stored in an HDF5 structure using the following
// functions:
// cuphyHDF5GetStruct()
// cuphyHDF5GetStructScalar()
// Compare values read to the expected value, when retrieved as the
// actual stored type (without conversion)
TEST(HDF5, StructValueAsTest)
{
    hdf5_file        f = hdf5_file::open("struct_example.h5");
    cuphyHDF5_struct s = cuphy::get_HDF5_struct(f, "A");
    //cuphyVariant_t   v = s.get_value("five_as_uint32");
    EXPECT_EQ(s.get_value_as<double>("one_as_double"),          1.0);
    EXPECT_EQ(s.get_value_as<float>("two_as_single"),           2.0f);
    EXPECT_EQ(s.get_value_as<unsigned char>("three_as_uint8"),  3);
    EXPECT_EQ(s.get_value_as<unsigned short>("four_as_uint16"), 4);
    EXPECT_EQ(s.get_value_as<unsigned int>("five_as_uint32"),   5);
    EXPECT_EQ(s.get_value_as<signed char>("six_as_int8"),       6);
    EXPECT_EQ(s.get_value_as<short>("seven_as_int16"),          7);
    EXPECT_EQ(s.get_value_as<int>("eight_as_int32"),            8);
    // (No int64 in cuPHY tensor types at the moment)
    //EXPECT_EQ(s.get_value_as<long int>("nine_as_int64"),        8);
}

////////////////////////////////////////////////////////////////////////
// StructValueAsConvertTest
// Reads values stored in an HDF5 structure using the following
// functions:
// cuphyHDF5GetStruct()
// cuphyHDF5GetStructScalar()
// Compare values read to the expected value, when retrieved as the
// a different type than the stored type (i.e. with conversion)
TEST(HDF5, StructValueAsConvertTest)
{
    hdf5_file        f = hdf5_file::open("struct_example.h5");
    cuphyHDF5_struct s = cuphy::get_HDF5_struct(f, "A");
    //cuphyVariant_t   v = s.get_value("five_as_uint32");
    EXPECT_EQ(s.get_value_as<float>("one_as_double"),           1.0f);
    EXPECT_EQ(s.get_value_as<double>("two_as_single"),          2.0);
    EXPECT_EQ(s.get_value_as<unsigned int>("three_as_uint8"),   3);
    EXPECT_EQ(s.get_value_as<unsigned char>("four_as_uint16"),  4);
    EXPECT_EQ(s.get_value_as<unsigned short>("five_as_uint32"), 5);
    EXPECT_EQ(s.get_value_as<unsigned char>("six_as_int8"),     6);
    EXPECT_EQ(s.get_value_as<unsigned int>("seven_as_int16"),   7);
    EXPECT_EQ(s.get_value_as<signed char>("eight_as_int32"),    8);
    // (No int64 in cuPHY tensor types at the moment)
    //EXPECT_EQ(s.get_value_as<long int>("nine_as_int64"),        8);
}

////////////////////////////////////////////////////////////////////////
// ArrayOfStructTest
// Reads values stored in an HDF5 "array of structures" (e.g. a dataset
// with a compound type), using the following functions:
// functions:
// cuphyHDF5GetStruct()
// cuphyHDF5GetStructScalar()
// Compare values read to the expected value, which for this case
// is a 'value' field with a value equal to the index.
TEST(HDF5, ArrayOfStructTest)
{
    hdf5_file        f = hdf5_file::open("struct_array_example.h5");
    hdf5_dataset     dset = f.open_dataset("A");
    EXPECT_EQ(dset.get_dataspace().get_rank(), 1);
    std::vector<hsize_t> dims = dset.get_dataspace().get_dimensions();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_TRUE(dims[0] > 0);
    for(hsize_t i = 0; i < dims[0]; ++i)
    {
        cuphyHDF5_struct s = cuphy::get_HDF5_struct_index(dset, i);
        EXPECT_EQ(s.get_value_as<unsigned int>("value"), static_cast<unsigned int>(i));
    }
}

////////////////////////////////////////////////////////////////////////
// ArrayOfStructOutOfRangeTest
// Reads values stored in an HDF5 "array of structures" (e.g. a dataset
// with a compound type), using the following functions:
// functions:
// cuphyHDF5GetStruct()
// cuphyHDF5GetStructScalar()
// Tests whether an exception is thrown when the array is accessed
// with an index outside of the dataset bounds. Verifies that an
// exception was thrown with the expected status value.
TEST(HDF5, ArrayOfStructOutOfRangeTest)
{
    hdf5_file        f = hdf5_file::open("struct_array_example.h5");
    hdf5_dataset     dset = f.open_dataset("A");
    EXPECT_EQ(dset.get_dataspace().get_rank(), 1);
    std::vector<hsize_t> dims = dset.get_dataspace().get_dimensions();
    EXPECT_EQ(dims.size(), 1);
    cuphyHDF5Status_t s = CUPHYHDF5_STATUS_SUCCESS;
    try
    {
        cuphyHDF5_struct s = cuphy::get_HDF5_struct_index(dset, dims[0]);
    }
    catch(cuphyHDF5_exception& e)
    {
        s = e.status();
    }
    EXPECT_EQ(s, CUPHYHDF5_STATUS_VALUE_OUT_OF_RANGE);
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
