/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(HDF5HPP_HPP_INCLUDED_)
#define HDF5HPP_HPP_INCLUDED_

#include "hdf5.h"
#include <exception>
#include <utility> // std::forward()
#include <array>
#include <vector>
#include <stdexcept> // std::runtime_error()
#include <numeric>   // std::accumulate()

namespace hdf5hpp
{

template <typename T>  struct hdf5_native_type;
template <>            struct hdf5_native_type<float>    { static hid_t copy() { return H5Tcopy(H5T_NATIVE_FLOAT);  } };
template <>            struct hdf5_native_type<double>   { static hid_t copy() { return H5Tcopy(H5T_NATIVE_DOUBLE); } };
template <>            struct hdf5_native_type<int8_t>   { static hid_t copy() { return H5Tcopy(H5T_NATIVE_INT8);   } };
template <>            struct hdf5_native_type<uint8_t>  { static hid_t copy() { return H5Tcopy(H5T_NATIVE_UINT8);  } };
template <>            struct hdf5_native_type<int16_t>  { static hid_t copy() { return H5Tcopy(H5T_NATIVE_INT16);  } };
template <>            struct hdf5_native_type<uint16_t> { static hid_t copy() { return H5Tcopy(H5T_NATIVE_UINT16); } };
template <>            struct hdf5_native_type<int32_t>  { static hid_t copy() { return H5Tcopy(H5T_NATIVE_INT32);  } };
template <>            struct hdf5_native_type<uint32_t> { static hid_t copy() { return H5Tcopy(H5T_NATIVE_UINT32); } };
template <>            struct hdf5_native_type<int64_t>  { static hid_t copy() { return H5Tcopy(H5T_NATIVE_INT64);  } };
template <>            struct hdf5_native_type<uint64_t> { static hid_t copy() { return H5Tcopy(H5T_NATIVE_UINT64); } };

template <typename T, size_t DIM> struct hdf5_native_array_type
{
    static hid_t copy()
    {
        hsize_t dim0       = DIM;
        hid_t   scalarType = hdf5_native_type<T>::copy();
        hid_t   arrayType  = H5Tarray_create(scalarType, 1, &dim0);
        H5Tclose(scalarType);
        return arrayType;
    }
};

template <typename T> struct hdf5_native_vlen_type
{
    static hid_t copy()
    {
        hid_t   scalarType = hdf5_native_type<T>::copy();
        hid_t   vlenType  = H5Tvlen_create(scalarType);
        H5Tclose(scalarType);
        return vlenType;
    }
};

////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_exception
// Default HDF5 error handling displays a call trace on stderr. We rely
// on that for information purposes, and just use this exception class
// for control flow.
class hdf5_exception : public std::exception //
{
public:
    virtual ~hdf5_exception() = default;
    virtual const char* what() const noexcept { return "HDF5 Error"; }
};

// clang-format off
//////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_object
class hdf5_object //
{
public:
    static const hid_t invalid_hid = -1;
    static bool id_is_valid(hid_t h) { return h >= 0; }
    bool        is_valid() const     { return id_is_valid(id_); }
    hid_t       id() const           { return id_; }
protected:
    void set_invalid() { id_ = invalid_hid; }
    hdf5_object(hid_t id = -1) : id_(id) {}
    hdf5_object(hdf5_object&& o) : id_(o.id_) { o.id_ = invalid_hid; }
    hdf5_object& operator=(hdf5_object&& o)
    {
        std::swap(id_, o.id_);
        return *this;
    }
private:
    hid_t  id_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_dataspace
class hdf5_dataspace : public hdf5_object //
{
public:
    hdf5_dataspace(hid_t id = invalid_hid) : hdf5_object(id) {}
    hdf5_dataspace(hdf5_dataspace&& d) : hdf5_object(std::forward<hdf5_object>(d)) {}
    ~hdf5_dataspace() { if(is_valid()) H5Sclose(id()); }
    int get_rank() const
    {
        int ndims = H5Sget_simple_extent_ndims(id());
        if(ndims < 0) throw std::runtime_error("Invalid dataspace (negative rank)");
        return ndims;
    }
    std::vector<hsize_t> get_dimensions() const
    {
        std::vector<hsize_t> dims(get_rank());
        if(H5Sget_simple_extent_dims(id(), dims.data(), nullptr) < 0)
        {
            throw std::runtime_error("Invalid dataspace (extents invalid)\n");
        }
        return dims;
    }
    hsize_t get_num_elements() const
    {
        std::vector<hsize_t> dims = get_dimensions();
        return std::accumulate(begin(dims), end(dims), 1, std::multiplies<hsize_t>());
    }
    void select_element(size_t         numDim,
                        const hsize_t* coord)
    {
        if(get_rank() != numDim)
        {
            throw std::runtime_error("Incorrect number of coordinates for dataspace select");
        }
        herr_t h = H5Sselect_elements(id(), H5S_SELECT_SET, 1, coord);
        //htri_t   is_valid = H5Sselect_valid(id());
        //hssize_t npoints  = H5Sget_select_npoints(id());
        if(h < 0)
        {
            throw std::runtime_error("H5Sselect_elements() error");
        }
    }
    static hdf5_dataspace create_simple(hsize_t dim0)
    {
        return hdf5_dataspace(H5Screate_simple(1, &dim0, nullptr));
    }
    static hdf5_dataspace create_simple(hsize_t dim0, hsize_t dim1)
    {
        hsize_t dims[2] = {dim0, dim1};
        return hdf5_dataspace(H5Screate_simple(2, dims, nullptr));
    }
    static hdf5_dataspace create_simple(hsize_t dim0, hsize_t dim1, hsize_t dim2)
    {
        hsize_t dims[3] = {dim0, dim1, dim2};
        return hdf5_dataspace(H5Screate_simple(3, dims, nullptr));
    }
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_datatype
class hdf5_datatype: public hdf5_object
{
public:
    hdf5_datatype(hid_t id = invalid_hid) : hdf5_object(id) {}
    hdf5_datatype(hdf5_datatype&& d) : hdf5_object(std::forward<hdf5_object>(d)) {}
    ~hdf5_datatype() { if(is_valid()) H5Tclose(id()); }
    H5T_class_t get_class() const                        { return H5Tget_class(id()); }
    const char* get_class_str() const
    {
        const char* s = "Unknown";
        switch(get_class())
        {
        case H5T_INTEGER:   s = "H5T_INTEGER";   break;
        case H5T_FLOAT:     s = "H5T_FLOAT";     break;
        case H5T_STRING:    s = "H5T_STRING";    break;
        case H5T_BITFIELD:  s = "H5T_BITFIELD";  break;
        case H5T_OPAQUE:    s = "H5T_OPAQUE";    break;
        case H5T_COMPOUND:  s = "H5T_COMPOUND";  break;
        case H5T_REFERENCE: s = "H5T_REFERENCE"; break;
        case H5T_ENUM:      s = "H5T_ENUM";      break;
        case H5T_VLEN:      s = "H5T_VLEN";      break;
        case H5T_ARRAY:     s = "H5T_ARRAY";     break;
        default:                                 break;
        }
        return s;
    }
    bool        is_compound() const                      { return (H5T_COMPOUND == get_class()); }
    bool        is_integer() const                       { return (H5T_INTEGER  == get_class()); }
    bool        is_float() const                         { return (H5T_FLOAT    == get_class()); }
    bool        is_signed() const                        { return (H5T_SGN_NONE != H5Tget_sign(id())); }
    bool        is_vlen() const                          { return (H5T_VLEN     == get_class()); }
    bool        is_array() const                         { return (H5T_ARRAY    == get_class()); }
    int         get_array_ndims() const                  { return H5Tget_array_ndims(id());      }
    void        get_array_dims(hsize_t dims[]) const
    {
        if(H5Tget_array_dims(id(), dims) < 0) throw std::runtime_error("H5Tget_array_dims() error");
    }
    size_t      get_size_bytes() const                   { return H5Tget_size(id()); }
    int         get_member_index(const char* name) const
    {
        int index = H5Tget_member_index(id(), name);
        if(index < 0)
        {
            throw std::runtime_error(std::string("Index not found for member name '") + name + std::string("'"));
        }
        return index;
    }
    std::string get_member_name(int idx) const
    {
        char* hdf5_str = H5Tget_member_name(id(), idx);
        std::string s(hdf5_str);
        H5free_memory(hdf5_str);
        return s;
    }
    const char* get_class_string() const
    {
        const char* c = "H5T_NO_CLASS";
        switch(get_class())
        {
        case H5T_INTEGER:   c = "H5T_INTEGER";   break;
        case H5T_FLOAT:     c = "H5T_FLOAT";     break;
        case H5T_STRING:    c = "H5T_STRING";    break;
        case H5T_BITFIELD:  c = "H5T_BITFIELD";  break;
        case H5T_OPAQUE:    c = "H5T_OPAQUE";    break;
        case H5T_COMPOUND:  c = "H5T_COMPOUND";  break;
        case H5T_REFERENCE: c = "H5T_REFERENCE"; break;
        case H5T_ENUM:      c = "H5T_ENUM";      break;
        case H5T_VLEN:      c = "H5T_VLEN";      break;
        case H5T_ARRAY:     c = "H5T_ARRAY";     break;
        default:                                 break;
        }
        return c;
    }
    // Create a compound type with a single member
    static hdf5_datatype create_compound(hid_t t, const char* name)
    {
        hid_t hComp = H5Tcreate(H5T_COMPOUND, H5Tget_size(t));
        if(!hdf5_object::id_is_valid(hComp))
        {
            throw std::runtime_error("Error creating compound type");
        }
        if(H5Tinsert(hComp, name, 0, t) < 0)
        {
            H5Tclose(hComp);
            throw std::runtime_error("Error inserting member into compound type");
        }
        return hdf5_datatype(hComp);
    }
};
// clang-format on

class hdf5_dataset;

////////////////////////////////////////////////////////////////////////
// hdf5_compound_element_reader
template <typename T>
struct hdf5_compound_element_reader
{
    static T read(hid_t dset, hid_t dspace, hid_t fileMemberType, const char* name)
    {
        T tDst{};
        // Create an in-memory compound datatype with a single member,
        // and the requested dataype
        hdf5_datatype memberType(hdf5_native_type<T>::copy());
        hdf5_datatype memCompoundType(hdf5_datatype::create_compound(memberType.id(), name));
        // Read into the local variable, converting to the requested,
        // in-memory destination type
        herr_t readStatus = H5Dread(dset,                                  // dataset
                                    memCompoundType.id(),                  // memory datatype
                                    hdf5_dataspace::create_simple(1).id(), // memory dataspace
                                    dspace,                                // file dataspace
                                    H5P_DEFAULT,                           // xfer property list
                                    &tDst);                                // destination memory
        if(readStatus < 0)
        {
            throw std::runtime_error("H5Dread() error for compound element");
        }
        return tDst;
    }
};

template <typename T, size_t DIM>
struct hdf5_compound_element_reader<std::array<T, DIM>>
{
    static std::array<T, DIM> read(hid_t dset, hid_t dspace, hid_t fileMemberType, const char* name)
    {
        hdf5_dataspace memDataspace = hdf5_dataspace::create_simple(1);
        // Allow reading of array types into a std::array<>
        if(H5T_ARRAY == H5Tget_class(fileMemberType))
        {
            std::array<T, DIM> tDst{};
            // Create an in-memory compound datatype with a single member,
            // and the requested dataype
            hdf5_datatype memberType(hdf5_native_array_type<T, DIM>::copy());
            hdf5_datatype memCompoundType(hdf5_datatype::create_compound(memberType.id(), name));
            // Read into the local variable, converting to the requested,
            // in-memory destination type
            herr_t readStatus = H5Dread(dset,                 // dataset
                                        memCompoundType.id(), // memory datatype
                                        memDataspace.id(),    // memory dataspace
                                        dspace,               // file dataspace
                                        H5P_DEFAULT,          // xfer property list
                                        tDst.data());         // destination memory
            if(readStatus < 0)
            {
                throw std::runtime_error("H5Dread() error for compound element");
            }
            return tDst;
        }
        // Allow reading of variable length array types into std::array, but
        // only if the size is correct.
        else if(H5T_VLEN == H5Tget_class(fileMemberType))
        {
            hvl_t          rdata{};
            // Create an in-memory compound datatype with a single member,
            // and the requested dataype
            hdf5_datatype  memberType(hdf5_native_vlen_type<T>::copy());
            hdf5_datatype  memCompoundType(hdf5_datatype::create_compound(memberType.id(), name));
            // Read variable length data. The HDF5 library will allocate
            // memory, and place the address in the hvl_t.p member. We
            // free the allocated memory with H5Dvlen_reclaim() below.
            herr_t readStatus = H5Dread(dset,                 // dataset
                                        memCompoundType.id(), // memory datatype
                                        memDataspace.id(),    // memory dataspace
                                        dspace,               // file dataspace
                                        H5P_DEFAULT,          // xfer property list
                                        &rdata);              // dest description (vlen)
            if(readStatus < 0)
            {
                throw std::runtime_error("H5Dread() error for compound element");
            }
            // Make sure that the size matches the caller's expected size
            if(DIM != rdata.len)
            {
                throw std::runtime_error("std::array<> size mismatch for compound VLEN member");
            }
            // Construct a vector with the data read
            std::array<T, DIM> tDst;
            std::copy(static_cast<T*>(rdata.p),              // input begin
                      static_cast<T*>(rdata.p) + rdata.len,  // input end
                      tDst.begin());                         // output begin
            // Release memory allocated by the HDF5 library
            H5Dvlen_reclaim(memberType.id(), memDataspace.id(), H5P_DEFAULT, &rdata);
            return tDst;
        }
        else
        {
            throw std::runtime_error("Invalid class for std::array<> read of H5T_COMPOUND member");
        }
    }
};

template <typename T>
struct hdf5_compound_element_reader<std::vector<T>>
{
    static std::vector<T> read(hid_t dset, hid_t dspace, hid_t fileMemberType, const char* name)
    {
        hdf5_dataspace memDataspace = hdf5_dataspace::create_simple(1);
        // Allow reading of variable length types into a std::vector<>
        if(H5T_VLEN == H5Tget_class(fileMemberType))
        {
            hvl_t          rdata{};
            // Create an in-memory compound datatype with a single member,
            // and the requested dataype
            hdf5_datatype  memberType(hdf5_native_vlen_type<T>::copy());
            hdf5_datatype  memCompoundType(hdf5_datatype::create_compound(memberType.id(), name));
            // Read variable length data. The HDF5 library will allocate
            // memory, and place the address in the hvl_t.p member. We
            // free the allocated memory with H5Dvlen_reclaim() below.
            herr_t readStatus = H5Dread(dset,                 // dataset
                                        memCompoundType.id(), // memory datatype
                                        memDataspace.id(),    // memory dataspace
                                        dspace,               // file dataspace
                                        H5P_DEFAULT,          // xfer property list
                                        &rdata);              // dest description (vlen)
            if(readStatus < 0)
            {
                throw std::runtime_error("H5Dread() error for compound element");
            }
            // Construct a vector with the data read
            std::vector<T> tDst(static_cast<T*>(rdata.p), static_cast<T*>(rdata.p) + rdata.len);
            // Release memory allocated by the HDF5 library
            H5Dvlen_reclaim(memberType.id(), memDataspace.id(), H5P_DEFAULT, &rdata);
            return tDst;
        }
        // Allow reading of array types into a std::vector<>
        else if(H5T_ARRAY == H5Tget_class(fileMemberType))
        {
            // Make sure that the array is rank 1 for vector output
            if(1 != H5Tget_array_ndims(fileMemberType))
            {
                throw std::runtime_error("std::vector<> read attempted for non-rank 1 H5T_COMPOUND member");
            }
            // Get the size of the array so that we can allocate space
            hsize_t dim0{};
            if((H5Tget_array_dims(fileMemberType, &dim0) < 0) || (dim0 == 0))
            {
                throw std::runtime_error("Invalid dimensions for H5T_COMPOUND member");
            }
            // Construct a vector with the appropriate size
            std::vector<T> tDst(dim0, T{});

            // Create an in-memory compound datatype with a single member,
            // and the requested dataype
            hdf5_datatype  baseType(hdf5_native_type<T>::copy());
            hdf5_datatype  memberType(H5Tarray_create(baseType.id(), 1, &dim0));
            hdf5_datatype  memCompoundType(hdf5_datatype::create_compound(memberType.id(), name));
            // Read variable length data. The HDF5 library will allocate
            // memory, and place the address in the hvl_t.p member. We
            // free the allocated memory with H5Dvlen_reclaim() below.
            herr_t readStatus = H5Dread(dset,                 // dataset
                                        memCompoundType.id(), // memory datatype
                                        memDataspace.id(),    // memory dataspace
                                        dspace,               // file dataspace
                                        H5P_DEFAULT,          // xfer property list
                                        tDst.data());         // output address
            if(readStatus < 0)
            {
                throw std::runtime_error("H5Dread() error for compound element");
            }
            return tDst;
        }
        else
        {
            // Support reading scalar types into a vector of length 1
            std::vector<T> tDst(1, hdf5_compound_element_reader<T>::read(dset,
                                                                         dspace,
                                                                         fileMemberType,
                                                                         name));
            return tDst;
        }
    }
};


// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_compound_member
// Read-only access to a member of an element in an HDF5 dataset with
// the H5T_COMPOUND type.
// Unlike many other classes in this file, this class does not simply
// wrap a native HDF5 object type. Instead, this class adds the ability
// to access structure elements
class hdf5_compound_member
{
public:
    hdf5_compound_member(hid_t dset, hid_t dspace, int mem_index) :
        dataset_(dset),
        dataspace_(dspace),
        index_(mem_index)
    {
        dataset_datatype_ = H5Dget_type(dset);
        if(dataset_datatype_ < 0)
        {
            throw std::runtime_error("H5Dget_type() error");
        }
        member_datatype_  = H5Tget_member_type(dataset_datatype_, mem_index);
        if(member_datatype_ < 0)
        {
            throw std::runtime_error("H5Dget_member_type() error");
        }
        H5Iinc_ref(dset);
        H5Iinc_ref(dspace);
    }
    hdf5_compound_member(hdf5_compound_member&& e) :
        dataset_(e.dataset_),
        dataspace_(e.dataspace_),
        dataset_datatype_(e.dataset_datatype_),
        member_datatype_(e.member_datatype_),
        index_(e.index_)
    {
        e.dataset_          = hdf5_object::invalid_hid;
        e.dataspace_        = hdf5_object::invalid_hid;
        e.dataset_datatype_ = hdf5_object::invalid_hid;
        e.member_datatype_  = hdf5_object::invalid_hid;
    }
    ~hdf5_compound_member()
    {
        if(hdf5_object::id_is_valid(dataset_))          { H5Idec_ref(dataset_);        }
        if(hdf5_object::id_is_valid(dataspace_))        { H5Idec_ref(dataspace_);      }
        if(hdf5_object::id_is_valid(dataset_datatype_)) { H5Tclose(dataset_datatype_); }
        if(hdf5_object::id_is_valid(member_datatype_))  { H5Tclose(member_datatype_);  }
    }
    std::string get_name() const
    {
        return hdf5_datatype(H5Dget_type(dataset_)).get_member_name(index_);
    }

    template <typename T>
    T as() const
    {
        return hdf5_compound_element_reader<T>::read(dataset_,
                                                     dataspace_,
                                                     member_datatype_,
                                                     get_name().c_str());
    }
private:
    hid_t dataset_;
    hid_t dataspace_;
    hid_t dataset_datatype_;
    hid_t member_datatype_;
    int   index_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_dataset_elem
// Read-only access to an element in an HDF5 dataset.
// Unlike many other classes in this file, this class does not simply
// wrap a native HDF5 object type. This class is used to simplify access
// to a single element in an HDF5 dataset. In particular, this class is
// expected to be useful to fetch values from HDF5 compound types (which
// are similar to C structures), which can be used to store
// configuration data, and which are sometimes stored in relatively
// small arrays. Accessing dataset elements using this class is expected
// to be simple, but not necessarily efficient.
class hdf5_dataset_elem
{
public:
    hdf5_dataset_elem(hid_t dset, hid_t elem_dspace) :
        dataset_(dset),
        element_dataspace_(elem_dspace)
    {
        H5Iinc_ref(dset);
        H5Iinc_ref(elem_dspace);
    }
    hdf5_dataset_elem(hdf5_dataset_elem&& e) :
        dataset_(e.dataset_),
        element_dataspace_(e.element_dataspace_)
    {
        e.dataset_           = hdf5_object::invalid_hid;
        e.element_dataspace_ = hdf5_object::invalid_hid;
    }
    ~hdf5_dataset_elem()
    {
        if(hdf5_object::id_is_valid(dataset_))           { H5Idec_ref(dataset_);           }
        if(hdf5_object::id_is_valid(element_dataspace_)) { H5Idec_ref(element_dataspace_); }
    }
    hdf5_compound_member operator[](const char* memberName) const
    {
        if(!hdf5_datatype(H5Dget_type(dataset_)).is_compound())
        {
            throw std::runtime_error("Attempted to retrieve compound member from non-compound type");
        }
        return hdf5_compound_member(dataset_,
                                    element_dataspace_,
                                    hdf5_datatype(H5Dget_type(dataset_)).get_member_index(memberName));
    }
private:
    hid_t dataset_;
    hid_t element_dataspace_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_dataset
class hdf5_dataset: public hdf5_object
{
public:
    hdf5_dataset(hid_t id = invalid_hid) : hdf5_object(id) {}
    hdf5_dataset(hdf5_dataset&& d) : hdf5_object(std::forward<hdf5_object>(d)) {}
    ~hdf5_dataset() { if(is_valid()) H5Dclose(id()); }
    hdf5_dataspace get_dataspace() const { return hdf5_dataspace(H5Dget_space(id())); }
    hdf5_datatype  get_datatype()  const { return hdf5_datatype(H5Dget_type(id()));   }
    size_t         get_buffer_size_bytes() const
    {
        return (get_dataspace().get_num_elements() * get_datatype().get_size_bytes());
    }
    hsize_t        get_num_elements() const { return get_dataspace().get_num_elements(); }
    void read(void* buffer)
    {
        herr_t h = H5Dread(id(),                // dataset ID
                           get_datatype().id(), // in-memory datatype
                           H5S_ALL,             // in-memory dataspace
                           H5S_ALL,             // in-file dataspace
                           H5P_DEFAULT,         // transfer property list
                           buffer);             // destination buffer
        if(h < 0) throw hdf5_exception();
    }
    hdf5_dataset_elem operator[](int idx)
    {
        hdf5_dataspace dspace   = get_dataspace();
        if(idx >= dspace.get_num_elements())
        {
            throw std::runtime_error("Dataset index out of bounds");
        }
        hsize_t        coord    = idx;
        dspace.select_element(1, &coord);

        return hdf5_dataset_elem(id(), dspace.id());
    }
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// hdf5hpp::hdf5_file
class hdf5_file : public hdf5_object
{
public:
    hdf5_file(hid_t id = invalid_hid) : hdf5_object(id) {}
    hdf5_file(hdf5_file&& f) : hdf5_object(std::forward<hdf5_object>(f)) {}
    ~hdf5_file() { if(is_valid()) H5Fclose(id()); }
    hdf5_file& operator=(hdf5_file&& f)
    {
        if(is_valid()) H5Fclose(id());
        set_invalid();
        hdf5_object::operator=(std::move(f));
        return *this;
    }

    bool is_valid_dataset(const char* name, hid_t dapl_id = H5P_DEFAULT)
    {
        return (H5Lexists(id(), name, dapl_id) > 0);
    }

    hdf5_dataset open_dataset(const char* name, hid_t dapl_id = H5P_DEFAULT)
    {
        hid_t h = H5Dopen(id(), name, dapl_id);
        if(!id_is_valid(h)){
            printf("H5Dopen(): Unable to open file %s\n",name);
            throw hdf5_exception();
        }
        return hdf5_dataset(h);
    }
    static hdf5_file create(const char* name, unsigned flags = H5F_ACC_TRUNC, hid_t cpl = H5P_DEFAULT, hid_t apl = H5P_DEFAULT)
    {
        hid_t f = H5Fcreate(name, flags, cpl, apl);
        if(!id_is_valid(f))
        {
            printf("H5Fcreate(): Unable to create file %s\n",name);
            throw hdf5_exception();
        }
        return hdf5_file(f);
    }
    static hdf5_file open(const char* name, unsigned flags = H5F_ACC_RDONLY, hid_t apl = H5P_DEFAULT)
    {
        hid_t f = H5Fopen(name, flags, apl);
        if(!id_is_valid(f))
        {
            printf("H5Fopen(): Unable to open file %s\n",name);
            throw hdf5_exception();
        }
        return hdf5_file(f);
    }
    void close()
    {
        if(is_valid()) H5Fclose(id());
        set_invalid();
    }
};
// clang-format on

} // namespace hdf5hpp

#endif // !defined(HDF5HPP_HPP_INCLUDED_)
