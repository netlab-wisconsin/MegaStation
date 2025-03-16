/**
 * @file cuphy_tensor.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief The file contains basic cuphy tensor buffer used in the mega station.
 * It is a wrapper of cuphyTensorDescriptor_t.
 * @version 0.1
 * @date 2023-11-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuphy.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "matrix.h"

namespace mega {

/**
 * @brief Inherit from Matrix for LDPC tensors of cuphy
 *
 */
class CuphyTensor : public Matrix {
 public:
  /**
   * @brief Data type for cuphy tensors
   *
   */
  enum cuphy_dtype_t : uint8_t {
    kBit = CUPHY_BIT,      //!< bit
    kHalf = CUPHY_R_16F,   //!< float16
    kFloat = CUPHY_R_32F,  //!< float32
  };

  /**
   * @brief Alignment flags for cuphy tensors
   *
   */
  enum cuphy_align_t : uint8_t {
    kDefault = CUPHY_TENSOR_ALIGN_DEFAULT,   //!< default alignment
    kTight = CUPHY_TENSOR_ALIGN_TIGHT,       //!< tight alignment
    kCoalesce = CUPHY_TENSOR_ALIGN_COALESCE  //!< coalesce alignment
  };

 private:
  using shared_chuphy_tensor = std::shared_ptr<cuphyTensorDescriptor>;

  shared_chuphy_tensor desc;  //!< cuphy tensor descriptor
  cuphy_dtype_t dtype;        //!< cuphy data type
  cuphy_align_t align;        //!< cuphy alignment flags

 private:
  /**
   * @brief Initialize a cuphy tensor descriptor
   *
   * @param desc_ptr_ cuphy tensor descriptor (pointer)
   * @param dtype_ cuphy data type
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   * @param strides_ strides
   * @param align_ alignment flags
   */
  inline void init_tensorDesc(const cuphyDataType_t dtype_, const int nDims_,
                              const int *dims_, const int *strides_,
                              const unsigned int align_) {
    cuphyTensorDescriptor_t desc_;
    cuphyCreateTensorDescriptor(&desc_);
    cuphySetTensorDescriptor(desc_, dtype_, nDims_, dims_, strides_, align_);
    cuphyGetTensorDescriptor(
        desc_, nDims_, reinterpret_cast<cuphyDataType_t *>(&dtype),
        reinterpret_cast<int *>(&nDims), reinterpret_cast<int *>(dims.data()),
        reinterpret_cast<int *>(strides.data()));
    cuphyGetTensorSizeInBytes(desc_, &sz_bytes[nDims_]);

    align = static_cast<cuphy_align_t>(align_);
    desc = shared_chuphy_tensor(desc_, cuphyDestroyTensorDescriptor);

    size_t sz_elem_bit =
        (sz_bytes[nDims_] * 8) / (strides[nDims_ - 1] * dims[nDims_ - 1]);
    for (uint8_t idx = 0; idx < nDims_; ++idx) {
      sz_bytes[idx] = (strides[idx] * sz_elem_bit) / 8;
    }

    for (uint8_t idx = nDims_; idx < kDimMax; ++idx) {
      dims[idx] = 0;
      strides[idx] = 0;
    }

    for (uint8_t idx = nDims_ + 1; idx <= kDimMax; ++idx) {
      sz_bytes[idx] = 0;
    }
  }

  /**
   * @brief Copy a cuphy tensor descriptor
   *
   * @param dtype_
   * @param nDims_
   * @param dims_
   * @param strides_
   * @param align_
   * @return cuphyTensorDescriptor_t
   */
  static inline shared_chuphy_tensor copy_tensorDesc(
      const cuphy_dtype_t dtype_, const int nDims_, const int *dims_,
      const int *strides_, const unsigned int align_) {
    cuphyTensorDescriptor_t desc_;
    cuphyCreateTensorDescriptor(&desc_);
    cuphySetTensorDescriptor(desc_, static_cast<cuphyDataType_t>(dtype_),
                             nDims_, dims_, strides_, align_);
    return shared_chuphy_tensor(desc_, cuphyDestroyTensorDescriptor);
  }

 protected:
  /**
   * @brief Construct a new Matrix object by direct assignment (internal use)
   *
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   * @param strides_ strides
   * @param sz_bytes_ size in bytes
   * @param alloc_ whether to allocate memory on host or device
   * @param data_ data pointer on host or device
   */
  CuphyTensor(const cuphy_dtype_t dtype_, const uint8_t nDims_,
              const uint32_t *const dims_, const uint32_t *const strides_,
              const uint64_t *const sz_bytes_, const alloc_type_t alloc_,
              const cuphy_align_t align_, const std::shared_ptr<void> &data_)
      : Matrix(nDims_, dims_, strides_, sz_bytes_, alloc_, data_),
        desc(copy_tensorDesc(dtype_, nDims_,
                             reinterpret_cast<const int *>(dims_),
                             reinterpret_cast<const int *>(strides_), align_)),
        dtype(dtype_),
        align(align_) {}

 public:
  /**
   * @brief Default construction of a new CuphyTensor object
   *
   */
  CuphyTensor() = default;
  /**
   * @brief Default destruction of a CuphyTensor object
   *
   */
  ~CuphyTensor() = default;

  /**
   * @brief Construct a new CuphyTensor object
   *
   * @param dtype_ cuphy data type
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   * @param alloc_ whether to allocate memory on host or device
   * @param align_ alignment flags
   * @param data_ data pointer on host or device
   */
  CuphyTensor(const cuphy_dtype_t dtype_, const uint8_t nDims_,
              const uint32_t *const dims_,
              const uint32_t *const strides_ = nullptr,
              const alloc_type_t alloc_ = kDesc,
              const cuphy_align_t align_ = kDefault,
              const std::shared_ptr<void> &data_ = nullptr) {
    init_tensorDesc(static_cast<cuphyDataType_t>(dtype_), nDims_,
                    reinterpret_cast<const int *>(dims_),
                    reinterpret_cast<const int *>(strides_), align_);
    alloc_data(alloc_, data_, sz_bytes[nDims]);
  }

  /**
   * @brief Construct a new CuphyTensor object of 1 dimension vector
   *
   * @param dtype_ cuphy data type
   * @param nVec_ length of the vector
   * @param alloc_ whether to allocate memory on host or device
   * @param align_ alignment flags
   * @param data_ data pointer on host or device
   */
  CuphyTensor(cuphy_dtype_t dtype_, const uint32_t nVec_,
              const alloc_type_t alloc_ = kDesc,
              const cuphy_align_t align_ = kDefault,
              const std::shared_ptr<void> &data_ = nullptr)
      : CuphyTensor(dtype_, 1, &nVec_, nullptr, alloc_, align_, data_) {}

  /**
   * @brief Construct a new CuphyTensor object of 2 dimension array
   *
   * @param dtype_ cuphy data type
   * @param nRows_ number of rows
   * @param nCols_ number of columns
   * @param alloc_ whether to allocate memory on host or device
   * @param align_ alignment flags
   * @param data_ data pointer on host or device
   */
  CuphyTensor(cuphy_dtype_t dtype_, const uint32_t nCols_,
              const uint32_t nRows_, const alloc_type_t alloc_ = kDesc,
              const cuphy_align_t align_ = kDefault,
              const std::shared_ptr<void> &data_ = nullptr)
      : CuphyTensor(dtype_, 2, std::array<uint32_t, 2>{nCols_, nRows_}.data(),
                    nullptr, alloc_, align_, data_) {}

  /**
   * @brief Construct a new CuphyTensor object of 3 dimension cube
   *
   * @param dtype_ cuphy data type
   * @param nRows_ number of rows
   * @param nCols_ number of columns
   * @param nDepths_ number of depths
   * @param alloc_ whether to allocate memory on host or device
   * @param align_ alignment flags
   * @param data_ data pointer on host or device
   */
  CuphyTensor(cuphy_dtype_t dtype_, const uint32_t nCols_,
              const uint32_t nRows_, const uint32_t nDepths_,
              const alloc_type_t alloc_ = kDesc,
              const cuphy_align_t align_ = kDefault,
              const std::shared_ptr<void> &data_ = nullptr)
      : CuphyTensor(dtype_, 3,
                    std::array<uint32_t, 3>{nCols_, nRows_, nDepths_}.data(),
                    nullptr, alloc_, align_, data_) {}

  /**
   * @brief Get the desc object
   *
   * @return cuphyTensorDescriptor_t
   */
  inline cuphyTensorDescriptor_t get_desc() const { return desc.get(); }

  /**
   * @brief Get the data type of the cuphy tensor
   *
   * @return cuphy_dtype_t
   */
  inline cuphy_dtype_t dType() const { return dtype; }

  /**
   * @brief Get the alignment flags of the cuphy tensor
   *
   * @return cuphy_align_t
   */
  inline cuphy_align_t alignment() const { return align; }

  /**
   * @brief Access the element at the given index
   *
   * @param idx index of the element
   */
  inline CuphyTensor operator[](const uint32_t idx) const {
    if (nDims == 0 || idx >= dims[nDims - 1]) {
      throw std::out_of_range("Index out of range");
    }

    if (sz_bytes[nDims - 1] == 0) {
      throw std::underflow_error("Size of element is less than a byte");
    }
    uint8_t *access_data = this->ptr<uint8_t>() + idx * sz_bytes[nDims - 1];
    return CuphyTensor(dtype, nDims - 1, dims.data(), strides.data(),
                       sz_bytes.data(), kRef, align,
                       std::shared_ptr<void>(data, access_data));
  }

  /**
   * @brief Wrap the data pointer with a new cuphy tensor
   *
   * @param data_ data pointer on host or device
   * @return CuphyTensor new cuphy tensor with the given data pointer
   */
  inline CuphyTensor wrap(void *ptr) const {
    std::shared_ptr<void> data_ =
        std::shared_ptr<void>(ptr, [](void *) -> void {});
    return CuphyTensor(dtype, nDims, dims.data(), strides.data(),
                       sz_bytes.data(), kRef, align, data_);
  }
};  // class CuphyTensor

}  // namespace mega