/**
 * @file matrix.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief The file contains basic matrix buffer used in the mega station.
 * @version 0.1
 * @date 2023-11-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>

namespace mega {

/**
 * @brief The Matrix class is a generic representation of a multi-dimensional
 * array.
 *
 */
class Matrix {
  /**
   * @brief Now only support up to 3 dimensions
   * nCols = \p dims[0], nRows = \p dims[1], nDepths = \p dims[2]
   * Row-major order by default, i.e. \p strides[0] = 1, \p strides[1] = nCols,
   * \p strides[2] = nCols * nRows
   */
 public:
  /**
   * @brief Flags for allocate memory on host or device
   *
   */
  enum alloc_type_t : uint8_t {
    kNull,   //!< empty matrix (for default constructor)
    kDesc,   //!< don't allocate memory, only create descriptor
    kRef,    //!< don't operate memory (free), only create reference
    kHost,   //!< allocate memory on host
    kHPin,   //!< allocate pinned memory on host
    kDevice  //!< allocate memory on device
  };

 protected:
  static constexpr uint8_t kDimMax = 3;  //!< maximum number of dimensions

  uint8_t nDims;                          //!< number of dimensions
  std::array<uint32_t, kDimMax> dims;     //!< dimensions
  std::array<uint32_t, kDimMax> strides;  //!< strides
  std::array<uint64_t, kDimMax + 1>
      sz_bytes;  //!< size in bytes of each stride, nDims-th
                 //!< is the size in bytes of the entire array

  alloc_type_t alloc;          //!< whether to allocate memory on host or device
  std::shared_ptr<void> data;  //!< pointer to the data (host/device)

 private:
  /**
   * @brief internal function to return the dimension of the given index for
   * constructor
   *
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   */
  inline void init_dim(const uint8_t nDims_, const uint32_t *const dims_) {
    if (nDims_ > kDimMax) {
      throw std::invalid_argument(
          "Number of dimensions exceeds maximum or below minimum");
    }

    for (uint8_t idx = 0; idx < nDims_; idx++) {
      dims[idx] = dims_[idx];
    }

    for (uint8_t idx = nDims_; idx < kDimMax; idx++) {
      dims[idx] = 0;
    }
  }

  /**
   * @brief internal function to return the stride of the given index for
   * constructor
   *
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   * @param strides_ strides
   */
  inline void init_stride(const uint8_t nDims_, const uint32_t *const dims_,
                          const uint32_t *const strides_) {
    if (nDims_ > kDimMax) {
      throw std::invalid_argument(
          "Number of dimensions exceeds maximum or below minimum");
    }

    if (strides_ == nullptr) {
      uint32_t stride = 1;
      for (uint8_t idx = 0; idx < nDims_; ++idx) {
        strides[idx] = stride;
        stride *= dims_[idx];
      }
    } else {
      for (uint8_t idx = 0; idx < nDims_; ++idx) {
        strides[idx] = strides_[idx];
      }
    }

    for (uint8_t idx = nDims_; idx < kDimMax; idx++) {
      strides[idx] = 0;
    }
  }

  /**
   * @brief internal function to return the size in bytes of the given index for
   * constructor
   *
   * @param sz_elem_ size of each element in bytes
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   * @param strides_ strides
   * @param sz_bytes_ size in bytes
   */
  inline void init_szBytes(const uint8_t sz_elem_, const uint8_t nDims_,
                           const uint32_t *const dims_,
                           const uint32_t *const strides_,
                           const uint64_t *const sz_bytes_) {
    if (nDims_ > kDimMax) {
      throw std::invalid_argument(
          "Number of dimensions exceeds maximum or below minimum");
    }

    if (sz_bytes_ == nullptr) {
      for (uint8_t idx = 0; idx < nDims_; ++idx) {
        sz_bytes[idx] = sz_elem_ * strides_[idx];
      }
      sz_bytes[nDims_] = sz_elem_ * strides_[nDims_ - 1] * dims_[nDims_ - 1];
    } else {
      for (uint8_t idx = 0; idx <= nDims_; ++idx) {
        sz_bytes[idx] = sz_bytes_[idx];
      }
    }

    for (uint8_t idx = nDims_ + 1; idx <= kDimMax; idx++) {
      sz_bytes[idx] = 0;
    }
  }

 protected:
  /**
   * @brief
   *
   * @param alloc_
   * @param data_
   * @param size_
   */
  inline void alloc_data(const alloc_type_t alloc_,
                         const std::shared_ptr<void> data_,
                         const uint64_t size_) {
    alloc = alloc_;
    if (data_ == nullptr) {
      void *data_ptr;
      if (alloc_ == kHost) {
        data_ptr = std::malloc(size_);
        data = std::shared_ptr<void>(data_ptr, std::free);
        goto check;
      } else if (alloc_ == kHPin) {
        cudaMallocHost(&data_ptr, size_);
        data = std::shared_ptr<void>(data_ptr, cudaFreeHost);
        goto check;
      } else if (alloc_ == kDevice) {
        cudaMalloc(&data_ptr, size_);
        data = std::shared_ptr<void>(data_ptr, cudaFree);
        goto check;
      } else {
        data = nullptr;
      }
    } else {
      data = data_;
    }
    return;
  check:
    if (data == nullptr) {
      throw std::bad_alloc();
    }
  }

  /**
   * @brief Construct a new Matrix object without computation of strides and
   * size (for internal use)
   *
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   * @param strides_ strides
   * @param sz_bytes_ size in bytes
   * @param alloc_ whether to allocate memory on host or device
   * @param data_ pointer to the data, if nullptr, allocate new memory
   */
  Matrix(const uint8_t nDims_, const uint32_t *const dims_,
         const uint32_t *const strides_, const uint64_t *const sz_bytes_,
         const alloc_type_t alloc_ = kDesc,
         const std::shared_ptr<void> &data_ = nullptr)
      : nDims(nDims_) {
    init_dim(nDims_, dims_);
    init_stride(nDims_, dims_, strides_);
    init_szBytes(0, nDims_, dims_, strides_, sz_bytes_);
    alloc_data(alloc_, data_, sz_bytes[nDims]);
  }

 public:
  /**
   * @brief Default construction of a new Matrix object
   *
   */
  Matrix() = default;
  /**
   * @brief Default destruction of a Matrix object
   *
   */
  ~Matrix() = default;

  /**
   * @brief Construct a new Matrix object
   *
   * @param sz_elem_ size of each element in bytes
   * @param nDims_ number of dimensions
   * @param dims_ dimensions
   * @param strides_ strides
   * @param alloc_ whether to allocate memory on host or device
   * @param data_ pointer to the data, if nullptr, allocate new memory
   */
  Matrix(const uint8_t sz_elem_, const uint8_t nDims_, const uint32_t *dims_,
         const uint32_t *const strides_ = nullptr,
         const alloc_type_t alloc_ = kDesc,
         const std::shared_ptr<void> &data_ = nullptr)
      : nDims(nDims_) {
    init_dim(nDims_, dims_);
    init_stride(nDims_, dims.data(), strides_);
    init_szBytes(sz_elem_, nDims_, dims.data(), strides.data(), nullptr);
    alloc_data(alloc_, data_, sz_bytes[nDims]);
  }

  /**
   * @brief Construct a new Matrix object of 1 dimension vector
   *
   * @param sz_elem_ size of each element in bytes
   * @param nVec_ length of the vector
   * @param alloc_ whether to allocate memory on host or device
   * @param data_ pointer to the data, if nullptr, allocate new memory
   */
  Matrix(const uint8_t sz_elem_, const uint32_t nVec_,
         const alloc_type_t alloc_ = kDesc,
         const std::shared_ptr<void> &data_ = nullptr)
      : Matrix(sz_elem_, 1, &nVec_, nullptr, alloc_, data_) {}

  /**
   * @brief Construct a new Matrix object of 2 dimension array
   *
   * @param sz_elem_ size of each element in bytes
   * @param nCols_ number of columns
   * @param nRows_ number of rows
   * @param alloc_ whether to allocate memory on host or device
   * @param data_ pointer to the data, if nullptr, allocate new memory
   */
  Matrix(const uint8_t sz_elem_, const uint32_t nCols_, const uint32_t nRows_,
         const alloc_type_t alloc_ = kDesc,
         const std::shared_ptr<void> &data_ = nullptr)
      : Matrix(sz_elem_, 2, std::array<uint32_t, 2>{nCols_, nRows_}.data(),
               nullptr, alloc_, data_) {}

  /**
   * @brief Construct a new Matrix object of 3 dimension cube
   *
   * @param sz_elem_ size of each element in bytes
   * @param nCols_ number of columns
   * @param nRows_ number of rows
   * @param nDepths_ number of depths
   * @param alloc_ whether to allocate memory on host or device
   * @param data_ pointer to the data, if nullptr, allocate new memory
   */
  Matrix(const uint8_t sz_elem_, const uint32_t nCols_, const uint32_t nRows_,
         const uint32_t nDepths_, const alloc_type_t alloc_ = kDesc,
         const std::shared_ptr<void> &data_ = nullptr)
      : Matrix(sz_elem_, 3,
               std::array<uint32_t, 3>{nCols_, nRows_, nDepths_}.data(),
               nullptr, alloc_, data_) {}

  /**
   * @brief Get Data Pointer of a certain type
   *
   * @tparam T Type of the data
   */
  template <typename T = void>
  inline T *ptr() const {
    return reinterpret_cast<T *>(data.get());
  }

  /**
   * @brief Get Data of a certain type at a certain index
   *
   * @tparam T Type of the data
   */
  template <typename T>
  inline T &deref(const uint64_t idx) const {
    return this->ptr<T>()[idx];
  }

  /**
   * @brief Get the number of dimensions
   *
   * @return uint8_t
   */
  inline uint8_t nDim() const { return nDims; }

  /**
   * @brief Get the size of the given dimension
   *
   * @param idx index of the dimension
   * @return uint32_t
   */
  inline uint32_t dim(const uint8_t idx) const {
    if (idx >= kDimMax) {
      throw std::out_of_range("Index out of range");
    }
    return dims[idx];
  }

  /**
   * @brief Get the stride of the given dimension
   *
   * @param idx index of the dimension
   * @return uint32_t
   */
  inline uint32_t stride(const uint8_t idx) const {
    if (idx >= kDimMax) {
      throw std::out_of_range("Index out of range");
    }
    return strides[idx];
  }

  /**
   * @brief Get the size in bytes of the given dimension
   *
   * @param idx index of the dimension
   * @return uint64_t
   */
  inline uint64_t szBytes(const uint8_t idx) const {
    if (idx > kDimMax) {
      throw std::out_of_range("Index out of range");
    }
    return sz_bytes[idx];
  }

  /**
   * @brief Get the size in bytes of the entire array
   *
   * @return uint64_t
   */
  inline uint64_t szBytes() const { return sz_bytes[nDims]; }

  /**
   * @brief Get the allocation type
   *
   * @return alloc_type_t
   */
  inline alloc_type_t allocType() const { return alloc; }

  /**
   * @brief Get the shared pointer to the data
   *
   * @return std::shared_ptr<void>
   */
  inline std::shared_ptr<void> raw() const { return data; }

  /**
   * @brief Access the element at the given index
   *
   * @param idx index of the element
   * @return Matrix A sub-matrix of the given index
   */
  inline Matrix operator[](const uint32_t idx) const {
    if (nDims == 0 || idx >= dims[nDims - 1]) {
      throw std::out_of_range("Index out of range");
    }
    uint8_t *access_data = this->ptr<uint8_t>() + idx * sz_bytes[nDims - 1];
    return Matrix(nDims - 1, dims.data(), strides.data(), sz_bytes.data(), kRef,
                  std::shared_ptr<void>(data, access_data));
  }

  /**
   * @brief Wrap the given data pointer to a matrix
   *
   * @param data_ pointer to the data
   * @return Matrix A matrix wrapping the given data pointer
   */
  inline Matrix wrap(void *ptr) const {
    std::shared_ptr<void> data_ =
        std::shared_ptr<void>(ptr, [](void *) -> void {});
    return Matrix(nDims, dims.data(), strides.data(), sz_bytes.data(), kRef,
                  data_);
  }

  /**
   * @brief Get the sub-matrix of given start and end index
   * [start, end)
   *
   * @param start start index
   * @param end end index
   * @return Matrix Sub-matrix
   */
  inline Matrix sub(const uint32_t start, const uint32_t end) const {
    if (nDims == 0 || start >= end || end > dims[nDims - 1]) {
      throw std::out_of_range("Index out of range");
    }
    uint8_t *access_data = this->ptr<uint8_t>() + start * sz_bytes[nDims - 1];
    auto dims_ = dims;
    dims_[nDims - 1] = end - start;
    return Matrix(sz_bytes[0], nDims, dims_.data(), nullptr, kRef,
                  std::shared_ptr<void>(data, access_data));
  }
};  // class Matrix

}  // namespace mega