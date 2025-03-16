#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>
#include <gdrconfig.h>

#define PAGE_ROUND_UP(x, n) (((x) + ((n) - 1)) & ~((n) - 1))

namespace mega {

struct GDRMemHandle {
  CUdeviceptr ptr;            //!< aligned ptr
  CUdeviceptr unaligned_ptr;  //!< unaligned ptr
  size_t size;
  size_t allocated_size;

  CUresult alloc(const size_t size_, bool aligned_mapping,
                 bool set_sync_memops);  // allocate memory
  CUresult free();                       // free memory
};

struct GDRInfo {
  static constexpr uint64_t kGpuPageSize = GPU_PAGE_SIZE;

  gdr_t gdr;                //!< GDR Handle
  gdr_mh_t gdr_handle;      //!< GDR Memory Handle
  gdr_info_t gdr_info;      //!< GDR Info
  GDRMemHandle mem_handle;  //!< GPU Memory Handle
  char *mapped_ptr;         //!< Mapped Pointer

  void alloc_map(void *&host_map,
                 const size_t size = kGpuPageSize);  // allocate and pin memory
  void alloc_map(const size_t size = kGpuPageSize);  // allocate and pin memory
  void unmap_free(void *&host_map);                  // unmap and free memory
  void unmap_free();                                 // unmap and free memory
  inline int32_t *ptr(const uint64_t &offset = 0) const {
    return reinterpret_cast<int32_t *>(gdr_info.va + offset);
  }
  inline int32_t *offset(const uint64_t &offset = 0) const {
    return reinterpret_cast<int32_t *>(gdr_info.va) + offset;
  }
  inline char *offset(const int32_t *ptr) const {
    return mapped_ptr + (reinterpret_cast<uint64_t>(ptr) - mem_handle.ptr);
  }
};

}  // namespace mega