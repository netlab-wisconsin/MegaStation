#include "gdr_utils.h"

#include <cstring>
#include <stdexcept>

using namespace mega;

CUresult GDRMemHandle::alloc(const size_t size_, bool aligned_mapping,
                             bool set_sync_memops) {
  CUresult ret = CUDA_SUCCESS;
  CUdeviceptr ptr_, out_ptr;
  size_t allocated_size_;

  if (aligned_mapping)
    allocated_size_ = size_ + GPU_PAGE_SIZE - 1;
  else
    allocated_size_ = size_;

  ret = cuMemAlloc(&ptr_, allocated_size_);
  if (ret != CUDA_SUCCESS) return ret;
  cuMemsetD8(ptr_, 0, allocated_size_);

  if (set_sync_memops) {
    unsigned int flag = 1;
    ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr_);
    if (ret != CUDA_SUCCESS) {
      cuMemFree(ptr_);
      return ret;
    }
  }

  if (aligned_mapping)
    out_ptr = PAGE_ROUND_UP(ptr_, GPU_PAGE_SIZE);
  else
    out_ptr = ptr_;

  ptr = out_ptr;
  unaligned_ptr = ptr_;
  size = size_;
  allocated_size = allocated_size_;

  return CUDA_SUCCESS;
}

CUresult GDRMemHandle::free() {
  CUresult ret = CUDA_SUCCESS;
  CUdeviceptr ptr;

  ret = cuMemFree(unaligned_ptr);
  if (ret == CUDA_SUCCESS) memset(this, 0, sizeof(GDRMemHandle));

  return ret;
}

void GDRInfo::alloc_map(const size_t size) {
  CUresult status = this->mem_handle.alloc(size, true, true);
  this->gdr = gdr_open();
  if (status != CUDA_SUCCESS || this->gdr == nullptr) {
    throw std::runtime_error("GDR Open Failed, Check whether mod in installed");
  }
  gdr_pin_buffer(this->gdr, this->mem_handle.ptr, size, 0, 0,
                 &this->gdr_handle);
  gdr_map(this->gdr, this->gdr_handle,
          reinterpret_cast<void **>(&this->mapped_ptr), size);
  gdr_get_info(this->gdr, this->gdr_handle, &this->gdr_info);
}

void GDRInfo::alloc_map(void *&host_map, const size_t size) {
  this->alloc_map(size);

  cudaMallocHost(&host_map, size);
  memset(host_map, 0, size);
}

void GDRInfo::unmap_free() {
  gdr_unmap(this->gdr, this->gdr_handle, this->mapped_ptr,
            this->mem_handle.size);
  gdr_unpin_buffer(this->gdr, this->gdr_handle);
  gdr_close(this->gdr);

  this->mem_handle.free();
  memset(this, 0, sizeof(GDRInfo));
}

void GDRInfo::unmap_free(void *&host_map) {
  cudaFreeHost(host_map);
  this->unmap_free();
}