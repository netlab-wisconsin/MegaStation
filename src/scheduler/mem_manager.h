#pragma once

#include <readerwriterqueue.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "atomic_table.h"
#include "concurrentqueue.h"
#include "defs.h"
#include "matrix/cuphy_tensor.h"
#include "matrix/matrix.h"

namespace mega {

class MemTable : AtomicTable<void *> {
 public:
  MemTable(const uint32_t vframe_counts)
      : AtomicTable<void *>(vframe_counts * MemOpTypeProp::memops_per_frame()) {
  }

  inline void *load(const MemOpId &memop_id) const {
    const uint32_t offset = MemOpTypeProp::offset(memop_id);
    return AtomicTable<void *>::load(offset);
  }

  inline void *wait(const MemOpId &memop_id) const {
    const uint32_t offset = MemOpTypeProp::offset(memop_id);
    auto atomic_val = std::atomic_ref<void *const>(atomic_table.at(offset));
    return atomic_val.load(std::memory_order_relaxed);
  }

  inline void store(const MemOpId &memop_id, void *const &val) {
    const uint32_t offset = MemOpTypeProp::offset(memop_id);
    AtomicTable<void *>::store(offset, val);
  }

  inline void *exchange(const MemOpId &memop_id, void *const &val) {
    const uint32_t offset = MemOpTypeProp::offset(memop_id);
    auto atomic_val = std::atomic_ref<void *>(atomic_table.at(offset));
    return atomic_val.exchange(val, std::memory_order_release);
  }
};

class RefTable : AtomicTable<uint8_t> {
 public:
  RefTable(const uint32_t vframe_counts)
      : AtomicTable<uint8_t>(vframe_counts *
                             MemOpTypeProp::memops_per_frame()) {}

  inline void store(const MemOpId &memop_id, const uint8_t &val) {
    const uint32_t offset = MemOpTypeProp::offset(memop_id);
    AtomicTable<uint8_t>::store(offset, val);
  }

  inline uint8_t decrement(const MemOpId &memop_id) {
    const uint32_t offset = MemOpTypeProp::offset(memop_id);
    auto atomic_val = std::atomic_ref<uint8_t>(atomic_table.at(offset));
    return atomic_val.fetch_sub(1, std::memory_order_release) - 1;
  }

  inline uint8_t increment(const MemOpId &memop_id) {
    const uint32_t offset = MemOpTypeProp::offset(memop_id);
    auto atomic_val = std::atomic_ref<uint8_t>(atomic_table.at(offset));
    return atomic_val.fetch_add(1, std::memory_order_release) + 1;
  }
};

using MemWrapper = std::variant<Matrix, CuphyTensor>;

class MemManager {
 private:
  uint64_t available_frames = 0;  //!< Available Frames

  std::vector<std::shared_ptr<void>>
      all_memop_list;  //!< All Memory Operation (To avoid memory leak)
  std::vector<moodycamel::ConcurrentQueue<void *>>
      free_memop_list;  //!< MemOpType -> Free MemOp Queue

  MemTable mem_table;  //!< Mem Table: memop_id -> MemOp Ptr

  static std::unique_ptr<RefTable>
      ref_table;  //!< Ref Table: memop_id -> Ref Count

  inline moodycamel::ConcurrentQueue<void *> &get_free_list(
      const MemOpType &memop_type) {
    return free_memop_list[static_cast<uint32_t>(memop_type) -
                           static_cast<uint32_t>(MemOpType::kPreFFT)];
  }

  static std::array<MemWrapper, MemOpTypeProp::kNumMemType>
      mem_warp;  //!< Matrix / Tensor Wrapper (mem type -> wrapper)

 public:
  static void init(const uint32_t vframe_counts);

  MemManager(const uint32_t vframe_counts);

  inline void *get_memop(const MemOpId &memop_id) {
    if (memop_id.type_id == MemOpType::kNullMO) return nullptr;

    void *mem_op_ptr;
    if (get_free_list(memop_id.type_id).try_dequeue(mem_op_ptr) == false)
      return nullptr;

    mem_table.store(memop_id, mem_op_ptr);
    return mem_op_ptr;
  }

  inline void *query_memop(const MemOpId &memop_id) {
    if (memop_id.type_id == MemOpType::kNullMO) return nullptr;

    return mem_table.load(memop_id);
  }

  inline void *wait_memop(const MemOpId &memop_id) {
    if (memop_id.type_id == MemOpType::kNullMO) return nullptr;

    return mem_table.wait(memop_id);
  }

  static inline void store_ref(const MemOpId &memop_id) {
    ref_table->store(memop_id, MemOpTypeProp::mem_refcount(memop_id.type_id));
  }

  static inline uint8_t dec_ref(const MemOpId &memop_id) {
    return ref_table->decrement(memop_id);
  }

  inline void free_memop(const MemOpId &memop_id) {
    void *mem_op_ptr = mem_table.exchange(memop_id, nullptr);
    if (mem_op_ptr == nullptr) return;

    get_free_list(memop_id.type_id).enqueue(mem_op_ptr);
  }

  static inline MemWrapper &get_mem_wrapper(const MemOpType &memop_type) {
    return mem_warp[static_cast<uint32_t>(memop_type) -
                    static_cast<uint32_t>(MemOpType::kPreFFT)];
  }

  inline uint64_t size() const { return available_frames; }
};

}  // namespace mega