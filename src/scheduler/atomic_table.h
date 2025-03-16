#pragma once

#include <atomic>
#include <vector>

namespace mega {

template <typename T>
class AtomicTable {
 protected:
  std::vector<T> atomic_table;  // task_id -> device_id

 public:
  AtomicTable(const uint32_t size) : atomic_table(size) {}

  inline T load(const uint32_t index) const {
    auto atomic_val = std::atomic_ref<const T>(atomic_table.at(index));
    return atomic_val.load(std::memory_order_consume);
  }

  inline void store(const uint32_t index, const T &val) {
    auto atomic_val = std::atomic_ref<T>(atomic_table.at(index));
    atomic_val.store(val, std::memory_order_release);
  }
};

}  // namespace mega
