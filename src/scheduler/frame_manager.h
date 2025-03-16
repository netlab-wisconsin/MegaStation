#pragma once

#include <concurrentqueue.h>
#include <parallel_hashmap/phmap.h>

#include <vector>

#include "atomic_table.h"
#include "defs.h"
#include "gdr_utils.h"
#include "notif_table.h"

namespace mega {

class CountTable : AtomicTable<uint32_t> {
 public:
  CountTable(const uint32_t vframe_counts)
      : AtomicTable<uint32_t>(vframe_counts) {}

  inline void store(const FrameId &frame_id) {
    AtomicTable<uint32_t>::store(frame_id.frm_id,
                                 TaskTypeProp::jobs_per_frame());
  }

  inline uint32_t decrement(const FrameId &frame_id) {
    auto atomic_val =
        std::atomic_ref<uint32_t>(atomic_table.at(frame_id.frm_id));
    return atomic_val.fetch_sub(1, std::memory_order_release) - 1;
  }
};

class FrameTable {
 public:
  static constexpr FrameId NFrameId = {static_cast<uint32_t>(~0)};

 private:
  const uint32_t
      frame_bytes;  //!< Frame Size in bytes in each notification array
  const uint32_t frame_counts;  //!< Number of frames in the notification table

  phmap::flat_hash_map<FrameId, FrameId>
      frame_table;  //!< Frame Table, phy_frame_id -> vframe_id
  std::vector<FrameId>
      reverse_frame_table;  //!< Reverse Frame Table, vframe_id -> phy_frame_id

  CountTable count_table;  //!< Count Table for each vframe

  moodycamel::ConcurrentQueue<FrameId> free_vframes;  //!< Free Virtual Frames
  moodycamel::ConsumerToken ctok;  //!< Consumer Token for free_vframes

 public:
  FrameTable()
      : frame_bytes(TaskTypeProp::jobs_per_frame() * sizeof(int32_t)),
        frame_counts(GDRInfo::kGpuPageSize / frame_bytes),
        frame_table(frame_counts),
        reverse_frame_table(frame_counts, NFrameId),
        count_table(frame_counts),
        free_vframes(frame_counts),
        ctok(free_vframes) {
    for (uint32_t i = 0; i < frame_counts; i++) {
      free_vframes.enqueue({i});
    }
  }

  inline FrameId register_frame(const FrameId &phy_frame_id) {
    // if (frame_table.contains(phy_frame_id)) [[unlikely]]
    //   return get_v(phy_frame_id);

    FrameId vfrm_id;
    if (free_vframes.try_dequeue(ctok, vfrm_id) == false) return NFrameId;

    FrameId &old_phy_frame_id = reverse_frame_table.at(vfrm_id.frm_id);
    frame_table.erase(old_phy_frame_id);

    frame_table[phy_frame_id] = vfrm_id;
    old_phy_frame_id = phy_frame_id;

    return vfrm_id;
  }

  inline void unregister_frame(const FrameId &vfrm_id, const FrameId &pfrm_id) {
    free_vframes.enqueue(vfrm_id);
    frame_table.erase(pfrm_id);
    reverse_frame_table.at(vfrm_id.frm_id) = NFrameId;
  }

  inline void init_count(const FrameId &vframe_id) {
    count_table.store(vframe_id);
  }

  inline bool dec_count(const FrameId &vframe_id, NotifTable &notif_table) {
    if (count_table.decrement(vframe_id) == 0) {
      notif_table.clear(vframe_id);
      free_vframes.enqueue(vframe_id);
      return true;
    }
    return false;
  }

  inline FrameId get_v(const FrameId &phy_frame_id) const {
    if (!frame_table.contains(phy_frame_id)) [[unlikely]]
      return NFrameId;
    return frame_table.at(phy_frame_id);
  }
  inline FrameId get_p(const FrameId &vframe_id) const {
    return reverse_frame_table.at(vframe_id.frm_id);
  }
  inline bool contains(const FrameId &phy_frame_id) const {
    return frame_table.contains(phy_frame_id);
  }

  inline uint32_t size() const { return frame_counts; }
};

}  // namespace mega