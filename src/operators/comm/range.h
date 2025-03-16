#pragma once

#include <cstdint>
#include <vector>

struct Range {
  // [start_batch, end_batch)
  uint16_t start;
  uint16_t end;
};

using RankRange = std::vector<Range>;