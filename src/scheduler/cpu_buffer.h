#pragma once

#include "defs.h"
#include "utils/consts.h"
#include "utils/types.h"

namespace mega {

struct RingBuffer {
  ArrFrameSymbol<Matrix> buffer;
  inline Matrix &operator[](const SymbolId &frame_sym) {
    return buffer[frame_sym.frm_id % kFrameWindow][frame_sym.sym_id];
  }
};

struct CPUMemBuffer {
  RingBuffer buf_in;
  RingBuffer buf_out;
  CPUMemBuffer();
};

}  // namespace mega