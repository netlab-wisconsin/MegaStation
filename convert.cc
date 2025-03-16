#include <cstddef>
#include <fstream>
#include <limits>

template <typename QuanT>
static constexpr float kQuantFloat() {
  return static_cast<float>(std::numeric_limits<QuanT>::max()) + 1.0f;
}

template <typename QuanT, typename QuanS>
static inline void ConvertQuantToQuant(const QuanT* in_buf, QuanS* out_buf,
                                       size_t n_elems) {
  for (size_t i = 0; i < n_elems; i++) {
    float trans = static_cast<float>(in_buf[i]) / kQuantFloat<QuanT>() *
                  kQuantFloat<QuanS>();
    if (trans > std::numeric_limits<QuanS>::max()) {
      trans = std::numeric_limits<QuanS>::max();
    }
    if (trans < std::numeric_limits<QuanS>::min()) {
      trans = std::numeric_limits<QuanS>::min();
    }
    out_buf[i] = static_cast<QuanS>(trans);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <input_file> <output_file>\n", argv[0]);
    return -1;
  }
  std::string input_file = argv[1];
  std::string output_file = argv[2];

  std::ifstream input(input_file, std::ios::binary);
  input.seekg(0, std::ios::end);
  std::streampos fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  const std::size_t n_elems = fileSize / sizeof(short);

  short* h_recv = new short[n_elems];
  input.read(reinterpret_cast<char*>(h_recv), fileSize);
  input.close();

  char* h_recv_char = new char[n_elems];
  ConvertQuantToQuant(h_recv, h_recv_char, n_elems);

  std::ofstream output(output_file, std::ios::binary);
  output.write(reinterpret_cast<char*>(h_recv_char), n_elems * sizeof(char));
  output.close();

  delete[] h_recv;
  delete[] h_recv_char;
  return 0;
}