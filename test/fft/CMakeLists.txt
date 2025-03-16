add_executable(
  test_fft
  test_fft.cc
)

target_link_libraries(
  test_fft
  GTest::gtest_main
  ${CUDA_LIBS}
  cuda_objects
)

include(GoogleTest)
gtest_discover_tests(test_fft)