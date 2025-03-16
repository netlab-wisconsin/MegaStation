add_executable(
  test_scrambler
  test_scrambler.cc
)

target_link_libraries(
  test_scrambler
  GTest::gtest_main
  ${CUDA_LIBS}
  cuda_objects
)

include(GoogleTest)
gtest_discover_tests(test_scrambler)