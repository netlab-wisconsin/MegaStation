add_executable(
  test_modulation
  test_modulation.cc
)

target_link_libraries(
  test_modulation
  GTest::gtest_main
  ${CUDA_LIBS}
  cuda_objects
)

include(GoogleTest)
gtest_discover_tests(test_modulation)