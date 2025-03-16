add_executable(integral_test integral_test.cc)

target_link_libraries(
  integral_test
  GTest::gtest_main
  ${CUDA_LIBS}
  ${CUPHY_LIBS}
  baseband_objects
  common_objects
  cuda_objects)

include(GoogleTest)
gtest_discover_tests(integral_test)
