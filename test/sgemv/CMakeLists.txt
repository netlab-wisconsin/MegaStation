add_executable(
  test_sgemv
  test_sgemv.cc
)

target_link_libraries(
  test_sgemv
  GTest::gtest_main
  ${CUDA_LIBS}
  cuda_objects
)

include(GoogleTest)
gtest_discover_tests(test_sgemv)