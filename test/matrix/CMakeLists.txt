add_executable(test_matrix test_matrix.cc)

target_link_libraries(test_matrix GTest::gtest_main ${CUDA_LIBS} ${CUPHY_LIBS})

add_executable(test_cuphy_tensor test_cuphy_tensor.cc)

# include(CMakePrintHelpers) cmake_print_variables(CUPHY_LIBS)

target_link_libraries(test_cuphy_tensor GTest::gtest_main ${CUDA_LIBS}
                      ${CUPHY_LIBS})

include(GoogleTest)
gtest_discover_tests(test_matrix)
gtest_discover_tests(test_cuphy_tensor)
