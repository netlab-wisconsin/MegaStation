cmake_minimum_required(VERSION 3.27)

project(
  MegaStation
  VERSION 0.1
  LANGUAGES C CXX CUDA)

option(MEGA_TESTS "Configure CMake to build tests (or not)" ON)
option(MEGA_BENCHMARKS "Configure CMake to build benchmarks (or not)" ON)

include(CMakePrintHelpers)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG 7e635fca68d014934b4af8a1cf874f63989352b7)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
  readerwriterqueue
  GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue
  GIT_TAG 2dee33ae3edd1e454ac34fea0a27017613355eff)
FetchContent_MakeAvailable(readerwriterqueue)

FetchContent_Declare(
  concurrentqueue
  GIT_REPOSITORY https://github.com/cameron314/concurrentqueue
  GIT_TAG 6dd38b8a1dbaa7863aa907045f32308a56a6ff5d)
FetchContent_MakeAvailable(concurrentqueue)

FetchContent_Declare(
  parallel_hashmap
  GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
  GIT_TAG 65775fa09fecaa65d0b0022ab6bf091c0e509445)
FetchContent_MakeAvailable(parallel_hashmap)

set(CUTLASS_NVCC_ARCHS
    "70;80"
    CACHE STRING "" FORCE)
set(CUTLASS_ENABLE_HEADERS_ONLY
    ON
    CACHE BOOL "" FORCE)
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG 7d49e6c7e2f8896c47f586706e67e1fb215529dc)
FetchContent_MakeAvailable(cutlass)

FetchContent_Declare(cuPHY SOURCE_DIR
                           ${CMAKE_SOURCE_DIR}/third_party/cuBB/cuPHY)
FetchContent_MakeAvailable(cuPHY)

FetchContent_Declare(
  gdrcopy
  GIT_REPOSITORY https://github.com/NVIDIA/gdrcopy.git
  GIT_TAG fbb6f924e0b6361c382bcb0aaef595f08a2cb61f)
include(${CMAKE_SOURCE_DIR}/cmake/gdrcopy.cmake)

FetchContent_Declare(
  datagen SOURCE_DIR ${CMAKE_SOURCE_DIR}/matlab/codegen/lib/data_generator)
FetchContent_MakeAvailable(datagen)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd

# enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc) # for multiple cuda
find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 70;80)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --ptxas-options=-v --expt-relaxed-constexpr --default-stream per-thread"
)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

if(DEFINED ENV{CUDA_DEVICE_MAX_CONNECTIONS})
  add_compile_definitions(MAX_CONNECT=$ENV{CUDA_DEVICE_MAX_CONNECTIONS})
else()
  add_compile_definitions(MAX_CONNECT=8)
endif()

set(CUDA_LIBS
    cuda
    cudart
    cublas
    cusolver
    cufft_static
    culibos
    nccl
    CUTLASS)
set(CUPHY_LIBS cuphy)
set(THIRD_LIBS spdlog::spdlog numa phmap readerwriterqueue concurrentqueue)

include_directories(src/operators)
add_subdirectory(src/operators)

include_directories(src)
include_directories(third_party/cuBB/cuPHY/src/cuphy)

set(COMMON_SOURCE_FILES src/config/config.cc src/utils/pin_threads.cc)
add_library(common_objects OBJECT ${COMMON_SOURCE_FILES})
target_link_libraries(common_objects spdlog::spdlog numa ${CUPHY_LIBS})

add_subdirectory(src/baseband)
add_subdirectory(src/scheduler)

set(MEGA_OBJECTS cuda_objects baseband_objects ldpcfac_objects
                 scheduler_objects common_objects)

add_executable(mega main.cc)
target_link_libraries(mega ${CUDA_LIBS} ${CUPHY_LIBS} ${THIRD_LIBS} gdrapi
                      ${MEGA_OBJECTS})

add_executable(dgen matlab/main.c)
target_link_libraries(dgen data_generator)

add_executable(convert convert.cc)

cmake_print_variables(MEGA_TESTS)
if(MEGA_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

cmake_print_variables(MEGA_BENCHMARKS)
if(MEGA_BENCHMARKS)
  add_subdirectory(benchmark)
endif()
