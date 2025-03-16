###########################################################################
# CMakeLists.txt generated for component data_generator
# Product type: STATIC library
###########################################################################
cmake_minimum_required(VERSION 3.12)
project(data_generator)

# Propagate the CMAKE_EXPORT_COMPILE_COMMANDS variable from the 
# environment if it is defined as an environment variable, but not as a 
# CMake variable. This is to work around a bug in CMake 3.19 when the 
# "NMake Makefiles" generator is selected.
if(DEFINED ENV{CMAKE_EXPORT_COMPILE_COMMANDS} AND NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS $ENV{CMAKE_EXPORT_COMPILE_COMMANDS})
endif()

# Define common variables that are used within the whole project.
set(SYSLIB_PREFIX $<IF:$<AND:$<BOOL:${WIN32}>,$<NOT:$<BOOL:${MINGW}>>>,lib,>)

###########################################################################
## Path variables
###########################################################################
# Derive an absolute path to the code generation anchor folder.
get_filename_component(START_DIR ../../.. ABSOLUTE)

# Special directories defined by using CACHE variables can be overridden 
# by setting the variable from the command line, e.g.,
# 
# 	cmake . -DMATLAB_ROOT=/path/to/another/matlab/root
set(MATLAB_ROOT /usr/local/MATLAB/MATLAB_Runtime/R2024a CACHE PATH "")

# Additional variables that are defined conditionally.
if("${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    set(BINARY_START_DIR "${START_DIR}")
else()
    set(BINARY_START_DIR "${CMAKE_BINARY_DIR}")
endif()
if(APPLE)
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL x86_64)
        list(APPEND MATLAB_ROOT_SYSLIB_PATHS ${MATLAB_ROOT}/bin/maci64)
        list(APPEND MATLAB_ROOT_SYSLIB_PATHS ${MATLAB_ROOT}/sys/os/maci64)
        list(APPEND MATLAB_ROOT_SYSINCLUDE_PATHS ${MATLAB_ROOT}/toolbox/eml/externalDependency/omp/maci64/include)
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL arm64)
        list(APPEND MATLAB_ROOT_SYSLIB_PATHS ${MATLAB_ROOT}/bin/maca64)
        list(APPEND MATLAB_ROOT_SYSINCLUDE_PATHS ${MATLAB_ROOT}/toolbox/eml/externalDependency/omp/maca64/include)
    endif()
elseif(UNIX AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL x86_64) AND ("${CMAKE_SYSTEM_NAME}" STREQUAL Linux))
    list(APPEND MATLAB_ROOT_SYSLIB_PATHS ${MATLAB_ROOT}/bin/glnxa64)
elseif(WIN32 AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL AMD64))
    if(MSVC)
        list(APPEND MATLAB_ROOT_SYSLIB_PATHS ${MATLAB_ROOT}/extern/lib/win64/microsoft)
    elseif(MINGW)
        list(APPEND MATLAB_ROOT_SYSLIB_PATHS ${MATLAB_ROOT}/extern/lib/win64/mingw64)
    endif()
    list(APPEND MATLAB_ROOT_SYSLIB_PATHS ${MATLAB_ROOT}/bin/win64
        ${MATLAB_ROOT}/lib/win64)
endif()


if(DEFINED MATLAB_ROOT_SYSLIB_PATHS)
    list(APPEND CMAKE_LIBRARY_PATH ${MATLAB_ROOT_SYSLIB_PATHS})
    link_directories(${MATLAB_ROOT_SYSLIB_PATHS})
endif()
if(DEFINED MATLAB_ROOT_SYSINCLUDE_PATHS)
    list(APPEND CMAKE_INCLUDE_PATH ${MATLAB_ROOT_SYSINCLUDE_PATHS})
endif()



###########################################################################
## Required packages
###########################################################################
find_package(OpenMP REQUIRED)

###########################################################################
## System Libraries
###########################################################################
find_library(FOUND_LIBM m NO_SYSTEM_ENVIRONMENT_PATH PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

###########################################################################
## Target definition and commands
###########################################################################

# Definition of target "data_generator".
add_library(data_generator STATIC ${START_DIR}/codegen/lib/data_generator/data_generator_data.c
    ${START_DIR}/codegen/lib/data_generator/rt_nonfinite.c
    ${START_DIR}/codegen/lib/data_generator/rtGetNaN.c
    ${START_DIR}/codegen/lib/data_generator/rtGetInf.c
    ${START_DIR}/codegen/lib/data_generator/data_generator_initialize.c
    ${START_DIR}/codegen/lib/data_generator/data_generator_terminate.c
    ${START_DIR}/codegen/lib/data_generator/data_generator.c
    ${START_DIR}/codegen/lib/data_generator/i64ddiv.c
    ${START_DIR}/codegen/lib/data_generator/eml_i64dmul.c
    ${START_DIR}/codegen/lib/data_generator/rand.c
    ${START_DIR}/codegen/lib/data_generator/fileManager.c
    ${START_DIR}/codegen/lib/data_generator/randn.c
    ${START_DIR}/codegen/lib/data_generator/wlanScramble.c
    ${START_DIR}/codegen/lib/data_generator/nrLDPCEncode.c
    ${START_DIR}/codegen/lib/data_generator/find.c
    ${START_DIR}/codegen/lib/data_generator/encode.c
    ${START_DIR}/codegen/lib/data_generator/ldpcQuasiCyclicMatrix.c
    ${START_DIR}/codegen/lib/data_generator/colon.c
    ${START_DIR}/codegen/lib/data_generator/sparse.c
    ${START_DIR}/codegen/lib/data_generator/combineVectorElements.c
    ${START_DIR}/codegen/lib/data_generator/ldpcEncoderConfig.c
    ${START_DIR}/codegen/lib/data_generator/sparse1.c
    ${START_DIR}/codegen/lib/data_generator/locBsearch.c
    ${START_DIR}/codegen/lib/data_generator/tril.c
    ${START_DIR}/codegen/lib/data_generator/triu.c
    ${START_DIR}/codegen/lib/data_generator/eml_setop.c
    ${START_DIR}/codegen/lib/data_generator/ldpcEncode.c
    ${START_DIR}/codegen/lib/data_generator/qammod.c
    ${START_DIR}/codegen/lib/data_generator/log2.c
    ${START_DIR}/codegen/lib/data_generator/getSquareConstellation.c
    ${START_DIR}/codegen/lib/data_generator/ifft.c
    ${START_DIR}/codegen/lib/data_generator/FFTImplementationCallback.c
    ${START_DIR}/codegen/lib/data_generator/fwrite.c
    ${START_DIR}/codegen/lib/data_generator/bit2int.c
    ${START_DIR}/codegen/lib/data_generator/eml_rand_mt19937ar_stateful.c
    ${START_DIR}/codegen/lib/data_generator/eml_rand_mt19937ar.c
    ${START_DIR}/codegen/lib/data_generator/exp.c
    ${START_DIR}/codegen/lib/data_generator/insertionsort.c
    ${START_DIR}/codegen/lib/data_generator/introsort.c
    ${START_DIR}/codegen/lib/data_generator/heapsort.c
    ${START_DIR}/codegen/lib/data_generator/fillIn.c
    ${START_DIR}/codegen/lib/data_generator/sort.c
    ${START_DIR}/codegen/lib/data_generator/sortIdx.c
    ${START_DIR}/codegen/lib/data_generator/circshift.c
    ${START_DIR}/codegen/lib/data_generator/data_generator_emxutil.c
    ${START_DIR}/codegen/lib/data_generator/data_generator_rtwutil.c)

# Set properties for target "data_generator".
set_target_properties(data_generator PROPERTIES PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY "${BINARY_START_DIR}/codegen/lib/data_generator/$<0:>"
    LIBRARY_OUTPUT_DIRECTORY "${BINARY_START_DIR}/codegen/lib/data_generator/$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${BINARY_START_DIR}/codegen/lib/data_generator/$<0:>")

# Specify language features required for target "data_generator".
target_compile_features(data_generator PUBLIC c_std_99)

# Specify compiler preprocessor definitions for target "data_generator".
target_compile_definitions(data_generator PRIVATE -DMODEL=data_generator)

# Specify include directories for target "data_generator".
target_include_directories(data_generator PUBLIC $<BUILD_INTERFACE:${START_DIR}/codegen/lib/data_generator>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/codegen/lib/data_generator>
    $<BUILD_INTERFACE:${START_DIR}>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>>
    $<BUILD_INTERFACE:${MATLAB_ROOT}/extern/include>)

# Specify library link dependencies for target "data_generator".  CMake 
# generator expressions are used to create a CMakeLists.txt file that 
# supports multiple platforms with differently named system library 
# dependencies.
target_link_libraries(data_generator INTERFACE $<$<BOOL:${FOUND_LIBM}>:m>
    OpenMP::OpenMP_C
    OpenMP::OpenMP_CXX)


###########################################################################
## Build success message
###########################################################################
add_custom_command(TARGET data_generator POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "\\#\\#\\# Created STATIC library: $<TARGET_FILE:data_generator>")

###########################################################################
## Call toolchain hook function if defined
###########################################################################
if(COMMAND toolchain_target_hook)
    toolchain_target_hook(data_generator)
endif()
