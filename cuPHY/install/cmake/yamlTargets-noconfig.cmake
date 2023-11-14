#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "yaml" for configuration ""
set_property(TARGET yaml APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(yaml PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libyaml.so"
  IMPORTED_SONAME_NOCONFIG "libyaml.so"
  )

list(APPEND _cmake_import_check_targets yaml )
list(APPEND _cmake_import_check_files_for_yaml "${_IMPORT_PREFIX}/lib/libyaml.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
