FetchContent_GetProperties(gdrcopy)
if(NOT gdrcopy_POPULATED)
  FetchContent_MakeAvailable(gdrcopy)

  set(GDR_DIR "${gdrcopy_SOURCE_DIR}")
  set(GDR_OUT "${gdrcopy_BINARY_DIR}")

  add_custom_command(
    OUTPUT ${GDR_OUT}/libgdrapi.so
    WORKING_DIRECTORY ${GDR_DIR}
    COMMENT Building libgdrapi
    COMMAND make -B lib_install DESTINC=${GDR_OUT} DESTLIB=${GDR_OUT}
    VERBATIM)

  add_custom_target(gdrcopy ALL DEPENDS ${GDR_OUT}/libgdrapi.so)

  set_target_properties(gdrcopy PROPERTIES PUBLIC_HEADER "${GDR_DIR}/include")
  install(PROGRAMS sudo ${GDR_DIR}/insmod.sh DESTINATION .)

  # Bring the populated content into the build
  # add_subdirectory(${gdrcopy_SOURCE_DIR} ${gdrcopy_BINARY_DIR})
  include_directories(${GDR_DIR}/include)
  link_directories(${GDR_OUT})
endif()
