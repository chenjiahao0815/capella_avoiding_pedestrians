# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_capella_avoiding_pedestrians_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED capella_avoiding_pedestrians_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(capella_avoiding_pedestrians_FOUND FALSE)
  elseif(NOT capella_avoiding_pedestrians_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(capella_avoiding_pedestrians_FOUND FALSE)
  endif()
  return()
endif()
set(_capella_avoiding_pedestrians_CONFIG_INCLUDED TRUE)

# output package information
if(NOT capella_avoiding_pedestrians_FIND_QUIETLY)
  message(STATUS "Found capella_avoiding_pedestrians: 0.0.1 (${capella_avoiding_pedestrians_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'capella_avoiding_pedestrians' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${capella_avoiding_pedestrians_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(capella_avoiding_pedestrians_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${capella_avoiding_pedestrians_DIR}/${_extra}")
endforeach()
