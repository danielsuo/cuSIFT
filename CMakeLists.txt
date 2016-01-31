cmake_minimum_required (VERSION 3.1)

project (cuSIFT)
set (cuSIFT_VERSION_MAJOR 2)
set (cuSIFT_VERSION_MINOR 0)
set (cuSIFT_VERSION_PATCH 0)

set (CPACK_PACKAGE_VERSION_MAJOR "${cuSIFT_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${cuSIFT_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${cuSIFT_VERSION_PATCH}")
set (CPACK_GENERATOR "ZIP")
include (CPack)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option (test "Build all tests." OFF) # Makes boolean 'test' available.

if  (APPLE)
  set (OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")
endif ()
find_package (OpenCV REQUIRED)

find_package (CUDA)
if  (NOT CUDA_FOUND)
  message (STATUS "CUDA not found. Project will not be built.")
endif (NOT CUDA_FOUND)

if  (WIN32)
  set (EXTRA_CXX_FLAGS "/D_CRT_SECURE_NO_WARNINGS ")
  list (APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;") 
endif ()
if  (UNIX)
  if  (APPLE)
    set (EXTRA_CXX_FLAGS "-msse2")
    list (APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;") 
  else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
    list (APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;") 
  endif ()
endif ()

SET (CUDA_SEPARABLE_COMPILATION ON)

set (cuda_sources
  cuImage.cu
  cuImage.h
  cuSIFT.cu
  cuSIFT_H.cu
  cuSIFT_H.h
  cuSIFT_D.h
  cuSIFT.h
  cutils.h
)  

set (sources
  extras/matching.cu
  extras/rigidTransform.cu
  extras/homography.cu
  extras/math_utils.cu
  extras/debug.cpp
)

set (${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}
                                 ${PROJECT_SOURCE_DIR}/extras
    CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

include_directories (/usr/local/include)
include_directories (${${PROJECT_NAME}_INCLUDE_DIRS})

################################
# Library
################################
cuda_add_library (cuSIFT ${cuda_sources} ${sources})
set_target_properties (cuSIFT PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"         
)
target_link_libraries (cuSIFT ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_include_directories (cuSIFT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (cuSIFT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extras)

################################
# Demo
################################
cuda_add_executable  (demo EXCLUDE_FROM_ALL main.cpp ${cuda_sources} ${sources})
target_link_libraries  (demo ${CUDA_LIBRARIES} ${OpenCV_LIBS})


################################
# Testing
################################
if (test)
  include (ExternalProject)

  ExternalProject_Add (
    GTest
    GIT_REPOSITORY https://github.com/google/googletest
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -Dgtest_force_shared_crt=ON
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    )

  ExternalProject_Get_Property (GTest source_dir)
  include_directories (${source_dir}/include)
  add_subdirectory (${source_dir})

  enable_testing ()

  cuda_add_executable (tests EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/test/test.cpp)
  target_link_libraries (tests cuSIFT gtest gtest_main)

  add_test(NAME tests COMMAND tests)

endif ()

# cuda_add_executable (cuSIFT ${cuda_sources} ${sources})

# set_target_properties (cuSIFT PROPERTIES
#   COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"         
# )

# target_link_libraries (cuSIFT
#   ${CUDA_LIBRARIES} ${OpenCV_LIBS} 
# )
 
# install (FILES 
#   ${cuda_sources} 
#   ${sources}
#   cuSIFTD.cu
#   CMakeLists.txt
#   Copyright.txt
#   DESTINATION .
# )
# install (FILES data/left.pgm data/righ.pgm
#   DESTINATION data
# )