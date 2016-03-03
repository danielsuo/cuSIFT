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
  set (OpenCV_DIR "/usr/local/Cellar/opencv3/3.1.0_1/share/OpenCV")
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
cuda_add_executable (demo EXCLUDE_FROM_ALL main.cpp ${cuda_sources} ${sources})
target_link_libraries (demo ${CUDA_LIBRARIES} ${OpenCV_LIBS})


################################
# Testing
################################
if (test)
  include (ExternalProject)

  # ExternalProject_Add (
  #   GTest
  #   GIT_REPOSITORY https://github.com/google/googletest
  #   INSTALL_COMMAND ""
  #   UPDATE_COMMAND ""
  #   CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  #              -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
  #              -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
  #              -Dgtest_force_shared_crt=ON
  #   LOG_DOWNLOAD ON
  #   LOG_CONFIGURE ON
  #   LOG_BUILD ON
  #   )

  ExternalProject_Add(vlfeat
    GIT_REPOSITORY https://github.com/danielsuo/vlfeat
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/vlfeat"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    )

  SET(VLROOT ${CMAKE_CURRENT_BINARY_DIR}/vlfeat)

  include_directories (${VLROOT})
  link_directories (${VLROOT}/bin/glnxa64)
  link_directories (${VLROOT}/bin/maci64)

  set(GTEST_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/gtest")
  ExternalProject_Add(GTestExternal
      SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk
      SVN_REVISION -r HEAD
      TIMEOUT 10
      PREFIX "${GTEST_PREFIX}"
      UPDATE_COMMAND ""
      INSTALL_COMMAND "")

  set(LIBPREFIX "${CMAKE_STATIC_LIBRARY_PREFIX}")
  set(LIBSUFFIX "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(GTEST_LOCATION "${GTEST_PREFIX}/src/GTestExternal-build")
  set(GTEST_INCLUDES "${GTEST_PREFIX}/src/GTestExternal/include")
  set(GTEST_LIBRARY  "${GTEST_LOCATION}/${LIBPREFIX}gtest${LIBSUFFIX}")
  set(GTEST_MAINLIB  "${GTEST_LOCATION}/${LIBPREFIX}gtest_main${LIBSUFFIX}")

  add_library(GTest IMPORTED STATIC GLOBAL)
  set_target_properties(GTest PROPERTIES
      IMPORTED_LOCATION                 "${GTEST_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${GTEST_INCLUDES}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")

  add_library(GTestMain IMPORTED STATIC GLOBAL)
  set_target_properties(GTestMain PROPERTIES
      IMPORTED_LOCATION "${GTEST_MAINLIB}"
      IMPORTED_LINK_INTERFACE_LIBRARIES
          "${GTEST_LIBRARY};${CMAKE_THREAD_LIBS_INIT}")

  add_dependencies(GTest GTestExternal vlfeat)

  # ExternalProject_Get_Property (GTest source_dir)
  # include_directories (${source_dir}/include)
  # add_subdirectory (${source_dir})

  enable_testing ()

  cuda_add_executable (tests EXCLUDE_FROM_ALL 
    ${PROJECT_SOURCE_DIR}/test/test.cpp
    ${PROJECT_SOURCE_DIR}/test/detector.cpp
    ${PROJECT_SOURCE_DIR}/test/descriptor.cpp
    )
  target_link_libraries (tests cuSIFT GTest GTestMain vl)

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
