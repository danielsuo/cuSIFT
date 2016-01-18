cmake_minimum_required(VERSION 3.1)

project(cuSIFT)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

if (APPLE)
  set(OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")
endif()
find_package(OpenCV REQUIRED)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;") 
  endif()
endif()

set(cuda_sources
  cuImage.cu
  cuImage.h
  cuSIFT_H.cu
  cuSIFT_H.h
  cuSIFT_D.h
  cuSIFT.h
  cutils.h
)  

set(sources
  extras/matching.cu
  extras/rigidTransform.cu
  extras/homography.cu
  extras/debug.cpp
  main.cpp
)

include_directories(/usr/local/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(CUDA_SEPARABLE_COMPILATION ON)

# cuda_add_executable(cuSIFT ${cuda_sources} ${sources})

# set_target_properties(cuSIFT PROPERTIES
#   COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
# )

# target_link_libraries(cuSIFT
#   ${CUDA_LIBRARIES} ${OpenCV_LIBS} 
# )
 
# install(FILES 
#   ${cuda_sources} 
#   ${sources}
#   cudaSiftD.cu
#   CMakeLists.txt
#   Copyright.txt
#   DESTINATION .
# )
# install(FILES data/left.pgm data/righ.pgm
#   DESTINATION data
# )

cuda_add_library (cuSIFT ${cuda_sources} ${sources})
set_target_properties(cuSIFT PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"         
)
target_link_libraries (cuSIFT ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_include_directories (cuSIFT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (cuSIFT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extras)