#CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

#project information
project(acl_e2e_demo)

include(cmake/detect_device.cmake)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (NOT DEFINED ATLAS_DEVICE)
        detect_device()
        message(STATUS "ATLAS_DEVICE not set, detected device type: ${ATLAS_DEVICE}")
else()
        message(STATUS "user set device type: ${ATLAS_DEVICE}")
endif()

if (${ATLAS_DEVICE} STREQUAL "Atlas200DK")
        set (HAS_LIBMEDIA TRUE)
else()
        set (HAS_LIBMEDIA FALSE)
endif()

message(STATUS "build with libmedia ${HAS_LIBMEDIA}")

#Compile options
#add_compile_options(-std=c++17 -O0 -g)
add_compile_options(-std=c++17 -O2)

find_package(Freetype REQUIRED)

add_definitions(-DENABLE_DVPP_INTERFACE)

#set(CMAKE_SKIP_RPATH TRUE)

if (DEFINED ACL_PATH)
        message(STATUS "user set ACL_PATH: ${ACL_PATH}")
elseif (EXISTS /usr/local/Ascend/acllib/)
        set(ACL_PATH "/usr/local/Ascend/acllib")
        message(STATUS "set ACL_PATH: /usr/local/Ascend/acllib")
elseif (EXISTS /usr/local/Ascend/ascend-toolkit/latest/acllib)
        set(ACL_PATH "/usr/local/Ascend/ascend-toolkit/latest/acllib")
        message(STATUS "set ACL_PATH to default path: /usr/local/Ascend/ascend-toolkit/latest/acllib")
elseif (EXISTS /usr/local/Ascend/nnrt/latest/acllib)
        set(ACL_PATH "/usr/local/Ascend/nnrt/latest/acllib")
        message(STATUS "set ACL_PATH to default path: /usr/local/Ascend/nnrt/latest/acllib")    
else ()
        set(ACL_PATH "/home/HwHiAiUser/Ascend/acllib")
        message(STATUS "set ACL_PATH to default path: /home/HwHiAiUser/Ascend/acllib")    
endif()

#Header path
include_directories(
	/usr/local/Ascend/include # for peripheral_api.h
	${ACL_PATH}/include # acl hearders
        /usr/include/python3.6 # for python headers
        /usr/local/lib/python3.6/dist-packages/numpy/core/include/ # for pip installed numpy headers
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/bytetrack_csrc/include
        ${FREETYPE_INCLUDE_DIRS})

if(BUILD_DEEP_SORT)
    include_directories(/usr/include/python3.6)
endif(BUILD_DEEP_SORT)

#add host lib path
link_directories(
	/usr/lib64/
	${ACL_PATH}/lib64/
)

set(DEMO_LIB_SRC
        src/dvpp_decoder.cpp
        src/ffmpeg_input.cpp
        src/acl_model.cpp
        src/ffmpeg_output.cpp
        src/vpc_resize.cpp
        src/vpc_batch_crop.cpp
        src/jpeg_encode.cpp
        src/dvpp_encoder.cpp
        src/freetype_helper.cpp
        src/camera_input.cpp
        src/app_profiler.cpp
        src/yolov3_stream.cpp
        src/yolov4_stream.cpp
        src/yolov5_stream.cpp
        src/yolox_stream.cpp
        src/yolox_bytetrack_stream.cpp
        src/focus_op.cpp
        src/signal_handler.cpp
        src/deep_sort_stream.cpp
        src/stream_factory.cpp
        src/device_manager.cpp
        bytetrack_csrc/src/bytetrack.cpp
        bytetrack_csrc/src/BYTETracker.cpp
        bytetrack_csrc/src/kalmanFilter.cpp
        bytetrack_csrc/src/lapjv.cpp
        bytetrack_csrc/src/STrack.cpp
        bytetrack_csrc/src/utils.cpp)

add_library(acl_demo SHARED
        ${DEMO_LIB_SRC})

if (${HAS_LIBMEDIA})
        target_compile_definitions(acl_demo PUBLIC HAS_CAMERA)
endif()

set(DEMO_DEP_LIBS
        ascendcl
        acl_dvpp
        runtime
        pthread
        opencv_core
        opencv_imgproc
        opencv_videoio
        opencv_imgcodecs
        avformat
        avutil
        avcodec
        python3.6m
        ${FREETYPE_LIBRARIES})

if (${ATLAS_DEVICE} STREQUAL "Atlas200DK")
        set(DEMO_DEP_LIBS
                ${DEMO_DEP_LIBS}
                /usr/lib64/libslog.so
                /usr/lib64/libc_sec.so
                /usr/lib64/libmedia_mini.so)
endif()

target_link_libraries(acl_demo
        ${DEMO_DEP_LIBS}
        )

add_executable(acl_demo_app
        src/acl_demo_app.cpp)

target_link_libraries(acl_demo_app
        acl_demo)

add_executable(benchmark_demo
        src/benchmark_demo.cpp)

target_link_libraries(benchmark_demo
        acl_demo)

