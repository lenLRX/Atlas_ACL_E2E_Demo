# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_cam)

include(cmake/detect_device.cmake)

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

option(BUILD_DEEP_SORT "build deepsort demo" OFF)
option(BUILD_ALL_DEMO "build all useless demo" OFF)

# Compile options
add_compile_options(-std=c++17 -g -O0)

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
else ()
        set(ACL_PATH "/home/HwHiAiUser/Ascend/acllib")
        message(STATUS "set ACL_PATH to default path: /home/HwHiAiUser/Ascend/acllib")    
endif()

# Header path
include_directories(
	/usr/local/Ascend/include # for peripheral_api.h
	${ACL_PATH}/include # acl hearders
        ${CMAKE_SOURCE_DIR}/src
        ${FREETYPE_INCLUDE_DIRS})

if(BUILD_DEEP_SORT)
    include_directories(/usr/include/python3.6)
endif(BUILD_DEEP_SORT)

# add host lib path
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
        src/yolov3_post.cpp
        src/freetype_helper.cpp
        src/camera_input.cpp)

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
        avformat
        avutil
        avcodec
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

if(BUILD_ALL_DEMO)

add_executable(acl_cam
        src/main.cpp)

target_link_libraries(acl_cam
        acl_demo)


add_executable(acl_cam_yolov3_pp
        src/main_yolov3_pp.cpp)

target_link_libraries(acl_cam_yolov3_pp
        acl_demo)

add_executable(acl_video_yolov3_pp
        src/main_yolov3_video.cpp)

target_link_libraries(acl_video_yolov3_pp
        acl_demo)

add_executable(acl_hw_enc
        src/main_hw_encode.cpp)

target_link_libraries(acl_hw_enc
        acl_demo)


add_executable(rtsp_test
        src/main_rtsp_client_test.cpp)

target_link_libraries(rtsp_test
        acl_demo)

endif(BUILD_ALL_DEMO)

add_executable(acl_multi_stream_demo
        src/multi_stream_demo.cpp)

target_link_libraries(acl_multi_stream_demo
        acl_demo)


if(BUILD_DEEP_SORT)
    add_executable(deepsort_test
        src/deep_sort/deep_sort_main.cpp
        src/deep_sort/deep_sort_py.cpp)

    target_link_libraries(deepsort_test
        acl_demo
        python3.6m)
endif(BUILD_DEEP_SORT)
