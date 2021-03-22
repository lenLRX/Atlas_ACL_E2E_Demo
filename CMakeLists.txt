# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_cam)

option(BUILD_DEEP_SORT "build deepsort demo" OFF)
option(BUILD_ALL_DEMO "build all useless demo" OFF)

# Compile options
add_compile_options(-std=c++17 -g -O0)

find_package(Freetype REQUIRED)

add_definitions(-DENABLE_DVPP_INTERFACE)

#set(CMAKE_SKIP_RPATH TRUE)

if (EXISTS /usr/local/Ascend/acllib/)
        set(ACL_PATH "/usr/local/Ascend/acllib")
        message(STATUS "set ACL_PATH: /usr/local/Ascend/acllib")
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

add_library(acl_demo SHARED
        src/dvpp_decoder.cpp
        src/ffmpeg_input.cpp
        src/acl_model.cpp
        src/ffmpeg_output.cpp
        src/vpc_resize.cpp
        src/vpc_batch_crop.cpp
        src/jpeg_encode.cpp
        src/dvpp_encoder.cpp
        src/yolov3_post.cpp
        src/camera_input.cpp
        src/freetype_helper.cpp)

target_link_libraries(acl_demo
        ascendcl
        acl_dvpp
        runtime
        /usr/lib64/libslog.so
        /usr/lib64/libc_sec.so
        /usr/lib64/libmedia_mini.so
        pthread
        opencv_core
        opencv_imgproc
        opencv_videoio
        avformat
        avutil
        avcodec
        ${FREETYPE_LIBRARIES})

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
