# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_cam)

# Compile options
add_compile_options(-std=c++17 -g -O0)

add_definitions(-DENABLE_DVPP_INTERFACE)

#set(CMAKE_SKIP_RPATH TRUE)



# Header path
include_directories(
	/usr/local/Ascend/include # for peripheral_api.h
	/usr/local/Ascend/acllib/include # acl hearders
)

# add host lib path
link_directories(
	/usr/lib64/
	/usr/local/Ascend/acllib/lib64/
)

add_executable(acl_cam
        src/main.cpp
        src/acl_model.cpp
        src/rtmp_stream.cpp
        src/vpc_resize.cpp
        src/yolov3_post.cpp)

target_link_libraries(acl_cam
        ascendcl
        acl_dvpp
        runtime
        /usr/lib64/libslog.so
        /usr/lib64/libc_sec.so
        /usr/lib64/libmedia_mini.so
        opencv_core
        avformat
        avutil
        avcodec)


add_executable(acl_cam_yolov3_pp
        src/main_yolov3_pp.cpp
        src/acl_model.cpp
        src/rtmp_stream.cpp
        src/vpc_resize.cpp
        src/yolov3_post.cpp)

target_link_libraries(acl_cam_yolov3_pp
        ascendcl
        acl_dvpp
        runtime
        /usr/lib64/libslog.so
        /usr/lib64/libc_sec.so
        /usr/lib64/libmedia_mini.so
        opencv_core
        opencv_imgproc
        avformat
        avutil
        avcodec)

add_executable(acl_video_yolov3_pp
        src/main_yolov3_video.cpp
        src/acl_model.cpp
        src/rtmp_stream.cpp
        src/vpc_resize.cpp
        src/yolov3_post.cpp)

target_link_libraries(acl_video_yolov3_pp
        ascendcl
        acl_dvpp
        runtime
        /usr/lib64/libslog.so
        /usr/lib64/libc_sec.so
        /usr/lib64/libmedia_mini.so
        opencv_core
        opencv_imgproc
        opencv_videoio
        avformat
        avutil
        avcodec)

add_executable(acl_hw_enc
        src/main_hw_encode.cpp
        src/acl_model.cpp
        src/rtmp_stream.cpp
        src/vpc_resize.cpp
        src/dvpp_encoder.cpp
        src/yolov3_post.cpp)

target_link_libraries(acl_hw_enc
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
        avcodec)


add_executable(rtsp_test
        src/main_rtsp_client_test.cpp
        src/dvpp_decoder.cpp
        src/rtsp_input.cpp
        src/acl_model.cpp
        src/rtmp_stream.cpp
        src/vpc_resize.cpp
        src/dvpp_encoder.cpp)

target_link_libraries(rtsp_test
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
        avcodec)