# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_cam)

# Compile options
add_compile_options(-std=c++11 -g -O0)


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
        src/rtmp_stream.cpp)

target_link_libraries(acl_cam
        ascendcl
        /usr/lib64/libslog.so
        /usr/lib64/libc_sec.so
        /usr/lib64/libmedia_mini.so
        avformat
        avutil
        avcodec)
