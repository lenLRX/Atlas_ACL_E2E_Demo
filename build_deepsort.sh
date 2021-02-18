#!/bin/bash
rm -rf build
mkdir build
cd build
cmake .. -DBUILD_DEEP_SORT=ON
make -j`nproc`
cd ..
