#!/bin/bash
rm -rf build
mkdir build
cd build
cmake .. -DATLAS_DEVICE=OrangePi
make -j`nproc`
cd ..
