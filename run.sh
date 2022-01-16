#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`
export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64/:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_AICPU_PATH=/home/HwHiAiUser/Ascend
# https://stackoverflow.com/questions/45640573/gulp-node-error-while-loading-shared-libraries-cannot-allocate-memory-in-stati
GOMP=/usr/local/lib/python3.6/dist-packages/torch/lib/libgomp-d22c30c5.so.1
if [[ -f "$GOMP" ]]; then
    export LD_PRELOAD=$GOMP
fi
./build/acl_demo_app -c $1
