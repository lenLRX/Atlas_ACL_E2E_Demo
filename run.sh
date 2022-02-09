#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`
export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64/:/usr/lib64:$LD_LIBRARY_PATH
NNRT_ENV_FILE=/usr/local/Ascend/nnrt/set_env.sh
if [[ -f "$NNRT_ENV_FILE" ]]; then
    source $NNRT_ENV_FILE
else
    export ASCEND_AICPU_PATH=/home/HwHiAiUser/Ascend  
fi
# https://stackoverflow.com/questions/45640573/gulp-node-error-while-loading-shared-libraries-cannot-allocate-memory-in-stati
GOMP=/usr/local/lib/python3.6/dist-packages/torch/lib/libgomp-d22c30c5.so.1
if [[ -f "$GOMP" ]]; then
    export LD_PRELOAD=$GOMP
fi
./build/acl_demo_app -c $1
