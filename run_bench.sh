#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`
export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64/:/usr/lib64:$LD_LIBRARY_PATH
NNRT_ENV_FILE=/usr/local/Ascend/nnrt/set_env.sh
if [[ -f "$NNRT_ENV_FILE" ]]; then
    source $NNRT_ENV_FILE
else
    export ASCEND_AICPU_PATH=/home/HwHiAiUser/Ascend  
fi
./build/benchmark_demo -c $1
