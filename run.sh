#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`
export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64/:$LD_LIBRARY_PATH
export ASCEND_AICPU_PATH=/home/HwHiAiUser/Ascend
./build/acl_demo_app -c $1
