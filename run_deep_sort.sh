#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`
export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64/:$LD_LIBRARY_PATH
#./build/acl_multi_stream_demo --input rtsp://192.168.1.9:8554/tt.mp4 --output rtmp://127.0.0.1:1935/myapp/stream1 --input rtsp://192.168.1.9:8554/tt.mp4 --output rtmp://127.0.0.1:1935/myapp/stream2
./build/deepsort_test --input deep_sort_short_test.mp4 --output deep_sort_out.mp4
