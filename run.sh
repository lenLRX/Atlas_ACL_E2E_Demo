#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64/:$LD_LIBRARY_PATH
#./build/acl_multi_stream_demo --input rtsp://192.168.1.9:8554/tt.mp4 --output rtmp://127.0.0.1:1935/myapp/stream1 --input rtsp://192.168.1.9:8554/tt.mp4 --output rtmp://127.0.0.1:1935/myapp/stream2
./build/acl_multi_stream_demo --input rtsp://192.168.1.9:8554/tt1.mp4 --output rtmp://127.0.0.1:1935/myapp/stream1 --input rtsp://192.168.1.9:8555/tt2.mp4 --output rtmp://127.0.0.1:1935/myapp/stream2