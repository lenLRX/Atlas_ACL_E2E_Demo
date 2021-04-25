# ACL多路端到端推理样例
这个demo主要包括以下功能点 
* ffmpeg RTSP/mp4/摄像头视频流输入
* ffmpeg RTMP/RTSP/mp4视频流输出
* DVPP H264解码
* DVPP H264编码
* ACL yolov3推理
    * DVPP resize
* [ACL yolov3+deepsort推理](deepsort.md)
    * DVPP crop resize
* 在YUV420SP图像上的[画框](src/drawing.h)和[中文字符显示](src/freetype_helper.cpp)
* 性能分析

## 支持硬件
* Atlas200DK
* Atlas300I

## 支持软件版本
* Atlas200DK: 20.0,20.1,20.2(3.2.0),3.3.0
* Atlas300I: 20.1

## 环境搭建
* [环境搭建文档(Atlas200DK)](env_atlas200dk.md)
* [环境搭建文档(Atlas300I)](env.md)
## YOLO V3 模型转换
[yolov3 模型转换文档](yolov3_model_cvt.md)
## DeepSort 模型转换
[deepsort 模型转换文档](deep_sort_model_cvt.md)
## 运行测试(RTSP输入RTMP输出)
1. 在Atlas200DK上启动nginx
2. 准备两个rtsp视频流，这里使用ubuntu上的vlc命令行为例。这里的视频文件请替换为你对应的测试文件。(vlc在windows上也可以在界面上完成类似的操作。)
```
nohup vlc -v road_traffic_test1.mp4 --sout '#rtp{sdp=rtsp://:8554/tt1.mp4}' &
nohup vlc -v road_traffic_test2.mp4 --sout '#rtp{sdp=rtsp://:8555/tt2.mp4}' &
```
3. 运行run.sh,请将input的地址替换为实际的地址
```
./run.sh
```
4. 使用vlc或其他视频播放器检查推理结果。
```
媒体->打开网络串流->输入run.sh中output对应的rtmp地址。(127.0.0.1替换为Atlas200DK在你的网络中对应的IP)
```

### MP4输入输出
现在新增了对MP4输入输出的支持，可以在run.sh中将rtsp或rtmp的输入输出地址替换为Atlas200 DK本地的MP4文件路径(请尽量使用绝对路径)。

示例:
```
./build/acl_multi_stream_demo --input input.mp4 --output output.mp4
```
### RTSP输出
现在新增了对RTSP输出的支持，可以在run.sh中将rtmp的输出地址替换为rtsp的路径。

示例：
```
./build/acl_multi_stream_demo --input input.mp4 --output rtsp://192.168.1.9/stream1
```
### 摄像头输入
现在新增了对Atlas200DK摄像头输入的支持，可以在run.sh中将输入地址替换为camera0或camera1。

摄像头输入规格： 720P @ 20fps
示例：
```
./build/acl_multi_stream_demo --input camera0 --output rtsp://192.168.1.9/stream1
```
### 其他格式
由于使用的是ffmpeg通用的API所以其他输入输出格式可能也是天然支持的，你可以直接试一下，说不定可以直接使用呢。

## TODO
* 现在测试能够支持2路实时推理，支持更多的推理可能还需要做一些修改(yolov3模型现在是1batch的，效率较低)。
* 现在所有框都是红色的，可以根据不同类别画不同颜色的框
