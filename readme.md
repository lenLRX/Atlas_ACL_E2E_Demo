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
* [性能分析](profiling.md)

## 支持硬件
* Atlas200DK
* Atlas300I

## 支持软件版本
* Atlas200DK:
  * 20.0
  * 20.1
  * 20.2(3.2.0)
  * 3.3.0.alpha001
  * 3.3.0.alpha006
* Atlas300I:
  * 20.1(华为云镜像环境)

## 环境搭建
* [环境搭建文档(Atlas200DK)](env_atlas200dk.md)
* [环境搭建文档(Atlas300I)](env_atlas300I.md)
## YOLO V3 模型转换
[yolov3 模型转换文档](yolov3_model_cvt.md)
## DeepSort 模型转换
[deepsort 模型转换文档](deep_sort_model_cvt.md)
## 运行测试
[demo使用说明](run.md)

## TODO
* 支持多batch，提升推理性能
* 支持Atlas300I多device
* 限制输出码率
* 提升deepsort性能
* 支持更多模型

使用过程中遇到的问题或者需求请直接提issue

