# ACL多路端到端推理样例(现已支持[OrangePi](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro(20T).html))
这个demo主要包括以下功能点 
* ffmpeg RTSP/mp4/摄像头视频流输入
* ffmpeg RTMP/RTSP/mp4视频流输出
* DVPP H264解码
* DVPP H264编码
* ACL yolov3推理
    * DVPP resize
* [AMCT yolov3 int8 量化](quant/caffe_quant.md)
* [ACL yolov3+deepsort推理](deepsort.md)
    * DVPP crop resize
* ACL yolov4推理
* ACL yolov5推理
* ACL yoloX推理
* ACL YOLOX+bytetrack多目标追踪
* 在YUV420SP图像上的[画框](src/drawing.h)和[中文字符显示](src/freetype_helper.cpp)
* [性能分析](profiling.md)

## 支持硬件
* OrangePi
* Atlas200DK
* Atlas300I

## 支持软件版本
* OrangePi:
  * 7.0.0+
* Atlas200DK:
  * 20.0
  * 20.1
  * 20.2(3.2.0)
  * 3.3.0.alpha001
  * 3.3.0.alpha006
  * 5.0.2.alpha002
  * 5.0.3.alpha002
  * 后续版本都会持续支持
* Atlas300I:
  * 5.0.5.alpha001(华为云镜像环境)

## 环境搭建
* [环境搭建文档(OrangePi)](env_orange_pi.md)
* [环境搭建文档(Atlas200DK)](env_atlas200dk.md)
* [环境搭建文档(Atlas300I)](env_atlas300I.md)
## YOLO V3 模型转换
[yolov3 模型转换文档](yolov3_model_cvt.md)
## YOLO V4 模型转换
[yolov4 模型转换文档](yolov4_model_cvt.md)
## YOLO V5 模型转换
[yolov5 模型转换文档](yolov5_model_cvt.md)
## YOLOX 模型转换
[yolox 模型转换文档](yolox_model_cvt.md)
## YOLOX+ByteTrack
[yolox+bytetrack 模型转换文档](bytetrack_model_cvt.md)
## DeepSort 模型转换
[deepsort 模型转换文档](deep_sort_model_cvt.md)
## 运行测试
[demo使用说明](run.md)

## TODO
* 支持多batch，提升推理性能
* 限制输出码率
* 提升deepsort性能
* 支持更多模型

使用过程中遇到的问题或者需求请直接提issue

