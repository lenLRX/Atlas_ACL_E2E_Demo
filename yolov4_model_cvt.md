# YOLOv4模型转换
1. 根据这个[仓库](https://github.com/Tianxiaomo/pytorch-YOLOv4)的指导下载配置Pytorch-YOLOV4环境,然后按照这个[知乎文章](https://zhuanlan.zhihu.com/p/384506398)指导生成yolov4_1_3_416_416_static.onnx
2. 按照官方文档或者[docker仓库](https://github.com/lenLRX/CANN-toolkit-docker)配置atc环境
3. 将yolov4_1_3_416_416_static.onnx和[yolov4_aipp.cfg](model/yolov4_aipp.cfg)放到atc环境中执行以下命令:
```
atc --mode=0 --model yolov4_1_3_416_416_static.onnx --framework=5 --output=yolov4_1_3_416_416_aipp --soc_version=Ascend310 --insert_op_conf=yolov4_aipp.cfg
```
4. 将得到的yolov4_1_3_416_416_aipp.om拷贝到model文件夹中