# 模型转换
1. 在官方文档的"开发环境"上首先导入atc依赖的环境变量,以这个[链接](https://support.huaweicloud.com/odevg-A200dk_3000/atlaste_10_0363.html)为例。注意install_path要设置为实际安装的路径。

2. 下载yolov3的[模型文件](https://gitee.com/Atlas200DK/sample-objectdetectionbyyolov3/blob/1.3x.0.0/MyModel/yolov3.caffemodel)
3. 转换模型,yolov3_pp.prototxt和aipp_yolov3_pp.cfg在model文件夹中
```
atc --model=yolov3_pp.prototxt --weight=yolov3.caffemodel --framework=0 --output=sample-yolov3_pp_416 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_pp.cfg
```
4. 将得到的sample-yolov3_pp_416.om拷贝到model文件夹中