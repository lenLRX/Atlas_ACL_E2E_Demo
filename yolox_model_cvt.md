# YOLOX模型转换
目前适配YOLOX 0.2.0版本, 未来YOLOX如果有不兼容的更新可以在issue中提醒我

1. 根据[官方仓库](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.2.0)指导在Host上下载安装YOLOX
```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX/
git checkout 0.2.0
pip install -r requirements.txt
python3 setup.py develop
```
2. 下载[yolox_s.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth),生成onnx
```
python3 tools/export_onnx.py --opset=11 --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```
3. 修改onnx
   * 将[model/yolov5_remove_nodes.py](model/yolov5_remove_nodes.py)拷贝到前一步得到的onnx目录下
   * 执行以下命令，将YOLOX的focus层的slice和concate算子删除
```
python yolov5_remove_nodes.py yolox_s.onnx -o yolox_s_truncate.onnx
```
4. 使用atc转换模型
```
atc --mode=0 --model yolox_s_truncate.onnx --framework=5 --output=yolox_s_truncate --soc_version=Ascend310
```
5. 将得到的yolox_s_truncate.om放到model文件夹备用