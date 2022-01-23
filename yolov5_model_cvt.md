# YOLOv5模型转换

Yolov5原作者一直在更新版本，每个版本的模型可能会发生变化，目前本仓库支持两个版本。

如果使用v6.0需要配置json: ```"yolov5_version": "v6"```

如果使用v5.0需要配置json: ```"yolov5_version": "v5"```

## YOLOv5 v6.0

1. 根据这个[仓库](https://github.com/ultralytics/yolov5)的指导下载配置YOLOv5环境,*建议使用virtual env*,[具体命令](https://github.com/ultralytics/yolov5/issues/251)
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout v6.0
pip install -r requirements.txt
pip install -U coremltools onnx scikit-learn==0.19.2
```
2. 生成onnx, 以yolov5s为例,最终得到yolov5s.onnx
```
python export.py --include onnx --weights yolov5s.pt --img 640 --batch 1 --opset=11
```
3. 使用atc转换模型
将[model/yuv420sp_aipp.cfg](model/yuv420sp_aipp.cfg)拷贝到yolov5文件夹下，使用atc转换模型。
```
atc --mode=0 --model yolov5s.onnx --framework=5 --output=yolov5s_v6 --soc_version=Ascend310 --insert_op_conf=yuv420sp_aipp.cfg
```
4. 将得到的yolov5s_v6.om放到model文件夹备用

## YOLOv5 v5.0后的某个版本

1. 根据这个[仓库](https://github.com/ultralytics/yolov5)的指导下载配置YOLOv5环境,*建议使用virtual env*,[具体命令](https://github.com/ultralytics/yolov5/issues/251)
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout e96c74b5a1c4a27934c5d8ad52cde778af248ed8
pip install -r requirements.txt
pip install -U coremltools onnx scikit-learn==0.19.2
```
2. 生成onnx, 以yolov5s为例,最终得到yolov5s.onnx
```
python export.py --include onnx --weights yolov5s.pt --img 640 --batch 1 --opset=11
```
3. 修改onnx
   * 将[model/yolov5_remove_nodes.py](model/yolov5_remove_nodes.py)拷贝到前一步得到的onnx目录下
   * 执行以下命令，将yolov5的focus层的slice和concate算子删除
```
python yolov5_remove_nodes.py yolov5s.onnx -o yolov5s_truncate.onnx
```
4. 使用atc转换模型
```
atc --mode=0 --model yolov5s_truncate.onnx --framework=5 --output=yolov5s_truncate --soc_version=Ascend310
```
5. 将得到的yolov5s_truncate.om放到model文件夹备用
