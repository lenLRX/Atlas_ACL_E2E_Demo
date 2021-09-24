# YOLOv5模型转换
1. 根据这个[仓库](https://github.com/ultralytics/yolov5)的指导下载配置YOLOv5环境,*建议使用virtual env*,[具体命令](https://github.com/ultralytics/yolov5/issues/251)
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
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
