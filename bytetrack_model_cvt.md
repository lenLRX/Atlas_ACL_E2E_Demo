# YOLOX-ByteTrack模型转换

目前测试支持的两种模型:
1. YOLOX的官方模型,转换方法参考YOLOX的模型转换[文档](yolox_model_cvt.md),对应配置文件:[config/yolox_bytetrack_demo.json](config/yolox_bytetrack_demo.json)
2. ByteTrack仓库的模型,只支持行人检测,对应配置文件:[config/yolox_bytetrack_pedestrian_demo.jsonn](config/yolox_bytetrack_pedestrian_demo.json)

## ByteTrack模型转换

1. 根据[官方仓库](https://github.com/ifzhang/ByteTrack.git)指导在Host上下载安装ByteTrack
```
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack/

pip3 install -r requirements.txt
python3 setup.py develop
pip3 install wheel cython
pip3 install cython_bbox
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
2. 下载ByteTrack模型bytetrack_s_mot17 [[google]](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing), [[baidu(code:qflm)]](https://pan.baidu.com/s/1PiP1kQfgxAIrnGUbFP6Wfg),生成onnx
```
python3 tools/export_onnx.py --output-name bytetrack_s.onnx -f exps/example/mot/yolox_s_mix_det.py --opset=11 -c bytetrack_s_mot17.pth.tar
```
3. 修改onnx
   * 将[model/yolov5_remove_nodes.py](model/yolov5_remove_nodes.py)拷贝到前一步得到的onnx目录下
   * 执行以下命令，将YOLOX的focus层的slice和concate算子删除
```
python3 yolov5_remove_nodes.py bytetrack_s.onnx -o bytetrack_s_truncate.onnx
```
4. 使用atc转换模型
```
atc --mode=0 --model bytetrack_s_truncate.onnx --framework=5 --output=bytetrack_s_truncate --soc_version=Ascend310
```
5. 将得到的bytetrack_s_truncate.om放到model文件夹备用