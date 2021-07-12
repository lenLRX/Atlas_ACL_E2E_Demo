# Caffe Int8量化(YOLOv3)
1. 根据这个[仓库](https://github.com/lenLRX/CANN-toolkit-docker)的指导安装量化工具AMCT的caffe版本的docker镜像
2. 启动docker的AMCT容器，将本仓库映射到容器内
```bash
# 请把/data/Atlas_ACL_E2E_Demo替换为实际的路径
sudo docker run -it -v=/data/Atlas_ACL_E2E_Demo:/work amct-caffe:latest
```
3. 切换目录,并且在仓库根目录下执行命令生成int8量化模型
```bash
# 请将road_traffic_test1.mp4替换成实际需要校准的视频
# 命令行参考:
# usage: yolov3.py [-h] [--prototxt PROTOTXT] [--caffemodel CAFFEMODEL]
#                 [--tmp_dir TMP_DIR] [--output_dir OUTPUT_DIR]
#                 [--output_model_name OUTPUT_MODEL_NAME] #--calib_video
#                 CALIB_VIDEO [--calib_frame CALIB_FRAME]
cd /work
python3.7.5 quant/yolov3.py --prototxt model/yolov3_no_pp.prototxt --calib_video road_traffic_test1.mp4
```
4. 在仓库根目录下执行命令生成为int8量化模型的prototxt添加后处理算子
```bash
# 将前一步生成的prototxt复制一下
cp model/yolov3_int8_deploy_model.prototxt model/yolov3_int8_pp.prototxt
# 将后处理算子粘贴到生成的prototxt后面
cat quant/yolov3_post_only.prototxt >> model/yolov3_int8_pp.prototxt
```
5. 使用atc生成量化后的离线模型
```bash
cd model
atc --model=yolov3_int8_pp.prototxt --weight=yolov3_int8_deploy_weights.caffemodel --framework=0 --output=amct_yolov3_int8_pp_416 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_pp.cfg
```
6. 量化后的int8模型可以直接替换原有的YOLOv3模型

