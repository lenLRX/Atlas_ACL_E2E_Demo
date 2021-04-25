# DeepSort模型转换
**由于转换模型流程太过复杂且这个模型本身不大，所以仓库中直接提供了二进制的模型(model/deepsort_mars.om).**

如果你要自己转换的话,以下是转换步骤:
1. 按照[YOLOV3模型转换文档](yolov3_model_cvt.md)中的方式配置好atc的环境变量
2. 克隆原作者的[仓库](https://github.com/nwojke/deep_sort)
3. 从谷歌网盘下载作者提供的[模型文件](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)
4. 将下载的文件解压到deepsort仓库中, 保证resources/networks/目录下是checkpoint和pb文件
5. 将deepsort仓库中的tools/freeze_model.py替换为我修改的这个[文件](https://github.com/lenLRX/deep_sort/blob/master/tools/freeze_model.py)
6. 准备一个tensorflow1.15的环境, 执行以下命令重新freeze模型:
```
python tools/freeze_model.py --graphdef_out=freeze_mars.pb
```
7. 将freeze_mars.pb和model/aipp_deepsort_feature.cfg放到一个文件夹中, 使用以下命令转换atc转换模型.
```
atc --mode=0 --framework=3 --insert_op_conf=aipp_deepsort_feature.cfg --output=deepsort_mars --model=freeze_mars.pb --soc_version=Ascend310 --out_nodes="features:0" --input_shape="images:16,128,64,3"
```
8. 将生成的deepsort_mars.om文件放到```model/```文件夹中
