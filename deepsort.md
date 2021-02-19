# DeepSORT + YOLOV3 模型示例

原作者代码仓库[地址](https://github.com/nwojke/deep_sort)

这只是一个串通流程的demo, python部分性能并不好。后续可以使用隔帧检测或使用C++甚至TIK加速Tracker部分代码。

![deepsort demo](deepsort_demo.gif)

## 环境搭建
首先搭建好[基础的环境](env.md)
### Python环境搭建
以下操作请使用 root用户操作
```bash
apt update
apt install python3-dev python3-sklearn python3-numpy python3-opencv
```

## 模型转换
**由于转换模型流程太过复杂且这个模型本身不大，所以仓库中直接提供了二进制的模型(model/deepsort_mars.om).**

如果你要自己转换的话,以下是转换步骤:
1. 按照[YOLOV3模型转换文档](https://github.com/lenLRX/Atlas200DK_ACL#%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2)中的方式配置好atc的环境变量
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
8. 将生成的deepsort_mars.om文件放到model/文件夹中

## 编译和运行
编译使用以下命令:
```
./build_deepsort.sh
```
运行请参考run_deep_sort.sh, 将输入输出替换为实际的输入输出。