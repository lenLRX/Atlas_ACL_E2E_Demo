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
由于转换模型流程太过复杂且这个模型本身不大，所以仓库中直接提供了二进制的模型.

如果你要自己转换的话,以下是转换步骤(待细化):
1. 按照[YOLOV3模型转换文档](https://github.com/lenLRX/Atlas200DK_ACL#%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2)中的方式配置好atc的环境变量
2. 克隆原作者的[仓库](https://github.com/nwojke/deep_sort)
3. 从谷歌网盘下载作者提供的[模型文件](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)
4. 将deepsort仓库中的tools/freeze_model.py替换为我修改的这个[文件](https://github.com/lenLRX/deep_sort/blob/master/tools/freeze_model.py),按照作者提供的方法重新freeze一次模型(需要单独配置一个tensorflow1.15的环境干这个事情)。
5. 使用atc转换模型(待细化)

## 编译和运行
编译使用以下命令:
```
./build_deepsort.sh
```
运行请参考run_deep_sort.sh, 将输入输出替换为实际的输入输出。