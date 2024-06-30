# 环境搭建
假设此时你刚刚完成制卡，OrangePi刚刚启动。。

以下步骤可以根据你已经配置好的环境跳过一些步骤。

## 安装acllib
OrangePi镜像中预装了CANN，可以使用预装的默认版本

## 安装demo
### 安装依赖
使用root用户安装以下依赖：
```
apt install cmake git libopencv-dev fonts-droid-fallback libfreetype6-dev libspdlog-dev nlohmann-json3-dev python3-dev python3-sklearn python3-numpy python3-opencv python3-pip libeigen3-dev
```
使用root用户安装pytorch-cpu:
```
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pip --upgrade
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install torch torchvision
```

如果还需要使用bytetrack的python版本,还需要额外使用root用户安装以下依赖:
```
python3 -m pip install cython --upgrade
python3 -m pip install wheel scipy cython_bbox lap --upgrade
```

### 下载并编译
<b>以下操作使用HwHiAiUser账户</b>
```
git clone https://github.com/lenLRX/Atlas_ACL_E2E_Demo.git
cd Atlas_ACL_E2E_Demo
./build_orange_pi.sh
```
如果github下载速度太慢，可以使用gitte的镜像:
```
git clone https://gitee.com/lenlrx/Atlas_ACL_E2E_Demo.git
```

### 转换模型
使用```npu-smi info```查看NPU的型号，替换atc命令中soc_version参数
```
# Atlas200DK
atc ... --soc_version=Ascend310
# OrangePi 20T 显示310B1
atc ... --soc_version=Ascend310B1
```
### 运行用例
使用run_orange_pi.sh替换run.sh

