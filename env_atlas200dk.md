# 环境搭建
假设此时你刚刚完成制卡，Atlas200DK刚刚启动。。

以下步骤可以根据你已经配置好的环境跳过一些步骤。
## 配置SSH root登录（可选）
首先使用HwHiAiUser登录Atlas200DK,密码默认为Mind@123. root的默认密码与HwHiAiUser相同
```bash
su root
vi /etc/ssh/sshd_config
```
将PermitRootLogin这一行改成PermitRootLogin yes,注意去掉行首的#
然后保存退出，重启sshd服务
```
service sshd restart
```
## 配置apt源
用root用户修改/etc/apt/sources.list
以清华tuna为例：
```
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe
```
## 配置DNS
假设这里使用阿里的DNS 223.5.5.5

以下操作在root用户下执行.

首先修改/etc/resolv.conf, 加入一行：
```
nameserver 223.5.5.5
```
然后更新apt源，安装resolvconf
```
apt update
apt install resolvconf
```
修改配置文件/etc/resolvconf/resolv.conf.d/head, 加入一行：
```
nameserver 223.5.5.5
```
最后重启服务:
```
service resolvconf restart
```

## 安装nginx和nginx-http-flv-module
1. 将代码下载并解压到/root目录下，这里使用的代码为:[nginx](https://nginx.org/download/nginx-1.18.0.tar.gz), [nginx-http-flv-module](https://github.com/winshining/nginx-http-flv-module/archive/v1.2.8.tar.gz)
2. 安装依赖: ```apt install build-essential libpcre3 libpcre3-dev zlib1g-dev openssl libssl-dev ```
3. 编译并安装:
```
cd nginx-1.18.0
./configure --add-module=../nginx-http-flv-module-1.2.8 
make -j`nproc`
make install
```
4. 修改配置文件,在/usr/local/nginx/conf/nginx.conf最后加上以下一段：
```
rtmp {
    server {
        application myapp {
            live on;
        }
    }
}
```
5. 启动nginx(如果希望nginx开启自启动需要额外配置)
```
cd /usr/local/nginx/sbin/
./nginx
```

## 安装acllib
### 20.0版本
将Ascend-acllib-1.73.5.1.b050-ubuntu18.04.aarch64-minirc.run传到/root目录下，执行以下命令安装：
```
chmod +x Ascend-acllib-1.73.5.1.b050-ubuntu18.04.aarch64-minirc.run
./Ascend-acllib-1.73.5.1.b050-ubuntu18.04.aarch64-minirc.run --full
```
### 20.1版本
以下操作使用root用户执行
1. 解压制卡时用到的Ascend-cann-minirc_20.1.rc1_ubuntu18.04-aarch64.zip，将压缩包中的Ascend-acllib-1.75.22.0.220-ubuntu18.04.aarch64-minirc.run文件上传到/root
2. 安装这个run包
```
chmod +x Ascend-acllib-1.75.22.0.220-ubuntu18.04.aarch64-minirc.run
./Ascend-acllib-1.75.22.0.220-ubuntu18.04.aarch64-minirc.run --full
```
实际上制卡时已经安装了这个run包，但是没有使用--full参数，导致没有安装头文件。(有必要节省那一点点空间吗？)

以下操作使用root用户执行，这是为了修复官方的一个BUG。
```
chmod u+w /etc/sudoers
vi /etc/sudoers
```
将最后一行改为：
```
HwHiAiUser ALL=(root) NOPASSWD:/opt/mini/minirc_install_phase1.sh,/bin/date -s *,/var/ide_cmd.sh *,/bin/sed -i * /etc/network/interfaces,/usr/bin/perf stat *,/usr/bin/perf record *,/usr/bin/perf script *,/usr/bin/pkill -2 perf,/var/tsdaemon_add_to_usermemory.sh
```
(加上了",/var/tsdaemon_add_to_usermemory.sh"这一段)

然后修改/var/tsdaemon_add_to_usermemory.sh文件，将echo那一行改为
```
echo $1 > /sys/fs/cgroup/memory/cgroup.procs
```
### 20.2(3.2.0)版本
以下操作使用root用户执行

将Ascend-cann-nnrt_20.2.rc1_linux-aarch64.run传到/root目录下,首先解压:
```
chmod +x Ascend-cann-nnrt_20.2.rc1_linux-aarch64.run
./Ascend-cann-nnrt_20.2.rc1_linux-aarch64.run --extract=. --noexec
```
然后安装acllib
```
cd run_package/
./Ascend-acllib-1.76.22.3.220-linux.aarch64.run --full
```
### 3.3.0.alpha001版本
以下操作使用root用户执行

将Ascend-cann-nnrt_3.3.0.alpha001_linux-aarch64.run传到/root目录下,首先解压:
```
chmod +x Ascend-cann-nnrt_3.3.0.alpha001_linux-aarch64.run
./Ascend-cann-nnrt_3.3.0.alpha001_linux-aarch64.run --extract=. --noexec
```
这里可能会报一个错，但是不影响，请忽略:
```
[NNRT] [20210313-05:51:01] [ERROR] Unsupported parameters: --keep
```
然后安装acllib
```
cd run_package/
./Ascend-acllib-1.77.t21.0.b210-linux.aarch64.run --full
```
### 3.3.0.alpha006版本
以下操作使用root用户执行

将Ascend-cann-nnrt_3.3.0.alpha006_linux-aarch64.run传到/root目录下,首先解压:
```
chmod +x Ascend-cann-nnrt_3.3.0.alpha006_linux-aarch64.run
./Ascend-cann-nnrt_3.3.0.alpha006_linux-aarch64.run --extract=. --noexec
```
然后安装acllib
```
cd run_package/
./Ascend-acllib-1.78.t3.0.b030-linux.aarch64.run --full
```
## 安装demo
### 安装依赖
使用root用户安装以下依赖：
```
apt install cmake git libopencv-dev fonts-droid-fallback libfreetype6-dev libspdlog-dev nlohmann-json-dev python3-dev python3-sklearn python3-numpy python3-opencv
```
### 下载并编译
<b>以下操作使用HwHiAiUser账户</b>
```
git clone https://github.com/lenLRX/Atlas_ACL_E2E_Demo.git
cd Atlas_ACL_E2E_Demo
./build.sh
```
如果github下载速度太慢，可以使用gitte的镜像:
```
git clone https://gitee.com/lenlrx/Atlas_ACL_E2E_Demo.git
```
