# 环境搭建
假设此时你刚刚完成制卡，Atlas200DK刚刚启动。。

以下步骤可以根据你已经配置好的环境跳过一些步骤。
## 配置SSH root登录（可选）
首先使用HwHiAiUser登录Atlas200DK,密码默认为Mind@123. root的默认密码与HwHiAiUser相同
```bash
su root
vi /etc/ssh/sshd_config
```
将PermetRootLogin这一行改成PermitRootLogin yes,注意去掉行首的#
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
将Ascend-acllib-1.73.5.1.b050-ubuntu18.04.aarch64-minirc.run传到/root目录下，执行以下命令安装：
```
chmod +x Ascend-acllib-1.73.5.1.b050-ubuntu18.04.aarch64-minirc.run
./Ascend-acllib-1.73.5.1.b050-ubuntu18.04.aarch64-minirc.run --full
```
## 安装demo
### 安装依赖
使用root用户安装以下依赖：
```
apt install cmake git libopencv-dev
```
### 下载并编译
<b>以下操作使用HwHiAiUser账户</b>
```
git clone https://github.com/lenLRX/Atlas200DK_ACL.git
cd Atlas200DK_ACL
./build.sh
```