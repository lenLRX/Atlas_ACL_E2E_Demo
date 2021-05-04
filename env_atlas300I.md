# 环境搭建
假设你使用的是华为云的ai1s实例:

<b>以下全部操作使用root用户操作</b>
## 配置apt源
执行以下命令配置镜像源
```bash
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
```
## 安装nginx和nginx-http-flv-module(可选)
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

## 安装依赖
```
apt install cmake git libopencv-dev fonts-droid-fallback libfreetype6-dev libspdlog-dev nlohmann-json-dev python3-dev python3-sklearn python3-numpy python3-opencv
```
## 安装demo
```
git clone https://github.com/lenLRX/Atlas_ACL_E2E_Demo.git
cd Atlas_ACL_E2E_Demo
./build.sh
```
如果github下载速度太慢，可以使用gitte的镜像:
```
git clone https://gitee.com/lenlrx/Atlas_ACL_E2E_Demo.git
```


