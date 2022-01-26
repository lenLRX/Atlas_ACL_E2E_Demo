# 环境搭建
假设你使用的是华为云的ai1s实例:

<b>以下全部操作使用root用户操作</b>
## 更新驱动+更新CANN版本+安装demo
将[安装脚本](script/ai1s_install_5.0.5.alpha001.sh)拷贝到ai1s实例的服务器上,直接运行即可自动下载安装。
```
chmod +x ai1s_install_5.0.5.alpha001.sh
./ai1s_install_5.0.5.alpha001.sh
```

## 安装nginx和nginx-http-flv-module(可选)
1. 将代码下载并解压到/root目录下，这里使用的代码为:[nginx](https://nginx.org/download/nginx-1.18.0.tar.gz), [nginx-http-flv-module](https://github.com/winshining/nginx-http-flv-module/archive/v1.2.8.tar.gz)
2. 编译并安装:
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



