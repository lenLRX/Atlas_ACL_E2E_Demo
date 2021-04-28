# demo使用说明

## 运行demo方式
执行一下命令运行demo
```
./run.sh config/yolov3_demo.json
```
按下ctrl+c可以停止demo。

其中```config/yolov3_demo.json```是配置文件

## 配置文件格式
下面是一个配置文件的例子:

```json
{
    "streams": [
        {
            "name": "yolov3_demo_stream1",
            "stream_type": "yolov3_demo",
            "src": "road_traffic_test1.mp4",
            "dst": "output1.mp4",
            "hw_encoder": false,
            "model_path": "./model/sample-yolov3_pp_416.om"
        },
        {
            "name": "yolov3_demo_stream2",
            "stream_type": "yolov3_demo",
            "src": "road_traffic_test2.mp4",
            "dst": "output2.mp4",
            "hw_encoder": true,
            "model_path": "./model/sample-yolov3_pp_416.om"
        }
    ],
    "config": {
        "app_perf": true,
        "perflog_path": "."
    }
}
```
配置文件说明:
* stream : 要处理的流
  * name: 流的名字
  * stream_type: 流的类型，现在支持```yolov3_demo```和```deep_sort_demo```
  * src: 输入路径,可以是摄像头，网络地址或者文件名
    * 摄像头: ```camera0```或```camera1```
    * 网络地址: 如```rtsp://192.168.1.9:8554/tt1.mp4```
    * 本地文件: ```output2.mp4```
  * dst: 输出路径
  * hw_encoder: 是否开启硬件编码。注意:*一个进程中只能有一个流开启硬件编码*
  * model_path: 模型的路径
* config: 通用配置项
  * app_perf: 是否输出性能日志
  * perflog_path: 性能日志输出路径
