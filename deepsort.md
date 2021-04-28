# DeepSORT + YOLOV3 模型示例

原作者代码仓库[地址](https://github.com/nwojke/deep_sort)

这只是一个串通流程的demo, python部分性能并不好。后续可以使用隔帧检测或使用C++甚至TIK加速Tracker部分代码。

测试的时候可以使用输入输出MP4的方式查看效果，使用视频串流可能顿卡。

![deepsort demo](deepsort_demo.gif)

## 编译和运行
运行请参考run_deep_sort.sh, 将输入输出替换为实际的输入输出。

json配置文件中stream_type属性为"deep_sort_demo"

运行使用以下命令:
```
./run.sh config/deep_sort_demo.json
```
