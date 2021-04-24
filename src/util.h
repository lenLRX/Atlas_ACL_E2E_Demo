#ifndef __ACL_UTIL_H__
#define __ACL_UTIL_H__

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <Python.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace std::chrono;

#define AVC1_TAG 0x31637661

#define CHECK_ACL(x)                                                           \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
    }                                                                          \
  } while (0);

class PerfTimer {
public:
  PerfTimer(const char *file, int line, const char *func) {
    file_ = file;
    line_ = line;
    func_ = func;
    start = steady_clock::now();
  }
  ~PerfTimer() {
    auto end = steady_clock::now();
    auto duration = end - start;
    microseconds duration_us = duration_cast<microseconds>(duration);

    std::stringstream ss;
    ss << file_ << ":" << line_ << " func:" << func_
       << " duration:" << duration_us.count() / 1000.f << "ms";
    std::cerr << ss.str() << std::endl;
  }

private:
  std::chrono::steady_clock::time_point start;
  const char *file_;
  int line_;
  const char *func_;
};

#define _CONCAT_(x, y) x##y
#define __CONCAT__(x, y) _CONCAT_(x, y)

#define PERF_TIMER()                                                           \
  auto __CONCAT__(temp_perf_obj_, __LINE__) =                                  \
      PerfTimer(__FILE__, __LINE__, __FUNCTION__)

const static std::vector<std::string> yolov3_label = {"background",
                                                      "person",
                                                      "bicycle",
                                                      "car",
                                                      "motorbike",
                                                      "aeroplane",
                                                      "bus",
                                                      "train",
                                                      "truck",
                                                      "boat",
                                                      "traffic light",
                                                      "fire hydrant",
                                                      "stop sign",
                                                      "parking meter",
                                                      "bench",
                                                      "bird",
                                                      "cat",
                                                      "dog",
                                                      "horse",
                                                      "sheep",
                                                      "cow",
                                                      "elephant",
                                                      "bear",
                                                      "zebra",
                                                      "giraffe",
                                                      "backpack",
                                                      "umbrella",
                                                      "handbag",
                                                      "tie",
                                                      "suitcase",
                                                      "frisbee",
                                                      "skis",
                                                      "snowboard",
                                                      "sports ball",
                                                      "kite",
                                                      "baseball bat",
                                                      "baseball glove",
                                                      "skateboard",
                                                      "surfboard",
                                                      "tennis racket",
                                                      "bottle",
                                                      "wine glass",
                                                      "cup",
                                                      "fork",
                                                      "knife",
                                                      "spoon",
                                                      "bowl",
                                                      "banana",
                                                      "apple",
                                                      "sandwich",
                                                      "orange",
                                                      "broccoli",
                                                      "carrot",
                                                      "hot dog",
                                                      "pizza",
                                                      "donut",
                                                      "cake",
                                                      "chair",
                                                      "sofa",
                                                      "potted plant",
                                                      "bed",
                                                      "dining table",
                                                      "toilet",
                                                      "TV monitor",
                                                      "laptop",
                                                      "mouse",
                                                      "remote",
                                                      "keyboard",
                                                      "cell phone",
                                                      "microwave",
                                                      "oven",
                                                      "toaster",
                                                      "sink",
                                                      "refrigerator",
                                                      "book",
                                                      "clock",
                                                      "vase",
                                                      "scissors",
                                                      "teddy bear",
                                                      "hair drier",
                                                      "toothbrush"};

const static std::vector<std::string> yolov3_label_zh_cn{
    u8"背景",     u8"人",           u8"自行车",
    u8"汽车",     u8"摩托车",       u8"飞机",
    u8"公交车",   u8"火车",         u8"卡车",
    u8"船",       u8"红绿灯",       u8"消防栓",
    u8"停止标志", u8"停车收费表",   u8"长椅",
    u8"鸟",       u8"猫",           u8"狗",
    u8"马",       u8"羊",           u8"牛",
    u8"象",       u8"熊",           u8"斑马",
    u8"长颈鹿",   u8"背包",         u8"雨伞",
    u8"手提包",   u8"领带",         u8"手提箱",
    u8"飞盘",     u8"滑雪板(skis)", u8"滑雪板(snowboard)",
    u8"运动球",   u8"风筝",         u8"棒球棒",
    u8"棒球手套", u8"滑板",         u8"冲浪板",
    u8"网球拍",   u8"瓶子",         u8"红酒杯",
    u8"杯子",     u8"叉子",         u8"刀",
    u8"勺子",     u8"碗",           u8"香蕉",
    u8"苹果",     u8"三明治",       u8"橘子",
    u8"西兰花",   u8"萝卜",         u8"热狗",
    u8"批萨",     u8"甜甜圈",       u8"蛋糕",
    u8"椅子",     u8"沙发",         u8"盆栽",
    u8"床",       u8"餐桌",         u8"厕所",
    u8"电视机",   u8"笔记本电脑",   u8"鼠标",
    u8"remote",   u8"键盘",         u8"手机",
    u8"微波炉",   u8"烤箱",         u8"烤面包机",
    u8"sink",     u8"冰箱",         u8"书",
    u8"钟",       u8"花瓶",         u8"剪刀",
    u8"泰迪熊",   u8"吹风机",       u8"牙刷"};

static int align_up(int size, int align) {
  return (size + (align - 1)) / align * align;
}

static int yuv420sp_size(int h, int w) { return (h * w * 3) / 2; }

static acldvppStreamFormat
h264_ffmpeg_profile_to_acl_stream_fromat(int profile) {
  switch (profile) {
  case 77: // h264 main level
    return H264_MAIN_LEVEL;
  case 66: // h264 baseline level
    return H264_BASELINE_LEVEL;
  case 100: // h264 high level
    return H264_HIGH_LEVEL;
  }
}

static int ParseCameraInput(const std::string &addr) {
  // Since Atlas200DK only support camera 0 and 1
  // we don't have to parse camera id
  if (addr == "camera0") {
    return 0;
  } else if (addr == "camera1") {
    return 1;
  }
  return -1;
}

static bool IsDeviceMode() {
  aclrtRunMode mode;
  CHECK_ACL(aclrtGetRunMode(&mode));
  return mode == ACL_DEVICE;
}

// from
// https://stackoverflow.com/questions/12805041/c-equivalent-to-javas-blockingqueue
template <typename T> class ThreadSafeQueue {
private:
  std::mutex d_mutex;
  std::condition_variable d_condition;
  std::deque<T> d_queue;

public:
  void push(T const &value) {
    {
      std::unique_lock<std::mutex> lock(this->d_mutex);
      d_queue.push_front(value);
    }
    this->d_condition.notify_one();
  }
  T pop() {
    std::unique_lock<std::mutex> lock(this->d_mutex);
    this->d_condition.wait(lock, [=] { return !this->d_queue.empty(); });
    T rc(std::move(this->d_queue.back()));
    this->d_queue.pop_back();
    return rc;
  }
};

template <typename T> class ThreadSafeQueueWithCapacity {
private:
  std::mutex d_mutex;
  std::condition_variable pop_cond;
  std::condition_variable full_cond;
  std::deque<T> d_queue;
  bool shutdowned{false};
  int capacity;

public:
  ThreadSafeQueueWithCapacity(int cap) : capacity(cap) {}
  void push(T const &value) {
    {
      std::unique_lock<std::mutex> lock(d_mutex);
      full_cond.wait(lock, [=] { return this->d_queue.size() < capacity; });
      d_queue.push_front(value);
    }
    pop_cond.notify_one();
  }

  void ShutDown() {
    {
      std::unique_lock<std::mutex> lock(d_mutex);
      shutdowned = true;
    }
    pop_cond.notify_all();
  }

  bool pop(T &item) {
    std::unique_lock<std::mutex> lock(d_mutex);
    while (true) {
      if (d_queue.empty()) {
        if (shutdowned) {
          return false;
        }
      } else {
        break;
      }
      pop_cond.wait(lock);
    }
    item = std::move(d_queue.back());
    d_queue.pop_back();
    full_cond.notify_one();
    return true;
  }
};

// https://docs.python.org/3/c-api/init.html#non-python-created-threads
class PyGILGuard {
public:
  PyGILGuard() {
    GetLock().lock();
    gstate = PyGILState_Ensure();
  }
  ~PyGILGuard() {
    PyGILState_Release(gstate);
    GetLock().unlock();
  }

  static std::mutex &GetLock() {
    static std::mutex mtx;
    return mtx;
  }

private:
  PyGILState_STATE gstate;
};

#endif //__ACL_UTIL_H__