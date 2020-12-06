#ifndef __ACL_UTIL_H__
#define __ACL_UTIL_H__

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <chrono>
#include <iostream>
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
       << " duration:" << duration_us.count() << "us";
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

#endif //__ACL_UTIL_H__