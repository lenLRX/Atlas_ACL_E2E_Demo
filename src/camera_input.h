#ifndef __CAMERA_INPUT_H__
#define __CAMERA_INPUT_H__

#include <atomic>
#include <iostream>
#include <string>

#include "acl_model.h"
#include "util.h"

class CameraInput {
public:
  CameraInput() = default;
  ~CameraInput();
  // 720P@20fps
  int Init(int id);
  void Run();
  void Process() { Run(); }
  int GetHeight();
  int GetWidth();
  int GetFPS();
  void SetOutputQueue(ThreadSafeQueueWithCapacity<DeviceBufferPtr> *queue);
  void Stop();
private:
  int camera_id;
  int cam_buffer_size;
  int height;
  int width;
  int fps;
  ThreadSafeQueueWithCapacity<DeviceBufferPtr> *output_queue{nullptr};
  std::atomic<bool> running{true};
};

#endif //__CAMERA_INPUT_H__