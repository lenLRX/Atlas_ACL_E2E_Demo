#ifndef __CAMERA_INPUT_H__
#define __CAMERA_INPUT_H__

#include <functional>
#include <iostream>
#include <string>

#include "acl_model.h"

class CameraInput {
public:
  CameraInput() = default;
  ~CameraInput();
  // 720P@20fps
  int Init(int id);
  void RegisterHandler(std::function<void(DeviceBufferPtr)> handler);
  void Run();
  int GetHeight();
  int GetWidth();
  int GetFPS();

private:
  std::function<void(DeviceBufferPtr)> buffer_handler;
  int camera_id;
  int cam_buffer_size;
  int height;
  int width;
  int fps;
};

#endif //__CAMERA_INPUT_H__