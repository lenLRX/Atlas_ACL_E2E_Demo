extern "C" {
#include "peripheral_api.h"
}

#include "camera_input.h"
#include "util.h"

int InitMediaLib() {
  static int media_lib_status = MediaLibInit();
  return media_lib_status;
}

CameraInput::~CameraInput() { free(camera_buffer); }

int CameraInput::Init(int camera_id) {
  if (camera_buffer) {
    throw std::runtime_error("camera_buffer already inited!");
  }
  int ret;
  ret = InitMediaLib();

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "MediaLibInit failed " << ret << std::endl;
    return -1;
  }

  ret = IsChipAlive(NULL);

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "IsChipAlive failed" << std::endl;
    return -1;
  }

  ret = OpenCamera(camera_id);

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "OpenCamera " << camera_id << " failed" << std::endl;
    return -1;
  }

  int fps = 20;
  ret = SetCameraProperty(0, CAMERA_PROP_FPS, &fps);
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty " << camera_id << " failed" << ret
              << std::endl;
    return -1;
  }

  struct CameraResolution resolution;

  height = resolution.height = 720;
  width = resolution.width = 1280;
  ret = SetCameraProperty(0, CAMERA_PROP_RESOLUTION, &resolution);
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty " << camera_id << " failed" << ret
              << std::endl;
    return -1;
  }

  cam_buffer_size = yuv420sp_size(720, 1280);

  camera_buffer = (uint8_t *)malloc(cam_buffer_size);

  CameraCapMode cap_mode = CAMERA_CAP_ACTIVE;

  ret = SetCameraProperty(0, CAMERA_PROP_CAP_MODE, &cap_mode);

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty " << camera_id << " failed " << ret
              << std::endl;
    ;
    return -1;
  }
  return 0;
}

void CameraInput::RegisterHandler(std::function<void(uint8_t *)> handler) {
  buffer_handler = handler;
}

void CameraInput::Run() {
  while (true) {
    int ret = ReadFrameFromCamera(0, camera_buffer, &cam_buffer_size);
    if (ret != LIBMEDIA_STATUS_OK) {
      std::cerr << "ReadFrameFromCamera failed" << std::endl;
      return;
    }
    buffer_handler(camera_buffer);
  }
}

int CameraInput::GetHeight() { return height; }

int CameraInput::GetWidth() { return width; }
