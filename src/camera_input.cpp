#include "camera_input.h"
#include "app_profiler.h"
#include "util.h"
#include "dev_mem_pool.h"

#ifdef HAS_CAMERA
extern "C" {
#include "peripheral_api.h"
}

static std::string StatusToStr(CameraStatus status) {
  switch (status) {
  case CAMERA_STATUS_OPEN:
    return "CAMERA_STATUS_OPEN";
    break;
  case CAMERA_STATUS_CLOSED:
    return "CAMERA_STATUS_CLOSED";
    break;
  case CAMERA_NOT_EXISTS:
    return "CAMERA_NOT_EXISTS";
    break;
  case CAMERA_STATUS_UNKOWN:
    return "CAMERA_STATUS_UNKOWN";
    break;
  }
  return "UNKNOWN";
}

int InitMediaLib() {
  static int media_lib_status = MediaLibInit();
  return media_lib_status;
}

CameraInput::~CameraInput() {}

int CameraInput::Init(int id) {
  camera_id = id;
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

  CameraStatus status = QueryCameraStatus(camera_id);

  if (status != CAMERA_STATUS_CLOSED) {
    std::cerr << "Check CameraStatus Failed reason: " << StatusToStr(status)
              << std::endl;
    return -1;
  }

  ret = OpenCamera(camera_id);

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "OpenCamera " << camera_id << " failed" << std::endl;
    return -1;
  }

  fps = 20;
  ret = SetCameraProperty(camera_id, CAMERA_PROP_FPS, &fps);
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty CAMERA_PROP_FPS " << camera_id << " failed "
              << ret << std::endl;
    return -1;
  }

  struct CameraResolution resolution;

  height = resolution.height = 720;
  width = resolution.width = 1280;
  ret = SetCameraProperty(camera_id, CAMERA_PROP_RESOLUTION, &resolution);
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty CAMERA_PROP_RESOLUTION " << camera_id
              << " failed " << ret << std::endl;
    return -1;
  }

  cam_buffer_size = yuv420sp_size(height, width);

  CameraCapMode cap_mode = CAMERA_CAP_ACTIVE;

  ret = SetCameraProperty(camera_id, CAMERA_PROP_CAP_MODE, &cap_mode);

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty CAMERA_PROP_CAP_MODE " << camera_id
              << " failed " << ret << std::endl;
    return -1;
  }
  return 0;
}

void CameraInput::Run() {
  while (running) {
    APP_PROFILE(CameraInput::Run);
    // uint8_t *camera_buffer = (uint8_t *)malloc(cam_buffer_size);
    void *camera_buffer = DevMemPool::AllocDvppMem(cam_buffer_size);
    int ret = ReadFrameFromCamera(camera_id, camera_buffer, &cam_buffer_size);
    if (ret != LIBMEDIA_STATUS_OK) {
      std::cerr << "ReadFrameFromCamera failed, camera id: " << camera_id
                << std::endl;
      free(camera_buffer);
      return;
    }
    auto dev_cam_buffer = std::make_shared<DeviceBuffer>(
        camera_buffer, cam_buffer_size, DeviceBuffer::DvppMemDeleter());
    output_queue->push(dev_cam_buffer);
  }
}

#else

CameraInput::~CameraInput() {}

int CameraInput::Init(int id) {
  throw std::runtime_error("Only Atlas200DK support camera!");
}

void CameraInput::Run() {
  throw std::runtime_error("Only Atlas200DK support camera!");
}

#endif

void CameraInput::SetOutputQueue(
    ThreadSafeQueueWithCapacity<DeviceBufferPtr> *queue) {
  output_queue = queue;
}

int CameraInput::GetHeight() { return height; }

int CameraInput::GetWidth() { return width; }

int CameraInput::GetFPS() { return fps; }

void CameraInput::Stop() { running = false; }
