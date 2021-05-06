#include "device_manager.h"
#include "util.h"

aclrtContext DeviceManager::AllocateCtx() {
  DeviceManager &manager = GetInstance();
  std::lock_guard<std::mutex> guard(manager.mtx);
  CHECK_ACL(aclrtSetDevice(manager.current_dev));
  aclrtContext ctx;
  CHECK_ACL(aclrtCreateContext(&ctx, manager.current_dev));
  manager.current_dev = (manager.current_dev + 1) % manager.dev_count;
  return ctx;
}

DeviceManager::DeviceManager() { CHECK_ACL(aclrtGetDeviceCount(&dev_count)); }

DeviceManager &DeviceManager::GetInstance() {
  static DeviceManager manager;
  return manager;
}
