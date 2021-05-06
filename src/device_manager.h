#ifndef __DEVICE_MANAGER_H__
#define __DEVICE_MANAGER_H__

#include <mutex>

#include "acl/acl.h"

class DeviceManager {
public:
  static aclrtContext AllocateCtx();

private:
  DeviceManager();
  static DeviceManager &GetInstance();
  uint32_t dev_count;
  uint32_t current_dev{0};
  std::mutex mtx;
};

#endif //__DEVICE_MANAGER_H__