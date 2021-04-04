#ifndef __ACL_MODEL_H__
#define __ACL_MODEL_H__

#include "acl/acl.h"

#include "util.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>

class DeviceBuffer {
public:
  using DeleteFn = std::function<void(void*)>;

  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(const DeviceBuffer&) = delete;

  DeviceBuffer(void* device_buffer, size_t buffer_size, DeleteFn delete_fn)
  :host_buffer(nullptr), device_buffer(device_buffer), buffer_size(buffer_size), delete_fn(delete_fn) {
    if (IsDeviceMode()) {
      host_buffer = device_buffer;
    }
  }

  void* GetHostPtr() {
    if (host_buffer == nullptr && !IsDeviceMode()) {
      // malloc and memcpy is delay to the first hit
      CHECK_ACL(aclrtMallocHost(&host_buffer, buffer_size));
      CHECK_ACL(aclrtMemcpy(host_buffer, buffer_size, device_buffer, buffer_size, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    return host_buffer;
  }

  void* GetDevicePtr() {
    return device_buffer;
  }

  ~DeviceBuffer() {
    if (!IsDeviceMode() && host_buffer != nullptr) {
      CHECK_ACL(aclrtFreeHost(host_buffer));
    }
    if (delete_fn) {
      delete_fn(device_buffer);
    }
  }

  static DeleteFn DevMemDeleter() {
    return [](void* dev_ptr) {
      CHECK_ACL(aclrtFree(dev_ptr));
    };
  }

  static DeleteFn DvppMemDeleter() {
    return [](void* dev_ptr) {
      CHECK_ACL(acldvppFree(dev_ptr));
    };
  }
private:
  void* host_buffer;
  void* device_buffer;
  size_t buffer_size;
  DeleteFn delete_fn;
};

using DeviceBufferPtr = std::shared_ptr<DeviceBuffer>;

class ACLModel {
public:
  using DevBufferVec = std::vector<DeviceBufferPtr>;
  ACLModel(aclrtStream stream);
  aclError Init(const char *model_path);
  DevBufferVec Infer(const DevBufferVec& inputs);
  const std::vector<size_t> &GetInputBufferSizes();
  const std::vector<size_t> &GetOutputBufferSizes();
  std::string ToString();
  //~ACLModel(); //TODO
private:
  std::string path;
  uint32_t model_id;
  void *model_mem;
  void *model_weight;
  bool loaded; // model load flag
  aclmdlDesc *model_desc;

  std::vector<size_t> input_buffer_sizes;
  std::vector<size_t> output_buffer_sizes;

  aclrtStream stream;
};

#endif //__ACL_MODEL_H__