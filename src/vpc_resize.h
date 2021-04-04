#ifndef __VPC_RESIZE_H__
#define __VPC_RESIZE_H__
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include "util.h"
#include "acl_model.h"

#include <functional>

class VPCResizeEngine {
public:
  typedef std::function<void(DeviceBufferPtr, DeviceBufferPtr)> CallBack;
  VPCResizeEngine(aclrtStream stream);
  ~VPCResizeEngine();
  void Destory();

  aclError Init(int src_h, int src_w, int dst_h, int dst_w);

  int GetOutputBufferSize();

  aclError Resize(DeviceBufferPtr pdata);
  void RegisterHandler(CallBack handler);

private:
  acldvppChannelDesc *channel_desc;
  acldvppResizeConfig *resize_config;
  aclrtStream stream;

  int output_buffer_size;
  int input_buffer_size;

  int src_h;
  int src_w;
  int dst_h;
  int dst_w;

  CallBack buffer_handler;
};

#endif //__VPC_RESIZE_H__