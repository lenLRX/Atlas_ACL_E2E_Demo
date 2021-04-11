#ifndef __VPC_RESIZE_H__
#define __VPC_RESIZE_H__
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include "util.h"
#include "acl_model.h"

#include <functional>
#include <tuple>

class VPCResizeEngine {
public:
  using OutTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  VPCResizeEngine(aclrtStream stream);
  ~VPCResizeEngine();
  void Destory();

  aclError Init(int src_h, int src_w, int dst_h, int dst_w);

  int GetOutputBufferSize();

  OutTy Process(DeviceBufferPtr pdata);

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
};

#endif //__VPC_RESIZE_H__