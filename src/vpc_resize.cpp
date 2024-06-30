#include "vpc_resize.h"
#include "app_profiler.h"
#include "util.h"

#include <string.h>

VPCResizeEngine::VPCResizeEngine(aclrtStream stream) : stream(stream) {
  channel_desc = acldvppCreateChannelDesc();
  resize_config = acldvppCreateResizeConfig();
  acldvppSetResizeConfigInterpolation(resize_config, 0);
}

aclError VPCResizeEngine::Init(int input_h, int input_w, int output_h,
                               int output_w) {
  src_h = input_h;
  src_w = input_w;
  dst_h = output_h;
  dst_w = output_w;

  CHECK_ACL(acldvppCreateChannel(channel_desc));
  input_buffer_size = yuv420sp_size(align_up(src_h, 2), align_up(src_w, 16));
  output_buffer_size = yuv420sp_size(dst_h, dst_w);
  return ACL_ERROR_NONE;
}

VPCResizeEngine::~VPCResizeEngine() {}

void VPCResizeEngine::Destory() {
  acldvppDestroyChannel(channel_desc);
  // std::cout << "VPCResizeEngine::~VPCResizeEngine End" << std::endl;
  // TODO: other clean up
}

VPCResizeEngine::OutTy VPCResizeEngine::Process(DeviceBufferPtr pdata) {
  APP_PROFILE(VPCResizeEngine);
  acldvppPicDesc *input_desc = acldvppCreatePicDesc();
  acldvppPicDesc *output_desc = acldvppCreatePicDesc();

  void *dvpp_mem = DevMemPool::AllocDvppMem(output_buffer_size);

  DeviceBufferPtr resized_data = std::make_shared<DeviceBuffer>(
      dvpp_mem, output_buffer_size, DeviceBuffer::DvppMemDeleter());

  CHECK_ACL(acldvppSetPicDescData(input_desc, pdata->GetDevicePtr()));
  CHECK_ACL(
      acldvppSetPicDescFormat(input_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(acldvppSetPicDescWidth(input_desc, src_w));
  CHECK_ACL(acldvppSetPicDescHeight(input_desc, src_h));
  CHECK_ACL(acldvppSetPicDescWidthStride(input_desc, align_up(src_w, 16)));
  CHECK_ACL(acldvppSetPicDescHeightStride(input_desc, align_up(src_h, 2)));
  CHECK_ACL(acldvppSetPicDescSize(input_desc, input_buffer_size));

  CHECK_ACL(acldvppSetPicDescData(output_desc, resized_data->GetDevicePtr()));
  CHECK_ACL(
      acldvppSetPicDescFormat(output_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(acldvppSetPicDescWidth(output_desc, dst_w));
  CHECK_ACL(acldvppSetPicDescHeight(output_desc, dst_h));
  CHECK_ACL(acldvppSetPicDescWidthStride(output_desc, dst_w));
  CHECK_ACL(acldvppSetPicDescHeightStride(output_desc, dst_h));
  CHECK_ACL(acldvppSetPicDescSize(output_desc, output_buffer_size));

  CHECK_ACL(acldvppVpcResizeAsync(channel_desc, input_desc, output_desc,
                                  resize_config, stream));
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(acldvppDestroyPicDesc(input_desc));
  CHECK_ACL(acldvppDestroyPicDesc(output_desc));

  return {pdata, resized_data};
}

int VPCResizeEngine::GetOutputBufferSize() { return output_buffer_size; }
