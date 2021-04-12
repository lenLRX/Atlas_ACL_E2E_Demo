#ifndef __VPC_BATCH_CROP_H__
#define __VPC_BATCH_CROP_H__
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include "util.h"

#include <functional>

class VPCBatchCrop {
public:
  VPCBatchCrop(aclrtStream stream);
  ~VPCBatchCrop();
  void Destory();

  aclError Init(int src_h, int src_w);
  aclError Crop(uint8_t *src_buffer, int *x1, int *y1, int *x2, int *y2,
                uint8_t **dst_array, int *dst_h, int *dst_w, int num);

  aclrtStream GetStream();

private:
  aclError CropBatch(int *x1, int *y1, int *x2, int *y2, uint8_t **dst_array,
                     int *dst_h, int *dst_w, int num);
  aclError SetPicDesc(acldvppPicDesc *pic_desc, int h, int w, uint8_t *buffer);
  acldvppChannelDesc *channel_desc;
  acldvppBatchPicDesc *src_batch_desc;
  aclrtStream stream;

  void *dvpp_input_mem;
  uint8_t *host_output_mem;
  int input_buffer_size;
  acldvppRoiConfig **crop_areas;
};

#endif //__VPC_BATCH_CROP_H__