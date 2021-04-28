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
  aclError CropBatch(uint8_t *src_buffer, int *x1, int *y1, int *x2, int *y2,
                     uint8_t **dst_array, int *dst_h, int *dst_w, int num);
  aclError SetPicDesc(acldvppPicDesc *pic_desc, int h, int w, uint8_t *buffer);
  acldvppChannelDesc *channel_desc;
  aclrtStream stream;

  int input_buffer_size;
  int src_w;
  int src_h;
  int src_w_stride;
  int src_h_stride;
  acldvppRoiConfig **crop_areas;
};

#endif //__VPC_BATCH_CROP_H__