#include "vpc_batch_crop.h"
#include "util.h"
#include <string.h>

static int vpc_crop_batch_size = 128;

VPCBatchCrop::VPCBatchCrop(aclrtStream stream) : stream(stream) {
  channel_desc = acldvppCreateChannelDesc();
  crop_areas = new acldvppRoiConfig *[vpc_crop_batch_size];
  for (int i = 0; i < vpc_crop_batch_size; ++i) {
    crop_areas[i] = acldvppCreateRoiConfig(0, 1, 0, 1);
  }
}

VPCBatchCrop::~VPCBatchCrop() {
  for (int i = 0; i < vpc_crop_batch_size; ++i) {
    acldvppDestroyRoiConfig(crop_areas[i]);
  }
  delete[] crop_areas;
  acldvppDestroyChannelDesc(channel_desc);
}

aclError VPCBatchCrop::Init(int src_h, int src_w) {
  CHECK_ACL(acldvppCreateChannel(channel_desc));
  input_buffer_size = yuv420sp_size(align_up(src_h, 2), align_up(src_w, 16));
  src_w = src_w;
  src_h = src_h;
  src_w_stride = align_up(src_w, 16);
  src_h_stride = align_up(src_h, 2);

  /*
  acldvppPicDesc *input_desc = acldvppGetPicDesc(src_batch_desc, 0);

  CHECK_ACL(acldvppSetPicDescData(input_desc, dvpp_input_mem));
  CHECK_ACL(
      acldvppSetPicDescFormat(input_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(acldvppSetPicDescWidth(input_desc, src_w));
  CHECK_ACL(acldvppSetPicDescHeight(input_desc, src_h));
  CHECK_ACL(acldvppSetPicDescWidthStride(input_desc, align_up(src_w, 16)));
  CHECK_ACL(acldvppSetPicDescHeightStride(input_desc, align_up(src_h, 2)));
  CHECK_ACL(acldvppSetPicDescSize(input_desc, input_buffer_size));
  */
}

aclError VPCBatchCrop::Crop(uint8_t *src_buffer, int *x1, int *y1, int *x2,
                            int *y2, uint8_t **dst_array, int *dst_h,
                            int *dst_w, int num) {
  int batch_num = num / vpc_crop_batch_size;
  int tail_size = num % vpc_crop_batch_size;
  int offset = 0;
  for (int i = 0; i < batch_num; ++i) {
    CHECK_ACL(CropBatch(src_buffer, x1 + offset, y1 + offset, x2 + offset,
                        y2 + offset, dst_array + offset, dst_h + offset,
                        dst_w + offset, vpc_crop_batch_size));
    offset += vpc_crop_batch_size;
  }

  if (tail_size > 0) {
    CHECK_ACL(CropBatch(src_buffer, x1 + offset, y1 + offset, x2 + offset,
                        y2 + offset, dst_array + offset, dst_h + offset,
                        dst_w + offset, tail_size));
  }
  return ACL_ERROR_NONE;
}

aclError VPCBatchCrop::CropBatch(uint8_t *src_buffer, int *x1, int *y1, int *x2,
                                 int *y2, uint8_t **dst_array, int *dst_h,
                                 int *dst_w, int num) {
  uint32_t roi_nums = num;
  acldvppBatchPicDesc *src_batch_desc = acldvppCreateBatchPicDesc(1);

  acldvppPicDesc *input_desc = acldvppGetPicDesc(src_batch_desc, 0);

  CHECK_ACL(acldvppSetPicDescData(input_desc, src_buffer));
  CHECK_ACL(
      acldvppSetPicDescFormat(input_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(acldvppSetPicDescWidth(input_desc, src_w));
  CHECK_ACL(acldvppSetPicDescHeight(input_desc, src_h));
  CHECK_ACL(acldvppSetPicDescWidthStride(input_desc, src_w_stride));
  CHECK_ACL(acldvppSetPicDescHeightStride(input_desc, src_h_stride));
  CHECK_ACL(acldvppSetPicDescSize(input_desc, input_buffer_size));

  acldvppBatchPicDesc *dst_batch_desc = acldvppCreateBatchPicDesc(num);
  for (int i = 0; i < num; ++i) {
    CHECK_ACL(SetPicDesc(acldvppGetPicDesc(dst_batch_desc, i), dst_h[i],
                         dst_w[i], dst_array[i]));
    // std::cout << "x1: " << x1[i] << " x2: " << x2[i]
    //  << " y1: " << y1[i] << " y2: " << y2[i] << std::endl;
    // std::cout << "dst_h: " << dst_h[i] << " dst_w: " << dst_w[i] <<
    // std::endl;
    CHECK_ACL(acldvppSetRoiConfig(crop_areas[i], x1[i], x2[i], y1[i], y2[i]));
    // std::cout << "CropBatch input pix:" << ((uint8_t*)dvpp_input_mem)[0] <<
    // std::endl;
  }

  CHECK_ACL(acldvppVpcBatchCropAsync(channel_desc, src_batch_desc, &roi_nums, 1,
                                     dst_batch_desc, crop_areas, stream));
  CHECK_ACL(aclrtSynchronizeStream(stream));
  /*
  for (int i = 0; i < num; ++i) {
    acldvppPicDesc *pic = acldvppGetPicDesc(dst_batch_desc, i);
    uint32_t dvpp_size = acldvppGetPicDescSize(pic);
    // std::cout << "dvpp size: " << dvpp_size << std::endl;
    void *dvpp_data = acldvppGetPicDescData(pic);
    // std::cout << "CropBatch output pix:" << ((uint8_t*)dvpp_data)[0] <<
    // std::endl;
    memcpy(dst_array[i], dvpp_data, dvpp_size);
    acldvppFree(dvpp_data);
  }
  */

  acldvppDestroyBatchPicDesc(dst_batch_desc);
  acldvppDestroyBatchPicDesc(src_batch_desc);

  return ACL_ERROR_NONE;
}

aclError VPCBatchCrop::SetPicDesc(acldvppPicDesc *pic_desc, int h, int w,
                                  uint8_t *buffer) {
  int buffer_size = yuv420sp_size(align_up(h, 2), align_up(w, 16));

  /*
  void *dvpp_output;
  acldvppMalloc(&dvpp_output, buffer_size);

  CHECK_ACL(acldvppSetPicDescData(pic_desc, dvpp_output));
  */
  CHECK_ACL(acldvppSetPicDescData(pic_desc, buffer));
  CHECK_ACL(acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(acldvppSetPicDescWidth(pic_desc, w));
  CHECK_ACL(acldvppSetPicDescHeight(pic_desc, h));
  CHECK_ACL(acldvppSetPicDescWidthStride(pic_desc, align_up(w, 16)));
  CHECK_ACL(acldvppSetPicDescHeightStride(pic_desc, align_up(h, 2)));
  CHECK_ACL(acldvppSetPicDescSize(pic_desc, buffer_size));
  return ACL_ERROR_NONE;
}

aclrtStream VPCBatchCrop::GetStream() { return stream; }
