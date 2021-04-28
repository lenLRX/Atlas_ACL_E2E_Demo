#include "jpeg_encode.h"
#include <fstream>
#include <iostream>

// TODO Thread safe
class JPEGChanContext {
public:
  static JPEGChanContext &GetInstance() {
    static JPEGChanContext ctx;
    return ctx;
  }

  acldvppChannelDesc *GetChanDesc() const { return channel_desc; }

private:
  JPEGChanContext() {
    channel_desc = acldvppCreateChannelDesc();
    acldvppCreateChannel(channel_desc);
  }

  ~JPEGChanContext() { acldvppDestroyChannelDesc(channel_desc); }

  acldvppChannelDesc *channel_desc;
};

aclError JPEGEncoder::Save(const std::string &path, acldvppPicDesc *pic_desc,
                           aclrtStream stream) {
  JPEGChanContext &ctx = JPEGChanContext::GetInstance();
  acldvppChannelDesc *channel_desc = ctx.GetChanDesc();
  acldvppJpegeConfig *cfg = acldvppCreateJpegeConfig();
  CHECK_ACL(acldvppSetJpegeConfigLevel(cfg, 100));

  uint32_t jpeg_buffer_size;

  CHECK_ACL(acldvppJpegPredictEncSize(pic_desc, cfg, &jpeg_buffer_size));

  void *jpeg_output_buffer;
  acldvppMalloc(&jpeg_output_buffer, jpeg_buffer_size);

  CHECK_ACL(acldvppJpegEncodeAsync(channel_desc, pic_desc, jpeg_output_buffer,
                                   &jpeg_buffer_size, cfg, stream));

  CHECK_ACL(aclrtSynchronizeStream(stream));

  std::ofstream ofs(path.c_str(), std::ios_base::binary);
  ofs.write((char *)jpeg_output_buffer, jpeg_buffer_size);

  std::cout << "write jpeg: " << path << " size: " << jpeg_buffer_size
            << std::endl;
  acldvppFree(jpeg_output_buffer);

  CHECK_ACL(acldvppDestroyJpegeConfig(cfg));
  return ACL_ERROR_NONE;
}