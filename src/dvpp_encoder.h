#ifndef __DVPP_ENCODER_H__
#define __DVPP_ENCODER_H__

#include <tuple>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "acl_cb_thread.h"

#include "util.h"
#include "acl_model.h"

class DvppEncoder {
public:
  using OutTy = std::tuple<void*, uint32_t>;
  DvppEncoder();
  ~DvppEncoder();
  void Destory();
  void ShutDown() { output_queue->ShutDown(); }
  aclError Init(const pthread_t thread_id, int h, int w);
  aclError SendFrame(uint8_t *data);
  void Process(DeviceBufferPtr buffer);

  void SetOutputQueue(ThreadSafeQueueWithCapacity<OutTy> *queue);
  ThreadSafeQueueWithCapacity<OutTy> *GetOutputQueue();

private:
  int height;
  int width;
  int size;
  aclvencChannelDesc *channel_desc;
  aclvencFrameConfig *frame_config;

  ThreadSafeQueueWithCapacity<OutTy> *output_queue{nullptr};
};

#endif //__DVPP_ENCODER_H__