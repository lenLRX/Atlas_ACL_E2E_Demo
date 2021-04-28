#ifndef __JPEG_ENCODE_H__
#define __JPEG_ENCODE_H__

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include "util.h"

#include <functional>

class JPEGEncoder {
public:
  static aclError Save(const std::string &path, acldvppPicDesc *pic_desc,
                       aclrtStream stream);
};

#endif //__JPEG_ENCODE_H__