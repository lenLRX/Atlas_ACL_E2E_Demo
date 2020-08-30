#ifndef __ACL_UTIL_H__
#define __ACL_UTIL_H__

#include "acl/acl.h"
#include <iostream>

#define CHECK_ACL(x) do {\
    aclError __ret = x;\
    if (__ret != ACL_ERROR_NONE) {\
        std::cerr << __FILE__ << ":" << __LINE__\
            << " aclError:" << __ret << std::endl;\
    }\
}while(0);

#endif//__ACL_UTIL_H__