#ifndef __ACL_UTIL_H__
#define __ACL_UTIL_H__

#include "acl/acl.h"
#include <iostream>
#include <sstream>
#include <chrono>

using namespace std::chrono;

#define CHECK_ACL(x) do {\
    aclError __ret = x;\
    if (__ret != ACL_ERROR_NONE) {\
        std::cerr << __FILE__ << ":" << __LINE__\
            << " aclError:" << __ret << std::endl;\
    }\
}while(0);


class PerfTimer
{
public:
    PerfTimer(const char* file, int line, const char* func) {
        file_ = file;
        line_ = line;
        func_ = func;
        start = steady_clock::now();
    }
    ~PerfTimer() {
        auto end = steady_clock::now();
        auto duration = end - start;
        microseconds duration_us = duration_cast<microseconds>(duration);

        std::stringstream ss;
        ss << file_ << ":" << line_ << " func:" << func_
        << " duration:" << duration_us.count() << "us";
        std::cerr << ss.str() << std::endl;
    }

private:
    std::chrono::steady_clock::time_point start;
    const char* file_;
    int line_;
    const char* func_;
};

#define _CONCAT_(x, y) x##y
#define __CONCAT__(x, y) _CONCAT_(x, y)

#define PERF_TIMER() \
    auto __CONCAT__(temp_perf_obj_, __LINE__) = PerfTimer(__FILE__, __LINE__, __FUNCTION__)

#endif//__ACL_UTIL_H__