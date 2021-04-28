#ifndef __ACL_CB_THREAD_H__
#define __ACL_CB_THREAD_H__

#include "acl/acl.h"
#include <pthread.h>
#include <string>
#include <thread>

#include "app_profiler.h"

class AclCallBackThread {
public:
  AclCallBackThread(const std::string &sname, const std::string &tname) {
    worker_thread = std::thread([sname, tname, this]() {
      SetStreamName(sname);
      SetThreadName(tname);
      // Notice: create context for this thread
      int deviceId = 0;
      aclrtContext context = nullptr;
      aclError ret = aclrtCreateContext(&context, deviceId);

      while (run_flag) {
        // Notice: timeout 1000ms
        ret = aclrtProcessReport(1000);
      }

      ret = aclrtDestroyContext(context);
    });
  }

  pthread_t GetPid() { return worker_thread.native_handle(); }

  void Join() {
    run_flag = false;
    worker_thread.join();
  }

  ~AclCallBackThread() {
    if (run_flag) {
      run_flag = false;
      worker_thread.join();
    }
  }

private:
  bool run_flag{true};
  std::thread worker_thread;
};

#endif //__ACL_CB_THREAD_H__