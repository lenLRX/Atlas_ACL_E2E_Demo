#ifndef __SIGNAL_HANDLER_H__
#define __SIGNAL_HANDLER_H__

#include <functional>
#include <mutex>
#include <vector>

class SingalHandler {
public:
  using CbVec = std::vector<std::function<void(void)>>;
  SingalHandler() = default;
  ~SingalHandler() = default;
  SingalHandler &operator=(const SingalHandler &) = delete;
  SingalHandler(const SingalHandler &) = delete;

  static SingalHandler &GetInstancce();
  static void RegisterSignal();
  static void Register(std::function<void(void)> callback);
  static CbVec &GetCallBacks();
  static std::mutex &GetMtx();

private:
  CbVec callbacks;
  std::mutex cb_mtx;
};

#endif //__SIGNAL_HANDLER_H__