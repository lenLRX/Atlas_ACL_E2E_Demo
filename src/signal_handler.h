#ifndef __SIGNAL_HANDLER_H__
#define __SIGNAL_HANDLER_H__

#include <functional>
#include <vector>

class SingalHandler {
public:
  using CbVec = std::vector<std::function<void(void)>>;
  SingalHandler() = default;
  ~SingalHandler() = default;

  static SingalHandler &GetInstancce();
  static void RegisterSignal();
  static void Register(std::function<void(void)> callback);
  static CbVec GetCallBacks();

private:
  CbVec callbacks;
};

#endif //__SIGNAL_HANDLER_H__