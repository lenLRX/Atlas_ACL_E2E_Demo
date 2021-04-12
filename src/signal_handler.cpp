#include <signal.h>
#include <unistd.h>

#include "signal_handler.h"

void app_signal_callback(int signum) {
  auto callbacks = SingalHandler::GetInstancce().GetCallBacks();
  for (auto &cb : callbacks) {
    cb();
  }
  _exit(0); // signal safe function
}

SingalHandler &SingalHandler::GetInstancce() {
  static SingalHandler handler;
  return handler;
}

void SingalHandler::RegisterSignal() { signal(SIGINT, app_signal_callback); }

void SingalHandler::Register(std::function<void(void)> callback) {
  SingalHandler::GetInstancce().callbacks.push_back(callback);
}

SingalHandler::CbVec SingalHandler::GetCallBacks() {
  return SingalHandler::GetInstancce().callbacks;
}
