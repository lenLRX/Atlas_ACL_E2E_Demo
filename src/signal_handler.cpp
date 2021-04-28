#include <signal.h>
#include <unistd.h>

#include <iostream>

#include "signal_handler.h"

void app_signal_callback(int signum) {
  auto &handler = SingalHandler::GetInstancce();
  std::lock_guard<std::mutex> guard(SingalHandler::GetMtx());
  auto &callbacks = SingalHandler::GetCallBacks();
  for (auto &cb : callbacks) {
    cb();
  }
  callbacks.clear();
}

SingalHandler &SingalHandler::GetInstancce() {
  static SingalHandler handler;
  return handler;
}

void SingalHandler::RegisterSignal() { signal(SIGINT, app_signal_callback); }

void SingalHandler::Register(std::function<void(void)> callback) {
  auto &handler = SingalHandler::GetInstancce();
  std::lock_guard<std::mutex> guard(handler.cb_mtx);
  handler.callbacks.push_back(callback);
}

SingalHandler::CbVec &SingalHandler::GetCallBacks() {
  return SingalHandler::GetInstancce().callbacks;
}

std::mutex &SingalHandler::GetMtx() {
  return SingalHandler::GetInstancce().cb_mtx;
}
