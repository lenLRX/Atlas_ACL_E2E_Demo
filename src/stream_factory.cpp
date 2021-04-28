#include <iostream>

#include "stream_factory.h"

std::thread StreamFactory::MakeStream(json config) {
  std::string stream_type = config.at("stream_type");
  auto &instance = GetInstance();
  return instance.factory.at(stream_type)(config);
}

bool StreamFactory::RegsiterStream(std::string name, FnTy make_fn) {
  auto &instance = GetInstance();
  instance.factory[name] = make_fn;
  return true;
}

StreamFactory &StreamFactory::GetInstance() {
  static StreamFactory factory;
  return factory;
}
