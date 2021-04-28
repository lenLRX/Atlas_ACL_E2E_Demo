#ifndef __STREAM_FACTORY_H__
#define __STREAM_FACTORY_H__

#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

using json = nlohmann::json;

class StreamFactory {
public:
  using FnTy = std::function<std::thread(json)>;
  static std::thread MakeStream(json config);
  static bool RegsiterStream(std::string name, FnTy make_fn);
  static StreamFactory &GetInstance();

private:
  std::map<std::string, FnTy> factory;
};

#define REGSITER_STREAM(type, fn)                                              \
  static bool __register_##type = StreamFactory::RegsiterStream(#type, fn)

#endif //__STREAM_FACTORY_H__