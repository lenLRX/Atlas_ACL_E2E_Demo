#ifndef __DEV_MEM_POOL_H__
#define __DEV_MEM_POOL_H__

#include <list>
#include <mutex>
#include <unordered_map>

#include "util.h"

#define USE_MEM_POOL

// a simple memory pool of device memory and dvpp memory

struct DevMemPoolEntry {
  DevMemPoolEntry() = default;
  DevMemPoolEntry(void *p, size_t sz, int32_t dev_id)
      : dev_ptr_(p), size_(sz), dev_id_(dev_id) {}
  void *dev_ptr_;
  size_t size_;
  int32_t dev_id_;
};

class DevMemPool {
public:
  static DevMemPool &GetInstance() {
    static DevMemPool pool;
    return pool;
  }

  static void *AllocDevMem(size_t size) {
#ifdef USE_MEM_POOL
    return GetInstance().AllocDevMemImpl(size);
#else
    void *dev_mem = nullptr;
    CHECK_ACL(aclrtMalloc(&dev_mem, size, ACL_MEM_MALLOC_HUGE_FIRST));
    return dev_mem;
#endif
  }

  static void FreeDevMem(void *p) {
#ifdef USE_MEM_POOL
    GetInstance().FreeDevMemImpl(p);
#else
    CHECK_ACL(aclrtFree(p));
#endif
  }

  static void *AllocDvppMem(size_t size) {
#ifdef USE_MEM_POOL
    return GetInstance().AllocDvppMemImpl(size);
#else
    void *dvpp_mem = nullptr;
    CHECK_ACL(acldvppMalloc(&dvpp_mem, size));
    return dvpp_mem;
#endif
  }

  static void FreeDvppMem(void *p) {
#ifdef USE_MEM_POOL
    GetInstance().FreeDvppMemImpl(p);
#else
    CHECK_ACL(acldvppFree(p));
#endif
  }

private:
  DevMemPool() {}
  ~DevMemPool() {
    // TODO: release device memory?
  }

  void *AllocDevMemImpl(size_t size) {
    void *dev_mem = nullptr;
    int32_t dev_id;
    CHECK_ACL(aclrtGetDevice(&dev_id));
    {
      std::lock_guard<std::mutex> g(dev_mem_mtx_);
      auto &list = dev_free_lists_[dev_id][size];
      if (!list.empty()) {
        dev_mem = list.front().dev_ptr_;
        list.pop_front();
      }
    }
    // TODO: if malloc failed, try to free dev mem in freelist
    if (dev_mem == nullptr) {
      CHECK_ACL(aclrtMalloc(&dev_mem, size, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    DevMemPoolEntry entry(dev_mem, size, dev_id);
    {
      std::lock_guard<std::mutex> g(dev_mem_mtx_);
      dev_using_memory_[dev_mem] = entry;
    }
    return dev_mem;
  }

  void FreeDevMemImpl(void *p) {
    {
      std::lock_guard<std::mutex> g(dev_mem_mtx_);
      DevMemPoolEntry entry = dev_using_memory_.at(p);
      dev_using_memory_.erase(p);
      dev_free_lists_[entry.dev_id_][entry.size_].push_back(entry);
    }
  }

  void *AllocDvppMemImpl(size_t size) {
    void *dvpp_mem = nullptr;
    int32_t dev_id;
    CHECK_ACL(aclrtGetDevice(&dev_id));
    {
      std::lock_guard<std::mutex> g(dvpp_mem_mtx_);
      auto &list = dvpp_free_lists_[dev_id][size];
      if (!list.empty()) {
        dvpp_mem = list.front().dev_ptr_;
        list.pop_front();
      }
    }
    // TODO: if malloc failed, try to free dev mem in freelist
    if (dvpp_mem == nullptr) {
      CHECK_ACL(acldvppMalloc(&dvpp_mem, size));
    }
    DevMemPoolEntry entry(dvpp_mem, size, dev_id);
    {
      std::lock_guard<std::mutex> g(dvpp_mem_mtx_);
      dvpp_using_memory_[dvpp_mem] = entry;
    }
    return dvpp_mem;
  }

  void FreeDvppMemImpl(void *p) {
    {
      std::lock_guard<std::mutex> g(dvpp_mem_mtx_);
      DevMemPoolEntry entry = dvpp_using_memory_.at(p);
      dvpp_using_memory_.erase(p);
      dvpp_free_lists_[entry.dev_id_][entry.size_].push_back(entry);
    }
  }

  std::mutex dev_mem_mtx_;
  std::mutex dvpp_mem_mtx_;
  std::unordered_map<void *, DevMemPoolEntry> dev_using_memory_;
  std::unordered_map<int32_t,
                     std::unordered_map<size_t, std::list<DevMemPoolEntry>>>
      dev_free_lists_;
  std::unordered_map<void *, DevMemPoolEntry> dvpp_using_memory_;
  std::unordered_map<int32_t,
                     std::unordered_map<size_t, std::list<DevMemPoolEntry>>>
      dvpp_free_lists_;
};

#endif //__DEV_MEM_POOL_H__