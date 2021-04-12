#ifndef __TASK_NODE_H__
#define __TASK_NODE_H__

#include <atomic>
#include <functional>
#include <thread>

#include "util.h"

template <typename ObjTy, typename InTy, typename OutTy> class TaskNode {
public:
  using InQueueTy = ThreadSafeQueueWithCapacity<InTy>;
  using OutQueueTy = ThreadSafeQueueWithCapacity<OutTy>;
  TaskNode(ObjTy *pobj, const std::string &thread_name,
           const std::string &stream_name)
      : pobj(pobj), thread_name(thread_name), stream_name(stream_name),
        input_queue(nullptr), output_queue(nullptr) {}

  void Start(aclrtContext ctx) {
    worker_thread = std::move(std::thread([ctx, this]() {
      aclrtSetCurrentContext(ctx);
      SetThreadName(thread_name);
      SetStreamName(stream_name);
      while (true) {
        InTy input;
        if (!input_queue->pop(input)) {
          output_queue->ShutDown();
          break;
        }
        OutTy output = pobj->Process(input);
        output_queue->push(output);
      }
    }));
  }

  void Join() { worker_thread.join(); }

  void SetInputQueue(InQueueTy *in_queue) { input_queue = in_queue; }

  void SetOutputQueue(OutQueueTy *out_queue) { output_queue = out_queue; }

private:
  ObjTy *pobj;

  std::string thread_name;
  std::string stream_name;
  std::thread worker_thread;
  InQueueTy *input_queue;
  OutQueueTy *output_queue;
};

template <typename ObjTy, typename InTy> class TaskNode<ObjTy, InTy, void> {
public:
  using InQueueTy = ThreadSafeQueueWithCapacity<InTy>;

  TaskNode(ObjTy *pobj, const std::string &thread_name,
           const std::string &stream_name)
      : pobj(pobj), thread_name(thread_name), stream_name(stream_name),
        input_queue(nullptr) {}

  void Start(aclrtContext ctx) {
    worker_thread = std::move(std::thread([ctx, this]() {
      aclrtSetCurrentContext(ctx);
      SetThreadName(thread_name);
      SetStreamName(stream_name);
      while (true) {
        InTy input;
        if (!input_queue->pop(input)) {
          pobj->ShutDown();
          break;
        }
        pobj->Process(input);
      }
    }));
  }

  void Join() { worker_thread.join(); }

  void SetInputQueue(InQueueTy *in_queue) { input_queue = in_queue; }

private:
  ObjTy *pobj;

  std::string thread_name;
  std::string stream_name;
  std::thread worker_thread;
  InQueueTy *input_queue;
};

template <typename ObjTy> class TaskNode<ObjTy, void, void> {
public:
  TaskNode(ObjTy *pobj, const std::string &thread_name,
           const std::string &stream_name)
      : pobj(pobj), thread_name(thread_name), stream_name(stream_name) {}

  void Start(aclrtContext ctx) {
    worker_thread = std::move(std::thread([ctx, this]() {
      aclrtSetCurrentContext(ctx);
      SetThreadName(thread_name);
      SetStreamName(stream_name);
      pobj->Process();
      pobj->ShutDown();
    }));
  }

  void Join() { worker_thread.join(); }

private:
  ObjTy *pobj;

  std::string thread_name;
  std::string stream_name;
  std::thread worker_thread;
};

#endif //__TASK_NODE_H__