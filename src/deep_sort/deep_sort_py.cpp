#include "deep_sort_py.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ndarrayobject.h"


#define CHECK_PY_ERR(obj) if (obj == NULL) {\
  PyErr_Print();\
  throw std::runtime_error("");\
  return;\
}


// https://github.com/numpy/numpy/issues/11925
class PyEnv {
public:
  static PyEnv& GetInstance() {
    static PyEnv env;
    return env;
  }

  PyObject* GetUpdateTrackerFn() {
    return update_tracker_fn_;
  }

  PyObject* GetMakeDetectionsFn() const {
    return make_detection_fn_;
  }

  PyObject* GetQueryTrackingFn() const {
    return query_tracking_fn_;
  }

  PyObject* GetTrackerCtx() const {
    return tracker_ctx_;
  }

private:
  PyEnv() {
    Py_Initialize();
    {// expansion of macro import_array
      if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
      }
    }
    numpy_ = PyImport_ImportModule("numpy");
    deepsort_module_ = PyImport_ImportModule("acl_deepsort_app");
    if (deepsort_module_ == NULL) {
      PyErr_Print();
      return;
    }

    PyObject* init_fn = PyObject_GetAttrString(deepsort_module_, "init_tracker");
    if (init_fn == NULL) {
      PyErr_Print();
      return;
    }

    PyObject* init_arg = PyTuple_New(0);
    tracker_ctx_ = PyObject_Call(init_fn, init_arg, NULL);

    if (tracker_ctx_ == NULL) {
      PyErr_Print();
      return;
    }

    update_tracker_fn_ = PyObject_GetAttrString(tracker_ctx_, "update");

    if (update_tracker_fn_ == NULL) {
      PyErr_Print();
      return;
    }

    Py_XDECREF(init_fn);
    Py_XDECREF(init_arg);

    make_detection_fn_ = PyObject_GetAttrString(deepsort_module_, "make_detections");

    query_tracking_fn_ = PyObject_GetAttrString(tracker_ctx_, "query_tracking_result");
  }

  ~PyEnv() {
    Py_FinalizeEx();
  }
  PyObject* numpy_;
  PyObject* deepsort_module_;
  PyObject* tracker_ctx_;
  PyObject* update_tracker_fn_;
  PyObject* make_detection_fn_;
  PyObject* query_tracking_fn_;
};


void deep_sort_py_func(int feature_num,
                       void* boxes,
                       void* scores,
                       void* feature_data,
                       std::vector<std::vector<int>>& trackings) {
  PyEnv& env = PyEnv::GetInstance();

  npy_intp boxes_dim[2] = {feature_num, 4};
  const int boxes_nd = 2;

  PyObject* boxes_arr = PyArray_SimpleNewFromData(
      boxes_nd, boxes_dim, NPY_INT32, boxes);
  PyArray_CLEARFLAGS((PyArrayObject*)boxes_arr, NPY_ARRAY_OWNDATA);

  npy_intp scores_dim[1] = {feature_num};
  const int scores_nd = 1;

  PyObject* scores_arr = PyArray_SimpleNewFromData(
      scores_nd, scores_dim, NPY_FLOAT32, scores);
  PyArray_CLEARFLAGS((PyArrayObject*)scores_arr, NPY_ARRAY_OWNDATA);

  npy_intp deepsort_dim[2] = {feature_num, 128};
  const int deepsort_nd = 2;
  PyObject* feature_arr = PyArray_SimpleNewFromData(
      deepsort_nd, deepsort_dim, NPY_FLOAT32, feature_data);
  PyArray_CLEARFLAGS((PyArrayObject*)feature_arr, NPY_ARRAY_OWNDATA);

  PyObject* detection_fn = env.GetMakeDetectionsFn();

  PyObject* detection_arg = Py_BuildValue("(O,O,O)", boxes_arr, scores_arr, feature_arr);

  if (detection_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("");
    return;
  }

  PyObject* detections = PyObject_Call(detection_fn, detection_arg, NULL);

  if (detections == NULL) {
    PyErr_Print();
    throw std::runtime_error("");
    return;
  }

  Py_XDECREF(detection_arg);

  PyObject* update_fn = env.GetUpdateTrackerFn();

  PyObject* update_arg = Py_BuildValue("(O)", detections);

  if (update_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("");
    return;
  }

  PyObject* upd_result = PyObject_Call(update_fn, update_arg, NULL);

  if (upd_result == NULL) {
    PyErr_Print();
    throw std::runtime_error("");
    return;
  }

  Py_XDECREF(detections);
  Py_XDECREF(upd_result);

  PyObject* query_fn = env.GetQueryTrackingFn();
  PyObject* query_arg = PyTuple_New(0);
  if (query_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("");
    return;
  }

  // List[List[tracking_id, x1, y1 ,x2 ,y2]]
  PyObject* tracking_result = PyObject_Call(query_fn, query_arg, NULL);
  if (tracking_result == NULL) {
    PyErr_Print();
    throw std::runtime_error("");
    return;
  }

  Py_XDECREF(query_arg);

  Py_ssize_t track_size = PyList_Size(tracking_result);
  for (Py_ssize_t i = 0;i < track_size; ++i) {
    PyObject* track = PyList_GetItem(tracking_result, i);
    CHECK_PY_ERR(track);
    PyObject* py_track_id = PyList_GetItem(track, 0);
    CHECK_PY_ERR(py_track_id);
    int track_id = PyLong_AsLong(py_track_id);
    PyObject* py_x1 = PyList_GetItem(track, 1);
    CHECK_PY_ERR(py_x1);
    int x1 = std::round(PyFloat_AsDouble(py_x1));
    PyObject* py_y1 = PyList_GetItem(track, 2);
    CHECK_PY_ERR(py_y1);
    int y1 = std::round(PyFloat_AsDouble(py_y1));
    PyObject* py_x2 = PyList_GetItem(track, 3);
    CHECK_PY_ERR(py_x2);
    int x2 = std::round(PyFloat_AsDouble(py_x2));
    PyObject* py_y2 = PyList_GetItem(track, 4);
    CHECK_PY_ERR(py_y2);
    int y2 = std::round(PyFloat_AsDouble(py_y2));
    trackings.push_back({track_id, x1, y1, x2, y2});
  }

  Py_XDECREF(tracking_result);
}
