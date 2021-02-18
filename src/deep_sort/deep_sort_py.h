#ifndef __DEEP_SORY_PY_H__
#define __DEEP_SORY_PY_H__

#include <Python.h>

#include <iostream>
#include <vector>
#include <cmath>

void deep_sort_py_func(int feature_num,
                       void* boxes,
                       void* scores,
                       void* feature_data,
                       std::vector<std::vector<int>>& trackings);

#endif//__DEEP_SORY_PY_H__