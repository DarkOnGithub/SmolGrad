#include <Python.h>

#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor {
    float* data;
    int* shape;
    int ndim;
    int size;
} Tensor;

PyObject* tensor_new(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* tensor_init(PyObject* self, PyObject* args, PyObject* kwds);
void tensor_dealloc(PyObject* self);
PyObject* tensor_repr(PyObject* self);

#endif