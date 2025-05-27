#include <Python.h>
#include <tensor.h>

static void copy_data(PyObject* list, int dim, int ndim, float* data, Py_ssize_t* index) {
    if (dim == ndim - 1) {
        for (Py_ssize_t i = 0; i < PyList_Size(list); i++) {
            PyObject* item = PyList_GetItem(list, i);
            if (!PyNumber_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "All elements must be numbers");
                return;
            }
            data[(*index)++] = (float)PyFloat_AsDouble(item);
        }
    } else {
        for (Py_ssize_t i = 0; i < PyList_Size(list); i++) {
            PyObject* sublist = PyList_GetItem(list, i);
            if (!PyList_Check(sublist)) {
                PyErr_SetString(PyExc_TypeError, "Invalid nested list structure");
                return;
            }
            copy_data(sublist, dim + 1, ndim, data, index);
        }
    }
}

static PyObject* Tensor_new(PyObject* self, PyObject* args, PyObject* kwds) {
    Tensor* tensor;
    return (PyObject*)tensor;
}

static PyObject* Tensor_init(PyObject* self, PyObject* args, PyObject* kwds) {
    Tensor* tensor = (Tensor*)self;
    Py_ssize_t size = PyTuple_Size(args);
    
    if (size != 1) {
        PyErr_SetString(PyExc_TypeError, "Tensor() takes exactly one argument");
        return NULL;
    }

    PyObject* input = PyTuple_GetItem(args, 0);
    if (!PyList_Check(input)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a list");
        return NULL;
    }

    Py_ssize_t ndim = 0;
    Py_ssize_t total_size = 1;
    PyObject* current = input;
    int* shape = NULL;
    
    while (PyList_Check(current)) {
        Py_ssize_t dim_size = PyList_Size(current);
        if (dim_size == 0) {
            PyErr_SetString(PyExc_ValueError, "Empty lists are not allowed");
            return NULL;
        }
        
        shape = realloc(shape, (ndim + 1) * sizeof(int));
        shape[ndim] = dim_size;
        total_size *= dim_size;
        
        current = PyList_GetItem(current, 0);
        ndim++;
    }

    tensor->data = (float*)malloc(total_size * sizeof(float));
    tensor->shape = shape;
    tensor->ndim = ndim;
    tensor->size = total_size;

    Py_ssize_t index = 0;
    copy_data(input, 0, ndim, tensor->data, &index);
    
    if (PyErr_Occurred()) {
        free(tensor->data);
        free(tensor->shape);
        return NULL;
    }

    Py_RETURN_NONE;
}

static void Tensor_dealloc(PyObject* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Tensor_repr(PyObject* self) {
    Tensor* tensor = (Tensor*)self;
    return Py_BuildValue("s", tensor->data);
}

static PyMethodDef tensor_methods[] = {
    {"__new__", Tensor_new, METH_VARARGS | METH_KEYWORDS, "Create a new Tensor object"},
    {"__init__", Tensor_init, METH_VARARGS | METH_KEYWORDS, "Initialize a Tensor object"},
    {"__repr__", Tensor_repr, METH_VARARGS | METH_KEYWORDS, "Represent a Tensor object"},
    {"__del__", Tensor_dealloc, METH_VARARGS | METH_KEYWORDS, "Delete a Tensor object"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "SmolGrad.Tensor",
    .tp_basicsize = sizeof(Tensor),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Tensor_new,
    .tp_init = (initproc)Tensor_init,
    .tp_dealloc = (destructor)Tensor_dealloc,
    .tp_repr = (reprfunc)Tensor_repr,
    .tp_methods = tensor_methods,
};

