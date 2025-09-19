#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include "symnmf.h"

static int as_2d_double(PyObject* obj, PyArrayObject** out, int* n, int* m) {
    *out = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!*out) return 0;
    if (PyArray_NDIM(*out) != 2) { Py_DECREF(*out); return 0; }
    npy_intp const* dims = PyArray_DIMS(*out);
    *n = (int)dims[0]; *m = (int)dims[1];
    return 1;
}

static PyObject* py_sym(PyObject* self, PyObject* args) {

    PyObject* Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    PyArrayObject* Xarr; int n, d;
    if (!as_2d_double(Xobj, &Xarr, &n, &d)) goto error;

    npy_intp dd[2] = {n, n};
    PyArrayObject* A = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!A) goto error;

    compute_similarity((const double*)PyArray_DATA(Xarr), n, d, (double*)PyArray_DATA(A));
    Py_DECREF(Xarr);
    return (PyObject*)A;

error:
    Py_XDECREF(Xarr);
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    PyArrayObject* Xarr; int n, d;
    if (!as_2d_double(Xobj, &Xarr, &n, &d)) goto error;

    npy_intp dd[2] = {n, n};
    PyArrayObject* A = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    PyArrayObject* D = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!A || !D) goto error2;

    compute_similarity((const double*)PyArray_DATA(Xarr), n, d, (double*)PyArray_DATA(A));
    compute_ddg((const double*)PyArray_DATA(A), n, (double*)PyArray_DATA(D));

    Py_DECREF(Xarr); Py_DECREF(A);
    return (PyObject*)D;

error2:
    Py_XDECREF(A);
error:
    Py_XDECREF(Xarr);
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    PyArrayObject* Xarr; int n, d;
    if (!as_2d_double(Xobj, &Xarr, &n, &d)) goto error;

    npy_intp dd[2] = {n, n};
    PyArrayObject* A = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    PyArrayObject* D = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    PyArrayObject* W = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!A || !D || !W) goto error3;

    compute_similarity((const double*)PyArray_DATA(Xarr), n, d, (double*)PyArray_DATA(A));
    compute_ddg((const double*)PyArray_DATA(A), n, (double*)PyArray_DATA(D));
    compute_norm((const double*)PyArray_DATA(A), (const double*)PyArray_DATA(D), n, (double*)PyArray_DATA(W));

    Py_DECREF(Xarr); Py_DECREF(A); Py_DECREF(D);
    return (PyObject*)W;

error3:
    Py_XDECREF(A); Py_XDECREF(D);
error:
    Py_XDECREF(Xarr);
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *H0obj = NULL, *Wobj = NULL;
    PyArrayObject *H0 = NULL, *W = NULL, *H = NULL;
    int k, max_iter, n1, k1, n2, n2b;
    double eps, beta;

    if (!PyArg_ParseTuple(args, "OOidi", &H0obj, &Wobj, &k, &eps, &max_iter)) {
        PyErr_Clear();
        if (!PyArg_ParseTuple(args, "OOidid", &H0obj, &Wobj, &k, &eps, &max_iter, &beta))
            return NULL;
    } else {
        beta = 0.5;
    }

    if (!as_2d_double(H0obj, &H0, &n1, &k1)) goto error;
    if (!as_2d_double(Wobj, &W, &n2, &n2b)) goto error;
    if (n1 != n2 || n2 != n2b || k1 != k) goto error;

    npy_intp dd[2] = {n1, k};
    H = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!H) goto error;

    memcpy(PyArray_DATA(H), PyArray_DATA(H0), (size_t)n1 * k * sizeof(double));
    symnmf_optimize((const double*)PyArray_DATA(W), n1, k, max_iter, eps, beta, (double*)PyArray_DATA(H));

    Py_DECREF(H0);
    Py_DECREF(W);
    return (PyObject*)H;

error:
    if (H0) Py_DECREF(H0);
    if (W) Py_DECREF(W);
    if (H) Py_DECREF(H);
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

static PyMethodDef Methods[] = {
    {"sym",    (PyCFunction)py_sym,    METH_VARARGS, "A = sym(X)"},
    {"ddg",    (PyCFunction)py_ddg,    METH_VARARGS, "D = ddg(X)"},
    {"norm",   (PyCFunction)py_norm,   METH_VARARGS, "W = norm(X)"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "H = symnmf(H0, W, k, eps, max_iter[, beta])"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
