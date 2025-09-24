#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include "symnmf.h"

/*
  Convert a Python object to a 2D NumPy array of doubles.
  - Checks dtype == double and that it's 2D.
  - On success: returns 1 and fills out, n (rows), m (cols).
  - On failure: returns 0 (and cleans up).
*/
static int as_2d_double(PyObject* obj, PyArrayObject** out, int* n, int* m) {
    *out = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!*out) return 0;                          /*couldn't make/view array */
    if (PyArray_NDIM(*out) != 2) {                /*must be 2D - as required */ 
        Py_DECREF(*out);
        return 0;
    }
    const npy_intp* dims = PyArray_DIMS(*out);    /*to clarify: dims[0]=rows, dims[1]=cols*/ 
    *n = (int)dims[0];
    *m = (int)dims[1];
    return 1;
}

/* A = sym(X)  where X is (n x d) and A is (n x n) similarity matrix */
static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject* Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;  /*expect one object as required */

    PyArrayObject* Xarr; int n, d;
    if (!as_2d_double(Xobj, &Xarr, &n, &d)) goto error;

    npy_intp dd[2] = { (npy_intp)n, (npy_intp)n };
    PyArrayObject* A = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!A) goto error;

    /*Fill A using the C impl from the symnmf.h/symnmf.c */ 
    compute_similarity((const double*)PyArray_DATA(Xarr), n, d, (double*)PyArray_DATA(A));

    Py_DECREF(Xarr);
    return (PyObject*)A;

error:
    Py_XDECREF(Xarr);
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

/* D = ddg(X): computes similarity first, then degree matrix (n x n diagonal) */
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    PyArrayObject* Xarr; int n, d;
    if (!as_2d_double(Xobj, &Xarr, &n, &d)) goto error;

    npy_intp dd[2] = { (npy_intp)n, (npy_intp)n };
    PyArrayObject* A = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    PyArrayObject* D = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!A || !D) goto error2;

    compute_similarity((const double*)PyArray_DATA(Xarr), n, d, (double*)PyArray_DATA(A));
    compute_ddg((const double*)PyArray_DATA(A), n, (double*)PyArray_DATA(D));

    Py_DECREF(Xarr);
    Py_DECREF(A);
    return (PyObject*)D;

error2:
    Py_XDECREF(A);
error:
    Py_XDECREF(Xarr);
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

/* W = norm(X): W = D^{-1/2} A D^{-1/2}, where A=similarity(X), D=degree(A) */
static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    PyArrayObject* Xarr; int n, d;
    if (!as_2d_double(Xobj, &Xarr, &n, &d)) goto error;

    npy_intp dd[2] = { (npy_intp)n, (npy_intp)n };
    PyArrayObject* A = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    PyArrayObject* D = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    PyArrayObject* W = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!A || !D || !W) goto error3;

    compute_similarity((const double*)PyArray_DATA(Xarr), n, d, (double*)PyArray_DATA(A));
    compute_ddg((const double*)PyArray_DATA(A), n, (double*)PyArray_DATA(D));
    compute_norm((const double*)PyArray_DATA(A), (const double*)PyArray_DATA(D),
                 n, (double*)PyArray_DATA(W));

    Py_DECREF(Xarr);
    Py_DECREF(A);
    Py_DECREF(D);
    return (PyObject*)W;

error3:
    Py_XDECREF(A);
    Py_XDECREF(D);
error:
    Py_XDECREF(Xarr);
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

/* symnmf(H0,W,k,eps,max_iter[,beta]) -> H (n×k).
   H0:(n×k), W:(n×n). beta=0.5 if missing. */
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *H0obj=NULL, *Wobj=NULL;
    PyArrayObject *H0=NULL, *W=NULL, *H=NULL;
    int k, max_iter, n1, k1, n2, n2b;
    double eps, beta;
    if (!PyArg_ParseTuple(args,"OOidi",&H0obj,&Wobj,&k,&eps,&max_iter)) {
        PyErr_Clear();
        if (!PyArg_ParseTuple(args,"OOidid",&H0obj,&Wobj,&k,&eps,&max_iter,&beta))
            return NULL;
    } else beta = 0.5;
    if (!as_2d_double(H0obj,&H0,&n1,&k1)) goto error;
    if (!as_2d_double(Wobj,&W,&n2,&n2b)) goto error;
    if (n1!=n2 || n2!=n2b || k1!=k) goto error;
    npy_intp dd[2] = { (npy_intp)n1, (npy_intp)k };
    H = (PyArrayObject*)PyArray_SimpleNew(2, dd, NPY_DOUBLE);
    if (!H) goto error;
    memcpy(PyArray_DATA(H), PyArray_DATA(H0), (size_t)n1*(size_t)k*sizeof(double));
    symnmf_optimize((const double*)PyArray_DATA(W), n1, k, max_iter, eps, beta,
                    (double*)PyArray_DATA(H));
    Py_DECREF(H0); Py_DECREF(W);
    return (PyObject*)H;
error:
    if (H0) Py_DECREF(H0);
    if (W)  Py_DECREF(W);
    if (H)  Py_DECREF(H);
    PyErr_SetString(PyExc_RuntimeError,"An Error Has Occurred");
    return NULL;
}


/* Python method table: names visible from Python and their docstrings */
static PyMethodDef Methods[] = {
    {"sym",    (PyCFunction)py_sym,    METH_VARARGS, "A = sym(X)\nCompute n×n similarity from X (n×d)."},
    {"ddg",    (PyCFunction)py_ddg,    METH_VARARGS, "D = ddg(X)\nCompute degree matrix from similarity(X)."},
    {"norm",   (PyCFunction)py_norm,   METH_VARARGS, "W = norm(X)\nCompute normalized similarity: D^{-1/2} A D^{-1/2}."},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "H = symnmf(H0, W, k, eps, max_iter[, beta])\nRun SymNMF starting from H0."},
    {NULL, NULL, 0, NULL}
};

/* Basic module definition and init */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,     /*no module-level docstring */ 
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    import_array();   /*as required for NumPy C-API */ 
    return PyModule_Create(&moduledef);
}
