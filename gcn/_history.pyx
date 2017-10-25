cimport numpy as np
import numpy as np
from libc.string cimport memcpy
from scipy.sparse import coo_matrix, csr_matrix
from time import time

cdef extern from "history.h":
#    cdef void compute_history(float *, int *, int *, int, int, float*, int, float*)
    cdef void c_slice(int, int*, float *, int *, int *, float *, int *, int *)
    cdef void c_indptr(int, int*, int*, int*)
    cdef void c_dense_slice(int, int, int*, float*, float*)

#def mean_history(fadj, history):
#    cdef float[:] ad = fadj.data
#    cdef int[:]   ai = fadj.indices
#    cdef int[:]   ap = fadj.indptr
#    cdef float[:,:] hd = history
#    output = np.zeros((fadj.shape[0], history.shape[1]), dtype=np.float32)
#    cdef float[:,:] od = output
#
#    compute_history(&ad[0], &ai[0], &ap[0], fadj.shape[0], fadj.nnz, 
#                    &hd[0,0], history.shape[1], &od[0,0])
#    return output

def slice(a, r):    
    cdef int N = len(r)
    indptr  = np.zeros(N+1, dtype=np.int32)
    cdef int[:] a_indptr    = a.indptr
    cdef int[:] o_indptr    = indptr
    cdef int[:] rv          = r
    c_indptr(N, &rv[0], &a_indptr[0], &o_indptr[0])
    cdef int nnz = o_indptr[N]

    if nnz == 0:
        return csr_matrix((N, a.shape[1]), dtype=a.dtype)

    data    = np.zeros(nnz, dtype=np.float32)
    indices = np.zeros((nnz, 2), dtype=np.int32)
    dense_shape = np.array([N, a.shape[1]], dtype=np.int32)

    cdef float[:] a_data    = a.data
    cdef int[:] a_indices   = a.indices

    # COO
    cdef float[:] o_data    = data
    cdef int[:,:] o_indices = indices

    c_slice(N, &rv[0], &a_data[0], &a_indices[0], &a_indptr[0],
                       &o_data[0], &o_indices[0,0], &o_indptr[0])

    return (indices, data, dense_shape)

def dense_slice(a, r):
    cdef int N = len(r)
    cdef int C = a.shape[1]
    output = np.zeros((N, C), dtype=np.float32)

    cdef float[:,:] a_data = a
    cdef float[:,:] o_data = output
    cdef int[:]     rv     = r
    c_dense_slice(N, C, &rv[0], &a_data[0,0], &o_data[0,0])
    return output

