cimport numpy as np
import numpy as np

cdef extern from "history.h":
    cdef void compute_history(float *, int *, int *, int, int, float*, int, float*)

def mean_history(fadj, history):
    cdef float[:] ad = fadj.data
    cdef int[:]   ai = fadj.indices
    cdef int[:]   ap = fadj.indptr
    cdef float[:,:] hd = history
    output = np.zeros((fadj.shape[0], history.shape[1]), dtype=np.float32)
    cdef float[:,:] od = output

    compute_history(&ad[0], &ai[0], &ap[0], fadj.shape[0], fadj.nnz, 
                    &hd[0,0], history.shape[1], &od[0,0])
    return output
