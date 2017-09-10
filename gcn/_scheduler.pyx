from scipy.sparse import coo_matrix
from libcpp.vector cimport vector
from libc.string cimport memcpy
cimport numpy as np
import numpy as np

cdef extern from "scheduler.h":
    void schedule_c(int L, int V, int E, int N, int batch_size, float dropconnect,
            int *d, int *e_s, int *e_t, float *e_w, 
            vector[int]& b_rows, vector[int]& b_cols, 
            vector[float]& b_data, vector[int]& b_offsets,
            vector[int]& r_fields, vector[int]& r_offsets)

cdef copy_int(int[:] buffer, int* ptr, int size):
    memcpy(&buffer[0], ptr, size*sizeof(int))

cdef copy_float(float[:] buffer, float* ptr, int size): 
    memcpy(&buffer[0], ptr, size*sizeof(float))


class Batch:
    def __init__(self, fields, adjs):
        self.fields = fields
        self.adjs = adjs

def schedule(A, d, L, batch_size, dropconnect):
    # A: weighted adjacency matrix
    # d: array of data indices
    # L: number of levels
    # output: L+1 receptive fields and L adjacency matrices
    A_row = A.row
    A_col = A.col
    A_data = A.data
    cdef int[:]   A_row_v  = A_row
    cdef int[:]   A_col_v  = A_col
    cdef float[:] A_data_v = A_data
    cdef int      V        = A.shape[0]
    cdef int      nnz      = A.nnz
    cdef int[:]   npd      = d

    cdef vector[int]   b_rows
    cdef vector[int]   b_cols
    cdef vector[float] b_data
    cdef vector[int]   b_offsets
    cdef vector[int]   r_fields
    cdef vector[int]   r_offsets

    schedule_c(L, V, nnz, d.shape[0], batch_size, dropconnect,
               &npd[0], &A_row_v[0], &A_col_v[0], &A_data_v[0],
               b_rows, b_cols, b_data, b_offsets, r_fields, r_offsets)

    batches = []
    cdef size_t n_batches = (b_offsets.size()-1) / L
    for i in range(n_batches):
        adjs = []
        fields = []
        # Process fields
        for l in range(L+1):
            s = r_offsets[i*(L+1)+l]
            t = r_offsets[i*(L+1)+l+1]
            f = np.zeros((t-s), dtype=np.int32)
            copy_int(f, &r_fields[s], t-s)
            fields.append(f)
        # Process adjs
        for l in range(L):
            s = b_offsets[i*L+l]
            t = b_offsets[i*L+l+1]
            r = np.zeros((t-s), dtype=np.int32)
            c = np.zeros((t-s), dtype=np.int32)
            d = np.zeros((t-s), dtype=np.float32)
            copy_int(r, &b_rows[s], t-s)
            copy_int(c, &b_cols[s], t-s)
            copy_float(d, &b_data[s], t-s)
            shape = (len(fields[l]), len(fields[l+1]))
            adjs.append(coo_matrix((d, (r, c)), shape=shape))

        adjs.reverse()
        fields.reverse()
        batches.append(Batch(fields=fields, adjs=adjs))

    return batches
        
