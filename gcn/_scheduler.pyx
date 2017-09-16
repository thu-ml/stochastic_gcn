from scipy.sparse import coo_matrix, csr_matrix
from libcpp.vector cimport vector
from libc.string cimport memcpy
cimport numpy as np
import numpy as np
from utils import sparse_to_tuple

cdef extern from "scheduler.h":
    cdef cppclass Scheduler:
        Scheduler() except +
        Scheduler(float*, int*, int*, int, int, int) except +
        void start_batch(int, int*)
        void expand(int)
        vector[float] adj_w, edg_w
        vector[int] field, new_field, adj_i, adj_p, edg_s, edg_t, visited

cdef copy_int(int[:] buffer, int* ptr, int size):
    memcpy(&buffer[0], ptr, size*sizeof(int))

cdef copy_float(float[:] buffer, float* ptr, int size): 
    memcpy(&buffer[0], ptr, size*sizeof(float))


class Batch:
    def __init__(self, fields, adjs):
        self.fields = fields
        self.adjs = adjs

cdef class PyScheduler:
    cdef Scheduler c_sch
    cdef object labels, data, degrees, placeholders
    cdef int L
    cdef int start

    def __init__(self, adj, labels, L, degrees, placeholders, data=None):
        cdef float[:] ad = adj.data
        cdef int[:]   ai = adj.indices
        cdef int[:]   ap = adj.indptr
        self.c_sch = Scheduler(&ad[0], &ai[0], &ap[0],
                               labels.shape[0], adj.data.shape[0], L)
        self.labels = labels
        self.data = data
        self.degrees = degrees
        self.L = L
        self.start = 0
        self.placeholders = placeholders

    def shuffle(self):
        np.random.shuffle(self.data)
        self.start = 0

    def batch(self, data):
        cdef int    fsz
        cdef int[:] dv = data
        fields   = [data]
        adjs     = []
        self.c_sch.start_batch(len(data), &dv[0])
        for l in range(self.L):
            self.c_sch.expand(self.degrees[self.L-l-1])

            # fields
            fsz = self.c_sch.field.size()
            field = np.zeros((fsz), dtype=np.int32)
            copy_int(field, self.c_sch.field.data(), fsz)
            fields.append(field)

            # adjs 
            ne    = self.c_sch.edg_s.size()
            edg_s = np.zeros((ne), dtype=np.int32)
            edg_t = np.zeros((ne), dtype=np.int32)
            edg_w = np.zeros((ne), dtype=np.float32)
            copy_int  (edg_s, self.c_sch.edg_s.data(), ne)
            copy_int  (edg_t, self.c_sch.edg_t.data(), ne)
            copy_float(edg_w, self.c_sch.edg_w.data(), ne)
            shape = (fields[-2].shape[0], fields[-1].shape[0])
            adj   = csr_matrix((edg_w, (edg_s, edg_t)), shape)
            adjs.append(adj)

        fields.reverse()
        adjs.reverse()
        return self.get_feed_dict(fields, adjs)

    def minibatch(self, batch_size):
        if self.start == self.data.shape[0]:
            return None
        end = min(self.data.shape[0], self.start+batch_size)
        batch = self.data[self.start:end]
        self.start = end
        return self.batch(batch)

    def get_feed_dict(self, fields, adjs):
        labels = self.labels[fields[-1]]
        adjs   = sparse_to_tuple(adjs)
        feed_dict = {self.placeholders['adj'][i] : adjs[i] for i in range(self.L)}
        feed_dict[self.placeholders['labels']] = labels
        for i in range(self.L+1):
            feed_dict[self.placeholders['fields'][i]] = fields[i]
        return feed_dict


