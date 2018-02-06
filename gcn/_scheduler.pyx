from scipy.sparse import coo_matrix, csr_matrix
from libcpp.vector cimport vector
from libc.string cimport memcpy
cimport numpy as np
import numpy as np
from utils import sparse_to_tuple
from time import time
from libcpp cimport bool

cdef extern from "scheduler.h":
    cdef cppclass Scheduler:
        Scheduler() except +
        Scheduler(float*, int*, int*, int, int, int, bool, bool) except +
        void start_batch(int, int*)
        void expand(int)
        void seed(int)
        vector[float] adj_w, edg_w, medg_w, fedg_w, scales
        vector[int] field, new_field, ffield, adj_i, adj_p, edg_s, edg_t, fedg_s, fedg_t, visited
        bool cv

cdef copy_int(int[:] buffer, int* ptr, int size):
    memcpy(&buffer[0], ptr, size*sizeof(int))

cdef copy_float(float[:] buffer, float* ptr, int size): 
    memcpy(&buffer[0], ptr, size*sizeof(float))


cdef class PyScheduler:
    cdef Scheduler c_sch
    cdef object labels, data, degrees, placeholders
    cdef int L
    cdef int start
    cdef float t

    def __init__(self, adj, labels, L, degrees, placeholders, seed, data=None, cv=False, importance=False):
        cdef float[:] ad = adj.data
        cdef int[:]   ai = adj.indices
        cdef int[:]   ap = adj.indptr
        self.c_sch = Scheduler(&ad[0], &ai[0], &ap[0],
                               labels.shape[0], adj.data.shape[0], L, cv, importance)
        self.c_sch.seed(seed)
        self.labels = labels
        self.data = data
        self.degrees = degrees
        self.L = L
        self.start = 0
        self.placeholders = placeholders
        self.t = 0

    def shuffle(self):
        np.random.shuffle(self.data)
        self.start = 0
        self.t = 0

    def batch(self, data):
        cdef int    fsz
        cdef int[:] dv = data
        fields   = [data]
        ffields  = []
        adjs     = []
        madjs    = []
        fadjs    = []
        scales   = []
        self.c_sch.start_batch(len(data), &dv[0])
        for l in range(self.L):
            self.c_sch.expand(self.degrees[self.L-l-1])

            # fields
            fsz = self.c_sch.field.size()
            field = np.zeros((fsz), dtype=np.int32)
            copy_int(field, self.c_sch.field.data(), fsz)
            fields.append(field)

            ssz = self.c_sch.scales.size()
            scale = np.zeros((ssz), dtype=np.float32)
            if ssz > 0:
                copy_float(scale, self.c_sch.scales.data(), ssz)
            scales.append(scale)

            # adjs 
            ne    = self.c_sch.edg_s.size()
            edg_s = np.zeros((ne), dtype=np.int32)
            edg_t = np.zeros((ne), dtype=np.int32)
            edg_w = np.zeros((ne), dtype=np.float32)
            edg_i = np.zeros((ne, 2), dtype=np.int32)
            copy_int  (edg_s, self.c_sch.edg_s.data(), ne)
            copy_int  (edg_t, self.c_sch.edg_t.data(), ne)
            copy_float(edg_w, self.c_sch.edg_w.data(), ne)
            shape = (fields[-2].shape[0], fields[-1].shape[0])
            edg_i[:,0] = edg_s
            edg_i[:,1] = edg_t
            adj   = (edg_i, edg_w, shape)
            adjs.append(adj)

            if self.c_sch.cv:
                fsz = self.c_sch.ffield.size()
                ffield = np.zeros((fsz), dtype=np.int32)
                copy_int(ffield, self.c_sch.ffield.data(), fsz)
                ffields.append(ffield)

                ne2 = self.c_sch.fedg_s.size()
                medg_w = np.zeros((ne), dtype=np.float32)
                fedg_s = np.zeros((ne2), dtype=np.int32)
                fedg_t = np.zeros((ne2), dtype=np.int32)
                fedg_w = np.zeros((ne2), dtype=np.float32)
                copy_float(medg_w, self.c_sch.medg_w.data(), ne)
                copy_int  (fedg_s, self.c_sch.fedg_s.data(), ne2)
                copy_int  (fedg_t, self.c_sch.fedg_t.data(), ne2)
                copy_float(fedg_w, self.c_sch.fedg_w.data(), ne2)

                fedg_i = np.zeros((ne2, 2), dtype=np.int32)
                fedg_i[:,0] = fedg_s
                fedg_i[:,1] = fedg_t
                fshape = (fields[-2].shape[0], ffields[-1].shape[0])

                madj = (np.copy(edg_i), medg_w, np.copy(shape))
                madjs.append(madj)
                fadj = (fedg_i, fedg_w, fshape)
                fadjs.append(fadj)

        fields.reverse()
        ffields.reverse()
        adjs.reverse()
        madjs.reverse()
        fadjs.reverse()
        scales.reverse()
        return self.get_feed_dict(fields, ffields, adjs, madjs, fadjs, scales)

    def minibatch(self, batch_size):
        if self.start == self.data.shape[0]:
            return None
        end = min(self.data.shape[0], self.start+batch_size)
        batch = self.data[self.start:end]
        self.start = end
        return self.batch(batch)

    def get_feed_dict(self, fields, ffields, adjs, madjs, fadjs, scales):
        labels = self.labels[fields[-1]]
        feed_dict = {self.placeholders['adj'][i] : adjs[i] for i in range(self.L)}
        feed_dict.update({self.placeholders['scales'][i]: scales[i] for i in range(len(scales))})
        if self.c_sch.cv:
            feed_dict.update({self.placeholders['madj'][i] : madjs[i] for i in range(len(madjs))})
            feed_dict.update({self.placeholders['fadj'][i] : fadjs[i] for i in range(len(fadjs))})
            feed_dict.update({self.placeholders['ffields'][i] : ffields[i] for i in range(len(ffields))})
        feed_dict[self.placeholders['labels']] = labels
        for i in range(self.L+1):
            feed_dict[self.placeholders['fields'][i]] = fields[i]
        return feed_dict

    def get_t(self):
        return self.t

