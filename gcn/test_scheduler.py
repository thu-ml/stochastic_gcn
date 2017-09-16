from scipy.sparse import coo_matrix, csr_matrix, diags
from scheduler import PyScheduler
import numpy as np

# A = coo_matrix(np.array([[10, 1, 0, 0, 2],
#                          [1, 10, 0, 0, 3],
#                          [0, 0, 10, 0, 4],
#                          [0, 0, 0, 10, 5],
#                          [2, 3, 4, 5, 10]]).astype(np.float32))
edges = np.array([(0, 1, 1), (0, 2, 1), (0, 3, 1),
                 (1, 4, 1), (1, 5, 1), (1, 6, 1),
                 (2, 7, 1), (2, 8, 1), (2, 9, 1),
                 (3, 10, 1)])
adj = csr_matrix((edges[:,2], (edges[:,0], edges[:,1])), 
                 shape=(11, 11), dtype=np.float32)
adj = adj + adj.transpose()
# Row normalize
deg = np.array(adj.sum(axis=0)).flatten()
deg = diags(1.0/deg, 0)
adj = deg.dot(adj)
print(adj)


labels = np.zeros((11, 2))
sch = PyScheduler(adj, labels, 2, [1, 2])
fields, adjs = sch.batch(np.array([0], dtype=np.int32))
print(fields[0])
print(fields[1])
print(fields[2])
print(adjs[0])
print('-')
print(adjs[1])
