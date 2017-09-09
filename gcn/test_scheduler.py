from scipy.sparse import coo_matrix
from scheduler import schedule
import numpy as np

A = coo_matrix(np.array([[10, 1, 0, 0, 2],
                         [1, 10, 0, 0, 3],
                         [0, 0, 10, 0, 4],
                         [0, 0, 0, 10, 5],
                         [2, 3, 4, 5, 10]]).astype(np.float32))

d = np.array([0, 1, 2, 3, 4]).astype(np.int32)
L = 1
batch_size = 2

batches = schedule(A, d, L, batch_size)
for batch in batches:
    print('Batch')
    print('Fields')
    for f in batch.fields:
        print(f)
    print('Adjs')
    for a in batch.adjs:
        print(a)

