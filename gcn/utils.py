import numpy as np
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.metrics import f1_score
import sys
import tensorflow as tf
import json
from time import time
import os

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    npz_file = 'data/{}_{}.npz'.format(dataset_str, FLAGS.normalization)
    if os.path.exists(npz_file):
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(npz_file)
        num_data     = data['num_data']
        full_v       = data['full_v']
        full_coords  = data['full_coords']
        train_v      = data['train_v']
        train_coords = data['train_coords']
        feats        = data['feats']
        labels       = data['labels']
        train_data   = data['train_data']
        val_data     = data['val_data']
        test_data    = data['test_data']
        print('Finished in {} seconds.'.format(time() - start_time))
    else:
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        #print(y.shape, ty.shape, ally.shape)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        # num_data, (v, coords), feats, labels, train_d, val_d, test_d
        num_data = features.shape[0]
        def _normalize_adj(adj):
            rowsum = np.array(adj.sum(1)).flatten()
            d_inv  = 1.0 / (rowsum+1e-20)
            d_mat_inv = sp.diags(d_inv, 0)
            adj = d_mat_inv.dot(adj).tocoo()
            coords = np.array((adj.row, adj.col)).astype(np.int32)
            return adj.data.astype(np.float32), coords

        def gcn_normalize_adj(adj):
            adj = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj.sum(1)) + 1e-20
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
            adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            adj = adj.tocoo()
            coords = np.array((adj.row, adj.col)).astype(np.int32)
            return adj.data.astype(np.float32), coords

        # Normalize features
        rowsum = np.array(features.sum(1)) + 1e-9
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        features = r_mat_inv.dot(features)

        if FLAGS.normalization == 'gcn':
            full_v, full_coords = gcn_normalize_adj(adj)
        else:
            full_v, full_coords = _normalize_adj(adj)
        full_v = full_v.astype(np.float32)
        full_coords = full_coords.astype(np.int32)
        train_v, train_coords = full_v, full_coords
        feats = features.todense().astype(np.float32)
        labels = (y_train + y_val + y_test).astype(np.float32)
        train_data = np.nonzero(train_mask)[0].astype(np.int32)
        val_data   = np.nonzero(val_mask)[0].astype(np.int32)
        test_data  = np.nonzero(test_mask)[0].astype(np.int32)

        with open(npz_file, 'wb') as fwrite:
            np.savez(fwrite, num_data=num_data, 
                             full_v=full_v,   full_coords=full_coords,
                             train_v=train_v, train_coords=train_coords,
                             feats=feats, labels=labels,
                             train_data=train_data, val_data=val_data, 
                             test_data=test_data)

    def _get_adj(data, coords):
        adj = sp.csr_matrix((data, (coords[0,:], coords[1,:])), 
                            shape=(num_data, num_data))
        return adj

    train_adj = _get_adj(train_v, train_coords)
    full_adj  = _get_adj(full_v,  full_coords)

    return num_data, train_adj, full_adj, feats, labels, train_data, val_data, test_data


def old_load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_nell_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']

    objects = []
    for i in range(len(names)):
        with open("data/{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print tx.shape
    
    test_idx_reorder = parse_index_file("data/{}.test.index".format(dataset_str))
    features = allx.tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = ally
    idx_test = test_idx_reorder
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+969)
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_graphsage_data(prefix, normalize=True):
    # Save normalized version
    npz_file = prefix + '.npz'
    if os.path.exists(npz_file):
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(prefix + '.npz')
        num_data     = data['num_data']
        full_v       = data['full_v']
        full_coords  = data['full_coords']
        train_v      = data['train_v']
        train_coords = data['train_coords']
        feats        = data['feats']
        labels       = data['labels']
        train_data   = data['train_data']
        val_data     = data['val_data']
        test_data    = data['test_data']
        print('Finished in {} seconds.'.format(time() - start_time))
    else:
        print('Loading data...')
        start_time = time()
    
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)
        if isinstance(G.nodes()[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n
    
        feats = np.load(prefix + "-feats.npy").astype(np.float32)
        id_map = json.load(open(prefix + "-id_map.json"))
        id_map = {conversion(k):int(v) for k,v in id_map.iteritems()}
        walks = []
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(class_map.values()[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)
    
        class_map = {conversion(k): lab_conversion(v) for k,v in class_map.iteritems()}
    
        # Construct adjacency matrix
        print("Loaded data ({} seconds).. now preprocessing..".format(time()-start_time))
        start_time = time()
    
        edges      = [(id_map[edge[0]], id_map[edge[1]]) for edge in G.edges_iter()]
        num_data   = len(id_map)
        val_data   = np.array([id_map[n] for n in G.nodes_iter() 
                                 if G.node[n]['val']], dtype=np.int32)
        test_data  = np.array([id_map[n] for n in G.nodes_iter() 
                                 if G.node[n]['test']], dtype=np.int32)
        is_train   = np.ones((num_data), dtype=np.bool)
        is_train[val_data] = False
        is_train[test_data] = False
        train_data = np.array([n for n in range(num_data) if is_train[n]], dtype=np.int32)
        
        train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]
        edges       = np.array(edges, dtype=np.int32)
        train_edges = np.array(train_edges, dtype=np.int32)
    
        # Process labels
        if isinstance(class_map.values()[0], list):
            num_classes = len(class_map.values()[0])
            labels = np.zeros((num_data, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], :] = np.array(class_map[k])
        else:
            num_classes = len(set(class_map.values()))
            labels = np.zeros((num_data, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], class_map[k]] = 1
    
        if normalize:
            from sklearn.preprocessing import StandardScaler
            train_ids = np.array([id_map[n] for n in G.nodes() 
                          if not G.node[n]['val'] and not G.node[n]['test']])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)

        def _normalize_adj(edges):
            adj = sp.csr_matrix((np.ones((edges.shape[0]), dtype=np.float32),
                (edges[:,0], edges[:,1])), shape=(num_data, num_data))
            adj += adj.transpose()

            rowsum = np.array(adj.sum(1)).flatten()
            d_inv  = 1.0 / (rowsum+1e-20)
            d_mat_inv = sp.diags(d_inv, 0)
            adj = d_mat_inv.dot(adj).tocoo()
            coords = np.array((adj.row, adj.col)).astype(np.int32)
            return adj.data, coords

        train_v, train_coords = _normalize_adj(train_edges)
        full_v,  full_coords  = _normalize_adj(edges)
    
        print("Done. {} seconds.".format(time()-start_time))
        with open(npz_file, 'wb') as fwrite:
            np.savez(fwrite, num_data=num_data, 
                             full_v=full_v,   full_coords=full_coords,
                             train_v=train_v, train_coords=train_coords,
                             feats=feats, labels=labels,
                             train_data=train_data, val_data=val_data, 
                             test_data=test_data)

    def _get_adj(data, coords):
        adj = sp.csr_matrix((data, (coords[0,:], coords[1,:])), 
                            shape=(num_data, num_data))
        return adj

    train_adj = _get_adj(train_v, train_coords)
    full_adj  = _get_adj(full_v,  full_coords)

    return num_data, train_adj, full_adj, feats, labels, train_data, val_data, test_data


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        # print(mx.sum(1))
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords.astype(np.int64), values, np.array(shape).astype(np.int64)

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def tuple_to_coo(tuple_mx):
    def to_coo(t):
        return sp.coo_matrix((t[1], (t[0][:,0],t[0][:,1])), t[2])
    if isinstance(tuple_mx, list):
        for i in range(len(tuple_mx)):
            tuple_mx[i] = to_tuple(tuple_mx[i])
    else:
        tuple_mx = to_coo(tuple_mx)
    return tuple_mx



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) + 1e-9
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv, 0)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def fast_construct_feed_dict(features, batch, labels, placeholders):
    """Construct feed dictionary."""
    supports = sparse_to_tuple(batch.adjs)
    f = sparse_to_tuple(features[batch.fields[0]])
    y = labels[batch.fields[-1]]

    feed_dict = dict()
    feed_dict.update({placeholders['labels']: y})
    feed_dict.update({placeholders['features']: f})
    feed_dict.update({placeholders['support_%d'%(i+1)][0]: supports[i] for i in range(len(supports))})
    feed_dict.update({placeholders['num_features_nonzero']: f[1].shape})
    return feed_dict


def vr_construct_feed_dict(Ax, support, batch, labels, placeholders):
    features = sparse_to_tuple(Ax[batch.fields[1]])
    hidden_fields = batch.fields[1]
    subsampled_support = sparse_to_tuple(batch.adjs[1])
    support = sparse_to_tuple(support[batch.fields[2]])
    y = labels[batch.fields[-1]]

    feed_dict = dict()
    feed_dict.update({placeholders['labels']: y})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['hidden_fields']: hidden_fields})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['subsampled_support']: subsampled_support})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def generate_batches(train_mask, batch_size):
    # Find all non-zeros
    observed = np.array(np.nonzero(train_mask)).flatten()
    observed = np.random.permutation(observed)
    masks = []
    N = len(observed)
    for i in range(0, N, batch_size):
        obs_batch = observed[i:i+batch_size]
        mask = np.zeros_like(train_mask)
        mask[obs_batch] = True
        masks.append(mask)
    return masks

def L_neighbour(mask, adj, L):
    eadj = adj + sp.eye(adj.shape[0])
    current_mask = mask
    for i in range(L):
        current_mask = eadj.dot(current_mask) > 0
    return current_mask


def least_squares(A, B, reg):
    D = int(A.get_shape()[1])
    ATA = tf.matmul(A, A, transpose_a=True) + tf.eye(D) * reg
    ATB = tf.matmul(A, B, transpose_a=True)
    return tf.cholesky_solve(tf.cholesky(ATA), ATB)


def least_squares_A(H, Z, W, gamma, beta):
    # min_A gamma ||A-H||^2 + beta ||Z-AW||^2
    D = int(W.get_shape()[0])
    lhs = tf.transpose(gamma * tf.eye(D) + beta * tf.matmul(W, W, transpose_b=True))
    rhs = tf.transpose(gamma * H + beta * tf.matmul(Z, W, transpose_b=True))
    AT  = tf.cholesky_solve(tf.cholesky(lhs), rhs)
    return tf.transpose(AT)


def hinge_loss(Y, Z):
    # return tf.maximum(0.0, 1.0 - (2*Y-1)*Z)
    return tf.losses.hinge_loss(labels=Y, logits=Z, reduction=tf.losses.Reduction.NONE)


class Averager:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window      = []

    def add(self, n):
        self.window.append(n)
        if len(self.window) > self.window_size:
            self.window = self.window[1:]

    def mean(self):
        return np.mean(self.window)


def calc_f1(y_pred, y_true, multitask):
    if multitask:
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
    else:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")
