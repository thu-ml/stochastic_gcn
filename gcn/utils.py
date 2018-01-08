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
import os, copy

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


def load_gcn_data(dataset_str):
    npz_file = 'data/{}_{}.npz'.format(dataset_str, FLAGS.normalization)
    if os.path.exists(npz_file):
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(npz_file)
        num_data     = data['num_data']
        labels       = data['labels']
        train_data   = data['train_data']
        val_data     = data['val_data']
        test_data    = data['test_data']
        train_adj = sp.csr_matrix((data['train_adj_data'], data['train_adj_indices'], data['train_adj_indptr']), shape=data['train_adj_shape'])
        full_adj = sp.csr_matrix((data['full_adj_data'], data['full_adj_indices'], data['full_adj_indptr']), shape=data['full_adj_shape'])
        feats = sp.csr_matrix((data['feats_data'], data['feats_indices'], data['feats_indptr']), shape=data['feats_shape'])
        train_feats = sp.csr_matrix((data['train_feats_data'], data['train_feats_indices'], data['train_feats_indptr']), shape=data['train_feats_shape'])
        test_feats = sp.csr_matrix((data['test_feats_data'], data['test_feats_indices'], data['test_feats_indptr']), shape=data['test_feats_shape'])
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

        if dataset_str != 'nell':
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
        else:
            test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
            features = allx.tocsr()
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
        labels = (y_train + y_val + y_test).astype(np.float32)
        train_data = np.nonzero(train_mask)[0].astype(np.int32)
        val_data   = np.nonzero(val_mask)[0].astype(np.int32)
        test_data  = np.nonzero(test_mask)[0].astype(np.int32)

        feats = (features.data, features.indices, features.indptr, features.shape)

        def _get_adj(data, coords):
            adj = sp.csr_matrix((data, (coords[0,:], coords[1,:])), 
                                shape=(num_data, num_data))
            return adj

        train_adj = _get_adj(train_v, train_coords)
        full_adj  = _get_adj(full_v,  full_coords)
        feats = sp.csr_matrix((feats[0], feats[1], feats[2]), 
                              shape=feats[-1], dtype=np.float32)

        train_feats = train_adj.dot(feats)
        test_feats  = full_adj.dot(feats)

        with open(npz_file, 'wb') as fwrite:
            np.savez(fwrite, num_data=num_data, 
                             train_adj_data=train_adj.data, train_adj_indices=train_adj.indices, train_adj_indptr=train_adj.indptr, train_adj_shape=train_adj.shape,
                             full_adj_data=full_adj.data, full_adj_indices=full_adj.indices, full_adj_indptr=full_adj.indptr, full_adj_shape=full_adj.shape,
                             feats_data=feats.data, feats_indices=feats.indices, feats_indptr=feats.indptr, feats_shape=feats.shape,
                             train_feats_data=train_feats.data, train_feats_indices=train_feats.indices, train_feats_indptr=train_feats.indptr, train_feats_shape=train_feats.shape,
                             test_feats_data=test_feats.data, test_feats_indices=test_feats.indices, test_feats_indptr=test_feats.indptr, test_feats_shape=test_feats.shape,
                             labels=labels,
                             train_data=train_data, val_data=val_data, 
                             test_data=test_data)

    return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data


def load_graphsage_data(prefix, normalize=True):
    version_info = map(int, nx.__version__.split('.'))
    major = version_info[0]
    minor = version_info[1]
    assert (major <= 1) and (minor <= 11), "networkx major version must be <= 1.11 in order to load graphsage data"

    # Save normalized version
    if FLAGS.max_degree==-1:
        npz_file = prefix + '.npz'
    else:
        npz_file = '{}_deg{}.npz'.format(prefix, FLAGS.max_degree)

    if os.path.exists(npz_file):
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(npz_file)
        num_data     = data['num_data']
        feats        = data['feats']
        train_feats  = data['train_feats']
        test_feats   = data['test_feats']
        labels       = data['labels']
        train_data   = data['train_data']
        val_data     = data['val_data']
        test_data    = data['test_data']
        train_adj = sp.csr_matrix((data['train_adj_data'], data['train_adj_indices'], data['train_adj_indptr']), shape=data['train_adj_shape'])
        full_adj  = sp.csr_matrix((data['full_adj_data'], data['full_adj_indices'], data['full_adj_indptr']), shape=data['full_adj_shape'])
        print('Finished in {} seconds.'.format(time() - start_time))
    else:
        print('Loading data...')
        start_time = time()
    
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)
    
        feats = np.load(prefix + "-feats.npy").astype(np.float32)
        id_map = json.load(open(prefix + "-id_map.json"))
        if id_map.keys()[0].isdigit():
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n
        id_map = {conversion(k):int(v) for k,v in id_map.iteritems()}

        walks = []
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(class_map.values()[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)
    
        class_map = {conversion(k): lab_conversion(v) for k,v in class_map.iteritems()}

        ## Remove all nodes that do not have val/test annotations
        ## (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        to_remove = []
        for node in G.nodes():
            if not id_map.has_key(node):
            #if not G.node[node].has_key('val') or not G.node[node].has_key('test'):
                to_remove.append(node)
                broken_count += 1
        for node in to_remove:
            G.remove_node(node)
        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
    
        # Construct adjacency matrix
        print("Loaded data ({} seconds).. now preprocessing..".format(time()-start_time))
        start_time = time()
    
        edges = []
        for edge in G.edges():
            if id_map.has_key(edge[0]) and id_map.has_key(edge[1]):
                edges.append((id_map[edge[0]], id_map[edge[1]]))
        print('{} edges'.format(len(edges)))
        num_data   = len(id_map)

        if FLAGS.max_degree != -1:
            print('Subsampling edges...')
            edges = subsample_edges(edges, num_data, FLAGS.max_degree)

        val_data   = np.array([id_map[n] for n in G.nodes() 
                                 if G.node[n]['val']], dtype=np.int32)
        test_data  = np.array([id_map[n] for n in G.nodes() 
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

        def _get_adj(data, coords):
            adj = sp.csr_matrix((data, (coords[0,:], coords[1,:])),
                                shape=(num_data, num_data))
            return adj
        
        train_adj = _get_adj(train_v, train_coords)
        full_adj  = _get_adj(full_v,  full_coords)
        train_feats = train_adj.dot(feats)
        test_feats  = full_adj.dot(feats)

        print("Done. {} seconds.".format(time()-start_time))
        with open(npz_file, 'wb') as fwrite:
            print('Saving {} edges'.format(full_adj.nnz))
            np.savez(fwrite, num_data=num_data, 
                             train_adj_data=train_adj.data, train_adj_indices=train_adj.indices, train_adj_indptr=train_adj.indptr, train_adj_shape=train_adj.shape,
                             full_adj_data=full_adj.data, full_adj_indices=full_adj.indices, full_adj_indptr=full_adj.indptr, full_adj_shape=full_adj.shape,
                             feats=feats, train_feats=train_feats, test_feats=test_feats,
                             labels=labels,
                             train_data=train_data, val_data=val_data, 
                             test_data=test_data)

    return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data


def load_youtube_data(prefix, ptrain):
    npz_file = 'data/{}_{}.npz'.format(prefix, ptrain)
    if os.path.exists(npz_file):
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(npz_file)
        num_data     = data['num_data']
        labels       = data['labels']
        train_data   = data['train_data']
        val_data     = data['val_data']
        test_data    = data['test_data']
        adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), 
                            shape=data['adj_shape'])
        feats = sp.csr_matrix((data['feats_data'], data['feats_indices'], data['feats_indptr']), 
                            shape=data['feats_shape'])
        feats1 = sp.csr_matrix((data['feats1_data'], data['feats1_indices'], data['feats1_indptr']), 
                            shape=data['feats1_shape'])
        print('Finished in {} seconds.'.format(time() - start_time))
    else:
        start_time = time()
        # read edges
        with open('data/'+prefix+'/edges.csv') as f:
            links = [link.split(',') for link in f.readlines()]
            links = [(int(link[0])-1, int(link[1])-1) for link in links]
        links = np.array(links).astype(np.int32)
        num_data = np.max(links)+1
        adj = sp.csr_matrix((np.ones(links.shape[0], dtype=np.float32), 
                             (links[:,0], links[:,1])),
                             shape=(num_data, num_data))
        adj = adj + adj.transpose()

        def _normalize_adj(adj):
            rowsum = np.array(adj.sum(1)).flatten()
            d_inv  = 1.0 / (rowsum+1e-20)
            d_mat_inv = sp.diags(d_inv, 0)
            adj = d_mat_inv.dot(adj)
            return adj

        adj = _normalize_adj(adj)

        feats = sp.eye(num_data, dtype=np.float32).tocsr()
        feats1 = adj.dot(feats)
        num_classes = 47

        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        with open('data/'+prefix+'/group-edges.csv') as f:
            for line in f.readlines():
                line = line.split(',')
                labels[int(line[0])-1, int(line[1])-1] = 1

        data = np.nonzero(labels.sum(1))[0].astype(np.int32)

        np.random.shuffle(data)
        n_train = int(len(data)*ptrain)
        train_data = np.copy(data[:n_train])
        val_data   = np.copy(data[n_train:])
        test_data  = np.copy(data[n_train:])

        num_data, adj, feats, feats1, labels, train_data, val_data, test_data = \
                data_augmentation(num_data, adj, adj, feats, labels, 
                                  train_data, val_data, test_data)

        print("Done. {} seconds.".format(time()-start_time))
        with open(npz_file, 'wb') as fwrite:
            np.savez(fwrite, num_data=num_data, 
                             adj_data=adj.data, adj_indices=adj.indices,
                             adj_indptr=adj.indptr, adj_shape=adj.shape,
                             feats_data=feats.data, feats_indices=feats.indices,
                             feats_indptr=feats.indptr, feats_shape=feats.shape,
                             feats1_data=feats1.data, feats1_indices=feats1.indices,
                             feats1_indptr=feats1.indptr, feats1_shape=feats1.shape,
                             labels=labels,
                             train_data=train_data, val_data=val_data, 
                             test_data=test_data)

    return num_data, adj, feats, feats1, labels, train_data, val_data, test_data


def data_augmentation(num_data, train_adj, full_adj, feats, labels, train_data, val_data, test_data, n_rep=1):
    if isinstance(feats, np.ndarray):
        feats  = np.tile(feats,  [n_rep+1, 1])
    else:
        feats  = sp.vstack([feats] * (n_rep+1))
    labels = np.tile(labels, [n_rep+1, 1])

    train_adj  = train_adj.tocoo()
    full_adj   = full_adj.tocoo()

    i    = []
    j    = []
    data = []

    def add_adj(adj, t):
        i.append(adj.row + t*num_data)
        j.append(adj.col + t*num_data)
        data.append(adj.data)

    for t in range(n_rep):
        add_adj(train_adj, t)
    add_adj(full_adj, n_rep)

    adj = sp.csr_matrix((np.concatenate(data), (np.concatenate(i), np.concatenate(j))),
                        shape=np.array(train_adj.shape)*(n_rep+1), dtype=train_adj.dtype)

    new_train = []
    for t in range(n_rep):
        new_train.append(train_data + t*num_data)
    train_data = np.concatenate(new_train)

    val_data  += n_rep * num_data
    test_data += n_rep * num_data
    return num_data*(n_rep+1), adj, feats, adj.dot(feats), labels, train_data, val_data, test_data


def np_dropout(feats, keep_prob):
    mask = np.random.rand(feats.shape[0], feats.shape[1]) < keep_prob
    return feats * mask.astype(np.float32) * (1.0 / keep_prob)


def np_sparse_dropout(feats, keep_prob):
    feats = feats.tocoo()
    mask  = np.random.rand(feats.data.shape[0]) < keep_prob
    feats = sp.csr_matrix((feats.data[mask], (feats.row[mask], feats.col[mask])),
                          shape=feats.shape, dtype=feats.dtype) 
    feats = feats * (1.0 / keep_prob)
    return feats


def load_data(dataset):
    gcn_datasets = set(['cora', 'citeseer', 'pubmed', 'nell'])
    if dataset in gcn_datasets:
        return load_gcn_data(dataset)
    elif dataset == 'youtube':
        return load_youtube_data(dataset, 0.9)
    else:
        return load_graphsage_data('data/{}'.format(dataset))


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


def subsample_edges(edges, num_data, max_degree):
    edges = np.array(edges, dtype=np.int32)
    np.random.shuffle(edges)
    degree = np.zeros(num_data, dtype=np.int32)

    new_edges = []
    for e in edges:
        if degree[e[0]]<max_degree and degree[e[1]]<max_degree:
            new_edges.append((e[0], e[1]))
            degree[e[0]]+=1
            degree[e[1]]+=1
    return new_edges
