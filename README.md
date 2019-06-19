# Stochastic Training for Graph Convolutional Networks

Code for the paper [Stochastic Training for Graph Convolutional Networks](https://arxiv.org/abs/1710.10568). The implementation is based on Thomas Kipf's [implementation for graph convolutional networks](https://github.com/tkipf/gcn).

## Requirements
* tensorflow (>0.12)
* networkx (<=1.11 for loading GraphSAGE datasets)

## Installation
Make sure that you have valid c++11 and cuda compiler.

```bash
pip install --upgrade pip
pip install numpy scipy networkx==1.11 tensorflow-gpu scikit-learn
python setup.py install
cd gcn
./build.sh
```

## Data

We support both [GCN](https://github.com/tkipf/gcn)'s data format and [GraphSAGE](https://github.com/williamleif/GraphSAGE)'s data format.

To download the Citeseer, Cora, PubMed, PPI and Reddit dataset

    mkdir data
    cd data
    git clone git@github.com:tkipf/gcn.git
    mv gcn/gcn/data/* .

    wget http://snap.stanford.edu/graphsage/ppi.zip
    unzip ppi.zip
    mv ppi/* .

    wget http://snap.stanford.edu/graphsage/reddit.zip
    unzip reddit.zip
    mv reddit/* .


## Run the demo

We provide recipes for the Citeseer, Cora, PubMed, NELL, PPI, and Reddit dataset in the `config` folder. Please refer to our paper for the algorithms. Typically, CVD+PP has similar accuracy with Exact, but is faster.

```bash
# Exact algorithm
config/cora.config
# NS+PP algorithm
config/cora.config --degree=1 --test_degree=1
# CV+PP algorithm
config/cora.config --cv --test_cv --degree=1 --test_degree=1 
# CVD+PP algorithm
config/cora.config --cv --cvd --test_cv --degree=1 --test_degree=1
```

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{chen2018stochastic,
  title={Stochastic Training of Graph Convolutional Networks with Variance Reduction},
  author={Chen, Jianfei and Zhu, Jun and Song, Le},
  booktitle={International Conference on Machine Learning},
  pages={941--949},
  year={2018}
}
```
