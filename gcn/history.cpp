#include "history.h"
#include <omp.h>
#include <chrono>
#include <iostream>
#include <memory.h>
//#include <mkl.h>
using namespace std;
using namespace std::chrono;

//void compute_history(float *adj_w, int *adj_i, int *adjp, int num_data, int num_edges,
//                     int *f, int fsize,
//                     float *history, int dims, float *output) {
//    std::vector<int> adj_p(adjp, adjp+num_data);
//    adj_p.push_back(num_edges);
//
//    int flops = 0;
//    auto start_t = chrono::high_resolution_clock::now();
//    #pragma omp parallel for
//    for (int i = 0; i < fsize; i++) {
//        int r = f[i];
//        int rnnz = adjp[r+1] - adjp[r];
//        int *idx = adj_i + adjp[r];
//        float *val = adj_w + adjp[r];
//        float *o   = output + i*dims;
//        for (int j = 0; j < rnnz; j++) {
//            float *c = history + idx[j] * dims;
//            float v  = val[j];
//            for (int k = 0; k < dims; k++) {
//                o[k] += v * c[k];
//            }
//        }
//        flops += rnnz * dims * 2;
//    }
//    auto end_t = chrono::high_resolution_clock::now();
//    auto t = chrono::duration<double>(end_t-start_t).count();
//    cout << float(flops) / 1024 / 1024 / t << ' ' << flops << ' ' << t << endl;
//}

//void compute_history(float *adj_w, int *adj_i, int *adjp, int num_data, int num_edges,
//                     float *history, int dims, float *output) {
//    vector<int> ptrb(adjp, adjp+num_data);
//    vector<int> ptre(adjp+1, adjp+num_data); ptre.push_back(num_edges);
//    float alpha = 1;
//    float beta = 0;
//    mkl_scsrmm("N", &num_data, &dims, &num_data, 
//               &alpha, "G00C00", adj_w, adj_i, ptrb.data(), ptre.data(), 
//               history, &dims, &beta, output, &dims);
//}

void c_indptr(int N, int *r, int *a_i, int *o_i) {
    int nnz = 0;
    for (int i = 0; i < N; i++) {
        o_i[i] = nnz;
        nnz += a_i[r[i]+1] - a_i[r[i]];
    }
    o_i[N] = nnz;
}

void c_slice(int N, int *r, float *a_d, int *a_i, int *a_p, float *o_d, int *o_i, int *o_p) {
    for (int i = 0; i <N; i++){
        auto sz = o_p[i+1] - o_p[i];
        memcpy(o_d+o_p[i], a_d+a_p[r[i]], sz*sizeof(float));

        int *oi = o_i + o_p[i] * 2;
        int *ai = a_i + a_p[r[i]];
        int len = o_p[i+1] - o_p[i];
        for (int j = 0; j < len; j++) {
            oi[j*2]   = i;
            oi[j*2+1] = ai[j];
        }
    }
}

void c_dense_slice(int N, int C, int *r, float *i_data, float *o_data) {
    auto start_t = high_resolution_clock::now();
    int  flops   = N * C * sizeof(float);
//#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        auto *id = i_data + r[i] * C;
        auto *od = o_data + i * C;
        for (int c = 0; c < C; c++)
            od[c] = id[c];
        //memcpy(o_data+i*C, i_data+r[i]*C, C*sizeof(float));
    }
    //auto end_t   = high_resolution_clock::now();
    //auto t       = duration<double>(end_t-start_t).count();
    //cout << "Finished in " << t << " seconds, " << flops/t/1024/1024 << " MB/s " << N << " " << C << endl;
}
