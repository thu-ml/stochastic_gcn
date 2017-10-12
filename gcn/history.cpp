#include "history.h"
#include <omp.h>
#include <chrono>
#include <iostream>
#include <mkl.h>
using namespace std;

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
void compute_history(float *adj_w, int *adj_i, int *adjp, int num_data, int num_edges,
                     float *history, int dims, float *output) {
    vector<int> ptrb(adjp, adjp+num_data);
    vector<int> ptre(adjp+1, adjp+num_data); ptre.push_back(num_edges);
    float alpha = 1;
    float beta = 0;
    mkl_scsrmm("N", &num_data, &dims, &num_data, 
               &alpha, "G00C00", adj_w, adj_i, ptrb.data(), ptre.data(), 
               history, &dims, &beta, output, &dims);
}
