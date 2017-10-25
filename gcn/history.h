#include <vector>

//void compute_history(float *adj_w, int *adj_i, int *adjp, int num_data, int num_edges,
//                     float *history, int dims, float *output);

void c_indptr(int N, int *r, int *a_i, int *o_i);
void c_slice(int N, int *r, float *a_d, int *a_i, int *a_p, float *o_d, int *o_i, int *o_p);

void c_dense_slice(int N, int C, int *r, float *i_data, float *o_data);
