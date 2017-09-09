#include <vector>
using namespace std;

void schedule_c(int L, int V, int E, int N, int batch_size,
        int *d, int *e_s, int *e_t, float *e_w, 
        vector<int>& b_rows, vector<int>& b_cols, 
        vector<float>& b_data, vector<int>& b_offsets,
        vector<int>& r_fields, vector<int>& r_offsets);
