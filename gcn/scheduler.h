#include <vector>
using namespace std;

// 做一个stale SGD，每次随机更新activation，其他用这个activation上一轮的值
// 分析activation更新传来的导数，应该比起weight更新造成的导数是高阶无穷小？

void schedule_c(int L, int V, int E, int N, int batch_size, float dropconnect,
        int *d, int *e_s, int *e_t, float *e_w, 
        vector<int>& b_rows, vector<int>& b_cols, 
        vector<float>& b_data, vector<int>& b_offsets,
        vector<int>& r_fields, vector<int>& r_offsets);
