#include <vector>
#include <memory>
using namespace std;

class Scheduler {
public:
    Scheduler() {}
    Scheduler(float *adj_w, int *adj_i, int *adj_p, int num_data, int num_edges, 
              int L);

    void start_batch(int num_data, int *data);
    void expand(int degree);

    void _expand(int degree);
    void _power_expand();

public:
    vector<float> adj_w;
    // weight of (i, j) is scale[j]
    vector<int>  adj_i, adj_p;
    int L, num_data, mode; 
    vector<int>  field, new_field, edg_s, edg_t;
    vector<float> edg_w;
    vector<int> visited;
    vector<float> degree, scale, node_sum;
};
