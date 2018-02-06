#include <vector>
#include <memory>
#include <random>
using namespace std;

class Scheduler {
public:
    Scheduler() {}
    Scheduler(float *adj_w, int *adj_i, int *adj_p, int num_data, int num_edges, 
              int L, bool cv, bool is);

    void start_batch(int num_data, int *data);
    void expand(int degree);
    void seed(int seed);

public:
    bool cv, is;
    vector<int>  adj_i, adj_p; vector<float> adj_w;
    int L, num_data, mode; 
    vector<int>  field, ffield, new_field;
    vector<float> scales;
    vector<float> importance;

    vector<int> edg_s, edg_t; vector<float> edg_w, medg_w;
    vector<int> fedg_s, fedg_t; vector<float> fedg_w;
    vector<int> visited, fvisited;
    std::mt19937 generator;
};
