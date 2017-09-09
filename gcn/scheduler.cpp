#include "scheduler.h"
#include <random>
#include <algorithm>

std::mt19937 generator;

struct Edge {
    int t;
    float w;
};

void schedule_c(int L, int V, int E, int N, int batch_size,
        int *d, int *e_s, int *e_t, float *e_w, 
        vector<int>& b_rows, vector<int>& b_cols, 
        vector<float>& b_data, vector<int>& b_offsets,
        vector<int>& r_fields, vector<int>& r_offsets)
{
    // Construct adjacency matrix
    vector<vector<Edge>> adj(V);
    for (int i = 0; i < E; i++)
        adj[e_s[i]].push_back(Edge{e_t[i], e_w[i]});

    // Construct mini batches
    vector<int> shuf_d(d, d+N);
    shuffle(shuf_d.begin(), shuf_d.end(), generator);

    vector<int> new_rf;
    vector<int> visited(V, -1);
    b_offsets.push_back(0);
    r_offsets.push_back(0);
    for (int b_s = 0; b_s < N; b_s += batch_size) {
        int b_t = min(N, b_s+batch_size);

        // rf for the last layer
        vector<int> current_rf(shuf_d.begin()+b_s, shuf_d.begin()+b_t);
        auto insert_r = [&]() {
            r_fields.insert(r_fields.end(), current_rf.begin(), current_rf.end());
            r_offsets.push_back(r_fields.size());
        };
        insert_r();
        for (int l=0; l < L; l++) {
            new_rf.clear();

            for (int i=0; i<current_rf.size(); i++) {
                int s = current_rf[i];
                for (auto e: adj[s]) {
                    if (visited[e.t] == -1) {
                        visited[e.t] = new_rf.size();
                        new_rf.push_back(e.t);
                    }
                    b_rows.push_back(i);
                    b_cols.push_back(visited[e.t]);
                    b_data.push_back(e.w);
                }
            }
            for (auto t: new_rf)
                visited[t] = -1;
            b_offsets.push_back(b_rows.size());
            current_rf = new_rf;
            insert_r();
        }
    }
}
