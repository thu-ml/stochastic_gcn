#include "scheduler.h"
#include <iostream>
#include <random>
#include <algorithm>

std::mt19937 generator;
uniform_real_distribution<float> u01;


Scheduler::Scheduler(float *adj_w, int *adj_i, int *adj_p, int num_data, int num_edges,
          int L, bool cv) :
    cv(cv),
    adj_w(adj_w, adj_w+num_edges),
    adj_i(adj_i, adj_i+num_edges),
    adj_p(adj_p, adj_p+num_data),
    L(L), num_data(num_data), 
    visited(num_data, -1), fvisited(num_data, -1)
{
    this->adj_p.push_back(num_edges);
}

void Scheduler::start_batch(int num_data, int *data) {
    field.clear();
    field.insert(field.begin(), data, data+num_data);
}

void Scheduler::expand(int degree) {
    // Add self edges
    new_field.clear();
    ffield.clear();
    new_field.insert(new_field.begin(), field.begin(), field.end());
    for (int i=0; i<new_field.size(); i++)
        visited[new_field[i]] = i;
    edg_s.clear();
    edg_t.clear();
    edg_w.clear();
    medg_w.clear();
    fedg_s.clear();
    fedg_t.clear();
    fedg_w.clear();

    // Add neighbour edges
    for (int i = 0; i < field.size(); i++) {
        int    s     = field[i];
        int   *row_i = adj_i.data() + adj_p[s];
        float *row_w = adj_w.data() + adj_p[s];
        int   adj_range = adj_p[s+1] - adj_p[s];
        int   adj_size  = min(adj_range, degree);
        float scale = (float)adj_range / adj_size;
        // cout << scale << endl;

        for (int it = 0; it < adj_size; it++) {
            int num_remaining = adj_range - it;
            int idx = min((int)(it + num_remaining * u01(generator)), 
                          adj_range-1);
            swap(row_i[it], row_i[idx]);
            swap(row_w[it], row_w[idx]);
            int   t = row_i[it];
            float w = row_w[it] * scale;
            if (visited[t] == -1) {
                visited[t] = new_field.size();
                new_field.push_back(t);
            }
            edg_s.push_back(i);
            edg_t.push_back(visited[t]);
            edg_w.push_back(w);
            if (cv)
                medg_w.push_back(row_w[it] * w);
        }

        if (cv) {
            for (int it = 0; it < adj_range; it++) {
                int t   = row_i[it];
                float w = row_w[it];
                if (fvisited[t] == -1) {
                    fvisited[t] = ffield.size();
                    ffield.push_back(t);
                }
                fedg_s.push_back(i);
                fedg_t.push_back(fvisited[t]);
                fedg_w.push_back(w);
            }
        }
    }

    field.swap(new_field);
    for (int s: field)
        visited[s] = -1;
    if (cv) {
        for (int s: ffield)
            fvisited[s] = -1;
    }
}
