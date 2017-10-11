#include "scheduler.h"
#include <iostream>
#include <random>
#include <algorithm>

std::mt19937 generator;
uniform_real_distribution<float> u01;


Scheduler::Scheduler(float *adj_w, int *adj_i, int *adj_p, int num_data, int num_edges,
          int L) :
    adj_w(adj_w, adj_w+num_edges),
    adj_i(adj_i, adj_i+num_edges),
    adj_p(adj_p, adj_p+num_data),
    L(L), num_data(num_data), 
    visited(num_data, -1), 
    degree(num_data), scale(num_data), node_sum(num_data)
{
    this->adj_p.push_back(num_edges);

    // Compute degree
    for (int i = 0; i < num_data; i++)
        degree[i] = adj_p[i+1] - adj_p[i];
    // Compute node scale
    for (int i = 0; i < num_data; i++) {
        float s = 1.0 / degree[i];
        for (int it = adj_p[i]; it < adj_p[i+1]; it++) {
            int j = adj_i[it];
            scale[j] += s;
        }
    }
    for (int i = 0; i < num_data; i++)
        scale[i] = 1.0 / scale[i];
    // Compute node sum
    for (int i = 0; i < num_data; i++)
        for (int it = adj_p[i]; it < adj_p[i+1]; it++) {
            int j = adj_i[it];
            node_sum[i] += scale[j];
        }
}

void Scheduler::start_batch(int num_data, int *data) {
    field.clear();
    field.insert(field.begin(), data, data+num_data);
}

void Scheduler::expand(int degree) {
    if (degree > 0)
        _expand(degree);
    else
        _power_expand();
}

void Scheduler::_expand(int degree) {
    //cout << "Expanding" << endl;
    //for (auto e: adj_p) cout << e << ' '; cout << endl;
    //for (auto e: adj_w) cout << e << ' '; cout << endl;
    //for (auto e: adj_i) cout << e << ' '; cout << endl;

    // Add self edges
    new_field.clear();
    new_field.insert(new_field.begin(), field.begin(), field.end());
    for (int i=0; i<new_field.size(); i++)
        visited[new_field[i]] = i;
    edg_s.clear();
    edg_t.clear();
    edg_w.clear();

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
        }
    }

    field.swap(new_field);
    for (int s: field)
        visited[s] = -1;
}

void Scheduler::_power_expand() {
    // Add self edges
    new_field.clear();
    new_field.insert(new_field.begin(), field.begin(), field.end());
    for (int i=0; i<new_field.size(); i++)
        visited[new_field[i]] = i;
    edg_s.clear();
    edg_t.clear();
    edg_w.clear();

    // Add neighbour edges
    for (int i = 0; i < field.size(); i++) {
        // Sample a neighbour
        int s        = field[i];
        int   *row_i = adj_i.data() + adj_p[s];
        float *row_w = adj_w.data() + adj_p[s];
        int   adj_range = adj_p[s+1] - adj_p[s];

        float u = u01(generator) * node_sum[s];       
        int it = 0; 
        for (it = 0; it < adj_range-1; it++) {
            u -= scale[row_i[it]];
            if (u <= 0) 
                break;
        }
        int   t = row_i[it];
        float w = row_w[it] * adj_range * node_sum[s] / scale[t];
        if (visited[t] == -1) {
            visited[t] = new_field.size();
            new_field.push_back(t);
        }
        edg_s.push_back(i);
        edg_t.push_back(visited[t]);
        edg_w.push_back(w);
    }

    field.swap(new_field);
    for (int s: field)
        visited[s] = -1;
}
