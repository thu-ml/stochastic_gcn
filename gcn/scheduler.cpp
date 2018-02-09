#include "scheduler.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include "mult.h"

uniform_real_distribution<float> u01;


Scheduler::Scheduler(float *adj_w, int *adj_i, int *adj_p, int num_data, int num_edges,
          int L, bool cv, bool is) :
    cv(cv), is(is),
    adj_w(adj_w, adj_w+num_edges),
    adj_i(adj_i, adj_i+num_edges),
    adj_p(adj_p, adj_p+num_data),
    L(L), num_data(num_data), 
    visited(num_data, -1), fvisited(num_data, -1), importance(num_data, 1e-6)
{
    this->adj_p.push_back(num_edges);
    // The importance is proportional with the out degree, i.e., adj[:,c]
    if (is) {
        for (int i = 0; i < num_data; i++)
            for (int p = adj_p[i]; p < adj_p[i+1]; p++)
                importance[adj_i[p]] += adj_w[p] * adj_w[p];
        //float sum = 0;
        //for (int i = 0; i < num_data; i++)
        //    sum += importance[i] = sqrt(importance[i]);
        //for (int i = 0; i < num_data; i++)
        //    importance[i] = 1;
    } else {
        for (int i = 0; i < num_data; i++)
            importance[i] = 1;
    }
}

void Scheduler::seed(int seed) {
    generator.seed(seed);
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
    scales.clear();


    if (is) {
        // Figure out all the neighbors and probs
        std::vector<int> neighbors;
        std::vector<float> probs;
        std::vector<bool> v2(num_data);
        std::vector<int> times(num_data);
        float total_importance = 0;
        for (int i: field)
            for (int p = adj_p[i]; p < adj_p[i+1]; p++) 
                if (!v2[adj_i[p]]) {
                    int t = adj_i[p];
                    v2[t] = true;
                    neighbors.push_back(t);
                    total_importance += importance[t];
                    probs.push_back(importance[t]);
                }

        // Importance sampling (TODO without replacement)
        //std::discrete_distribution<> mult(probs.begin(), probs.end());
        Mult mult(probs);
        int num_nbrs = neighbors.size();
        //int num_samples = num_nbrs;
        int num_samples = min(field.size() * degree, neighbors.size());
        //int num_samples = field.size() * degree;
        
        //probs.clear();
        //shuffle(neighbors.begin(), neighbors.end(), generator);
        for (int it = 0; it < num_samples; it++) {
            //int t = neighbors[mult(generator)];
            int t = neighbors[mult.Query()];
            //int t = neighbors[generator() % neighbors.size()];
            //int t = neighbors[it];
            times[t]++;
            if (visited[t] == -1) {
                visited[t] = new_field.size();
                new_field.push_back(t);
            }
        }

        // Add edges
        for (int i = 0; i < field.size(); i++)
            for (int p = adj_p[field[i]]; p < adj_p[field[i]+1]; p++) {
                if (times[adj_i[p]]) {
                    int t = adj_i[p];
                    float weight = times[t] * adj_w[p] * 
                                   total_importance / (importance[t] * num_samples);
                    //float weight = times[t] * adj_w[p] * num_nbrs / num_samples;
                    //cout << (float)times[t] * num_nbrs / num_samples << endl;
                    edg_s.push_back(i); 
                    edg_t.push_back(visited[t]);
                    edg_w.push_back(weight);
                    if (std::isnan(weight))
                        throw std::runtime_error("nan");
                }
            }

        field.swap(new_field);
        for (int s: field)
            visited[s] = -1;
        return;
    }

    // Add neighbour edges
    for (int i = 0; i < field.size(); i++) {
        int    s     = field[i];
        int   *row_i = adj_i.data() + adj_p[s];
        float *row_w = adj_w.data() + adj_p[s];
        int   adj_range = adj_p[s+1] - adj_p[s];
        int   adj_size  = min(adj_range, degree);
        float scale = (float)adj_range / adj_size;
        if (adj_range==0) scale = 1;
        scales.push_back(1.0 / sqrt(scale));
        
        // cout << scale << endl;

        float total_weight = 0;
        
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
            int vt = visited[t];
            edg_s.push_back(i);
            edg_t.push_back(vt);
            edg_w.push_back(w);

            if (is) {
                float iw = importance[vt];
                edg_w.push_back(w * iw);
                total_weight += iw;
            }

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
