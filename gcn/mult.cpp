#include "mult.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
using namespace std;

Mult::Mult(const std::vector<float> &prob) : prob(prob) {
    N = prob.size();
    while (N != lowbit(N)) N += lowbit(N);
    bit.resize(N+1);
    fill(bit.begin(), bit.end(), 0);
    sum = 0;

    for (int i = 0; i < prob.size(); i++)
        Add(i+1, prob[i]);

    if (prob.empty())
        throw runtime_error("Prob is empty");
    max_result = prob.size() - 1;
}

void Mult::Add(int idx, float val) {
    while (idx <= N) {
        bit[idx] += val;
        idx += lowbit(idx);
    }
    sum += val;
}

int Mult::Query() {
    float u = u01(generator) * sum;
    auto result = min(Query(u), max_result);
    Add(result+1, -prob[result]);
    prob[result] = 0;
    return result;
}

int Mult::Query(float u) {
    int curr_idx = 0;
    int step_size = N;
    while (step_size > 0) {
        if (curr_idx + step_size > N || bit[curr_idx + step_size] > u)
            step_size /= 2;
        else {
            u -= bit[curr_idx + step_size];
            curr_idx += step_size;
            step_size /= 2;
        }
    }
    return curr_idx;
}
