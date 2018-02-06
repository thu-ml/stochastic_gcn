#ifndef __MULT
#define __MULT

#include <vector>
#include <random>

// Sample multinomial without replacement
struct Mult {
    Mult(const std::vector<float> &prob);

    int Sample();

    int lowbit(int x) { return x & (-x); }

    void Add(int idx, float val);

    int Query();

    int Query(float u);

    std::vector<float> prob;
    std::vector<float> bit;
    float sum;
    int N, max_result;
    std::uniform_real_distribution<float> u01;
    std::mt19937 generator;
};

#endif
