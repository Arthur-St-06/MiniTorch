#pragma once

// Needed in random functions for nn_init
#define MERSENNE_STATE_N 624u
#define MERSENNE_STATE_M 397u
#define LMASK 0x7ffffffful
#define UMASK 0x80000000ul

typedef struct mt19937_state
{
    unsigned long long seed_;
    int left_;
    unsigned int next_;
    unsigned int state_[MERSENNE_STATE_N];
    unsigned int MATRIX_A[2];
};

inline void manual_seed(mt19937_state* state, unsigned int seed) {
    state->MATRIX_A[0] = 0x0u;
    state->MATRIX_A[1] = 0x9908b0df;
    state->state_[0] = seed & 0xffffffff;
    for (unsigned int j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->left_ = 1;
    state->next_ = 0;
}