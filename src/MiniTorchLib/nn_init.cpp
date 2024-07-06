#include "nn_init.h"

#define M_PI 3.1415926535897932384
namespace nn
{
    void init::uniform_(float* data, unsigned int numel, float from, float to, mt19937_state* state)
    {
        for (unsigned int t = 0; t < numel; t++) {
            data[t] = randfloat32(state) * (to - from) + from;
        }
    }

    void init::normal_(float* data, unsigned int numel, float mean, float std, mt19937_state* state) {
#define EPSILONE 1e-12
        if (numel >= 16) {
            normal_fill(data, numel, mean, std, state);
        }
        else {
            double next_double_normal_sample = 0.0; // make compiler warning happy, won't be used
            int has_next_double_normal_sample = 0;
            for (unsigned int t = 0; t < numel; t++) {
                if (has_next_double_normal_sample) {
                    data[t] = (float)(next_double_normal_sample * std + mean);
                    has_next_double_normal_sample = 0;
                    continue;
                }
                // for numel < 16 we draw a double (float64)
                float u1 = randfloat64(state);
                float u2 = randfloat64(state);
                float radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
                float theta = 2.0 * M_PI * u1;
                next_double_normal_sample = radius * sinf(theta);
                has_next_double_normal_sample = 1;
                data[t] = (radius * cosf(theta) * std + mean);
            }
        }
    }

    void init::next_state(mt19937_state* state) {
        state->left_ = MERSENNE_STATE_N;
        state->next_ = 0;
        unsigned int y, j;
        for (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
            y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
            state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
        }
        for (; j < MERSENNE_STATE_N - 1; j++) {
            y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
            state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
        }
        y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
        state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }

    unsigned int init::randint32(mt19937_state* state) {
        if (!state) return 0;
        if (state->MATRIX_A[0] != 0 || state->MATRIX_A[1] != 0x9908b0df) manual_seed(state, 5489); // auto-initialize
        if (--state->left_ <= 0) {
            next_state(state);
        }
        unsigned int y = state->state_[state->next_++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    inline unsigned long long init::randint64(mt19937_state* state) {
        return (((unsigned long long)(randint32(state)) << 32) | randint32(state));
    }

    inline float init::randfloat32(mt19937_state* state) {
        return (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
    }

    inline double init::randfloat64(mt19937_state* state) {
        return (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
    }

    // Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    void init::normal_fill_16(float* data, float mean, float std, mt19937_state* state) {
#define EPSILONE 1e-12
        for (unsigned int t = 0; t < 8; t++) {
            float u1 = 1 - data[t];
            float u2 = data[t + 8];
            float radius = sqrtf(-2 * logf(u1 + EPSILONE));
            float theta = 2.0 * M_PI * u2;
            data[t] = (radius * cosf(theta) * std + mean);
            data[t + 8] = (radius * sinf(theta) * std + mean);
        }
    }

    void init::normal_fill(float* data, unsigned int numel, float mean, float std, mt19937_state* state) {
        for (unsigned int t = 0; t < numel; t++) {
            data[t] = randfloat32(state);
        }
        for (unsigned int i = 0; i < numel - 15; i += 16) {
            normal_fill_16(data + i, mean, std, state);
        }
        if (numel % 16 != 0) {
            // recompute the last 16 values
            data = data + numel - 16;
            for (unsigned int i = 0; i < 16; i++) {
                data[i] = randfloat32(state);
            }
            normal_fill_16(data, mean, std, state);
        }
    }
}