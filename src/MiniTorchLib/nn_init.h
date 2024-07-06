#pragma once

#include <math.h>
#include "rand.h"

namespace nn
{
	// Copyright(c) Makoto Matsumoto and Takuji Nishimura
	class init
	{
	public:
		void uniform_(float* data, unsigned int numel, float from, float to, mt19937_state* state);
		void normal_(float* data, unsigned int numel, float mean, float std, mt19937_state* state);
	private:
		void next_state(mt19937_state* state);
		unsigned int randint32(mt19937_state* state);
		inline unsigned long long randint64(mt19937_state* state);
		inline float randfloat32(mt19937_state* state);
		inline double randfloat64(mt19937_state* state);

		void normal_fill_16(float* data, float mean, float std, mt19937_state* state);
		void normal_fill(float* data, unsigned int numel, float mean, float std, mt19937_state* state);
	};
}