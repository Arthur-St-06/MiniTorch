#pragma once

#include "cuda_common.h"

void add_cpu(floatX* _data1, floatX* _data2, floatX* _result_data, int _size);
void arange_cpu(floatX* _data, int _start, int _size);