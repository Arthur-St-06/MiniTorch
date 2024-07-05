#pragma once

#include <chrono>

template<typename Func, typename... Args>
double timeit(Func _func, Args... _args)
{
    auto start = std::chrono::high_resolution_clock::now();
    _func(_args...);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return duration.count();
}