#pragma once

#include <chrono>

template<typename Func, typename... Args>
double timeit(Func func, Args... args)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(args...);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return duration.count();
}