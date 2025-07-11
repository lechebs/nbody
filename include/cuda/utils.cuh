#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdint>

typedef uint64_t morton_t;

constexpr int THREADS_PER_BLOCK = 32;
// TODO: compute it such that a SM is fully utilized
constexpr int MAX_THREADS_PER_BLOCK = 512;

struct LessOp
{
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs)
  {
    return lhs < rhs;
  }
};

// TODO: move to utils_gpu.cu
inline int log2(int n)
{
    int it = 0;
    int val = 1;
    while (val < n) {
        val *= 2;
        it++;
    }
    return it;
}

// TODO: move to utils_gpu.cu
inline int geometric_sum(int base, int n)
{
    int pow = 1;
    for (int i = 0; i < n + 1; i++) {
        pow *= base;
    }
    return (1 - pow) / (1 - base);
}

template<typename T> void swap_ptr(T **ptr1, T **ptr2)
{
    T *tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

#endif

