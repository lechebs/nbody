#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>

#include "vector.hpp"

template<typename T, size_t N, size_t M>
class Matrix
{
public:

    static Matrix identity() requires (N == M)
    {
        Matrix mat;

        for (size_t i = 0; i < N; ++i) {
            mat(i, i) = 1.0;
        }

        return mat;
    } 

    // friend keyword is needed here otherwise the compiler
    // will interpret the function as a member function that
    // is trying to overload operator*, which requires only
    // one parameter. So in this case it's not related to the
    // access to private data.
    // Remember that friend functions are non-member functions.
    template<size_t Q, size_t P>
    inline friend Matrix<T, N, Q>
    operator*(const Matrix<T, N, M> &lhs,
              const Matrix<T, P, Q> &rhs) requires (M == P)
    {
        Matrix<T, N, Q> res;

        for (size_t k = 0; k < M; ++k) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < M; ++j) {
                    res(i, k) += lhs(i, j) * rhs(j, k);
                }
            }
        }

        return res;
    }

    inline T &operator()(size_t i, size_t j)
    {
        return _data[i][j];
    }

    inline const T &operator()(size_t i, size_t j) const
    {
        return _data.at(i)[j];
    }

    // Provides read only access to underlying buffer
    const T *data() const {
        return &_data[0][0];
    }

private:
    std::array<Vector<T, M>, N> _data;
};

#endif
