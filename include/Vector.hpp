#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include <array>
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <iostream>
#include <vector>
// #include <cassert>

template <typename T, size_t N>
class Vector
{
private:
    std::array<T, N> comps;

public:
    // Default vectors
    static const Vector ZERO;
    // Constructor
    Vector() { comps.fill(0); }

    Vector(const std::array<T,N> &init) : comps(init) {}
    Vector(const Vector &init) : comps(init.comps) {}
    // Constructor that takes initializer_list
    Vector(const std::initializer_list<T> &init)
    {
        if (init.size() != N)
        {
            throw std::invalid_argument("Initializer list size must match vector dimension");
        }
        std::copy(init.begin(), init.end(), comps.begin());
    }

    inline Vector &operator=(const Vector &rhs) = default;

    Vector(const std::vector<T> &init)
    {
        if (init.size() != N)
        {
            throw std::invalid_argument("Initializer list size must match vector dimension");
        }

        std::copy(init.begin(), init.end(), comps.begin());
    }

    // // Constructor to initialize a Vector
    // Vector(const std::initializer_list<T>& init) {
    //     assert(init.size() == N && "Initializer list size must match vector dimensions.");
    //     std::copy(init.begin(), init.end(), comps.begin());
    // }

    // Arithmetic operators
    inline friend Vector operator+(const Vector &lhs, const Vector &rhs)
    {
        Vector res;
        for (size_t i = 0; i < N; i++)
        {
            res[i] = lhs[i] + rhs[i];
        }
        return res;
    }

    Vector &operator+=(const Vector &rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    inline friend Vector<T, N> operator-(const Vector &lhs, const Vector &rhs)
    {
        Vector res;
        for (size_t i = 0; i < N; i++)
        {
            res[i] = lhs[i] - rhs[i];
        }
        return res;
    }

    Vector &operator-=(const Vector &rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    inline friend Vector operator*(const T &scalar, const Vector &rhs)
    {
        Vector res;
        for (size_t i = 0; i < N; i++)
        {
            res[i] = rhs[i] * scalar;
        }

        return res;
    }

    Vector &operator*=(const T &scalar)
    {
        *this = scalar * *this;
        return *this;
    }

    // Cross product specialization for 3d vectors
    inline static Vector<T, 3> cross(const Vector<T, 3> &lhs,
                               const Vector<T, 3> &rhs)
    {
        Vector<T, 3> res;

        res[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
        res[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
        res[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];

        return res;
    }

    inline friend Vector operator*(const Vector &rhs,const T &scalar)
    {
        return scalar*rhs;
    }

    inline friend Vector operator/(const Vector &rhs,const T &scalar)
    {
        return (1.0/scalar)*rhs;
    }

    // WARNING: THIS IS DANGEROUS WHEN WORKING WITH FLOATS AND DOUBLES
    inline friend bool operator==(const Vector &lhs, const Vector &rhs)
    {
        for (size_t i = 0; i < N; i++)
            if (lhs[i] != rhs[i])
                return false;
        return true;
    }

    // Vector utilities
    inline const T norm() const
    {
        T norm = 0;
        for (size_t i = 0; i < N; i++)
        {
            norm += comps[i] * comps[i];
        }

        return sqrt(norm);
    }

    // Normalizes the vector in place
    inline Vector &normalize()
    {
        T _norm = norm();

        for (size_t i = 0; i < N; i++) {
            comps[i] /= _norm;
        }

        return *this;
    }

    // Access operators
    inline T &operator[](size_t coord)
    {
        return comps[coord];
    }
    inline const T &operator[](size_t coord) const
    {
        return comps[coord];
    }

    const T *data() const
    {
        return comps.data();
    }
        //
    // Stream operators
    inline friend std::ostream &operator<<(std::ostream &os, const Vector &vec)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            os << vec.comps[i];
            if (i < N - 1)
                os << " ";
        }
        return os;
    }
};

// Definition of some useful types
typedef Vector<double, 2> Vector2d;
typedef Vector<double, 3> Vector3d;

// Explicit definition of static members is needed
template<typename T, size_t N>
const Vector<T, N> Vector<T, N>::ZERO;

#endif
