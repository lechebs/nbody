#ifndef INITIAL_CONDITIONS_CUH
#define INITIAL_CONDITIONS_CUH

#include <random>
#include <vector>
#include <cmath>

#include "cuda/soa_vec3.cuh"

template<typename T> class Spawner
{
public:
    Spawner(SoAVec3<T> &pos, SoAVec3<T> &vel, int num_bodies, int seed) :
        pos_(pos),
        vel_(vel),
        num_bodies_(num_bodies),
        rng_(seed)
    {
        for (int i = 0; i < 3; ++i) {
            tmp_pos_[i].resize(num_bodies);
            tmp_vel_[i].resize(num_bodies);
        }
    }

    void sample_uniform_pos()
    {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < num_bodies_; ++j) {
                tmp_pos_[i][j] = rand();
            }
        }

        to_gpu(pos_, tmp_pos_);
    }

    void sample_spherical_pos(float rad)
    {
        for (int i = 0; i < num_bodies_; ++i) {
            T r = std::pow(rand(), 1.0 / 1.0) * rad;
            T theta = std::acos(2.0 * rand() - 1.0);
            T phi = rand() * 2 * M_PI;

            tmp_pos_[0][i] = 0.5 + r * std::sin(theta) * std::cos(phi);
            tmp_pos_[1][i] = 0.5 + r * std::sin(theta) * std::sin(phi);
            tmp_pos_[2][i] = 0.5 + r * std::cos(theta);
        }

        to_gpu(pos_, tmp_pos_);
    }

    /*
    void sample_plummer(float a)
    {
        for (int i = 0; i < num_bodies_; ++i) {
            T u = rand();
            T r = a * std::pow((std::pow(u, -2.0 / 3) - 1), -0.5);

            z[i] = rand() * 2 - 1.0;

            T r_xy = std::sqrt(1 - z[i] * z[i]);
            T theta = rand() * 2 * M_PI;

            tmp_pos_[0][i] = r_xy * std::cos(theta) * r * 0.5 + 0.5;
            tmp_pos_[1][i] = r_xy * std::sin(theta) * r * 0.5 + 0.5;
            tmp_pos_[2][i] = r * z[i] * 0.5 + 0.5;
        }

        to_gpu(pos_, tmp_pos_);
    }
    */

    void sample_rotating_disk(float rad)
    {
        std::normal_distribution dist_norm(0.0, 1.0);

        for (int i = 0; i < num_bodies_; ++i) {
            T r = std::pow(rand(), 0.5) * rad;
            T theta = rand() * 2 * M_PI;

            T x = r * std::cos(theta);
            T y = r * std::sin(theta);

            tmp_pos_[0][i] = 0.5 + x;
            tmp_pos_[1][i] = 0.5 + y;
            tmp_pos_[2][i] = 0.5 + dist_norm(rng_) * (1 - r) * (1 - r) * 0.01;

            T M = num_bodies_ * (r / rad) * (r / rad);
            T mag = std::sqrt(M / r);

            tmp_vel_[0][i] = -std::sin(theta) * mag;
            tmp_vel_[1][i] = std::cos(theta) * mag;
            tmp_vel_[2][i] = (rand() - 0.5);
        }

        to_gpu(pos_, tmp_pos_);
        to_gpu(vel_, tmp_vel_);
    }

    /*
    void sample_tangent_vel(float mag)
    {
        for (int i = 0; i < n; ++i) {
            T x = x_pos[i] - 0.5;
            T y = y_pos[i] - 0.5;
            T r = std::sqrt(x * x + y * y);
            x_dst[i] = -y * 100;
            y_dst[i] = x * 100;
            z_dst[i] = 0.0;//(std::rand() - RAND_MAX / 2 ) / RAND_MAX / 2 * 500;
        }

    }
    */

private:
    T rand()
    {
        return dist_u_(rng_);
    }

    void to_gpu(SoAVec3<T> &dst, std::array<std::vector<T>, 3> &src)
    {
        T *dst_ptrs[] = { dst.x(), dst.y(), dst.z() };

        for (int i = 0; i < 3; ++i) {
            cudaMemcpy(dst_ptrs[i],
                       src[i].data(),
                       num_bodies_ * sizeof(T),
                       cudaMemcpyHostToDevice);
        }
    }

    SoAVec3<T> &pos_;
    SoAVec3<T> &vel_;

    int num_bodies_;

    std::default_random_engine rng_;
    std::uniform_real_distribution<T> dist_u_;

    std::array<std::vector<T>, 3> tmp_pos_;
    std::array<std::vector<T>, 3> tmp_vel_;
};

#endif
