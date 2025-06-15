#ifndef INITIAL_CONDITIONS_CUH
#define INITIAL_CONDITIONS_CUH

#include <vector>
#include <cmath>
#include <cstdlib>

#include "cuda/soa_vec3.cuh"
#include "cuda/physics_common.cuh"

template<typename T>
class InitialConditions
{
public:
    static void set_seed(int seed)
    {
        std::srand(seed);
    }

    static void sample_uniform(float domain_size,
                               SoAVec3<T> &dst,
                               int num_bodies)
    {
        Vec3Buff buff;
        resize_buff_(buff, num_bodies);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < num_bodies; ++j) {
                buff[i][j] = rand_u_(0, domain_size);
            }
        }

        to_dev_(dst, buff);
    }

    static void sample_sphere(float radius,
                              float center_x,
                              float center_y,
                              float center_z,
                              float domain_size,
                              SoAVec3<T> &dst,
                              int num_bodies,
                              int offset = 0)
    {
        Vec3Buff buff;
        resize_buff_(buff, num_bodies);

        for (int i = 0; i < num_bodies; ++i) {
            T r = std::pow(rand_u_(0, 1), 1.0 / 3.0) * radius;
            T theta = std::acos(rand_u_(-1, 1));
            T phi = rand_u_(0, 2 * M_PI);

            buff[0][i] = center_x + r * std::sin(theta) * std::cos(phi);
            buff[1][i] = center_y + r * std::sin(theta) * std::sin(phi);
            buff[2][i] = center_z + r * std::cos(theta);
        }

        clamp_buff_values_(buff, 0, domain_size);
        to_dev_(dst, buff, offset);
    }

    static void sample_disk(float r_min,
                            float r_max,
                            float exp,
                            float center_x,
                            float center_y,
                            float center_z,
                            float domain_size,
                            SoAVec3<T> &pos_dst,
                            SoAVec3<T> &vel_dst,
                            int num_bodies,
                            int offset = 0)
    {
        Vec3Buff pos_buff;
        Vec3Buff vel_buff;
        resize_buff_(pos_buff, num_bodies);
        resize_buff_(vel_buff, num_bodies);

        for (int i = 0; i < num_bodies; ++i) {
            T r = rand_powerlaw_(r_min, r_max, exp);
            T theta = rand_u_(0, 2 * M_PI);

            pos_buff[0][i] = center_x + r * std::cos(theta);
            pos_buff[1][i] = center_y + r * std::sin(theta);
            pos_buff[2][i] = center_z + r * rand_norm_(0, 0.1);

            // Computing circular velocity mass

            T r_pow = std::pow(r, exp + 1);
            T r_min_pow = std::pow(r_min, exp + 1);
            T r_max_pow = std::pow(r_max, exp + 1);

            T star_mass = 3000000.0;
            T disk_mass = num_bodies;
            T enc_mass = star_mass + disk_mass * (r_pow - r_min_pow) /
                                                 (r_max_pow - r_min_pow);
            T vel_kep = std::sqrt(GRAVITY * enc_mass / r);

            vel_buff[0][i] = vel_kep * std::sin(theta);
            vel_buff[1][i] = -vel_kep * std::cos(theta);
            vel_buff[2][i] = 0.0;
        }

        clamp_buff_values_(pos_buff, 0, domain_size);
        to_dev_(pos_dst, pos_buff, offset);
        to_dev_(vel_dst, vel_buff, offset);
    }

private:
    using Vec3Buff = std::array<std::vector<T>, 3>;

    static T rand_u_(T min, T max)
    {
        return (T) std::rand() / RAND_MAX * (max - min) + min;
    }

    static T rand_powerlaw_(T min, T max, T exp)
    {
        T max_pow = std::pow(max, exp + 1);
        T min_pow = std::pow(min, exp + 1);

        return std::pow(rand_u_(0, 1) * (max_pow - min_pow) + min_pow,
                        1 / (exp + 1));
    }

    static T rand_norm_(T mean, T std)
    {
        // Box-muller trick
        T r = std::sqrt(-2 * std::log(rand_u_(0, 1)));
        T theta = rand_u_(0, 2 * M_PI);

        return r * std::cos(theta) * std + mean;
    }

    static void resize_buff_(Vec3Buff &buff, int size)
    {
        for (int i = 0; i < 3; ++i) {
            buff[i].resize(size);
        }
    }

    static void clamp_buff_values_(Vec3Buff &buff, T min, T max)
    {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < buff.size(); ++j) {
                buff[i][j] = std::max(min, std::min(max, buff[i][j]));
            }
        }
    }

    static void to_dev_(SoAVec3<T> &dst, Vec3Buff &src, int offset = 0)
    {
        std::array<T *, 3> dst_ptrs = { dst.x(), dst.y(), dst.z() };
        for (int i = 0; i < 3; ++i) {
            cudaMemcpy(dst_ptrs[i] + offset,
                       src[i].data(),
                       src[i].size() * sizeof(T),
                       cudaMemcpyHostToDevice);
        }
    }
};

/*
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
*/

#endif
