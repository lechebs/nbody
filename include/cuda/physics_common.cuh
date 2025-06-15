#ifndef PHYSICS_COMMON_CUH
#define PHYSICS_COMMON_CUH

#include <type_traits>

#include "cuda/soa_vec3.cuh"

// TODO: place in constant memory
#define GRAVITY 1.0f
#define SOFTENING_FACTOR 1e-3f
#define VELOCITY_DAMPENING 0.0f
//#define VELOCITY_DAMPENING (1.0f - 0.8236f)
//#define VELOCITY_DAMPENING (1.0f - 0.8136f) for 2mln
#define DIST_SCALE 1.0f

namespace
{
    template<typename T>
    __device__ void impose_boundary_conditions(T &x, T &y, T &z,
                                               T &vx, T &vy, T &vz)
    {
        // Reflective boundary conditions

        if (x < 0.0f || x > 1.0f) {
            vx *= -1.0f;
            x = max(0.0f, min(1.0f, x));
        }

        if (y < 0.0f || y > 1.0f) {
            vy *= -1.0f;
            y = max(0.0f, min(1.0f, y));
        }

        if (z < 0.0f || z > 1.0f) {
            vz *= -1.0f;
            z = max(0.0f, min(1.0f, z));
        }
    }
}

template<typename T>
__device__ __forceinline__ T compute_dist_sq(T p1x, T p1y, T p1z,
                                             T p2x, T p2y, T p2z)
{
    T dx = p1x - p2x;
    T dy = p1y - p2y;
    T dz = p1z - p2z;

    return dx * dx + dy * dy + dz * dz;
}

template<typename T>
__device__ __inline__
void accumulate_pairwise_force(T p1x, T p1y, T p1z,
                               T p2x, T p2y, T p2z,
                               T mass,
                               T &dst_x, T &dst_y, T &dst_z)
{
    T dist_x = p2x - p1x;
    T dist_y = p2y - p1y;
    T dist_z = p2z - p1z;

    T dist_sq = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

    T inv_den = DIST_SCALE * dist_sq + SOFTENING_FACTOR * SOFTENING_FACTOR;

    if constexpr (std::is_same_v<T, float>) {
        // Use fast inverse sqrt intrinsic
        inv_den = __frsqrt_rn(inv_den * inv_den * inv_den);
    } else {
        // Invert sqrt intrinsic
        inv_den = 1 / __dsqrt_rn(inv_den * inv_den * inv_den);
    }

    dst_x += mass * GRAVITY * dist_x * inv_den;
    dst_y += mass * GRAVITY * dist_y * inv_den;
    dst_z += mass * GRAVITY * dist_z * inv_den;
}

template<typename T>
__global__ void leapfrog_integrate_pos(SoAVec3<T> pos,
                                       const SoAVec3<T> vel,
                                       SoAVec3<T> vel_half,
                                       const SoAVec3<T> acc,
                                       float dt,
                                       int num_bodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    T x = pos.x(idx);
    T y = pos.y(idx);
    T z = pos.z(idx);

    T vx = vel.x(idx) + (T) 0.5f * acc.x(idx) * dt;
    T vy = vel.y(idx) + (T) 0.5f * acc.y(idx) * dt;
    T vz = vel.z(idx) + (T) 0.5f * acc.z(idx) * dt;

    x += vx * dt;
    y += vy * dt;
    z += vz * dt;

    impose_boundary_conditions(x, y, z, vx, vy, vz);

    pos.x(idx) = x;
    pos.y(idx) = y;
    pos.z(idx) = z;

    vel_half.x(idx) = vx;
    vel_half.y(idx) = vy;
    vel_half.z(idx) = vz;
}

template<typename T>
__global__ void leapfrog_integrate_vel(SoAVec3<T> vel,
                                       const SoAVec3<T> vel_half,
                                       const SoAVec3<T> acc,
                                       float dt,
                                       int num_bodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    T vx = vel_half.x(idx) + (T) 0.5f * acc.x(idx) * dt;
    T vy = vel_half.y(idx) + (T) 0.5f * acc.y(idx) * dt;
    T vz = vel_half.z(idx) + (T) 0.5f * acc.z(idx) * dt;

    vel.x(idx) = vx * ((T) 1.0f - VELOCITY_DAMPENING);
    vel.y(idx) = vy * ((T) 1.0f - VELOCITY_DAMPENING);
    vel.z(idx) = vz * ((T) 1.0f - VELOCITY_DAMPENING);
}

#endif
