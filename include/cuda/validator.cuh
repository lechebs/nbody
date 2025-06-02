#ifndef VALIDATOR_CUH
#define VALIDATOR_CUH

#include "cuda/soa_vec3.cuh"

template<typename T> class Validator
{
public:
    Validator(const SoAVec3<T> &pos,
              const SoAVec3<T> &vel,
              const SoAVec3<T> &acc,
              const int *d_sort_indices,
              int num_bodies,
              float dt,
              int max_timesteps);

    void copy_initial_conditions();

    void update_all_pairs();

    void dump_history_to_csv(const std::string &file_path);

    ~Validator();

private:
    const SoAVec3<T> &pos_;
    const SoAVec3<T> &vel_;
    const SoAVec3<T> &acc_;

    SoAVec3<T> pos_ap_;
    SoAVec3<T> vel_ap_;
    SoAVec3<T> vel_half_ap_;
    SoAVec3<T> acc_ap_;
    SoAVec3<T> tmp_acc_ap_;

    const int *sort_indices_;
    int num_bodies_;
    float dt_;
    int max_timesteps_;

    int curr_step_;

    T *energy_;
    T *energy_ap_;
    T *acc_err_;

    T *tmp_reduce_;
    size_t tmp_reduce_size_;

    T *sys_energy_;
    T *sys_energy_ap_;
    T *sys_momentum_mag_;
    T *sys_momentum_mag_ap_;
    T *avg_acc_err_;
};

#endif
