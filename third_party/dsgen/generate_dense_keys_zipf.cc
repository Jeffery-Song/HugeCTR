#include <cstdint>
#include <cmath>
#include <random>
#include <omp.h>
#include <iostream>

// def generate_dense_keys_zipf(num_samples, num_slots, vocabulary_range_per_slot, key_dtype, alpha):
//     dense_keys = np.empty((num_samples, num_slots), key_dtype)
//     for i in range(num_slots):
//         vocab_range = vocabulary_range_per_slot[i]
//         H = vocab_range[1] - vocab_range[0] + 1
//         L = 1
//         rnd_rst = np.random.uniform(0.0, 1.0, size=num_samples)
//         rnd_rst = ((-H**alpha * L**alpha) / (rnd_rst * (H**alpha - L**alpha) - H**alpha)) ** (1 / alpha)
//         keys_per_slot = rnd_rst.astype(key_dtype) + vocab_range[0] - 1
//         dense_keys[:, i] = keys_per_slot
//     return dense_keys
#define KDType uint32_t
inline KDType gen_one_key(double rnd, double H_p_a, double L_p_a, double alpha) {
  double rst = powl(((-H_p_a * L_p_a) / (rnd * (H_p_a - L_p_a) - H_p_a)), (1 / alpha));
  return (KDType)rst;
}

extern "C" {

void generate_dense_keys_zipf(size_t num_samples, size_t num_slots, size_t* vocabulary_range_per_slot, KDType* output, double alpha) {
  std::cout << "generating dense keys " << num_samples * num_slots << "\n";
  uint64_t seed_l1 = 0x1234876590f;
  std::mt19937_64 seed_l1_gen(seed_l1);
  std::uniform_int_distribution<uint64_t> seed_l1_dist;
  std::mt19937 generator[120];
  for (int i = 0; i < 120; i++) {
    generator[i] = std::mt19937(seed_l1_dist(seed_l1_gen));
  }

  std::vector<size_t> vocab_low_list(num_slots); //   = vocabulary_range_per_slot[i * 2];
  std::vector<size_t> vocab_high_list(num_slots); //  = vocabulary_range_per_slot[i * 2 + 1];
  std::vector<double> H_p_a_list(num_slots); //  = powl(H, alpha);
  std::vector<double> L_p_a_list(num_slots); //  = powl(L, alpha);

  for (int i = 0; i < num_slots; i++) {
    vocab_low_list[i]  = vocabulary_range_per_slot[i * 2];
    vocab_high_list[i] = vocabulary_range_per_slot[i * 2 + 1];

    double H = vocab_high_list[i] - vocab_low_list[i];
    double L = 1;
    H_p_a_list[i] = powl(H, alpha);
    L_p_a_list[i] = powl(L, alpha);
  }
  std::uniform_real_distribution<double> l2_dist(0,1);
  #pragma omp parallel for num_threads(12)
  for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx ++) {
    if ((sample_idx) % (1000 * 8192) == 0) {
      std::cout << "generated " << sample_idx << " rows\n";
    }
    for (int i = 0; i < num_slots; i++) {
      double rnd = l2_dist(generator[omp_get_thread_num()]);
      output[sample_idx * num_slots + i] = gen_one_key(rnd, H_p_a_list[i], L_p_a_list[i], alpha) + vocab_low_list[i] - 1;
    }
  }
}

}
