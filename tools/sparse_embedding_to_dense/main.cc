#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

void convert(const uint64_t* sparse_key, const uint8_t* sparse_val, size_t val_nbytes_per_key, size_t num_sparse_keys, size_t num_total_keys, uint8_t* output_val) {
  #pragma omp parallel for
  for (size_t i = 0; i < num_sparse_keys; i++) {
    uint64_t key = sparse_key[i];
    memcpy(output_val + key * val_nbytes_per_key, sparse_val + i * val_nbytes_per_key, val_nbytes_per_key);
  }
}

void init(size_t val_nbytes_per_key, size_t num_total_keys, uint8_t* output_val, int val) {
  memset(output_val, val, num_total_keys * val_nbytes_per_key);
}
void init_key(size_t num_total_keys, uint64_t* output_key) {
  #pragma omp parallel for
  for (size_t i = 0; i < num_total_keys; i++) {
    output_key[i] = i;
  }
}

int main(int argc, char** argv) {
  assert(argc == 4);
  const size_t num_total_keys = std::stoull(argv[1]);
  const std::string sparse_path = argv[2];
  const std::string dense_path = argv[3];

  const std::string sparse_key_file = sparse_path + "/key";
  const std::string sparse_val_file = sparse_path + "/emb_vector";

  const std::string dense_key_file = dense_path + "/key";
  const std::string dense_val_file = dense_path + "/emb_vector";

  int sparse_key_fd = open(sparse_key_file.c_str(), O_RDONLY);
  struct stat sparse_key_stat;
  fstat(sparse_key_fd, &sparse_key_stat);
  assert(sparse_key_stat.st_size % sizeof(uint64_t) == 0);
  const size_t num_sparse_keys = sparse_key_stat.st_size / sizeof(uint64_t);
  std::cout << "Detected num sparse keys is " << num_sparse_keys << "\n";

  int sparse_val_fd = open(sparse_val_file.c_str(), O_RDONLY);
  struct stat sparse_val_stat;
  fstat(sparse_val_fd, &sparse_val_stat);
  assert(sparse_val_stat.st_size % num_sparse_keys == 0);
  const size_t val_nbytes_per_key = sparse_val_stat.st_size / num_sparse_keys;
  std::cout << "Detected value nbytes per key is " << val_nbytes_per_key << "\n";

  int dense_key_fd = open(dense_key_file.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
  ftruncate64(dense_key_fd, num_total_keys * sizeof(uint64_t));

  int dense_val_fd = open(dense_val_file.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
  ftruncate64(dense_val_fd, num_total_keys * val_nbytes_per_key);

  const uint64_t* sparse_key = (uint64_t*)mmap(nullptr, num_sparse_keys * sizeof(uint64_t), PROT_READ, MAP_PRIVATE, sparse_key_fd, 0);
  const uint8_t* sparse_val = (uint8_t*)mmap(nullptr, num_sparse_keys * val_nbytes_per_key, PROT_READ, MAP_PRIVATE, sparse_val_fd, 0);

  uint64_t* dense_key = (uint64_t*)mmap(nullptr, num_total_keys * sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_SHARED, dense_key_fd, 0);
  uint8_t* dense_val = (uint8_t*)mmap(nullptr, num_total_keys * val_nbytes_per_key, PROT_READ | PROT_WRITE, MAP_SHARED, dense_val_fd, 0);

  init(val_nbytes_per_key, num_total_keys, dense_val, 0);
  init_key(num_total_keys, dense_key);
  convert(sparse_key, sparse_val, val_nbytes_per_key, num_sparse_keys, num_total_keys, dense_val);
  return 0;
}