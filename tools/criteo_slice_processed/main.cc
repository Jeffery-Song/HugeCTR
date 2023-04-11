#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <iostream>

void take_one_part_impl(uint32_t* from, uint32_t* to, size_t dim_begin, size_t dim_end, size_t total_dim, size_t num_entry) {
  #pragma omp parallel for
  for (size_t i = 0; i < num_entry; i++) {
    uint32_t *entry_from = from + i * total_dim, *entry_to = to + i * (dim_end - dim_begin);
    for (size_t dim = dim_begin; dim < dim_end; dim++) {
      entry_to[(dim - dim_begin)] = entry_from[dim];
    }
  }
}

std::string inputf;
size_t num_entry;
void take_one_part(uint32_t* old, std::string f_suffix, size_t dim_begin, size_t dim_end, size_t total_dim) {
  int fd = open((inputf + f_suffix).c_str(), O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
  size_t fsize = num_entry * (dim_end - dim_begin) * sizeof(uint32_t);
  ftruncate64(fd, fsize);
  uint32_t* output = (uint32_t*)mmap(nullptr, fsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  take_one_part_impl(old, output, dim_begin, dim_end, total_dim, num_entry);
}
int main(int argc, char** argv) {
  inputf = argv[1];
  size_t dense_dim = std::stoi(argv[2]);
  size_t sparse_dim = std::stoi(argv[3]);
  size_t label_dim = std::stoi(argv[4]);
  size_t total_dim = dense_dim + sparse_dim + label_dim;
  int input_fd = open(inputf.c_str(), O_RDONLY);
  struct stat sparse_key_stat;
  fstat(input_fd, &sparse_key_stat);

  if (sparse_key_stat.st_size % (total_dim * sizeof(uint32_t)) != 0) {
    std::cout << "wrong dim\n";
    abort();
  }
  num_entry = sparse_key_stat.st_size / (total_dim * sizeof(uint32_t));

  uint32_t* old = (uint32_t*)mmap(nullptr, sparse_key_stat.st_size, PROT_READ, MAP_SHARED, input_fd, 0);

  take_one_part(old, ".label",   0,                     label_dim,                          total_dim);
  take_one_part(old, ".dense",   label_dim,             label_dim + dense_dim,              total_dim);
  take_one_part(old, ".sparse",  label_dim + dense_dim, label_dim + dense_dim + sparse_dim, total_dim);
}