/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <hps/inference_utils.hpp>
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Overload CUDA atomic for other 64bit unsinged/signed integer type
__forceinline__ __device__ uint32_t atomicAdd(uint32_t* address, int val) {
  return (uint32_t)atomicAdd((unsigned int*)address, (unsigned int)val);
}

namespace HugeCTR {

// Kernels to combine the value buffer
__global__ void merge_emb_vec(float* d_output_emb_vec, const float* d_missing_emb_vec,
                              const uint64_t* d_missing_index, const size_t len,
                              const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_missing_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

// Kernels to fill the default value to the output buffer
__global__ void fill_default_emb_vec(float* d_output_emb_vec, const float default_emb_vec,
                                     const uint64_t* d_missing_index, const size_t len,
                                     const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] = default_emb_vec;
  }
}

// Kernels to decompress the value buffer
__global__ void decompress_emb_vec(const float* d_src_emb_vec, const uint64_t* d_src_index,
                                   float* d_dst_emb_vec, const size_t len,
                                   const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t dst_emb_vec = idx / emb_vec_size;
    size_t dst_float = idx % emb_vec_size;
    size_t src_emb_vec = d_src_index[dst_emb_vec];
    d_dst_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_src_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

// Kernels to transfer the missing key index to unique missing key vec
__global__ void transfer_missing_vec(const uint64_t* d_missing_index_, const size_t missing_len,
  uint32_t* d_unique_missing_keys) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < missing_len) d_unique_missing_keys[d_missing_index_[idx]] = 1;
}

// Kernels to count hit/missing keys
template <typename key_type>
__global__ void count_hit_keys(const key_type* d_keys, const size_t key_len,
                                const uint32_t* d_unique_missing_keys, const uint64_t* d_unique_index_ptr,
                                uint32_t* d_missing_key_counters, uint32_t* d_hit_key_counters) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < key_len) {
    key_type key = d_keys[idx];
    if (d_unique_missing_keys[d_unique_index_ptr[idx]] == 0) { // hit
      atomicAdd(&d_hit_key_counters[key], 1);
    } else {                                                   // miss
      atomicAdd(&d_missing_key_counters[key], 1);
    }
  }
}

// Kernels to get vec overlap
void __global__ vec_overlap(const uint32_t *d_src1, const uint32_t *d_src2,
                            uint64_t *d_dst, size_t n) {
	const size_t global_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (global_id < n) d_dst[global_id] = min(d_src1[global_id], d_src2[global_id]);
}

// Kernels to reduce key overlap
template <size_t reduce_size>
void __global__ vec_reduce(const uint64_t *d_src, uint64_t *d_dst, size_t n) {
	const size_t
		global_id = threadIdx.x + blockDim.x * blockIdx.x,
		reduce_id = global_id % reduce_size;
	unsigned val = global_id < n ? d_src[global_id] : 0;
	for (size_t offset = reduce_size >> 1; offset > 0; offset >>= 1)
		val += __shfl_xor_sync(FULL_MASK, val, offset, reduce_size);
	if (reduce_id == 0) d_dst[global_id / reduce_size] = val;
}

void merge_emb_vec_async(float* d_vals_merge_dst_ptr, const float* d_vals_retrieved_ptr,
                         const uint64_t* d_missing_index_ptr, const size_t missing_len,
                         const size_t emb_vec_size, const size_t BLOCK_SIZE, cudaStream_t stream) {
  if (missing_len == 0) {
    return;
  }
  size_t missing_len_in_float = missing_len * emb_vec_size;
  merge_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_vals_merge_dst_ptr, d_vals_retrieved_ptr, d_missing_index_ptr, missing_len, emb_vec_size);
}

void fill_default_emb_vec_async(float* d_vals_merge_dst_ptr, const float default_emb_vec,
                                const uint64_t* d_missing_index_ptr, const size_t missing_len,
                                const size_t emb_vec_size, const size_t BLOCK_SIZE,
                                cudaStream_t stream) {
  if (missing_len == 0) {
    return;
  }
  size_t missing_len_in_float = missing_len * emb_vec_size;
  fill_default_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_vals_merge_dst_ptr, default_emb_vec, d_missing_index_ptr, missing_len, emb_vec_size);
}

void decompress_emb_vec_async(const float* d_unique_src_ptr, const uint64_t* d_unique_index_ptr,
                              float* d_decompress_dst_ptr, const size_t decompress_len,
                              const size_t emb_vec_size, const size_t BLOCK_SIZE,
                              cudaStream_t stream) {
  if (decompress_len == 0) {
    return;
  }
  size_t decompress_len_in_float = decompress_len * emb_vec_size;
  decompress_emb_vec<<<((decompress_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_unique_src_ptr, d_unique_index_ptr, d_decompress_dst_ptr, decompress_len, emb_vec_size);
}

template <typename key_type>
void cache_access_statistic_util<key_type>::transfer_missing_vec_async(const uint64_t* d_missing_index_, const size_t missing_len,
                                uint32_t* d_unique_missing_keys,const size_t BLOCK_SIZE,
                                cudaStream_t stream) {
  if (missing_len == 0) return;
  transfer_missing_vec<<<((missing_len - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
    d_missing_index_, missing_len, d_unique_missing_keys);
}

template <typename key_type>
void cache_access_statistic_util<key_type>::count_hit_keys_async(const key_type* d_keys, const size_t key_len,
                          const uint32_t* d_unique_missing_keys, const uint64_t* d_unique_index_ptr,
                          uint32_t* d_missing_key_counters, uint32_t* d_hit_key_counters,
                          const size_t BLOCK_SIZE, cudaStream_t stream) {
  if (key_len == 0) return;
  count_hit_keys<<<((key_len - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
    d_keys, key_len, d_unique_missing_keys, d_unique_index_ptr, d_missing_key_counters, d_hit_key_counters);
}

template <typename key_type>
void cache_access_statistic_util<key_type>::vec_overlap_async(const uint32_t *d_src1, const uint32_t *d_src2, 
                                                              uint64_t *d_dst, size_t n, 
                                                              const size_t BLOCK_SIZE, cudaStream_t stream) {
  if (n == 0) return;
  vec_overlap<<<((n - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(d_src1, d_src2, d_dst, n);
}

template <typename key_type>
void cache_access_statistic_util<key_type>::vec_reduce_async(uint64_t *d_dst, size_t n, 
                                                            const size_t BLOCK_SIZE, cudaStream_t stream) {
  if (n == 0) return;
  uint64_t *dst = d_dst, *src = d_dst;
  for (size_t len = n; len > 1; len = (len + WARP_SIZE - 1) / WARP_SIZE)
    dst += (len + WARP_SIZE - 1) / WARP_SIZE;
  for (size_t len = n; len > 1; len = (len + WARP_SIZE - 1) / WARP_SIZE) {
    dst -= (len + WARP_SIZE - 1) / WARP_SIZE;
    vec_reduce<WARP_SIZE><<<((len - 1) / BLOCK_SIZE + 1), BLOCK_SIZE>>>(src, dst, len);
    src = dst;
  }
  cudaDeviceSynchronize();
}

template class cache_access_statistic_util<unsigned int>;
template class cache_access_statistic_util<long long>;

}  // namespace HugeCTR
