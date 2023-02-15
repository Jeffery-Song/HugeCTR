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

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>

#include <math.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <hps/embedding_cache.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/inference_utils.hpp>
#include <vector>
#include "base/debug/logger.hpp"
#include "coll_cache_lib/common.h"
#include "coll_cache_lib/logging.h"
#include "coll_cache_lib/run_config.h"

std::string config_file_name = "parameter_server_access_test.json";

using namespace HugeCTR;
using namespace coll_cache_lib::common;
namespace {

template <typename TypeHashKey>
void parameter_server_test(const std::string& config_file) {
  // initialize configurations
  parameter_server_config* ps_config = new parameter_server_config{config_file};
  InferenceParams inference_param = ps_config->inference_params_array[0];
  std::string model_name = inference_param.model_name;
  std::vector<int> deployed_devices = inference_param.deployed_devices;
  size_t embedding_vec_size = ps_config->embedding_vec_size_[model_name][0];
  size_t max_batchsize = inference_param.max_batchsize;
  float hit_rate_threshold = inference_param.hit_rate_threshold;
  RunConfig::num_device = deployed_devices.size();
  RunConfig::worker_id = 0;
  int lookup_cnt = 0;

  // initialize HierParameterServer and embedding caches
  std::shared_ptr<HierParameterServerBase> parameter_server =
      HierParameterServerBase::create(*ps_config, ps_config->inference_params_array);
  std::vector<std::shared_ptr<EmbeddingCacheBase>> embed_caches(2);
  embed_caches[0] = parameter_server->get_embedding_cache(model_name, deployed_devices[0]);
  embed_caches[1] = parameter_server->get_embedding_cache(model_name, deployed_devices[1]);
  printf("Total num of device: %lu\n", RunConfig::num_device);
  printf("Total num of unique keys: %lu\n", inference_param.max_vocabulary_size[0]);
  printf("Size of Cache: %lu\n", embed_caches[0]->get_slot_num());

  auto custom_lookup = [&] (int device_id, size_t batch_size, uint32_t gen(int i)) {
    auto cache = embed_caches[device_id];
    cudaStream_t stream = cache->get_refresh_streams()[0];
    TensorPtr d_embed_out = Tensor::Empty(DataType::kF32, {batch_size, embedding_vec_size}, GPU(deployed_devices[device_id]), "");
    TensorPtr h_keys = Tensor::Empty(DataType::kI32, {batch_size}, CPU(), "");
    for (int i = 0; i < h_keys->NumItem(); i++) h_keys->Ptr<uint32_t>()[i] = gen(i);
    cache->lookup(0, d_embed_out->Ptr<float>(), h_keys->Ptr<TypeHashKey>(), batch_size, hit_rate_threshold, stream);
  };

  auto custom_check = [&] (std::vector<size_t> total_lookups, std::vector<double> ratios) {
    auto equal = [] (double a, double b) {
      if (isnan(b)) return isnan(a);
      else return (fabs(a - b) <= double(1e-5));
    };

    for (int i = 0; i < 2; i++) {
      auto cache = embed_caches[i];
      HCTR_CHECK_HINT(equal(cache->total_lookups, total_lookups[i]), 
                      "Total lookup counts: %lu, should be %lu\n", cache->total_lookups, total_lookups[i]);
    }
    auto access_ratios = parameter_server->report_access_overlap();
    HCTR_CHECK_HINT(equal(access_ratios[0], ratios[0]), 
                    "Cache access hit ratio: %f, should be %f\n", access_ratios[0], ratios[0]);
    HCTR_CHECK_HINT(equal(access_ratios[1], ratios[1]), 
                    "Cache access hit overlap ratio: %f, should be %f\n", access_ratios[1], ratios[1]);
    HCTR_CHECK_HINT(equal(access_ratios[2], ratios[2]), 
                    "Cache access miss overlap ratio: %f, should be %f\n", access_ratios[2], ratios[2]);
    auto cache_ratios = parameter_server->report_cache_intersect();
    HCTR_CHECK_HINT(equal(cache_ratios, ratios[3]), 
                    "Cached keys overlap ratio: %f, should be %f\n", cache_ratios, ratios[3]);
  };

  auto regular_check = [&] (std::vector<size_t> cnt_infos) {
    // records
    static size_t total_lookups = 0;
    static size_t hit_cnts = 0, miss_cnts = 0;
    static size_t hit_overlaps = 0, miss_overlaps = 0;
    total_lookups += cnt_infos[0];
    hit_cnts += cnt_infos[1];
    miss_cnts += cnt_infos[2];
    hit_overlaps += cnt_infos[3];
    miss_overlaps += cnt_infos[4];

    custom_check({total_lookups, total_lookups}, {
      total_lookups? ((double)hit_cnts/total_lookups):NAN, 
      hit_cnts? ((double)hit_overlaps/hit_cnts):NAN, 
      miss_cnts? ((double)miss_overlaps/miss_cnts):NAN, 
      1});
  };

  // round 1: fill all cache
  printf("===========custom lookup %d===========\n", ++lookup_cnt);
  custom_lookup(0, max_batchsize - 1, [](int i) {return i<1? (uint32_t)i: (uint32_t)(i + 1);});
  custom_lookup(1, max_batchsize - 1, [](int i) {return (uint32_t)(i + 1);});
  regular_check({max_batchsize - 1, 0, max_batchsize - 1, 0, max_batchsize - 2});
  custom_lookup(0, 1, [](int i) {return (uint32_t)1;});
  custom_lookup(1, 1, [](int i) {return (uint32_t)0;});
  regular_check({1, 0, 1, 0, 2});

  // round 2: batch lookup
  printf("===========custom lookup %d===========\n", ++lookup_cnt);
  custom_lookup(0, max_batchsize, [](int i) {return (uint32_t)(i/2);});
  custom_lookup(1, max_batchsize, [](int i) {return (uint32_t)(i/4);});
  regular_check({max_batchsize, max_batchsize, 0, max_batchsize/2, 0});

  // round 3: mini-batch lookup
  printf("===========custom lookup %d===========\n", ++lookup_cnt);
  custom_lookup(0, 2, [](int i) {return (uint32_t)0;});
  custom_lookup(1, 2, [](int i) {return (uint32_t)1;});
  regular_check({2, 2, 0, 2, 0});
  printf("===========custom lookup %d===========\n", ++lookup_cnt);
  custom_lookup(0, 2, [](int i) {return (uint32_t)0;});
  custom_lookup(1, 2, [](int i) {return (uint32_t)1;});
  regular_check({2, 2, 0, 0, 0});
  printf("===========custom lookup %d===========\n", ++lookup_cnt);
  custom_lookup(0, 2, [](int i) {return (uint32_t)1;});
  custom_lookup(1, 2, [](int i) {return (uint32_t)0;});
  regular_check({2, 2, 0, 4, 0});
}
}  // namespace

TEST(parameter_server_access, LookupRecord) {
  parameter_server_test<unsigned int>(config_file_name);
}