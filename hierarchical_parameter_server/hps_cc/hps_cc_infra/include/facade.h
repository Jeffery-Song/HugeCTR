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

#pragma once
#include <cstdint>
#include "hps/inference_utils.hpp"
#include "lookup_manager.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace HierarchicalParameterServer {

using namespace HugeCTR;

enum StepProfileItem {
  // L1
  kLogL1NumSample = 0,
  kLogL1NumNode,
  kLogL1CopyTime,
  kLogL1RemoteBytes,
  // L2
  kLogL2ExtractTime,
  kLogL2FeatCopyTime,
  kLogL2CacheCopyTime,
  // L3
  kLogL3CacheGetIndexTime,
  KLogL3CacheCopyIndexTime,
  kLogL3CacheExtractMissTime,
  kLogL3CacheCopyMissTime,
  kLogL3CacheCombineMissTime,
  kLogL3CacheCombineCacheTime,
  kLogL3CacheCombineRemoteTime,
  kLogL3LabelExtractTime,
  // Number of items
  kNumLogStepItems
};

class Facade final {
 private:
  Facade();
  ~Facade() = default;
  Facade(const Facade&) = delete;
  Facade& operator=(const Facade&) = delete;
  Facade(Facade&&) = delete;
  Facade& operator=(Facade&&) = delete;

  std::once_flag lookup_manager_init_once_flag_;
  std::shared_ptr<LookupManager> lookup_manager_;

 public:
  static Facade* instance();
  void operator delete(void*);
  void init(const int32_t global_replica_id, tensorflow::OpKernelContext* ctx,
            const char* ps_config_file, const int32_t global_batch_size,
            const int32_t num_replicas_in_sync);
  void forward(const char* model_name, const int32_t table_id, const int32_t global_replica_id,
               const tensorflow::Tensor* values_tensor, tensorflow::Tensor* emb_vector_tensor,
               tensorflow::OpKernelContext* ctx);
  void report_avg();
  parameter_server_config* ps_config;

  // for profiler
  inline void set_step_profile_value(const int64_t epoch, const int64_t step, const int64_t type, double value) {
    this->lookup_manager_->set_step_profile_value(epoch, step, type, value);
  }

  inline void add_epoch_profile_value(const int64_t epoch, const int64_t type, const double value) {
    this->lookup_manager_->add_epoch_profile_value(epoch, type, value);
  }
};

}  // namespace HierarchicalParameterServer