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

#include <cstdint>
#include <exception>

#include "config.h"
#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using StepProfileItem = HierarchicalParameterServer::StepProfileItem;

template <typename Device>
class GetStepProfileValue : public OpKernel {
 public:
  explicit GetStepProfileValue(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
    // OP_REQUIRES(ctx, global_batch_size_ > 0,
    //             errors::Aborted(__FILE__, ":", __LINE__, " ", "global_batch_size must be > 0."));

    // OP_REQUIRES_OK(ctx, ctx->GetAttr("ps_config_file", &ps_config_file_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* epoch_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("epoch", &epoch_tensor));
    const Tensor* step_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("step", &step_tensor));
    const Tensor* profile_type_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("profile_type", &profile_type_tensor));

    try {
      int32_t epoch = epoch_tensor->scalar<int32_t>()(0);
      int32_t step = step_tensor->scalar<int32_t>()(0);
      int32_t profile_type = profile_type_tensor->scalar<int32_t>()(0);

      auto device_ctx = ctx->op_device_context();
      OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));

      Tensor* value_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &value_tensor));
      HierarchicalParameterServer::Facade::instance()->get_step_profile_value(
          epoch, step, profile_type, value_tensor->scalar<std::int32_t>()(0));
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("GetStepProfileValue")
                            // .Device(DEVICE_GPU)
                            .HostMemory("epoch")
                            .HostMemory("step")
                            .HostMemory("profile_type")
                            .HostMemory("value"),
                        GetStepProfileValue<CPUDevice>);

}  // namespace tensorflow
