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

#include <sys/wait.h>
#include <exception>

#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

namespace {
std::string ToReadableSize(size_t nbytes) {
  constexpr size_t kGigabytes = 1ull << 30;
  constexpr size_t kMegabytes = 1ull << 20;
  constexpr size_t kKilobytes = 1ull << 10;
  char buf[100];
  if (nbytes > kGigabytes) {
    double new_size = (float)nbytes / kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (nbytes > kMegabytes) {
    double new_size = (float)nbytes / kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (nbytes > kKilobytes) {
    double new_size = (float)nbytes / kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)nbytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}
};

template <typename Device>
class Report : public OpKernel {
 public:
  explicit Report(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    try {
      size_t free, total;
      cudaMemGetInfo(&free, &total);
      std::stringstream ss;
      ss << "[CUDA] worker" << "" << " cuda mem usage: " << ToReadableSize(total  - free) << "\n";
      std::cout << ss.str();
      SparseOperationKit::Facade::instance()->report_avg();
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }

    Tensor* status_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &status_tensor));
    status_tensor->flat<tstring>()(0) = "OK";
  }
};

REGISTER_KERNEL_BUILDER(Name("ReportSOK").Device(DEVICE_GPU).HostMemory("status"),
                        Report<GPUDevice>);


template <typename Device>
class SetStepProfileValue : public OpKernel {
 public:
  explicit SetStepProfileValue(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
    // OP_REQUIRES(ctx, global_batch_size_ > 0,
    //             errors::Aborted(__FILE__, ":", __LINE__, " ", "global_batch_size must be > 0."));

    // OP_REQUIRES_OK(ctx, ctx->GetAttr("ps_config_file", &ps_config_file_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const Tensor* profile_type_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("profile_type", &profile_type_tensor));
    const Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("value", &value_tensor));
    try {
      int global_replica_id = global_replica_id_tensor->scalar<int32_t>()(0);
      int64_t profile_type = profile_type_tensor->scalar<int64_t>()(0);
      double value = value_tensor->scalar<double>()(0);

      auto device_ctx = ctx->op_device_context();
      OP_REQUIRES(ctx, device_ctx == nullptr, errors::Aborted("should have no device context."));

      SparseOperationKit::Facade::instance()->set_step_profile_value(
          global_replica_id, profile_type, value);
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SetStepProfileValueSOK")
                            .Device(DEVICE_CPU)
                            .HostMemory("global_replica_id")
                            .HostMemory("profile_type")
                            .HostMemory("value"),
                        SetStepProfileValue<CPUDevice>);


extern "C" {
int wait_one_child() {
  int child_stat;
  pid_t pid = waitpid(-1, &child_stat, 0);
  if (WEXITSTATUS(child_stat) != 0) {
    std::cerr << "detect a terminated child " << pid << ", status is " << WEXITSTATUS(child_stat)
              << "\n";
    return 1;
  } else if (WIFSIGNALED(child_stat)) {
    std::cerr << "detect an abnormal terminated child, signal is "
              << strsignal(WTERMSIG(child_stat));
    return 1;
  } else
    return 0;
}
}

}  // namespace tensorflow
