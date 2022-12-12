#pragma once
#include <functional>
namespace SparseOperationKit {

extern std::function<void(const int64_t type, double value)> set_step_time;

}