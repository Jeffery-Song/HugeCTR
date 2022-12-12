#include <functional>
#include "coll_profiler.h"

namespace SparseOperationKit {

std::function<void(const int64_t type, double value)> set_step_time = [](const int64_t type, double value){};

}