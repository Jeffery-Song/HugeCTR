# 
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.17)

set(MAIN_SRC_DIR /hugectr_dev)

include_directories(
  /usr/local/cuda/include
  ${MAIN_SRC_DIR}/third_party/collcachelib
  ${MAIN_SRC_DIR}
  ${MAIN_SRC_DIR}/test
  ${MAIN_SRC_DIR}/HugeCTR/include
  $ENV{CONDA_PREFIX}/include
  $ENV{CONDA_PREFIX}/include/libcudf/libcudacxx
  ${CUDA_INCLUDE_DIRS}
  ${MAIN_SRC_DIR}/HugeCTR
  ${MAIN_SRC_DIR}/HugeCTR/include
  ${MAIN_SRC_DIR}/third_party
  ${MAIN_SRC_DIR}/third_party/argparse/include
  ${MAIN_SRC_DIR}/third_party/cutlass
  ${MAIN_SRC_DIR}/third_party/cpptqdm
  ${MAIN_SRC_DIR}/third_party/googletest/googletest/include
  ${MAIN_SRC_DIR}/third_party/googletest/googlemock/include
  ${MAIN_SRC_DIR}/third_party/json/single_include
  ${MAIN_SRC_DIR}/third_party/parallel-hashmap
  ${MAIN_SRC_DIR}/third_party/xxhash
  ${MAIN_SRC_DIR}/third_party/redis_pp/src
  ${MAIN_SRC_DIR}/third_party/redis_pp/src/sw/redis++/no_tls
  ${MAIN_SRC_DIR}/third_party/redis_pp/src/sw/redis++/cxx17
  ${MAIN_SRC_DIR}/third_party/rocksdb/include
  ${MAIN_SRC_DIR}/third_party/librdkafka/src
  ${MAIN_SRC_DIR}/gpu_cache/include
  ${MAIN_SRC_DIR}/HugeCTR/embedding_storage/dynamic_embedding_table

  ${CUDNN_INC_PATHS}
  ${NCCL_INC_PATHS}
  ${HWLOC_INC_PATHS}
  ${UCX_INC_PATHS}
  ${SHARP_INC_PATHS})

link_directories(
  /usr/local/cuda/lib64
  ${MAIN_SRC_DIR}/hierarchical_parameter_server/_skbuild/linux-x86_64-3.8/cmake-build/lib
  ${MAIN_SRC_DIR}/third_party/collcachelib/build
)

file(GLOB embedding_cache_test_src
  embedding_cache_test.cpp
)

file(GLOB embedding_cache_update_test_src
  embedding_cache_update_test.cpp
)

file(GLOB parameter_server_test_src
  parameter_server_test.cpp
)

file(GLOB parameter_server_access_test_src
  parameter_server_access_test.cpp
)

file(GLOB db_backend_test_src
  db_backend_test.cpp
)

add_executable(embedding_cache_test ${embedding_cache_test_src})
target_compile_features(embedding_cache_test PUBLIC cxx_std_17)
target_link_libraries(embedding_cache_test PUBLIC huge_ctr_hps cudart gtest gtest_main stdc++fs)

add_executable(embedding_cache_update_test ${embedding_cache_update_test_src})
target_compile_features(embedding_cache_update_test PUBLIC cxx_std_17)
target_link_libraries(embedding_cache_update_test PUBLIC huge_ctr_hps cudart gtest gtest_main stdc++fs)

add_executable(parameter_server_test ${parameter_server_test_src})
target_compile_features(parameter_server_test PUBLIC cxx_std_17)
target_link_libraries(parameter_server_test PUBLIC huge_ctr_hps cudart gtest gtest_main stdc++fs)

add_executable(parameter_server_access_test ${parameter_server_access_test_src})
target_compile_features(parameter_server_access_test PUBLIC cxx_std_17)
target_link_libraries(parameter_server_access_test PUBLIC huge_ctr_hps cudart gtest gtest_main stdc++fs coll_cache pthread)

add_executable(db_backend_test ${db_backend_test_src})
target_compile_features(db_backend_test PUBLIC cxx_std_17)
target_link_libraries(db_backend_test PUBLIC huge_ctr_hps cudart gtest gtest_main stdc++fs)
