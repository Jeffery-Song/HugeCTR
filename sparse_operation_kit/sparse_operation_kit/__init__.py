"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

print("[INFO]: %s is imported" % __name__)

from sparse_operation_kit.core._version import __version__

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
# TODO: this env should be left to users
# os.environ["TF_GPU_THREAD_COUNT"] = "16"
del os

# # ---------- import submodule ----------- #
import sparse_operation_kit.tf

# ------------ import items into root package -------- #
from sparse_operation_kit.core.initialize import Init
from sparse_operation_kit.core.initialize import Report
from sparse_operation_kit.core.initialize import wait_one_child
from sparse_operation_kit.core.initialize import SetStepProfileValue
from sparse_operation_kit.core.context_scope import OptimizerScope
from sparse_operation_kit.embeddings.distributed_embedding import DistributedEmbedding
from sparse_operation_kit.embeddings.all2all_dense_embedding import All2AllDenseEmbedding
from sparse_operation_kit.embeddings.get_embedding_op import get_embedding
from sparse_operation_kit.saver.Saver import Saver
from sparse_operation_kit.optimizers.utils import split_embedding_variable_from_others
from sparse_operation_kit.core.embedding_layer_handle import GraphKeys

def _get_next_enum_val(next_val):
    res = next_val[0]
    next_val[0] += 1
    return res

_step_log_val = [0]

# Step L1 Log
kLogL1NumSample        = _get_next_enum_val(_step_log_val)
kLogL1NumNode          = _get_next_enum_val(_step_log_val)
kLogL1SampleTotalTime  = _get_next_enum_val(_step_log_val)
kLogL1SampleTime       = _get_next_enum_val(_step_log_val)
kLogL1SendTime         = _get_next_enum_val(_step_log_val)
kLogL1RecvTime         = _get_next_enum_val(_step_log_val)
kLogL1CopyTime         = _get_next_enum_val(_step_log_val)
kLogL1ConvertTime      = _get_next_enum_val(_step_log_val)
kLogL1TrainTime        = _get_next_enum_val(_step_log_val)
kLogL1FeatureBytes     = _get_next_enum_val(_step_log_val)
kLogL1LabelBytes       = _get_next_enum_val(_step_log_val)
kLogL1IdBytes          = _get_next_enum_val(_step_log_val)
kLogL1GraphBytes       = _get_next_enum_val(_step_log_val)
kLogL1MissBytes        = _get_next_enum_val(_step_log_val)
kLogL1RemoteBytes      = _get_next_enum_val(_step_log_val)
kLogL1PrefetchAdvanced = _get_next_enum_val(_step_log_val)
kLogL1GetNeighbourTime = _get_next_enum_val(_step_log_val)
kLogL1SamplerId        = _get_next_enum_val(_step_log_val)
# Step L2 Log
kLogL2ShuffleTime    = _get_next_enum_val(_step_log_val)
kLogL2LastLayerTime  = _get_next_enum_val(_step_log_val)
kLogL2LastLayerSize  = _get_next_enum_val(_step_log_val)
kLogL2CoreSampleTime = _get_next_enum_val(_step_log_val)
kLogL2IdRemapTime    = _get_next_enum_val(_step_log_val)
kLogL2GraphCopyTime  = _get_next_enum_val(_step_log_val)
kLogL2IdCopyTime     = _get_next_enum_val(_step_log_val)
kLogL2ExtractTime    = _get_next_enum_val(_step_log_val)
kLogL2FeatCopyTime   = _get_next_enum_val(_step_log_val)
kLogL2CacheCopyTime  = _get_next_enum_val(_step_log_val)
# Step L3 Log
kLogL3KHopSampleCooTime          = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleKernelTime       = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleSortCooTime      = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleCountEdgeTime    = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleCompactEdgesTime = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkSampleCooTime    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKTime         = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep1Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep2Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep3Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep4Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep5Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep6Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep7Time    = _get_next_enum_val(_step_log_val)
kLogL3RemapFillUniqueTime        = _get_next_enum_val(_step_log_val)
kLogL3RemapPopulateTime          = _get_next_enum_val(_step_log_val)
kLogL3RemapMapNodeTime           = _get_next_enum_val(_step_log_val)
kLogL3RemapMapEdgeTime           = _get_next_enum_val(_step_log_val)
kLogL3CacheGetIndexTime          = _get_next_enum_val(_step_log_val)
KLogL3CacheCopyIndexTime         = _get_next_enum_val(_step_log_val)
kLogL3CacheExtractMissTime       = _get_next_enum_val(_step_log_val)
kLogL3CacheCopyMissTime          = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineMissTime       = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineCacheTime      = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineRemoteTime     = _get_next_enum_val(_step_log_val)
kLogL3LabelExtractTime           = _get_next_enum_val(_step_log_val)

# Epoch Log
_epoch_log_val = [0]
kLogEpochSampleTime                  = _get_next_enum_val(_epoch_log_val)
KLogEpochSampleGetCacheMissIndexTime = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleSendTime              = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleTotalTime             = _get_next_enum_val(_epoch_log_val)
kLogEpochCoreSampleTime              = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleCooTime               = _get_next_enum_val(_epoch_log_val)
kLogEpochIdRemapTime                 = _get_next_enum_val(_epoch_log_val)
kLogEpochShuffleTime                 = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleKernelTime            = _get_next_enum_val(_epoch_log_val)
kLogEpochCopyTime                    = _get_next_enum_val(_epoch_log_val)
kLogEpochConvertTime                 = _get_next_enum_val(_epoch_log_val)
kLogEpochTrainTime                   = _get_next_enum_val(_epoch_log_val)
kLogEpochTotalTime                   = _get_next_enum_val(_epoch_log_val)
kLogEpochFeatureBytes                = _get_next_enum_val(_epoch_log_val)
kLogEpochMissBytes                   = _get_next_enum_val(_epoch_log_val)

__all__ = [item for item in dir() if not item.startswith("__")]
