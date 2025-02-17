"""
 Copyright (c) 2022, NVIDIA CORPORATION.

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

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from sparse_operation_kit.experiment.communication import global_gpu_id, num_gpus


def Variable(*args, **kwargs):
    mode = kwargs.pop("mode") if "mode" in kwargs else None

    if mode is None or mode == "distributed":
        kwargs["global_gpu_id"] = global_gpu_id()
        kwargs["num_gpus"] = num_gpus()
        return DistributedVariable(*args, **kwargs)

    elif mode[: len("localized")] == "localized":
        target_gpu = int(mode.split(":")[1])
        kwargs["target_gpu"] = target_gpu
        kwargs["global_gpu_id"] = global_gpu_id()
        return LocalizedVariable(*args, **kwargs)

    else:
        raise RuntimeError("Not supported mode: %s" % mode)


class DistributedVariable(ResourceVariable):
    def __init__(
        self,
        initial_value=None,
        trainable=None,
        collections=None,
        validate_shape=True,
        caching_device=None,
        name=None,
        dtype=None,
        variable_def=None,
        import_scope=None,
        constraint=None,
        distribute_strategy=None,
        synchronization=None,
        aggregation=None,
        shape=None,
        initializer=None,
        global_gpu_id=None,
        num_gpus=None,
    ):
        self._global_gpu_id = global_gpu_id
        self._num_gpus = num_gpus

        if initial_value is not None:
            if isinstance(initial_value, list):
                initial_value = np.array(initial_value)
            self._global_shape = initial_value.shape

            local_size = self._global_shape[0] // num_gpus
            if global_gpu_id < self._global_shape[0] % num_gpus:
                local_size += 1

            if isinstance(initial_value, np.ndarray):
                device = "CPU"
            else:
                device = initial_value.device

            with tf.device(device):
                indices = tf.convert_to_tensor(np.arange(local_size), dtype=tf.int64)
                indices = indices * num_gpus + global_gpu_id
                initial_value = tf.nn.embedding_lookup(initial_value, indices)
                if dtype is not None:
                    initial_value = tf.cast(initial_value, dtype)
                else:
                    initial_value = tf.cast(initial_value, initial_value.dtype)
        else:
            self._global_shape = shape
            shape = None

            local_size = self._global_shape[0] // num_gpus
            if global_gpu_id < self._global_shape[0] % num_gpus:
                local_size += 1

            initial_value = initializer(shape=(local_size, self._global_shape[1]), dtype=dtype)

        super(DistributedVariable, self).__init__(
            initial_value=initial_value,
            trainable=trainable,
            collections=collections,
            validate_shape=validate_shape,
            caching_device=caching_device,
            name=name,
            dtype=dtype,
            variable_def=variable_def,
            import_scope=import_scope,
            constraint=constraint,
            distribute_strategy=distribute_strategy,
            synchronization=synchronization,
            aggregation=aggregation,
            shape=shape,
        )

    @property
    def global_shape(self):
        return self._global_shape

    @property
    def global_gpu_id(self):
        return self._global_gpu_id

    @property
    def target_gpu(self):
        return -1

    @property
    def num_gpus(self):
        return self._num_gpus

    def key_map(self, indices):
        return indices // self._num_gpus


class LocalizedVariable(ResourceVariable):
    def __init__(
        self,
        initial_value=None,
        trainable=None,
        collections=None,
        validate_shape=True,
        caching_device=None,
        name=None,
        dtype=None,
        variable_def=None,
        import_scope=None,
        constraint=None,
        distribute_strategy=None,
        synchronization=None,
        aggregation=None,
        shape=None,
        initializer=None,
        global_gpu_id=None,
        target_gpu=None,
    ):
        self._global_gpu_id = global_gpu_id
        self._target_gpu = target_gpu
        self._num_gpus = num_gpus()
        if target_gpu >= self._num_gpus:
            raise RuntimeError(
                "There are only %d GPU(s), cannot put embedding table on %dth(zero-indexed) GPU."
                % (self._num_gpus, target_gpu)
            )

        if initial_value is not None:
            if isinstance(initial_value, list):
                initial_value = np.array(initial_value)
            self._global_shape = initial_value.shape
        else:
            self._global_shape = shape
            shape = None

        if target_gpu != global_gpu_id:
            empty_value = np.ndarray(shape=[0, self._global_shape[1]], dtype=np.float32)
            if dtype is not None:
                initial_value = tf.cast(empty_value, dtype=dtype)
            else:
                initial_value = tf.convert_to_tensor(empty_value)
        elif initial_value is None:
            initial_value = initializer(shape=self._global_shape, dtype=dtype)

        super(LocalizedVariable, self).__init__(
            initial_value=initial_value,
            trainable=trainable,
            collections=collections,
            validate_shape=validate_shape,
            caching_device=caching_device,
            name=name,
            dtype=dtype,
            variable_def=variable_def,
            import_scope=import_scope,
            constraint=constraint,
            distribute_strategy=distribute_strategy,
            synchronization=synchronization,
            aggregation=aggregation,
            shape=shape,
        )

    @property
    def global_shape(self):
        return self._global_shape

    @property
    def global_gpu_id(self):
        return self._global_gpu_id

    @property
    def target_gpu(self):
        return self._target_gpu

    @property
    def num_gpus(self):
        return self._num_gpus

    def key_map(self, indices):
        return indices
