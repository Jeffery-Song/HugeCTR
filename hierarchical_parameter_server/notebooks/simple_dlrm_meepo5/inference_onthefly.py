import time
import os
# this must be set before hps is imported
os.environ["COLL_NUM_REPLICA"] = "4"
proxy = None
if "http_proxy" in os.environ:
    proxy = os.environ["http_proxy"]
    del os.environ["http_proxy"]
import hierarchical_parameter_server as hps
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
import atexit
import multiprocessing
from numba import njit
import numba
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../common_model")
from dlrm import DLRM

args = dict()
args["gpu_num"] = 4                               # the number of available GPUs
args["iter_num"] = 3000                           # the number of training iteration
args["slot_num"] = 50                             # the number of feature fields in this embedding layer
args["embed_vec_size"] = 128                      # the dimension of embedding vectors
args["dense_dim"] = 13                            # the dimension of dense features
args["global_batch_size"] = 32768                    # the globally batchsize for all GPUs
args["combiner"] = "mean"
args["ps_config_file"] = "dlrm.json"
# args["dense_model_path"] = "/nvme/songxiaoniu/hps-model/dlrm_simple/dense.model"
# args["dense_model_path"] = "/nvme/songxiaoniu/hps-model/dlrm_simple_slot100/dense.model"
# SOK requires 64bit key, but we may use different key type at inference
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32
args["optimizer"] = "plugin_adam"
args["max_vocabulary_size"] = 50000000

# load vocabulary_range_per_slot from yaml file
feature_spec = {"cat_" + str(i) + ".bin" : {'cardinality' : args["max_vocabulary_size"] // args["slot_num"]} for i in range(args["slot_num"])}

# specify vocabulary range
ranges = [[0, 0] for i in range(args["slot_num"])]
max_range = 0
for i in range(args["slot_num"]):
    feature_cat = feature_spec['cat_' + str(i) + '.bin']['cardinality']
    ranges[i][0] = max_range
    ranges[i][1] = max_range + feature_cat
    max_range += feature_cat
args["vocabulary_range_per_slot"] = ranges
assert(max_range == args["max_vocabulary_size"])
args["max_vocabulary_size_per_gpu"] = (max_range) // args["gpu_num"]
if ((max_range) % args["gpu_num"]) != 0:
    args["max_vocabulary_size_per_gpu"] += 1
print(args['max_vocabulary_size'])

# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def generate_random_samples(num_samples, vocabulary_range_per_slot, dense_dim):
    vocabulary_range_per_slot = np.array(vocabulary_range_per_slot)
    num_slots = vocabulary_range_per_slot.shape[0]
    @njit(parallel=True)
    def generate_dense_keys(num_samples, vocabulary_range_per_slot, key_dtype = args["np_key_type"]):
        # dense_keys = [None for _ in range(num_slots)]
        dense_keys = np.empty((num_samples, num_slots), key_dtype)
        for i in range(num_slots):
            vocab_range = vocabulary_range_per_slot[i]
            H = vocab_range[1] - vocab_range[0] + 1
            L = 1
            a = 1
            rnd_rst = np.random.uniform(0.0, 1.0, size=num_samples)
            rnd_rst = ((-H**a * L**a) / (rnd_rst * (H**a - L**a) - H**a)) ** (1 / a)
            # keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples)).astype(key_dtype)
            keys_per_slot = rnd_rst.astype(key_dtype) + vocab_range[0] - 1
            dense_keys[:, i] = keys_per_slot
            # dense_keys[i] = keys_per_slot
        # dense_keys = np.concatenate(dense_keys, axis = 1)
        return dense_keys
    cat_keys = generate_dense_keys(num_samples, vocabulary_range_per_slot)
    @njit(parallel=True)
    def generate_cont_feats(num_samples, dense_dim):
        dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
        labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
        return dense_features, labels
    dense_features, labels = generate_cont_feats(num_samples, dense_dim)
    return cat_keys, dense_features, labels

def inference_with_saved_model(args):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        hps.Init(global_batch_size = args["global_batch_size"],
                ps_config_file = args["ps_config_file"])
        # model = InferenceModel(args["slot_num"], args["embed_vec_size"], args["dense_dim"], args["dense_model_path"])
        model = DLRM("mean", args["max_vocabulary_size"] // args["gpu_num"], args["embed_vec_size"], args["slot_num"], args["dense_dim"], 
            arch_bot = [256, 128, args["embed_vec_size"]],
            arch_top = [256, 128, 1],
            tf_key_type = args["tf_key_type"], tf_vector_type = args["tf_vector_type"], 
            self_interaction=False)
        # model.summary()
    # from https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
    atexit.register(strategy._extended._cross_device_ops._pool.close) # type: ignore
    atexit.register(strategy._extended._host_cross_device_ops._pool.close) #type: ignore

    @tf.function
    def _infer_step(inputs, labels):
        logit = model(inputs, training=False)
        return logit
    @tf.function
    def _whole_infer_step(in_param):
        # return strategy.run(_infer_step, args=in_param)
        inputs, labels = in_param
        return _infer_step(inputs, labels)
    @tf.function
    def _warmup_step(inputs):
        return inputs

    embeddings_peek = list()
    inputs_peek = list()

    def _dataset_fn(num_replica, local_id):
        assert(args["global_batch_size"] % num_replica == 0)
        replica_batch_size = args["global_batch_size"] // num_replica
        sparse_keys, dense_features, labels = generate_random_samples(replica_batch_size * args["iter_num"], args["vocabulary_range_per_slot"], args["dense_dim"])
        dataset = tf.data.Dataset.from_tensor_slices((sparse_keys, dense_features, labels))
        dataset = dataset.batch(replica_batch_size, drop_remainder=True, num_parallel_calls=10)
        dataset = dataset.cache()
        dataset = dataset.prefetch(10)
        return dataset
    dataset = _dataset_fn(args["gpu_num"], int(os.environ["HPS_WORKER_ID"]))

    ret_list = []
    ds_iter = iter(dataset)
    # for sparse_keys, dense_features, labels in tqdm(dataset, "warmup run"):
    barrier.wait()
    for iter_num in tqdm(range(args["iter_num"]), "warmup run"):
        sparse_keys, dense_features, labels = next(ds_iter)
        inputs = [sparse_keys, dense_features]
        ret = _warmup_step((inputs, labels))
        # ret = strategy.run(_warmup_step, args=(inputs, labels))
        ret_list.append(ret)
    barrier.wait()
    for i in tqdm(ret_list, "warmup should be done"):
        ret = _warmup_step(i)
        # ret = strategy.run(_warmup_step, args=i)
    barrier.wait()
    for i in tqdm(ret_list, "warmup should be done"):
        ret = _warmup_step(i)
        # ret = strategy.run(_warmup_step, args=i)
    barrier.wait()

    ds_time = 0
    md_time = 0
    if proxy:
        os.environ["http_proxy"] = proxy
    barrier.wait()
    for i in range(args["iter_num"]):
        t0 = time.time()
        t1 = time.time()
        logits = _whole_infer_step(ret_list[i])
        # logits = _infer_step(ret_list[i])
        t2 = time.time()
        ds_time += t1 - t0
        md_time += t2 - t1
        if i % 500 == 0:
            barrier.wait()
            print(i, "time {:.6} {:.6}".format(ds_time / 500, md_time / 500))
            ds_time = 0
            md_time = 0
    return embeddings_peek, inputs_peek

def proc_func(id):
    print(f"worker {id} at process {os.getpid()}")
    os.environ["TF_CONFIG"] = '{"cluster": {"worker": ["localhost:12340", "localhost:12341", "localhost:12342", "localhost:12343"]}, "task": {"type": "worker", "index": ' + str(id) + '} }'
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[id], 'GPU')
    os.environ["HPS_WORKER_ID"] = str(id)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    embeddings_peek, inputs_peek = inference_with_saved_model(args)
    hps.Shutdown()

barrier = multiprocessing.Barrier(args["gpu_num"])

proc_list = [None for _ in range(args["gpu_num"])]
for i in range(args["gpu_num"]):
    proc_list[i] = multiprocessing.Process(target=proc_func, args=(i,))
    proc_list[i].start()
for i in range(args["gpu_num"]):
    proc_list[i].join()