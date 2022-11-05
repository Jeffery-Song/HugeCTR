import time
import os
# this must be set before hps is imported
os.environ["COLL_NUM_REPLICA"] = "8"
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

args = dict()
args["gpu_num"] = 8                               # the number of available GPUs
args["iter_num"] = 6000                           # the number of training iteration
args["slot_num"] = 26                             # the number of feature fields in this embedding layer
args["embed_vec_size"] = 128                      # the dimension of embedding vectors
args["dense_dim"] = 13                            # the dimension of dense features
args["global_batch_size"] = 65536                    # the globally batchsize for all GPUs
args["combiner"] = "mean"
args["ps_config_file"] = "dlrm.json"
args["dense_model_path"] = "/nvme/songxiaoniu/hps-model/dlrm_criteo/dense.model"
# SOK requires 64bit key, but we may use different key type at inference
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32
args["optimizer"] = "plugin_adam"
args["dataset_path"] = "/nvme/songxiaoniu/hps-dataset/criteo_like_uniform/saved_dataset"

# load vocabulary_range_per_slot from yaml file
file = open('/nvme/songxiaoniu/hps-dataset/criteo_like_uniform/desc.yaml', 'r', encoding="utf-8")
file_data = file.read()
file.close()
feature_spec = yaml.load(file_data, yaml.Loader)['feature_spec']

# specify vocabulary range
ranges = [[0, 0] for i in range(26)]
max_range = 0
for i in range(26):
    feature_cat = feature_spec['cat_' + str(i) + '.bin']['cardinality']
    ranges[i][0] = max_range
    ranges[i][1] = max_range + feature_cat
    max_range += feature_cat
args["vocabulary_range_per_slot"] = ranges
args["max_vocabulary_size"] = max_range
args["max_vocabulary_size_per_gpu"] = (max_range) // args["gpu_num"]
if ((max_range) % args["gpu_num"]) != 0:
    args["max_vocabulary_size_per_gpu"] += 1
print(args['max_vocabulary_size'])

# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class InferenceModel(tf.keras.models.Model):
    def __init__(self,
                 slot_num,
                 embed_vec_size,
                 dense_dim,
                 dense_model_path,
                 **kwargs):
        super(InferenceModel, self).__init__(**kwargs)
        
        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size
        self.dense_dim = dense_dim
        
        self.lookup_layer = hps.LookupLayer(model_name = "dlrm", 
                                            table_id = 0,
                                            emb_vec_size = self.embed_vec_size,
                                            emb_vec_dtype = args["tf_vector_type"])
        self.dense_model = tf.keras.models.load_model(dense_model_path, compile=False)
    
    def call(self, inputs):
        input_cat = inputs[0]
        input_dense = inputs[1]

        embeddings = tf.reshape(self.lookup_layer(input_cat),
                                shape=[-1, self.slot_num, self.embed_vec_size])
        logit = self.dense_model([embeddings, input_dense])
        return logit

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), sparse=False, dtype=args["tf_key_type"]), 
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def inference_with_saved_model(args):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        hps.Init(global_batch_size = args["global_batch_size"],
                ps_config_file = args["ps_config_file"])
        model = InferenceModel(args["slot_num"], args["embed_vec_size"], args["dense_dim"], args["dense_model_path"])
        model.summary()
    # from https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
    atexit.register(strategy._extended._cross_device_ops._pool.close) # type: ignore
    atexit.register(strategy._extended._host_cross_device_ops._pool.close) #type: ignore

    @tf.function
    def _infer_step(inputs, labels):
        logit = model(inputs, training=False)
        return logit
    @tf.function
    def _whole_infer_step(in_param):
        return strategy.run(_infer_step, args=in_param)
    @tf.function
    def _warmup_step(inputs, labels):
        return inputs, labels

    embeddings_peek = list()
    inputs_peek = list()

    def _dataset_fn(num_replica, local_id):
        assert(args["global_batch_size"] % num_replica == 0)
        replica_batch_size = args["global_batch_size"] // num_replica
        dataset = tf.data.experimental.load(args["dataset_path"], compression="GZIP")
        dataset = dataset.shard(num_replica, local_id)
        if dataset.element_spec[0].shape[0] != replica_batch_size:
            print("loaded dataset has batch size {}, but we need {}, so we have to rebatch it!".format(dataset.element_spec[0].shape[0], replica_batch_size))
            dataset = dataset.unbatch().batch(replica_batch_size, num_parallel_calls=56)
        else:
            print("loaded dataset has batch size we need, so directly use it")
        dataset = dataset.cache()
        dataset = dataset.prefetch(1000)
        return dataset
    dataset = _dataset_fn(args["gpu_num"], int(os.environ["HPS_WORKER_ID"]))

    ret_list = []
    for sparse_keys, dense_features, labels in tqdm(dataset, "warmup run"):
        inputs = [sparse_keys, dense_features]
        ret = strategy.run(_warmup_step, args=(inputs, labels))
        ret_list.append(ret)
    for i in tqdm(ret_list, "warmup should be done"):
        ret = strategy.run(_warmup_step, args=i)
    for i in tqdm(ret_list, "warmup should be done"):
        ret = strategy.run(_warmup_step, args=i)

    ds_time = 0
    md_time = 0
    os.environ["http_proxy"] = proxy
    for i in range(args["iter_num"]):
        t0 = time.time()
        t1 = time.time()
        logits = _whole_infer_step(ret_list[i])
        t2 = time.time()
        ds_time += t1 - t0
        md_time += t2 - t1
        if i % 500 == 0:
            print(i, "time {:.6} {:.6}".format(ds_time / 500, md_time / 500))
            ds_time = 0
            md_time = 0
    return embeddings_peek, inputs_peek

def proc_func(id):
    print(f"worker {id} at process {os.getpid()}")
    os.environ["TF_CONFIG"] = '{"cluster": {"worker": ["localhost:12340", "localhost:12341", "localhost:12342", "localhost:12343", "localhost:12344", "localhost:12345", "localhost:12346", "localhost:12347"]}, "task": {"type": "worker", "index": ' + str(id) + '} }'
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[i], 'GPU')
    os.environ["HPS_WORKER_ID"] = str(id)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    embeddings_peek, inputs_peek = inference_with_saved_model(args)
    hps.Shutdown()

proc_list = [None for _ in range(args["gpu_num"])]
for i in range(args["gpu_num"]):
    proc_list[i] = multiprocessing.Process(target=proc_func, args=(i,))
    proc_list[i].start()
for i in range(args["gpu_num"]):
    proc_list[i].join()
