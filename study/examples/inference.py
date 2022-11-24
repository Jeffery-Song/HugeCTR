import argparse
import time
import os
proxy = None
if "http_proxy" in os.environ:
    proxy = os.environ["http_proxy"]
    del os.environ["http_proxy"]
import hierarchical_parameter_server as hps
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import atexit
import multiprocessing
from common_config import *
import json

def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("RM INFERENCE")
    add_common_arguments(argparser, default_run_config)
    return vars(argparser.parse_args())

def get_run_config():
    run_config = {}
    run_config.update(get_default_common_config())
    run_config.update(parse_args(run_config))
    process_common_config(run_config)
    print_run_config(run_config)
    return run_config

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
    tf_config = {"task": {"type": "worker", "index": id}, "cluster": {"worker": []}}
    for i in range(args["gpu_num"]): tf_config['cluster']['worker'].append("localhost:" + str(12340+i))
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[id], 'GPU')
    os.environ["HPS_WORKER_ID"] = str(id)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    embeddings_peek, inputs_peek = inference_with_saved_model(args)
    hps.Shutdown()

args = get_run_config()
proc_list = [None for _ in range(args["gpu_num"])]
for i in range(args["gpu_num"]):
    proc_list[i] = multiprocessing.Process(target=proc_func, args=(i,))
    proc_list[i].start()
for i in range(args["gpu_num"]):
    proc_list[i].join()
