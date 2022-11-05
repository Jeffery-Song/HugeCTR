import time
import os
os.environ["COLL_NUM_REPLICA"] = "8"
os.environ["HPS_WORKER_ID"] = "0"
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
import atexit
from numba import njit

args = dict()
args["gpu_num"] = 1                               # the number of available GPUs
args["iter_num"] = 2400                           # the number of training iteration
args["slot_num"] = 26                             # the number of feature fields in this embedding layer
args["embed_vec_size"] = 128                      # the dimension of embedding vectors
args["dense_dim"] = 13                            # the dimension of dense features
args["global_batch_size"] = 8192                    # the globally batchsize for all GPUs
# SOK requires 64bit key, but we may use different key type at inference
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32
args["dataset_path"] = "saved_dataset_small"

# load vocabulary_range_per_slot from yaml file
file = open('criteo_full.yaml', 'r', encoding="utf-8")
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def generate_random_samples(num_samples, vocabulary_range_per_slot, dense_dim):
    print("generating random dataset - key")
    vocabulary_range_per_slot = np.array(vocabulary_range_per_slot)
    num_slots = vocabulary_range_per_slot.shape[0]
    @njit(parallel=True)
    def generate_dense_keys(num_samples, vocabulary_range_per_slot, key_dtype = args["np_key_type"]):
        dense_keys = np.empty((num_samples, num_slots), key_dtype)
        for i in range(num_slots):
            vocab_range = vocabulary_range_per_slot[i]
            keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples)).astype(key_dtype)
            dense_keys[:, i] = keys_per_slot
        return dense_keys
    cat_keys = generate_dense_keys(num_samples, vocabulary_range_per_slot)
    print("generating random dataset - feat")
    @njit(parallel=True)
    def generate_cont_feats(num_samples, dense_dim):
        dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
        labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
        return dense_features, labels
    dense_features, labels = generate_cont_feats(num_samples, dense_dim)
    print("generating random dataset done")
    return cat_keys, dense_features, labels

def tf_dataset(sparse_keys, dense_features, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((sparse_keys, dense_features, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True, num_parallel_calls=112)
    # # dataset = dataset.prefetch(8)
    dataset = dataset.cache()
    return dataset

def main():
    sparse_keys, dense_features, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot"], args["dense_dim"])
    dataset = tf_dataset(sparse_keys, dense_features, labels, args["global_batch_size"])
    print("saving dataset...")
    t = time.time()
    def custom_shard_func(cat, cont, label):
        # this is batched, so the first dim is batch, second dim is one of 13 dense features
        return tf.cast(cont[0][0] * 128, tf.int64)
    tf.data.experimental.save(dataset, args["dataset_path"], shard_func=custom_shard_func, compression="GZIP")
    print("saving dataset...done", time.time() - t)
main()