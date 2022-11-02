import time
import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
import atexit

args = dict()
args["gpu_num"] = 8                               # the number of available GPUs
args["iter_num"] = 200                           # the number of training iteration
args["slot_num"] = 26                             # the number of feature fields in this embedding layer
args["embed_vec_size"] = 128                      # the dimension of embedding vectors
args["dense_dim"] = 13                            # the dimension of dense features
args["global_batch_size"] = 65536                    # the globally batchsize for all GPUs
args["combiner"] = "mean"
args["ps_config_file"] = "dlrm.json"
args["dense_model_path"] = "dlrm_dense.model"
# SOK requires 64bit key, but we may use different key type at inference
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32
args["optimizer"] = "plugin_adam"

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

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def generate_random_samples(num_samples, vocabulary_range_per_slot, dense_dim):
    def generate_dense_keys(num_samples, vocabulary_range_per_slot, key_dtype = args["np_key_type"]):
        dense_keys = list()
        for vocab_range in vocabulary_range_per_slot:
            keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples, 1), dtype=key_dtype)
            dense_keys.append(keys_per_slot)
        dense_keys = np.concatenate(np.array(dense_keys), axis = 1)
        return dense_keys
    cat_keys = generate_dense_keys(num_samples, vocabulary_range_per_slot)
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return cat_keys, dense_features, labels

def tf_dataset(sparse_keys, dense_features, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((sparse_keys, dense_features, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    # dataset = dataset.prefetch(8)
    dataset = dataset.cache()
    return dataset


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
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        hps.Init(global_batch_size = args["global_batch_size"],
                ps_config_file = args["ps_config_file"])
        model = InferenceModel(args["slot_num"], args["embed_vec_size"], args["dense_dim"], args["dense_model_path"])
        model.summary()
    # from https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

    @tf.function
    def _infer_step(inputs, labels):
        logit = model(inputs, training=False)
        return logit
    @tf.function
    def _whole_infer_step(in_param):
        return strategy.run(_infer_step, args=in_param)

    embeddings_peek = list()
    inputs_peek = list()

    # def _dataset_fn(input_context):
    #     replica_batch_size = input_context.get_per_replica_batch_size(args["global_batch_size"])
    #     sparse_keys, dense_features, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot"], args["dense_dim"])
    #     dataset = tf_dataset(sparse_keys, dense_features, labels, replica_batch_size)
    #     dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    #     return dataset

    # # with strategy.scope():
    # dataset = strategy.distribute_datasets_from_function(_dataset_fn
    # , tf.distribute.InputOptions(
    #     experimental_fetch_to_device=True,
    #     experimental_replication_mode=tf.distribute.InputReplicationMode.PER_WORKER,
    #     experimental_place_dataset_on_device=False,
    #     experimental_per_replica_buffer_size=8
    # )
    # )

    def _dataset_fn():
        sparse_keys, dense_features, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot"], args["dense_dim"])
        dataset = tf_dataset(sparse_keys, dense_features, labels, args["global_batch_size"])
        return dataset
    dataset = strategy.experimental_distribute_dataset(_dataset_fn()
        # , tf.distribute.InputOptions(
        #     experimental_fetch_to_device=True,
        #     experimental_replication_mode=tf.distribute.InputReplicationMode.PER_WORKER,
        #     experimental_place_dataset_on_device=True,
        #     experimental_per_replica_buffer_size=2
        # )
    )
    for _ in tqdm(dataset, "warm dataset"):
        pass
    for _ in tqdm(dataset, "dataset shoulded be warm"):
        pass
    ds_time = 0
    md_time = 0
    dataset_iter = iter(dataset)

    for i in range(args["iter_num"]):
        t0 = time.time()
        sparse_keys, dense_features, labels = next(dataset_iter)
        t1 = time.time()
        inputs = [sparse_keys, dense_features]
        # logits = strategy.run(_infer_step, args=(inputs, labels))
        logits = _whole_infer_step((inputs, labels))
        t2 = time.time()
        ds_time += t1 - t0
        md_time += t2 - t1
        if i % 10 == 0:
            print(i, "time {:.6} {:.6}".format(ds_time / 10, md_time / 10))
            ds_time = 0
            md_time = 0
    return embeddings_peek, inputs_peek

embeddings_peek, inputs_peek = inference_with_saved_model(args)