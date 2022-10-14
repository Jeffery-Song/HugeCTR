import os
from time import sleep
import atexit
args = dict()

args["gpu_num"] = 8                                         # the number of available GPUs
args["iter_num"] = 10                                       # the number of training iteration
args["global_batch_size"] = 1024                            # the globally batchsize for all GPUs

args["slot_num_per_table"] = [3, 2]                         # the number of feature fields for two embedding tables
args["embed_vec_size_per_table"] = [16, 32]                 # the dimension of embedding vectors for two embedding tables
args["max_vocabulary_size_per_table"] = [30000, 2000]       # the vocabulary size for two embedding tables
args["vocabulary_range_per_slot_per_table"] = [ [[0,10000],[10000,20000],[20000,30000]], [[0, 1000], [1000, 2000]] ]
args["max_nnz_per_slot_per_table"] = [[4, 2, 3], [1, 1]]    # the max number of non-zeros for each slot for two embedding tables

args["dense_model_path"] = "multi_table_sparse_input_dense.model"
args["ps_config_file"] = "multi_table_sparse_input.json"
args["embedding_table_path"] = ["multi_table_sparse_input_sparse_0.model", "multi_table_sparse_input_sparse_1.model"]
args["saved_path"] = "multi_table_sparse_input_tf_saved_model"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import hierarchical_parameter_server as hps

args["np_key_type"] = np.int64
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int64
args["tf_vector_type"] = tf.float32

class InferenceModel(tf.keras.models.Model):
    def __init__(self,
                 slot_num_per_table,
                 embed_vec_size_per_table,
                 max_nnz_per_slot_per_table,
                 dense_model_path,
                 **kwargs):
        super(InferenceModel, self).__init__(**kwargs)
        
        self.slot_num_per_table = slot_num_per_table
        self.embed_vec_size_per_table = embed_vec_size_per_table
        self.max_nnz_per_slot_per_table = max_nnz_per_slot_per_table
        self.max_nnz_of_all_slots_per_table = [max(ele) for ele in self.max_nnz_per_slot_per_table]
        
        self.sparse_lookup_layer = hps.SparseLookupLayer(model_name = "multi_table_sparse_input", 
                                            table_id = 0,
                                            emb_vec_size = self.embed_vec_size_per_table[0],
                                            emb_vec_dtype = args["tf_vector_type"])
        self.lookup_layer = hps.LookupLayer(model_name = "multi_table_sparse_input", 
                                            table_id = 1,
                                            emb_vec_size = self.embed_vec_size_per_table[1],
                                            emb_vec_dtype = args["tf_vector_type"])
        self.dense_model = tf.keras.models.load_model(dense_model_path)
    
    def call(self, inputs):
        # SparseTensor of keys, shape: (batch_size*slot_num, max_nnz)
        embeddings0 = tf.reshape(self.sparse_lookup_layer(sp_ids=inputs[0], sp_weights = None, combiner="mean"),
                                shape=[-1, self.slot_num_per_table[0] * self.embed_vec_size_per_table[0]])
        # Tensor of keys, shape: (batch_size, slot_num)
        embeddings1 = tf.reshape(self.lookup_layer(inputs[1]), 
                                 shape=[-1, self.slot_num_per_table[1] * self.embed_vec_size_per_table[1]])
        
        logit = self.dense_model([embeddings0, embeddings1])
        return logit, embeddings0, embeddings1

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.max_nnz_of_all_slots_per_table[0], ), sparse=True, dtype=args["tf_key_type"]),
                  tf.keras.Input(shape=(self.slot_num_per_table[1], ), dtype=args["tf_key_type"])]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def generate_random_samples(num_samples, vocabulary_range_per_slot_per_table, max_nnz_per_slot_per_table):
    def generate_sparse_keys(num_samples, vocabulary_range_per_slot, max_nnz_per_slot, key_dtype = args["np_key_type"]):
        slot_num = len(max_nnz_per_slot)
        max_nnz_of_all_slots = max(max_nnz_per_slot)
        indices = []
        values = []
        for i in range(num_samples):
            for j in range(slot_num):
                vocab_range = vocabulary_range_per_slot[j]
                max_nnz = max_nnz_per_slot[j]
                nnz = np.random.randint(low=1, high=max_nnz+1)
                entries = sorted(np.random.choice(max_nnz, nnz, replace=False))
                for entry in entries:
                    indices.append([i, j, entry])
                values.extend(np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(nnz, )))
        values = np.array(values, dtype=key_dtype)
        return tf.sparse.SparseTensor(indices = indices,
                                    values = values,
                                    dense_shape = (num_samples, slot_num, max_nnz_of_all_slots))

    def generate_dense_keys(num_samples, vocabulary_range_per_slot, key_dtype = args["np_key_type"]):
        dense_keys = list()
        for vocab_range in vocabulary_range_per_slot:
            keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples, 1), dtype=key_dtype)
            dense_keys.append(keys_per_slot)
        dense_keys = np.concatenate(np.array(dense_keys), axis = 1)
        return dense_keys
    
    assert len(vocabulary_range_per_slot_per_table)==2, "there should be two embedding tables"
    assert max(max_nnz_per_slot_per_table[0])>1, "the first embedding table has sparse key input (multi-hot)"
    assert min(max_nnz_per_slot_per_table[1])==1, "the second embedding table has dense key input (one-hot)"
    
    sparse_keys = generate_sparse_keys(num_samples, vocabulary_range_per_slot_per_table[0], max_nnz_per_slot_per_table[0])
    dense_keys = generate_dense_keys(num_samples, vocabulary_range_per_slot_per_table[1])
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return sparse_keys, dense_keys, labels

def tf_dataset(sparse_keys, dense_keys, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((sparse_keys, dense_keys, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset

def inference_with_saved_model(args):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        hps.Init(global_batch_size = args["global_batch_size"],
                ps_config_file = args["ps_config_file"])
        model = InferenceModel(args["slot_num_per_table"], args["embed_vec_size_per_table"], args["max_nnz_per_slot_per_table"], args["dense_model_path"])
        model.summary()
    # dataset = strategy.distribute_datasets_from_function(_dataset_fn)
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

    inputs = [tf.keras.Input(shape=(max(args["max_nnz_per_slot_per_table"][0]), ), sparse=True, dtype=args["tf_key_type"]),
             tf.keras.Input(shape=(args["slot_num_per_table"][1], ), dtype=args["tf_key_type"])]
    _, _, _= model(inputs)
    # model.summary()
    # model = tf.keras.models.load_model(args["saved_path"])
    # model.summary(expand_nested=True,show_trainable=True)
    def _infer_step(inputs, labels):
        logit, embeddings0, embeddings1 = model(inputs)
        return logit, embeddings0, embeddings1
    embeddings0_peek = list()
    embeddings1_peek = list()
    inputs_peek = list()
    sparse_keys, dense_keys, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot_per_table"], args["max_nnz_per_slot_per_table"])
    dataset = tf_dataset(sparse_keys, dense_keys, labels, args["global_batch_size"])
    for i, (sparse_keys, dense_keys, labels) in enumerate(dataset):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        inputs = [sparse_keys, dense_keys]
        logit, embeddings0, embeddings1 = _infer_step(inputs, labels)
        embeddings0_peek.append(embeddings0)
        embeddings1_peek.append(embeddings1)
        inputs_peek.append(inputs)
        print("-"*20, "Step {}".format(i),  "-"*20)
        # 1st embedding table, input keys are SparseTensor 
    return embeddings0_peek, embeddings1_peek, inputs_peek
print("before inference")
embeddings0_peek, embeddings1_peek, inputs_peek = inference_with_saved_model(args)

# 1st embedding table, input keys are SparseTensor 
print(inputs_peek[-1][0].values)
print(embeddings0_peek[-1])

# 2nd embedding table, input keys are Tensor
print(inputs_peek[-1][1])
print(embeddings1_peek[-1])
