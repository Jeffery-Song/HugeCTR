
import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf
import atexit

args = dict()

args["gpu_num"] = 4                               # the number of available GPUs
args["iter_num"] = 10                             # the number of training iteration
args["slot_num"] = 10                             # the number of feature fields in this embedding layer
args["embed_vec_size"] = 16                       # the dimension of embedding vectors
args["dense_dim"] = 10                            # the dimension of dense features
args["global_batch_size"] = 1024                  # the globally batchsize for all GPUs
args["max_vocabulary_size"] = 100000
args["vocabulary_range_per_slot"] = [[i*10000, (i+1)*10000] for i in range(10)] 
args["max_nnz"] = 5                # the max number of non-zeros for all slots
args["combiner"] = "mean"

args["ps_config_file"] = "dnn.json"
args["dense_model_path"] = "dnn_dense.model"
args["embedding_table_path"] = "dnn_sparse.model"
args["saved_path"] = "dnn_tf_saved_model"
args["np_key_type"] = np.int64
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int64
args["tf_vector_type"] = tf.float32

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))

# %%
def generate_random_samples(num_samples, vocabulary_range_per_slot, max_nnz, dense_dim):
    def generate_sparse_keys(num_samples, vocabulary_range_per_slot, max_nnz, key_dtype = args["np_key_type"]):
        slot_num = len(vocabulary_range_per_slot)
        indices = []
        values = []
        for i in range(num_samples):
            for j in range(slot_num):
                vocab_range = vocabulary_range_per_slot[j]
                nnz = np.random.randint(low=1, high=max_nnz+1)
                entries = sorted(np.random.choice(max_nnz, nnz, replace=False))
                for entry in entries:
                    indices.append([i, j, entry])
                values.extend(np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(nnz, )))
        values = np.array(values, dtype=key_dtype)
        return tf.sparse.SparseTensor(indices = indices,
                                    values = values,
                                    dense_shape = (num_samples, slot_num, max_nnz))

    
    sparse_keys = generate_sparse_keys(num_samples, vocabulary_range_per_slot, max_nnz)
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return sparse_keys, dense_features, labels

def tf_dataset(sparse_keys, dense_features, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((sparse_keys, dense_features, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset

def inference_with_saved_model(args):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        hps.Init(global_batch_size = args["global_batch_size"], ps_config_file = args["ps_config_file"])
        model = tf.keras.models.load_model(args["saved_path"])
        model.summary()
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

    def _reshape_input(sparse_keys):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        return sparse_keys
    
    def _infer_step(inputs, labels):
        logit, embeddings = model(inputs)
        return logit, embeddings

    embeddings_peek = list()
    inputs_peek = list()

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args["global_batch_size"])
        sparse_keys, dense_features, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot"], args["max_nnz"], args["dense_dim"])
        dataset = tf_dataset(sparse_keys, dense_features, labels, replica_batch_size)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)
    for i, (sparse_keys, dense_features, labels) in enumerate(dataset):
        sparse_keys = strategy.run(_reshape_input, args=(sparse_keys,))
        inputs = [sparse_keys, dense_features]
        logit, embeddings = strategy.run(_infer_step, args=(inputs, labels))
        embeddings_peek.append(embeddings)
        inputs_peek.append(inputs)
        print("-"*20, "Step {}".format(i),  "-"*20)
    return embeddings_peek, inputs_peek

# %%
embeddings_peek, inputs_peek = inference_with_saved_model(args)

print(inputs_peek[-1][0].values)
print(embeddings_peek[-1])