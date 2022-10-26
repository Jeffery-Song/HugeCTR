
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
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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

class InferenceModel(tf.keras.models.Model):
    def __init__(self,
                 slot_num,
                 embed_vec_size,
                 max_nnz,
                 dense_dim,
                 dense_model_path,
                 **kwargs):
        super(InferenceModel, self).__init__(**kwargs)
        
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.dense_dim = dense_dim
        self.sparse_lookup_layer = hps.SparseLookupLayer(model_name="dnn", table_id=0, emb_vec_size=self.embed_vec_size, emb_vec_dtype=args["tf_vector_type"], name="lookup")
        self.dense_model = tf.keras.models.load_model(dense_model_path)

    def call(self, inputs):
        input_cat = inputs[0]
        input_dense = inputs[1]
        
        # SparseTensor of keys, shape: (batch_size*slot_num, max_nnz)
        embedding_vector = self.sparse_lookup_layer(input_cat, sp_weights = None, combiner="mean")
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.embed_vec_size])
        concat_feat = tf.concat([embedding_vector, input_dense], axis=1)
        logit = self.dense_model(concat_feat)
        return logit, embedding_vector

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.max_nnz, ), sparse=True, dtype=args["tf_key_type"]), 
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def inference_with_saved_model(args):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        hps.Init(global_batch_size = args["global_batch_size"], ps_config_file = args["ps_config_file"])
        model = InferenceModel(args["slot_num"], args["embed_vec_size"], args["max_nnz"], args["dense_dim"], args["dense_model_path"])
        model.summary()
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

    def _reshape_input(sparse_keys):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        return sparse_keys

    @tf.function
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