import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf

args = dict()

args["gpu_num"] = 1                               # the number of available GPUs
args["iter_num"] = 10                             # the number of training iteration
args["slot_num"] = 3                              # the number of feature fields in this embedding layer
args["embed_vec_size"] = 16                       # the dimension of embedding vectors
args["global_batch_size"] = 65536                 # the globally batchsize for all GPUs
args["max_vocabulary_size"] = 30000
args["vocabulary_range_per_slot"] = [[0,10000],[10000,20000],[20000,30000]]
args["ps_config_file"] = "naive_dnn.json"
args["dense_model_path"] = "naive_dnn_dense.model"
args["embedding_table_path"] = "naive_dnn_sparse.model"
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))


def generate_random_samples(num_samples, vocabulary_range_per_slot, key_dtype = args["np_key_type"]):
    keys = list()
    for vocab_range in vocabulary_range_per_slot:
        keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples, 1), dtype=key_dtype)
        keys.append(keys_per_slot)
    keys = np.concatenate(np.array(keys), axis = 1)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return keys, labels

def tf_dataset(keys, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((keys, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset

# %%
class InferenceModel(tf.keras.models.Model):
    def __init__(self,
                 slot_num,
                 embed_vec_size,
                 dense_model_path,
                 **kwargs):
        super(InferenceModel, self).__init__(**kwargs)

        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size
        self.lookup_layer = hps.LookupLayer(model_name = "naive_dnn", 
                                            table_id = 0,
                                            emb_vec_size = self.embed_vec_size,
                                            emb_vec_dtype = args["tf_vector_type"],
                                            name = "lookup")
        self.dense_model = tf.keras.models.load_model(dense_model_path)

    def call(self, inputs):
        embedding_vector = self.lookup_layer(inputs)
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.embed_vec_size])
        logit = self.dense_model(embedding_vector)
        return logit, embedding_vector

    def summary(self):
        inputs = tf.keras.Input(shape=(self.slot_num,), dtype=args["tf_key_type"])
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

# %%
def inference_with_saved_model(args):
    hps.Init(global_batch_size = args["global_batch_size"],
             ps_config_file = args["ps_config_file"])
    model = InferenceModel(args["slot_num"], args["embed_vec_size"], args["dense_model_path"])
    model.summary()
    def _infer_step(inputs, labels):
        logit, embedding_vector = model(inputs)
        return logit, embedding_vector
    embedding_vectors_peek = list()
    id_tensors_peek = list()
    keys, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot"],  args["np_key_type"])
    dataset = tf_dataset(keys, labels, args["global_batch_size"])
    for i, (id_tensors, labels) in enumerate(dataset):
        print("-"*20, "Step {}".format(i),  "-"*20)
        _, embedding_vector = _infer_step(id_tensors, labels)
        embedding_vectors_peek.append(embedding_vector)
        id_tensors_peek.append(id_tensors)
    return embedding_vectors_peek, id_tensors_peek

# %%
embedding_vectors_peek, id_tensors_peek = inference_with_saved_model(args)
print(embedding_vectors_peek[-1])
print(id_tensors_peek[-1])


