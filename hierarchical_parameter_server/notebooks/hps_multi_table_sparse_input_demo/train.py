# %% [markdown]
# <img src="http://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_hugectr_hps-multi-table-sparse-input-demo/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # HPS for Multiple Tables and Sparse Inputs

# %% [markdown]
# ## Overview
# 
# This notebook demonstrates how to use HPS when there are multiple embedding tables and sparse input. It is recommended to run [hierarchical_parameter_server_demo.ipynb](hierarchical_parameter_server_demo.ipynb) before diving into this notebook.
# 
# For more details about HPS APIs, please refer to [HPS APIs](https://nvidia-merlin.github.io/HugeCTR/master/hierarchical_parameter_server/api/index.html). For more details about HPS, please refer to [HugeCTR Hierarchical Parameter Server (HPS)](https://nvidia-merlin.github.io/HugeCTR/master/hierarchical_parameter_server/index.html).

# %% [markdown]
# ## Installation
# 
# ### Get HPS from NGC
# 
# The HPS Python module is preinstalled in the 22.09 and later [Merlin TensorFlow Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow): `nvcr.io/nvidia/merlin/merlin-tensorflow:22.09`.
# 
# You can check the existence of the required libraries by running the following Python code after launching this container.
# 
# ```bash
# $ python3 -c "import hierarchical_parameter_server as hps"
# ```

# %% [markdown]
# ## Configurations
# 
# First of all we specify the required configurations, e.g., the arguments needed for generating the dataset, the paths to save the model and the model parameters. We will use a deep neural network (DNN) model which has two embedding table and several dense layers in this notebook. Please note that there are two inputs here, one is the sparse key tensor (multi-hot) while the other is the dense key tensor (one-hot). 

# %%
import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf
import struct

args = dict()

args["gpu_num"] = 1                                         # the number of available GPUs
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
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32


os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))

# %%
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

# %% [markdown]
# ## Train with native TF layers
# 
# We define the model graph for training with native TF layers, i.e., `tf.nn.embedding_lookup_sparse`, `tf.nn.embedding_lookup` and `tf.keras.layers.Dense`.  We can then train the model and extract the trained weights of the two embedding tables. As for the dense layers, they are saved as a separate model graph, which can be loaded directly during inference.

# %%
class TrainModel(tf.keras.models.Model):
    def __init__(self,
                 init_tensors_per_table,
                 slot_num_per_table,
                 embed_vec_size_per_table,
                 max_nnz_per_slot_per_table,
                 **kwargs):
        super(TrainModel, self).__init__(**kwargs)
        
        self.slot_num_per_table = slot_num_per_table
        self.embed_vec_size_per_table = embed_vec_size_per_table
        self.max_nnz_per_slot_per_table = max_nnz_per_slot_per_table
        self.max_nnz_of_all_slots_per_table = [max(ele) for ele in self.max_nnz_per_slot_per_table]
        
        self.init_tensors_per_table = init_tensors_per_table
        self.params0 = tf.Variable(initial_value=tf.concat(self.init_tensors_per_table[0], axis=0))
        self.params1 = tf.Variable(initial_value=tf.concat(self.init_tensors_per_table[1], axis=0))
        
        self.reshape = tf.keras.layers.Reshape((self.max_nnz_of_all_slots_per_table[0],),
                                                input_shape=(self.slot_num_per_table[0], self.max_nnz_of_all_slots_per_table[0]))
        
        self.fc_1 = tf.keras.layers.Dense(units=256, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros",
                                                 name='fc_1')
        self.fc_2 = tf.keras.layers.Dense(units=256, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros",
                                                 name='fc_2')
        self.fc_3 = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros",
                                                 name='fc_3')

    def call(self, inputs):
        # SparseTensor of keys, shape: (batch_size*slot_num, max_nnz)
        embeddings0 = tf.reshape(tf.nn.embedding_lookup_sparse(params=self.params0, sp_ids=inputs[0], sp_weights = None, combiner="mean"),
                                shape=[-1, self.slot_num_per_table[0] * self.embed_vec_size_per_table[0]])
        # Tensor of keys, shape: (batch_size, slot_num)
        embeddings1 = tf.reshape(tf.nn.embedding_lookup(params=self.params1, ids=inputs[1]), 
                                 shape=[-1, self.slot_num_per_table[1] * self.embed_vec_size_per_table[1]])
        
        logit = self.fc_3(tf.math.add(self.fc_1(embeddings0), self.fc_2(embeddings1)))
        return logit, embeddings0, embeddings1

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.max_nnz_of_all_slots_per_table[0], ), sparse=True, dtype=args["tf_key_type"]),
                  tf.keras.Input(shape=(self.slot_num_per_table[1], ), dtype=args["tf_key_type"])]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

# %%
def train(args):
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, _, _ = model(inputs)
            loss = loss_fn(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return logit, loss

    init_tensors_per_table = [np.ones(shape=[args["max_vocabulary_size_per_table"][0], args["embed_vec_size_per_table"][0]], dtype=args["np_vector_type"]),
                              np.ones(shape=[args["max_vocabulary_size_per_table"][1], args["embed_vec_size_per_table"][1]], dtype=args["np_vector_type"])]

    model = TrainModel(init_tensors_per_table, args["slot_num_per_table"], args["embed_vec_size_per_table"], args["max_nnz_per_slot_per_table"])
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    sparse_keys, dense_keys, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot_per_table"], args["max_nnz_per_slot_per_table"])
    dataset = tf_dataset(sparse_keys, dense_keys, labels, args["global_batch_size"])
    for i, (sparse_keys, dense_keys, labels) in enumerate(dataset):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        inputs = [sparse_keys, dense_keys]
        _, loss = _train_step(inputs, labels)
        print("-"*20, "Step {}, loss: {}".format(i, loss),  "-"*20)
    return model

# %%
trained_model = train(args)
weights_list = trained_model.get_weights()
embedding_weights_per_table = weights_list[-2:]
dense_model = tf.keras.Model([trained_model.get_layer("fc_1").input, 
                              trained_model.get_layer("fc_2").input], 
                             trained_model.get_layer("fc_3").output)
dense_model.summary()
dense_model.save(args["dense_model_path"])

# %% [markdown]
# ## Create the inference graph with HPS SparseLookupLayer and LookupLayer
# In order to use HPS in the inference stage, we need to create a inference model graph which is almost the same as the train graph except that `tf.nn.embedding_lookup_sparse` is replaced by `hps.SparseLookupLayer` and `tf.nn.embedding_lookup` is replaced by `hps.LookupLayer`. The trained dense model graph can be loaded directly, while the weights of two embedding tables should be converted to the formats required by HPS. 
# 
# We can then save the inference model graph, which will be ready to be loaded for inference deployment.

# %%

# %%
def convert_to_sparse_model(embeddings_weights, embedding_table_path, embedding_vec_size):
    os.system("mkdir -p {}".format(embedding_table_path))
    with open("{}/key".format(embedding_table_path), 'wb') as key_file, \
        open("{}/emb_vector".format(embedding_table_path), 'wb') as vec_file:
      for key in range(embeddings_weights.shape[0]):
        vec = embeddings_weights[key]
        key_struct = struct.pack('q', key)
        vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
        key_file.write(key_struct)
        vec_file.write(vec_struct)

# %%
convert_to_sparse_model(embedding_weights_per_table[0], args["embedding_table_path"][0], args["embed_vec_size_per_table"][0])
convert_to_sparse_model(embedding_weights_per_table[1], args["embedding_table_path"][1], args["embed_vec_size_per_table"][1])
