# %% [markdown]
# <img src="http://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_hugectr_hps-hps-pretrained-model-training-demo/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # HPS Pretrained Model Training Demo

# %% [markdown]
# ## Overview
# 
# This notebook demonstrates how to use HPS to load pre-trained embedding tables. It is recommended to run [hierarchical_parameter_server_demo.ipynb](hierarchical_parameter_server_demo.ipynb) before diving into this notebook.
# 
# For more details about HPS APIs, please refer to [HPS APIs](https://nvidia-merlin.github.io/HugeCTR/master/hierarchical_parameter_server/api/index.html). For more details about HPS, please refer to [HugeCTR Hierarchical Parameter Server (HPS)](https://nvidia-merlin.github.io/HugeCTR/master/hierarchical_parameter_server/index.html).

# %% [markdown]
# ## Installation
# 
# ### Get HPS from NGC
# 
# The HPS Python module is preinstalled in the 22.09 and later [Merlin TensorFlow Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr): `nvcr.io/nvidia/merlin/merlin-tensorflow:22.09`.
# 
# You can check the existence of the required libraries by running the following Python code after launching this container.
# 
# ```bash
# $ python3 -c "import hierarchical_parameter_server as hps"
# ```

# %% [markdown]
# ## Configurations
# 
# First of all we specify the required configurations, e.g., the arguments needed for generating the dataset, the model parameters and the paths to save the model. We will use a deep neural network (DNN) model which has one embedding table and several dense layers. Please note that the input to the embedding layer will be a sparse key tensor.

# %%
import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf
import struct

args = dict()

args["gpu_num"] = 4                               # the number of available GPUs
args["iter_num"] = 10                             # the number of training iteration
args["slot_num"] = 10                             # the number of feature fields in this embedding layer
args["embed_vec_size"] = 16                       # the dimension of embedding vectors
args["dense_dim"] = 10                            # the dimension of dense features
args["global_batch_size"] = 1024                  # the globally batchsize for all GPUs
args["max_vocabulary_size"] = 100000
args["vocabulary_range_per_slot"] = [[i*10000, (i+1)*10000] for i in range(10)] 
args["combiner"] = "mean"

args["ps_config_file"] = "dnn.json"
args["dense_model_path"] = "dnn_dense.model"
args["embedding_table_path"] = "dnn_sparse.model"
args["np_key_type"] = np.int32
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))

# %%
def generate_random_samples(num_samples, vocabulary_range_per_slot, dense_dim):
    def generate_dense_keys(num_samples, vocabulary_range_per_slot, key_dtype = args["np_key_type"]):
        keys = list()
        for vocab_range in vocabulary_range_per_slot:
            keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples, 1), dtype=key_dtype)
            keys.append(keys_per_slot)
        keys = np.concatenate(np.array(keys), axis = 1)
        return keys
    cat_keys = generate_dense_keys(num_samples, vocabulary_range_per_slot)
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return cat_keys, dense_features, labels

def tf_dataset(cat_keys, dense_features, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((cat_keys, dense_features, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset

# %% [markdown]
# ## Train with native TF layers
# 
# We define the model graph for training with native TF layers, i.e., `tf.nn.embedding_lookup_sparse` and `tf.keras.layers.Dense`. Besides, the embedding weights are stored in `tf.Variable`. We can then train the model and extract the trained weights of the embedding table.

# %%
class DNN(tf.keras.models.Model):
    def __init__(self,
                 init_tensors,
                 embed_vec_size,
                 slot_num,
                 dense_dim,
                 **kwargs):
        super(DNN, self).__init__(**kwargs)
        
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.dense_dim = dense_dim
        self.params = tf.Variable(initial_value=tf.concat(init_tensors, axis=0))
        self.fc1 = tf.keras.layers.Dense(units=1024, activation="relu", name="fc1")
        self.fc2 = tf.keras.layers.Dense(units=256, activation="relu", name="fc2")
        self.fc3 = tf.keras.layers.Dense(units=1, activation="sigmoid", name="fc3")

    def call(self, inputs, training=True):
        input_cat = inputs[0]
        input_dense = inputs[1]

        embeddings = tf.reshape(tf.nn.embedding_lookup(params=self.params, ids=input_cat),
                                shape=[-1, self.slot_num * self.embed_vec_size])
        concat_feas = tf.concat([embeddings, input_dense], axis=1)
        logit = self.fc3(self.fc2(self.fc1(concat_feas)))
        return logit, embeddings

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), dtype=args["tf_key_type"]), 
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

# %%
def train(args):
    init_tensors = np.ones(shape=[args["max_vocabulary_size"], args["embed_vec_size"]], dtype=args["np_vector_type"])
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DNN(init_tensors, args["embed_vec_size"], args["slot_num"], args["dense_dim"])
        model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)    

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args["global_batch_size"])
    

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, _ = model(inputs)
            loss = _replica_loss(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return logit, loss

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args["global_batch_size"])
        cat_keys, dense_features, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot"], args["dense_dim"])
        dataset = tf_dataset(cat_keys, dense_features, labels, replica_batch_size)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)
    for i, (cat_keys, dense_features, labels) in enumerate(dataset):
        inputs = [cat_keys, dense_features]  
        _, loss = strategy.run(_train_step, args=(inputs, labels))
        print("-"*20, "Step {}, loss: {}".format(i, loss),  "-"*20)
    return model

# %%
trained_model = train(args)
weights_list = trained_model.get_weights()
embedding_weights = weights_list[-1]

# %% [markdown]

# %%
# store embedding vector to file
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

convert_to_sparse_model(embedding_weights, args["embedding_table_path"], args["embed_vec_size"])

# %%
dense_model = tf.keras.models.Model(trained_model.get_layer("fc1").input, trained_model.get_layer("fc3").output)
dense_model.summary()
dense_model.save(args["dense_model_path"])
