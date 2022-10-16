import sparse_operation_kit as sok
import sys
import os
import numpy as np
import tensorflow as tf
import struct
import yaml
import time
sys.path.append("../../../sparse_operation_kit/unit_test/test_scripts/tf2/")
import utils

args = dict()
args["gpu_num"] = 4                               # the number of available GPUs
args["iter_num"] = 1024                           # the number of training iteration
args["slot_num"] = 26                             # the number of feature fields in this embedding layer
args["embed_vec_size"] = 128                      # the dimension of embedding vectors
args["dense_dim"] = 13                            # the dimension of dense features
args["global_batch_size"] = 8                    # the globally batchsize for all GPUs
args["max_nnz"] = 10                # the max number of non-zeros for all slots
args["combiner"] = "mean"
args["ps_config_file"] = "dlrm.json"
args["dense_model_path"] = "dlrm_dense.model"
args["embedding_table_path"] = "dlrm_sparse.model"
args["saved_path"] = "dlrm_tf_saved_model"
args["np_key_type"] = np.int64
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int64
args["tf_vector_type"] = tf.float32
args["optimizer"] = "plugin_adam"

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))

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
args["max_vocabulary_size"] = max_range + 1
args["max_vocabulary_size_per_gpu"] = (max_range + 1) // args["gpu_num"]
if ((max_range + 1) % args["gpu_num"]) != 0:
    args["max_vocabulary_size_per_gpu"] += 1
print(args['max_vocabulary_size'])

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

def tf_dataset2(keys, dense, labels, batchsize, to_sparse_tensor=False, repeat=None):

    num_of_samples, slot_num, max_nnz = keys.shape

    def _convert_to_sparse(keys, dense, labels):
        if tf.rank(keys) != 2:
            keys = tf.reshape(keys, shape=[-1, max_nnz])
        indices = tf.where(keys != -1)
        values = tf.gather_nd(keys, indices)
        
        if keys.dtype == "int64":
            values = tf.cast(values, dtype=tf.int64)
        elif keys.dtype == "uint32":
            values = tf.cast(values, dtype=tf.uint32)
        else:
            raise ValueError("Not supported key_dtype.")
            
        if dense.dtype == "float32":
            dense = tf.cast(dense, dtype=tf.float32)
        else:
            raise ValueError("Not supported key_dtype.")
        
        return (
            tf.sparse.SparseTensor(
                indices=indices, values=values, dense_shape=[batchsize * slot_num, max_nnz]
            ),
            dense,
            labels,
        )

    def _cast_values(keys, dense, labels):
        if keys.dtype == "int64":
            keys = tf.cast(keys, dtype=tf.int64)
        elif keys.dtype == "uint32":
            keys = tf.cast(keys, dtype=tf.uint32)
        else:
            raise ValueError("Not supported key_dtype.")
        
        if dense.dtype == "float32":
            dense = tf.cast(dense, dtype=tf.float32)
        else:
            raise ValueError("Not supported key_dtype.")
        return keys, dense, labels

    dataset = tf.data.Dataset.from_tensor_slices((keys, dense, labels))
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    if to_sparse_tensor:
        dataset = dataset.map(
            lambda keys, dense, labels: _convert_to_sparse(keys, dense, labels), num_parallel_calls=1
        )
    else:
        dataset = dataset.map(lambda keys, dense, labels: _cast_values(keys, dense, labels), num_parallel_calls=1)
    return dataset

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                arch,
                activation='relu',
                out_activation=None,
                **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layers = []
        index = 0
        for units in arch[:-1]:
            self.layers.append(tf.keras.layers.Dense(units, activation=activation, name="{}_{}".format(kwargs['name'], index)))
            index+=1
        self.layers.append(tf.keras.layers.Dense(arch[-1], activation=out_activation, name="{}_{}".format(kwargs['name'], index)))

            
    def call(self, inputs, training=True):
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x

class SecondOrderFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self, self_interaction=False):
        super(SecondOrderFeatureInteraction, self).__init__()
        self.self_interaction = self_interaction

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_feas = tf.shape(inputs)[1]

        dot_products = tf.matmul(inputs, inputs, transpose_b=True)

        ones = tf.ones_like(dot_products)
        mask = tf.linalg.band_part(ones, 0, -1)
        out_dim = num_feas * (num_feas + 1) // 2

        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)
            out_dim = num_feas * (num_feas - 1) // 2
        flat_interactions = tf.reshape(tf.boolean_mask(dot_products, mask), (batch_size, out_dim))
        return flat_interactions

class DLRM(tf.keras.models.Model):
    def __init__(self,
                 combiner,
                 max_vocabulary_size_per_gpu,
                 embed_vec_size,
                 slot_num,
                 max_nnz,
                 dense_dim,
                 arch_bot,
                 arch_top,
                 self_interaction,
                 **kwargs):
        super(DLRM, self).__init__(**kwargs)
        
        self.combiner = combiner
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.dense_dim = dense_dim
 
        initializer = tf.keras.initializers.RandomUniform() # or "random_uniform" or "Zeros" or "ones"
        self.embedding_layer = sok.DistributedEmbedding(combiner=self.combiner,
                                                        max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                        embedding_vec_size=self.embed_vec_size,
                                                        slot_num=self.slot_num,
                                                        max_nnz=self.max_nnz,
                                                        embedding_initializer=initializer)
        self.bot_nn = MLP(arch_bot, name = "bottom", out_activation='relu')
        self.top_nn = MLP(arch_top, name = "top", out_activation='sigmoid')
        self.interaction_op = SecondOrderFeatureInteraction(self_interaction)
        if self_interaction:
            self.interaction_out_dim = (self.slot_num+1) * (self.slot_num+2) // 2
        else:
            self.interaction_out_dim = self.slot_num * (self.slot_num+1) // 2
        self.reshape_layer1 = tf.keras.layers.Reshape((1, arch_bot[-1]), name = "reshape1")
        self.concat1 = tf.keras.layers.Concatenate(axis=1, name = "concat1")
        self.concat2 = tf.keras.layers.Concatenate(axis=1, name = "concat2")
            
    def call(self, inputs, training=True):
        input_cat = inputs[0]
        input_dense = inputs[1]
        
        embedding_vector = self.embedding_layer(input_cat, training=training)
        dense_x = self.bot_nn(input_dense)
        concat_features = self.concat1([embedding_vector, self.reshape_layer1(dense_x)])
        
        Z = self.interaction_op(concat_features)
        z = self.concat2([dense_x, Z])
        logit = self.top_nn(z)
        return logit, embedding_vector

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.max_nnz, ), sparse=True, dtype=args["tf_key_type"]), 
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def train(args):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        sok.Init(global_batch_size=args["global_batch_size"])
        dlrm = DLRM(combiner = "mean", 
                    max_vocabulary_size_per_gpu = args["max_vocabulary_size"] // args["gpu_num"],
                    embed_vec_size = args["embed_vec_size"],
                    slot_num = args["slot_num"],
                    max_nnz = args["max_nnz"],
                    dense_dim = args["dense_dim"],
                    arch_bot = [256, 128, args["embed_vec_size"]],
                    arch_top = [256, 128, 1],
                    self_interaction = False)

        emb_opt = utils.get_embedding_optimizer(args["optimizer"])(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args["optimizer"])(learning_rate=0.1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args["global_batch_size"])

    @tf.function
    def _train_step(inputs, labels):
        inputs[0] = tf.sparse.reshape(inputs[0], [-1, args["max_nnz"]])
        with tf.GradientTape() as tape:
            logit, embedding_vector = dlrm(inputs, training=True)
            loss = _replica_loss(labels, logit)
        embedding_variables, other_variable = sok.split_embedding_variable_from_others(dlrm.trainable_variables)
        grads, emb_grads = tape.gradient(loss, [other_variable, embedding_variables])
        if 'plugin' not in args["optimizer"]:
            with sok.OptimizerScope(embedding_variables):
                emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                        experimental_aggregate_gradients=False)
        else:
            emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                    experimental_aggregate_gradients=False)
        dense_opt.apply_gradients(zip(grads, other_variable))
        return logit, embedding_vector, loss
    
    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args["global_batch_size"])
        print(replica_batch_size)
        random_samples = generate_random_samples(args["global_batch_size"], args["vocabulary_range_per_slot"], args["max_nnz"], args["dense_dim"])
        dataset = tf_dataset2(*random_samples, batchsize=replica_batch_size, to_sparse_tensor=False, repeat=1)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    for i in range(args["iter_num"]):
        dataset = strategy.distribute_datasets_from_function(_dataset_fn)
        for j, (sparse_keys, dense_features, labels) in enumerate(dataset):
            inputs = [sparse_keys, dense_features]
            logit, embedding_vector, loss = strategy.run(_train_step, args=(inputs, labels))
            print("-"*20, "Step {}, loss: {}".format(i, loss),  "-"*20)
    return dlrm

trained_model = train(args)
trained_model.summary()

dense_model = tf.keras.Model([trained_model.get_layer("distributed_embedding").output,
                             trained_model.get_layer("bottom").input],
                             trained_model.get_layer("top").output)
dense_model.summary()
dense_model.save(args["dense_model_path"])

embedding_saver = sok.Saver()
os.mkdir("dlrm_sparse.model")
embedding_saver.dump_to_file(trained_model.embedding_layer.embedding_variable, args["embedding_table_path"])
os.rename("dlrm_sparse.model/EmbeddingVariable_keys.file", "dlrm_sparse.model/key")
os.rename("dlrm_sparse.model/EmbeddingVariable_values.file", "dlrm_sparse.model/emb_vector")


class InferenceModel(tf.keras.models.Model):
    def __init__(self,
                 slot_num,
                 embed_vec_size,
                 max_nnz,
                 dense_dim,
                 dense_model_path,
                 **kwargs):
        super(InferenceModel, self).__init__(**kwargs)
        
        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size
        self.max_nnz = max_nnz
        self.dense_dim = dense_dim
        
        self.sparse_lookup_layer = hps.SparseLookupLayer(model_name = "dlrm", 
                                            table_id = 0,
                                            emb_vec_size = self.embed_vec_size,
                                            emb_vec_dtype = args["tf_vector_type"])
        self.dense_model = tf.keras.models.load_model(dense_model_path, compile=False)
    
    def call(self, inputs):
        input_cat = inputs[0]
        input_dense = inputs[1]

        embeddings = tf.reshape(self.sparse_lookup_layer(sp_ids=input_cat, sp_weights = None, combiner="mean"),
                                shape=[-1, self.slot_num, self.embed_vec_size])
        logit = self.dense_model([embeddings, input_dense])
        return logit, embeddings

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.max_nnz, ), sparse=True, dtype=args["tf_key_type"]), 
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def create_and_save_inference_graph(args): 
    model = InferenceModel(args["slot_num"], args["embed_vec_size"], args["max_nnz"], args["dense_dim"], args["dense_model_path"])
    model.summary()
    inputs = [tf.keras.Input(shape=(args["max_nnz"], ), sparse=True, dtype=args["tf_key_type"]), 
              tf.keras.Input(shape=(args["dense_dim"], ), dtype=tf.float32)]
    _, _ = model(inputs)
    model.save(args["saved_path"])

create_and_save_inference_graph(args)
