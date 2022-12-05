from numba import njit
import numpy as np

@njit(parallel=True)
def generate_dense_keys_zipf(num_samples, num_slots, vocabulary_range_per_slot, key_dtype, alpha):
    dense_keys = np.empty((num_samples, num_slots), key_dtype)
    for i in range(num_slots):
        vocab_range = vocabulary_range_per_slot[i]
        H = vocab_range[1] - vocab_range[0] + 1
        L = 1
        rnd_rst = np.random.uniform(0.0, 1.0, size=num_samples)
        rnd_rst = ((-H**alpha * L**alpha) / (rnd_rst * (H**alpha - L**alpha) - H**alpha)) ** (1 / alpha)
        keys_per_slot = rnd_rst.astype(key_dtype) + vocab_range[0] - 1
        dense_keys[:, i] = keys_per_slot
    return dense_keys

@njit(parallel=True)
def generate_dense_keys_uniform(num_samples, num_slots, vocabulary_range_per_slot, key_dtype):
    dense_keys = np.empty((num_samples, num_slots), key_dtype)
    for i in range(num_slots):
        vocab_range = vocabulary_range_per_slot[i]
        keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples)).astype(key_dtype)
        dense_keys[:, i] = keys_per_slot
    return dense_keys

@njit(parallel=True)
def generate_cont_feats(num_samples, dense_dim):
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return dense_features, labels

def generate_random_samples(num_samples, vocabulary_range_per_slot, dense_dim, key_dtype, alpha):
    vocabulary_range_per_slot = np.array(vocabulary_range_per_slot)
    num_slots = vocabulary_range_per_slot.shape[0]
    if alpha == 0:
        cat_keys = generate_dense_keys_uniform(num_samples, num_slots, vocabulary_range_per_slot, key_dtype)
    else:
        cat_keys = generate_dense_keys_zipf(num_samples, num_slots, vocabulary_range_per_slot, key_dtype, alpha)
    dense_features, labels = generate_cont_feats(num_samples, dense_dim)
    return cat_keys, dense_features, labels
