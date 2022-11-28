from numba import njit
import numpy as np
def generate_random_samples_zipf(num_samples, vocabulary_range_per_slot, dense_dim, key_dtype, alpha):
    vocabulary_range_per_slot = np.array(vocabulary_range_per_slot)
    num_slots = vocabulary_range_per_slot.shape[0]
    a = alpha
    @njit(parallel=True)
    def generate_dense_keys(num_samples, vocabulary_range_per_slot, key_dtype = key_dtype):
        # dense_keys = [None for _ in range(num_slots)]
        dense_keys = np.empty((num_samples, num_slots), key_dtype)
        for i in range(num_slots):
            vocab_range = vocabulary_range_per_slot[i]
            H = vocab_range[1] - vocab_range[0] + 1
            L = 1
            rnd_rst = np.random.uniform(0.0, 1.0, size=num_samples)
            rnd_rst = ((-H**a * L**a) / (rnd_rst * (H**a - L**a) - H**a)) ** (1 / a)
            # keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples)).astype(key_dtype)
            keys_per_slot = rnd_rst.astype(key_dtype) + vocab_range[0] - 1
            dense_keys[:, i] = keys_per_slot
            # dense_keys[i] = keys_per_slot
        # dense_keys = np.concatenate(dense_keys, axis = 1)
        return dense_keys
    cat_keys = generate_dense_keys(num_samples, vocabulary_range_per_slot)
    @njit(parallel=True)
    def generate_cont_feats(num_samples, dense_dim):
        dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
        labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
        return dense_features, labels
    dense_features, labels = generate_cont_feats(num_samples, dense_dim)
    return cat_keys, dense_features, labels