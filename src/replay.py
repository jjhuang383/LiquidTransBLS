import numpy as np

class ExperienceReplayBuffer:
    """
    Experience Replay Buffer to prevent catastrophic forgetting.
    Stores past (X, Y) samples and allows sampling for retraining.
    """

    def __init__(self, capacity=2000, is_bearing=False):
        self.capacity = capacity
        self.buffer_X = None
        self.buffer_Y = None
        self.count = 0
        self.is_bearing = is_bearing

    def _get_stratification_metric(self, X):
        if X.ndim == 3:
            return X[:, -1, -1]
        return X[:, -1]

    def add(self, X, Y):
        if len(X) == 0:
            return
        before_count = 0
        before_hash_set = set()
        input_hashes = []
        duplicate_against_before = 0
        duplicate_within_input = 0
        if self.buffer_X is None:
            self.buffer_X = X
            self.buffer_Y = Y
        else:
            self.buffer_X = np.vstack([self.buffer_X, X])
            self.buffer_Y = np.vstack([self.buffer_Y, Y])
        if len(self.buffer_X) > self.capacity:
            Y_flat = self.buffer_Y.flatten()
            if self.is_bearing:
                metric = self._get_stratification_metric(self.buffer_X)
                p33 = np.percentile(metric, 33)
                p66 = np.percentile(metric, 66)
                idx_high_energy = np.where(metric > p66)[0]
                idx_mid_energy = np.where((metric > p33) & (metric <= p66))[0]
                idx_low_energy = np.where(metric <= p33)[0]
                idx_low = idx_high_energy
                idx_mid = idx_mid_energy
                idx_high = idx_low_energy
            else:
                idx_high = np.where(Y_flat > 0.7)[0]
                idx_mid = np.where((Y_flat > 0.3) & (Y_flat <= 0.7))[0]
                idx_low = np.where(Y_flat <= 0.3)[0]
            target_per_bin = self.capacity // 3

            def safe_sample(indices, k):
                if len(indices) == 0:
                    return np.array([], dtype=int)
                if len(indices) <= k:
                    return indices
                n_latest = k // 2
                n_random = k - n_latest
                indices_sorted = np.sort(indices)
                latest_subset = indices_sorted[-n_latest:] if n_latest > 0 else np.array([], dtype=int)
                remaining_pool = indices_sorted[:-n_latest] if n_latest > 0 else indices_sorted
                if len(remaining_pool) > n_random:
                    random_subset = np.random.choice(remaining_pool, n_random, replace=False)
                else:
                    random_subset = remaining_pool
                return np.concatenate([latest_subset, random_subset])
            keep_high = safe_sample(idx_high, target_per_bin)
            keep_mid = safe_sample(idx_mid, target_per_bin)
            remaining_capacity = self.capacity - len(keep_high) - len(keep_mid)
            keep_low = safe_sample(idx_low, remaining_capacity)
            keep_indices = np.concatenate([keep_high, keep_mid, keep_low])
            keep_indices = keep_indices.astype(int)
            keep_indices = keep_indices[keep_indices < len(self.buffer_X)]
            keep_indices = keep_indices[keep_indices < len(self.buffer_Y)]
            keep_indices = np.sort(keep_indices)
            self.buffer_X = self.buffer_X[keep_indices]
            self.buffer_Y = self.buffer_Y[keep_indices]

    def sample(self, batch_size):
        if self.buffer_X is None:
            return (None, None)
        n = len(self.buffer_X)
        if n < batch_size:
            return (self.buffer_X, self.buffer_Y)
        Y_flat = self.buffer_Y.flatten()
        if self.is_bearing:
            metric = self._get_stratification_metric(self.buffer_X)
            p33 = np.percentile(metric, 33)
            p66 = np.percentile(metric, 66)
            idx_high = np.where(metric <= p33)[0]
            idx_mid = np.where((metric > p33) & (metric <= p66))[0]
            idx_low = np.where(metric > p66)[0]
        else:
            idx_high = np.where(Y_flat > 0.7)[0]
            idx_mid = np.where((Y_flat > 0.3) & (Y_flat <= 0.7))[0]
            idx_low = np.where(Y_flat <= 0.3)[0]
        target_per_bin = max(batch_size // 3, 1)

        def stratified_take(indices, k):
            if len(indices) == 0 or k <= 0:
                return np.array([], dtype=int)
            if len(indices) <= k:
                return indices
            return np.random.choice(indices, k, replace=False)
        take_high = stratified_take(idx_high, target_per_bin)
        take_mid = stratified_take(idx_mid, target_per_bin)
        remaining = batch_size - len(take_high) - len(take_mid)
        take_low = stratified_take(idx_low, remaining)
        indices = np.concatenate([take_high, take_mid, take_low])
        if len(indices) < batch_size:
            selected = set(indices.tolist())
            remaining_pool = np.array([i for i in range(n) if i not in selected], dtype=int)
            if len(remaining_pool) > 0:
                extra = np.random.choice(remaining_pool, min(batch_size - len(indices), len(remaining_pool)), replace=False)
                indices = np.concatenate([indices, extra])
        indices = np.sort(indices[:batch_size])
        sample_X = self.buffer_X[indices]
        sample_Y = self.buffer_Y[indices]
        return (sample_X, sample_Y)

    def get_data(self):
        """Returns all data in the buffer."""
        return (self.buffer_X, self.buffer_Y)

    def clear(self):
        """Clears the buffer."""
        before_hashes = []
        before_count = 0
        self.buffer_X = None
        self.buffer_Y = None
        self.count = 0
