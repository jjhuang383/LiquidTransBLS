import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
BEARING_DATASETS = {'XJTU', 'IMS', 'PHM 2012'}

class TaskSpecificScaler:
    """
    任务专属归一化器，确保每个任务的测试集只使用该任务训练集的统计信息

    关键原则:
    - 每个任务维护独立的scaler
    - 测试集归一化ONLY使用训练集的mean/std
    - 增量数据不影响测试集的归一化参数
    """

    def __init__(self, dataset_name=None):
        self.task_scalers = {}
        self.group_scalers = {}
        self.current_task_idx = None
        self.dataset_name = dataset_name
        self.use_signed_log = False
        self.scaler_cls = MinMaxScaler if dataset_name in BEARING_DATASETS else StandardScaler

    def _transform_features_before_scaling(self, X):
        arr = np.asarray(X, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_signed_log:
            arr = np.sign(arr) * np.log1p(np.abs(arr))
        return arr

    def _build_identity_scaler(self, n_features=1):
        scaler = self.scaler_cls()
        dummy = np.zeros((1, max(1, int(n_features))), dtype=np.float32)
        scaler.fit(dummy)
        return scaler

    def fit_task(self, task_idx, X_train):
        """
        为特定任务拟合scaler

        Args:
            task_idx: 任务索引
            X_train: 训练数据 (可以是list或ndarray)
        """
        if X_train is None:
            scaler = self._build_identity_scaler(1)
        elif isinstance(X_train, list):
            arrays = [self._transform_features_before_scaling(x) for x in X_train if x is not None and len(x) > 0]
            if arrays:
                scaler = self.scaler_cls()
                scaler.fit(np.vstack(arrays))
            else:
                scaler = self._build_identity_scaler(1)
        elif len(X_train) > 0:
            scaler = self.scaler_cls()
            scaler.fit(self._transform_features_before_scaling(X_train))
        else:
            n_features = X_train.shape[1] if getattr(X_train, 'ndim', 1) > 1 else 1
            scaler = self._build_identity_scaler(n_features)
        self.task_scalers[task_idx] = scaler
        self.current_task_idx = task_idx

    def transform_task(self, task_idx, X):
        """
        使用任务专属scaler转换数据

        Args:
            task_idx: 任务索引
            X: 待转换数据

        Returns:
            转换后的数据
        """
        if X is None:
            return None
        if task_idx not in self.task_scalers:
            raise ValueError(f'Task {task_idx} scaler not fitted yet! Available scalers: {list(self.task_scalers.keys())}')
        scaler = self.task_scalers[task_idx]
        if isinstance(X, list):
            if len(X) == 0:
                return []
            return [scaler.transform(self._transform_features_before_scaling(x)) for x in X]
        else:
            if len(X) == 0:
                return X
            return scaler.transform(self._transform_features_before_scaling(X))

def create_sliding_window(X, window_size):
    """
    Creates sliding windows from feature matrix X.
    Args:
        X: (N_samples, N_features)
        window_size: int
    Returns:
        X_windows: (N_samples - window_size + 1, window_size * N_features)
    """
    if window_size <= 0:
        raise ValueError('Window size must be positive.')
    if len(X) < window_size:
        return np.array([])
    windows = []
    for i in range(len(X) - window_size + 1):
        window = X[i:i + window_size].flatten()
        windows.append(window)
    return np.array(windows)

def postprocess_predictions(y_pred, dataset_name_resolved, y_std=None):
    if y_pred is None:
        return (y_pred, y_std)
    pred = np.asarray(y_pred, dtype=np.float32)
    pred_was_1d = pred.ndim == 1
    if pred_was_1d:
        pred = pred[:, None]
    if pred.shape[0] == 0:
        return (y_pred, y_std)
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    if dataset_name_resolved == 'C-MAPSS':
        alpha = 0.15
    elif dataset_name_resolved == 'IMS':
        alpha = 0.35
    else:
        alpha = 0.45
    pred_filtered = np.zeros_like(pred, dtype=np.float32)
    for col in range(pred.shape[1]):
        seq = np.clip(pred[:, col], 0.0, 1.0)
        ema = np.empty_like(seq, dtype=np.float32)
        ema[0] = seq[0]
        if dataset_name_resolved == 'C-MAPSS':
            alpha_base = 0.15
            for idx in range(1, len(seq)):
                if seq[idx] < ema[idx - 1] - 0.05:
                    dynamic_alpha = 0.5
                else:
                    dynamic_alpha = alpha_base
                ema[idx] = dynamic_alpha * seq[idx] + (1.0 - dynamic_alpha) * ema[idx - 1]
        else:
            for idx in range(1, len(seq)):
                ema[idx] = alpha * seq[idx] + (1.0 - alpha) * ema[idx - 1]
        pred_filtered[:, col] = ema
    return (pred_filtered, y_std)

def sample_coreset(X, Y, max_samples=1500):
    """Plan D: Extract a Coreset (most representative samples) to drastically speed up update."""
    if X is None or Y is None:
        return (X, Y)
    if isinstance(X, list):
        total_samples = sum((len(x[0] if isinstance(x, tuple) else x) for x in X))
        if total_samples <= max_samples:
            return (X, Y)
        ratio = max_samples / total_samples
        X_core, Y_core = ([], [])
        offset = 0
        for x_item in X:
            x_data = x_item[0] if isinstance(x_item, tuple) else x_item
            is_stateful = x_item[1] if isinstance(x_item, tuple) else False
            n_group = len(x_data)
            if n_group == 0:
                continue
            n_take = max(1, int(n_group * ratio))
            idx = np.linspace(0, n_group - 1, n_take, dtype=int)
            X_core.append((x_data[idx], is_stateful) if isinstance(x_item, tuple) else x_data[idx])
            Y_core.append(Y[offset:offset + n_group][idx])
            offset += n_group
        return (X_core, np.vstack(Y_core))
    else:
        total_samples = len(X)
        if total_samples <= max_samples:
            return (X, Y)
        idx = np.linspace(0, total_samples - 1, max_samples, dtype=int)
        return (X[idx], Y[idx])

def apply_odometer(X, dataset_name_resolved):
    is_bearing_dataset = dataset_name_resolved in BEARING_DATASETS
    if not is_bearing_dataset or X is None:
        return X

    def process_unit(x):
        if x.shape[1] > 25:
            rms = (x[:, 1] + x[:, 25]) / 2.0
        elif x.shape[1] > 1:
            rms = x[:, 1]
        else:
            rms = x[:, 0]
        rms = np.maximum(rms, 1e-06)
        alpha = 0.2
        smoothed_rms = np.zeros_like(rms)
        smoothed_rms[0] = rms[0]
        for i in range(1, len(rms)):
            smoothed_rms[i] = alpha * rms[i] + (1 - alpha) * smoothed_rms[i - 1]
        m_exponent = 3.0 if dataset_name_resolved == 'PHM2012' else 4.0
        damage_rate = smoothed_rms ** m_exponent
        healthy_len = int(len(damage_rate) * 0.05)
        if healthy_len > 0:
            damage_rate[:healthy_len] = 0.0
        cumulative_damage = np.cumsum(damage_rate).reshape(-1, 1)
        time_clock = np.arange(1, len(x) + 1, dtype=np.float32).reshape(-1, 1)
        base_rms = np.mean(smoothed_rms[:max(10, int(len(smoothed_rms) * 0.1))]) + 1e-08
        rel_stress = np.maximum(1.0, smoothed_rms / base_rms)
        time_speed = rel_stress ** 1.1
        if healthy_len > 0:
            time_speed[:healthy_len] = 1.0
        warped_time = np.cumsum(time_speed).reshape(-1, 1)
        return np.hstack([x, cumulative_damage, time_clock, warped_time])
    if isinstance(X, list):
        return [process_unit(x) for x in X]
    else:
        return process_unit(X)
