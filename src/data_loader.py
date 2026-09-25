import numpy as np
import scipy.signal
import scipy.stats
import logging
import os
import pandas as pd
import glob
BEARING_CHANNEL_FEATURE_DIM = 33
BEARING_PAIR_FEATURE_DIM = 4
BEARING_TWO_CHANNEL_FEATURE_DIM = 2 * BEARING_CHANNEL_FEATURE_DIM + BEARING_PAIR_FEATURE_DIM
BEARING_AGGREGATED_FEATURE_DIM = BEARING_CHANNEL_FEATURE_DIM + BEARING_PAIR_FEATURE_DIM
BEARING_CHANNEL_MASK_DIM = 2
BEARING_TWO_CHANNEL_MASKED_FEATURE_DIM = BEARING_TWO_CHANNEL_FEATURE_DIM + BEARING_CHANNEL_MASK_DIM
XJTU_SEQUENCE_CONTEXT_DIM = 4
XJTU_USE_SEQUENCE_CONTEXT = False
BEARING_HI_METRIC_WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
BEARING_LABEL_MODE = 'linear'
BEARING_DISABLE_GWI = False
IMS_SEGMENT_MODE = 'equal'
PHM_DOWNSAMPLE_STEP = None

def configure_bearing_experiment(label_mode='fpt', disable_gwi=False, ims_segment_mode='equal', phm_downsample_step=None, xjtu_use_sequence_context=None):
    global BEARING_LABEL_MODE, BEARING_DISABLE_GWI, IMS_SEGMENT_MODE, PHM_DOWNSAMPLE_STEP
    global XJTU_USE_SEQUENCE_CONTEXT
    BEARING_LABEL_MODE = str(label_mode).lower()
    BEARING_DISABLE_GWI = bool(disable_gwi)
    IMS_SEGMENT_MODE = str(ims_segment_mode).lower()
    if phm_downsample_step in (None, '', 0):
        PHM_DOWNSAMPLE_STEP = None
    else:
        PHM_DOWNSAMPLE_STEP = max(1, int(phm_downsample_step))
    if xjtu_use_sequence_context is not None:
        XJTU_USE_SEQUENCE_CONTEXT = bool(xjtu_use_sequence_context)

def _finite_scalar(value, default=0.0):
    return float(np.nan_to_num(value, nan=default, posinf=default, neginf=default))

def _safe_skew(signal):
    return _finite_scalar(scipy.stats.skew(signal))

def _safe_kurtosis(signal):
    return _finite_scalar(scipy.stats.kurtosis(signal))

def _safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return _finite_scalar(np.corrcoef(x, y)[0, 1])

def _sequence_slope(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        return 0.0
    x = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    centered = values - values.mean()
    denom = np.sum(x ** 2) + 1e-12
    return _finite_scalar(np.sum(x * centered) / denom)

def _moving_average_1d(values, window):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return values
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(values) < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(values)]

def _causal_ema_1d(values, alpha):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return values.copy()
    alpha = float(np.clip(alpha, 0.0001, 1.0))
    ema = np.empty_like(values, dtype=np.float64)
    ema[0] = values[0]
    for idx in range(1, len(values)):
        ema[idx] = alpha * values[idx] + (1.0 - alpha) * ema[idx - 1]
    return ema

def _median_abs_deviation(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return 1e-08
    median = np.median(values)
    return _finite_scalar(np.median(np.abs(values - median)), default=1e-08) + 1e-08

def _bearing_hi_metrics_from_feature_sequence(feature_seq):
    """
    Build a lightweight degradation-indicator matrix from extracted bearing features.
    Metrics per step:
    1) RMS
    2) Envelope RMS
    3) Kurtosis
    4) High-frequency energy ratio
    """
    feature_seq = np.asarray(feature_seq, dtype=np.float64)
    if feature_seq.ndim != 2 or feature_seq.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    usable_dim = feature_seq.shape[1]
    if usable_dim >= BEARING_PAIR_FEATURE_DIM:
        usable_dim -= BEARING_PAIR_FEATURE_DIM
    n_channels = max(1, usable_dim // BEARING_CHANNEL_FEATURE_DIM)
    channel_metrics = []
    for channel_idx in range(n_channels):
        start = channel_idx * BEARING_CHANNEL_FEATURE_DIM
        end = start + BEARING_CHANNEL_FEATURE_DIM
        if end > usable_dim:
            break
        channel_block = feature_seq[:, start:end]
        channel_metrics.append(np.column_stack([channel_block[:, 1], channel_block[:, 16], channel_block[:, 4], channel_block[:, 15]]))
    if not channel_metrics:
        fallback_indices = [min(feature_seq.shape[1] - 1, idx) for idx in (1, 16, 4, 15)]
        metrics = feature_seq[:, fallback_indices]
    else:
        metrics = np.mean(np.stack(channel_metrics, axis=0), axis=0)
    return np.nan_to_num(metrics.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def _build_bearing_fpt_hi_labels(feature_seq, return_meta=False):
    """
    Convert per-step bearing features into a descending RUL target using FPT detection.
    Label mapping:
       - t < FPT: Label = 1.0
       - t >= FPT: Linear interpolation to 0.0
    """
    n_steps = len(feature_seq)
    if BEARING_LABEL_MODE == 'linear':
        labels = _build_linear_degradation_labels(n_steps)
        meta = {'fpt_idx': 0, 'fpt_ratio': 0.0, 'mode': 'linear'}
        return (labels, meta) if return_meta else labels
    if n_steps == 0:
        empty = np.zeros((0, 1), dtype=np.float32)
        meta = {'fpt_idx': 0, 'fpt_ratio': 0.0, 'mode': 'empty'}
        return (empty, meta) if return_meta else empty
    base_len = max(3, int(n_steps * 0.1))
    if feature_seq.shape[1] > 15:
        rms = feature_seq[:, 1]
        kurt = feature_seq[:, 4]
        base_len = max(3, int(n_steps * 0.1))
        rms_base_mean = np.mean(rms[:base_len])
        rms_base_std = np.std(rms[:base_len]) + 1e-08
        rms_norm = (rms - rms_base_mean) / rms_base_std
        kurt_base_mean = np.mean(kurt[:base_len])
        kurt_base_std = np.std(kurt[:base_len]) + 1e-08
        kurt_norm = (kurt - kurt_base_mean) / kurt_base_std
        indicator = np.maximum(0, rms_norm) + np.maximum(0, kurt_norm)
    elif feature_seq.shape[1] > 1:
        indicator = feature_seq[:, 1]
    else:
        indicator = feature_seq[:, 0]
    window = max(1, min(10, n_steps // 10))
    smoothed = pd.Series(indicator).rolling(window=window, min_periods=1).mean().values
    base_mean = np.mean(smoothed[:base_len])
    base_std = np.std(smoothed[:base_len]) + 1e-08
    k_factor = 0.5
    threshold = base_mean + k_factor * base_std
    fpt_idx = 0
    min_fpt = max(3, int(n_steps * 0.05))
    for i in range(max(base_len, min_fpt), n_steps):
        if smoothed[i] > threshold:
            if i + 1 < n_steps and smoothed[i + 1] > threshold:
                fpt_idx = i
                break
            elif i + 1 == n_steps:
                fpt_idx = i
                break
    if fpt_idx == 0 or fpt_idx > n_steps * 0.95:
        if n_steps < 100:
            fpt_idx = int(n_steps * 0.5)
        else:
            fpt_idx = int(n_steps * 0.2)
    labels = np.ones(n_steps, dtype=np.float32)
    labels[fpt_idx:] = np.linspace(1.0, 0.0, n_steps - fpt_idx, dtype=np.float32)
    labels = labels.reshape(-1, 1)
    meta = {'fpt_idx': fpt_idx, 'fpt_ratio': fpt_idx / n_steps, 'mode': 'fpt_detected'}
    return (labels, meta) if return_meta else labels

def _build_linear_degradation_labels(n_steps):
    n_steps = int(max(0, n_steps))
    if n_steps == 0:
        return np.zeros((0, 1), dtype=np.float32)
    return np.linspace(1.0, 0.0, n_steps, dtype=np.float32).reshape(-1, 1)

def _extract_bearing_channel_features(signal, n_bands=4, n_chunks=4):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(signal) < 8:
        empty = np.zeros(BEARING_CHANNEL_FEATURE_DIM, dtype=np.float32)
        aux = {'rms': 0.0, 'env': np.zeros(1, dtype=np.float64), 'power': np.zeros(1, dtype=np.float64)}
        return (empty, aux)
    signal_centered = signal - np.mean(signal)
    abs_signal = np.abs(signal_centered)
    eps = 1e-12
    std_val = _finite_scalar(np.std(signal_centered))
    rms_val = _finite_scalar(np.sqrt(np.mean(signal_centered ** 2)))
    peak_val = _finite_scalar(np.max(abs_signal))
    p2p_val = _finite_scalar(np.ptp(signal_centered))
    kurt_val = _safe_kurtosis(signal_centered)
    skew_val = _safe_skew(signal_centered)
    crest_factor = _finite_scalar(peak_val / (rms_val + eps))
    impulse_factor = _finite_scalar(peak_val / (np.mean(abs_signal) + eps))
    shape_factor = _finite_scalar(rms_val / (np.mean(abs_signal) + eps))
    margin_factor = _finite_scalar(peak_val / (np.mean(np.sqrt(abs_signal)) ** 2 + eps))
    if len(signal_centered) > 2:
        tkeo = signal_centered[1:-1] ** 2 - signal_centered[:-2] * signal_centered[2:]
        tkeo_mean = _finite_scalar(np.mean(tkeo))
        tkeo_std = _finite_scalar(np.std(tkeo))
    else:
        tkeo_mean = 0.0
        tkeo_std = 0.0
    fft_mag = np.abs(np.fft.rfft(signal_centered))
    power_spec = fft_mag ** 2
    freqs = np.fft.rfftfreq(len(signal_centered), d=1.0)
    total_pow = power_spec.sum() + eps
    band_edges = np.linspace(0, len(power_spec), n_bands + 1, dtype=int)
    band_ratios = []
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        end = max(end, start + 1)
        band_ratios.append(_finite_scalar(power_spec[start:end].sum() / total_pow))
    centroid = _finite_scalar(np.sum(freqs * power_spec) / total_pow)
    rms_freq = _finite_scalar(np.sqrt(np.sum(freqs ** 2 * power_spec) / total_pow))
    p_norm = power_spec / total_pow
    spec_entropy = _finite_scalar(-np.sum(p_norm * np.log(p_norm + eps)))
    hf_start = int(len(power_spec) * 0.6)
    high_freq_ratio = _finite_scalar(power_spec[hf_start:].sum() / total_pow)
    spec_skew = _finite_scalar(np.sum((freqs - centroid) ** 3 * p_norm) / (rms_freq ** 3 + eps))
    spec_kurt = _finite_scalar(np.sum((freqs - centroid) ** 4 * p_norm) / (rms_freq ** 4 + eps))
    envelope = np.abs(scipy.signal.hilbert(signal_centered))
    env_rms = _finite_scalar(np.sqrt(np.mean(envelope ** 2)))
    env_kurt = _safe_kurtosis(envelope)
    env_peak = _finite_scalar(np.max(envelope))
    env_crest = _finite_scalar(env_peak / (env_rms + eps))
    shock_density = _finite_scalar(np.mean(envelope > np.mean(envelope) + 2.0 * np.std(envelope) + eps))
    env_fft = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
    if len(env_fft) >= 3:
        valid_env_fft = env_fft[min(5, len(env_fft) - 1):]
        if len(valid_env_fft) >= 3:
            sort_idx = np.argsort(valid_env_fft)[-3:]
            env_peak_amps = valid_env_fft[sort_idx]
        else:
            env_peak_amps = np.zeros(3)
    else:
        env_peak_amps = np.zeros(3)
    chunks = np.array_split(signal_centered, n_chunks)
    env_chunks = np.array_split(envelope, n_chunks)
    chunk_rms = [_finite_scalar(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0 for chunk in chunks]
    chunk_env_rms = [_finite_scalar(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0 for chunk in env_chunks]
    rms_slope = _sequence_slope(chunk_rms)
    rms_range = _finite_scalar(chunk_rms[-1] - chunk_rms[0]) if chunk_rms else 0.0
    env_slope = _sequence_slope(chunk_env_rms)
    env_range = _finite_scalar(chunk_env_rms[-1] - chunk_env_rms[0]) if chunk_env_rms else 0.0
    features = np.array([std_val, rms_val, peak_val, p2p_val, kurt_val, skew_val, crest_factor, impulse_factor, margin_factor, shape_factor, tkeo_mean, tkeo_std, band_ratios[0], band_ratios[1], band_ratios[2], band_ratios[3], centroid, rms_freq, spec_entropy, high_freq_ratio, spec_skew, spec_kurt, env_rms, env_kurt, env_crest, shock_density, env_peak_amps[0], env_peak_amps[1], env_peak_amps[2], rms_slope, rms_range, env_slope, env_range], dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    aux = {'rms': rms_val, 'env': envelope, 'power': power_spec}
    return (features, aux)

def _extract_bearing_pair_features(signal, aux_list):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] < 2 or len(aux_list) < 2:
        return np.zeros(BEARING_PAIR_FEATURE_DIM, dtype=np.float32)
    sig_h = signal[:, 0] - np.mean(signal[:, 0])
    sig_v = signal[:, 1] - np.mean(signal[:, 1])
    corr = _safe_corrcoef(sig_h, sig_v)
    env_corr = _safe_corrcoef(aux_list[0]['env'], aux_list[1]['env'])
    log_rms_ratio = _finite_scalar(np.log((aux_list[0]['rms'] + 1e-12) / (aux_list[1]['rms'] + 1e-12)))
    power_h = np.asarray(aux_list[0]['power'], dtype=np.float64).reshape(-1)
    power_v = np.asarray(aux_list[1]['power'], dtype=np.float64).reshape(-1)
    m = min(len(power_h), len(power_v))
    if m == 0:
        spectral_cos = 0.0
    else:
        denom = np.linalg.norm(power_h[:m]) * np.linalg.norm(power_v[:m]) + 1e-12
        spectral_cos = _finite_scalar(np.dot(power_h[:m], power_v[:m]) / denom)
    return np.array([corr, env_corr, log_rms_ratio, spectral_cos], dtype=np.float32)

def _extract_bearing_multichannel_features(signal, combine='concat', max_channels=2):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2:
        raise ValueError(f'Expected 1-D or 2-D signal array, got shape {signal.shape}.')
    channel_features = []
    aux_list = []
    for channel_idx in range(signal.shape[1]):
        features, aux = _extract_bearing_channel_features(signal[:, channel_idx])
        channel_features.append(features)
        aux_list.append(aux)
    effective_signal = signal[:, :max_channels]
    effective_aux = aux_list[:max_channels]
    pair_features = _extract_bearing_pair_features(effective_signal, effective_aux)
    if combine == 'concat':
        features = np.concatenate(channel_features + [pair_features], axis=0)
    elif combine == 'mean':
        features = np.concatenate([np.mean(np.vstack(channel_features), axis=0), pair_features], axis=0)
    elif combine == 'concat_masked':
        padded_channels = []
        channel_mask = []
        for channel_idx in range(max_channels):
            if channel_idx < len(channel_features):
                padded_channels.append(channel_features[channel_idx])
                channel_mask.append(1.0)
            else:
                padded_channels.append(np.zeros(BEARING_CHANNEL_FEATURE_DIM, dtype=np.float32))
                channel_mask.append(0.0)
        features = np.concatenate(padded_channels + [pair_features, np.asarray(channel_mask, dtype=np.float32)], axis=0)
    else:
        raise ValueError(f'Unsupported combine mode: {combine}')
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def _enhance_bearing_sequence(X):
    """
    Enhance raw frame-level features with sequence-level contextual and degradation information.
    This explicitly extracts the "continuous degradation information" (Cumulative wear, relative shift)
    requested for robust RUL prediction across bearing datasets.
    """
    X = np.asarray(X, dtype=np.float32)
    n_steps, n_feats = X.shape
    if n_steps < 10:
        return np.concatenate([X, X, X, X, X], axis=1)
    alpha = 0.2
    X_smooth = np.zeros_like(X)
    X_smooth[0] = X[0]
    for t in range(1, n_steps):
        X_smooth[t] = alpha * X[t] + (1 - alpha) * X_smooth[t - 1]
    base_len = max(5, min(n_steps // 10, 50))
    base_mean = np.mean(X_smooth[:base_len], axis=0)
    base_std = np.std(X_smooth[:base_len], axis=0) + 1e-08
    X_rel = (X_smooth - base_mean) / base_std
    interpretable_indices = [1, 4, 10, 19, 22, 26]
    if n_feats >= 66:
        interpretable_indices.extend([34, 37, 43, 52, 55, 59])
    X_cum_rate = np.zeros_like(X)
    gwi = np.zeros((n_steps, 1), dtype=np.float32)
    valid_count = 0
    for idx in interpretable_indices:
        if idx < n_feats:
            feat_raw = X[:, idx]
            feat_ultra_smooth = np.zeros_like(feat_raw)
            feat_ultra_smooth[0] = feat_raw[0]
            for t in range(1, n_steps):
                feat_ultra_smooth[t] = 0.05 * feat_raw[t] + 0.95 * feat_ultra_smooth[t - 1]
            b_mean = np.mean(feat_ultra_smooth[:base_len])
            b_std = np.std(feat_ultra_smooth[:base_len]) + 1e-08
            normalized_feat = (feat_ultra_smooth - b_mean) / b_std
            deviation = np.maximum(0, normalized_feat - 5.0)
            cum_dev = np.cumsum(deviation)
            compressed_cum_dev = cum_dev * 0.01
            gwi[:, 0] += compressed_cum_dev
            valid_count += 1
    if valid_count > 0:
        gwi /= valid_count
    for idx in range(n_feats):
        X_cum_rate[:, idx] = gwi[:, 0]
    if BEARING_DISABLE_GWI:
        X_cum_rate.fill(0.0)
    X_grad = np.zeros_like(X)
    X_grad[1:] = X_smooth[1:] - X_smooth[:-1]
    alpha_grad = 0.3
    for t in range(1, n_steps):
        X_grad[t] = alpha_grad * X_grad[t] + (1 - alpha_grad) * X_grad[t - 1]
    dist_to_base = np.sqrt(np.sum(X_rel ** 2, axis=1)).reshape(-1, 1)
    dist_smooth = np.zeros_like(dist_to_base)
    dist_smooth[0] = dist_to_base[0]
    alpha_dist = 0.15
    for t in range(1, n_steps):
        dist_smooth[t] = alpha_dist * dist_to_base[t] + (1 - alpha_dist) * dist_smooth[t - 1]
    log_dist = np.log1p(dist_smooth)
    max_dist = np.max(log_dist)
    if max_dist > 1e-06:
        log_dist = log_dist / max_dist
    else:
        log_dist = np.zeros_like(log_dist)
    X_log_dist = np.zeros_like(X)
    for idx in range(n_feats):
        X_log_dist[:, idx] = log_dist[:, 0]
    enhanced_X = np.concatenate([X_smooth, X_rel, X_cum_rate, X_grad, X_log_dist], axis=1)
    return np.nan_to_num(enhanced_X, nan=0.0, posinf=0.0, neginf=0.0)

def _extract_xjtu_sequence_context(feature_seq):
    feature_seq = np.asarray(feature_seq, dtype=np.float32)
    if feature_seq.ndim != 2 or feature_seq.shape[0] == 0:
        return np.zeros((0, XJTU_SEQUENCE_CONTEXT_DIM), dtype=np.float32)
    n_steps = feature_seq.shape[0]
    base_len = max(5, min(n_steps // 8, 80))
    hi_metrics = _bearing_hi_metrics_from_feature_sequence(feature_seq).astype(np.float64, copy=False)
    base_metrics = hi_metrics[:base_len]
    base_center = np.median(base_metrics, axis=0, keepdims=True)
    base_scale = np.median(np.abs(base_metrics - base_center), axis=0, keepdims=True) + 1e-08
    norm_metrics = np.maximum((hi_metrics - base_center) / base_scale, 0.0)
    severity = np.sum(norm_metrics * BEARING_HI_METRIC_WEIGHTS.reshape(1, -1), axis=1)
    severity = np.clip(severity, 0.0, 10.0)
    severity_short = _causal_ema_1d(severity, 0.3)
    severity_long = _causal_ema_1d(severity, 0.08)
    trend_gap = severity_short - severity_long
    delta = np.diff(severity, prepend=severity[0])
    cumulative_rise = np.cumsum(np.maximum(delta, 0.0)) / np.arange(1, n_steps + 1, dtype=np.float64)
    local_instability = _causal_ema_1d(np.abs(delta), 0.25)
    context = np.column_stack([np.log1p(severity), np.tanh(trend_gap), np.log1p(cumulative_rise), np.log1p(local_instability)])
    return np.nan_to_num(context.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def _enhance_xjtu_bearing_sequence(X):
    enhanced = _enhance_bearing_sequence(X)
    if not XJTU_USE_SEQUENCE_CONTEXT:
        return enhanced.astype(np.float32, copy=False)
    context = _extract_xjtu_sequence_context(X)
    return np.concatenate([enhanced, context], axis=1).astype(np.float32, copy=False)

class CATLLoader:
    """Loads local battery_data_{temperature}C.npz archives and a user-supplied tasks.json split."""

    def __init__(self, data_root):
        self.data_root = data_root
        self.logger = logging.getLogger('TCBLS_RUL.DataLoader')
        self.cache = {}
        self.data_25 = self._load_npz('battery_data_25C.npz')
        self.data_45 = self._load_npz('battery_data_45C.npz')
        self.data_60 = self._load_npz('battery_data_60C.npz')
        self.keys_25 = sorted(list(self.data_25.keys())) if self.data_25 else []
        self.keys_45 = sorted(list(self.data_45.keys())) if self.data_45 else []
        self.keys_60 = sorted(list(self.data_60.keys())) if self.data_60 else []
        import json
        with open(os.path.join(data_root, 'tasks.json'), encoding='utf-8') as stream:
            self.tasks = json.load(stream)

    def _load_npz(self, filename):
        path = os.path.join(self.data_root, filename)
        if not os.path.exists(path):
            self.logger.warning(f'File not found: {path}')
            return {}
        try:
            data = np.load(path, allow_pickle=True)
            if 'sequences' in data:
                return data['sequences'].item()
            return {}
        except Exception as e:
            self.logger.error(f'Error loading {path}: {e}')
            return {}

    def _process_battery(self, data_item):
        """
        Process single battery data.
        X: SOH (from dict)
        Y: RUL (Linearly decreasing from 1 to 0)
        """
        X = None
        if isinstance(data_item, dict) and 'soh' in data_item:
            X = data_item['soh']
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            X = X.reshape(-1, 1)
        elif hasattr(data_item, 'columns') and 'SOC' in data_item.columns:
            X = data_item['SOC'].values.reshape(-1, 1)
        if X is None:
            return (None, None)
        N = len(X)
        if N == 0:
            return (None, None)
        Y = np.linspace(1, 0, N).reshape(-1, 1)
        return (X, Y)

    def get_task_data(self, task_name):
        if task_name not in self.tasks:
            return ((None, None), (None, None), (None, None))
        config = self.tasks[task_name]
        temp = config['temp']
        if temp == 25:
            source = self.data_25
        elif temp == 45:
            source = self.data_45
        else:
            source = self.data_60

        def load_units(keys):
            Xs, Ys = ([], [])
            for k in keys:
                if k not in source:
                    continue
                x, y = self._process_battery(source[k])
                if x is not None:
                    if len(x) < 100:
                        continue
                    Xs.append(x)
                    Ys.append(y)
            if not Xs:
                return (None, None)
            return (Xs, Ys)
        train_keys = config['train']
        test_keys = config['test']
        is_incremental = task_name in ['B', 'D', 'F']
        data_train = load_units(train_keys)
        data_test = load_units(test_keys)
        if is_incremental:
            return ((None, None), data_train, data_test)
        else:
            return (data_train, (None, None), data_test)

    def get_task_update_type(self, task_name):
        if task_name == 'A':
            return 'initial'
        return 'same_condition' if task_name in ('B', 'D', 'F') else 'new_condition'

    def get_task_scaler_group(self, task_name):
        return 'CATL_' + str(self.tasks[task_name]['temp'])
NDBatteryLoader = CATLLoader

class TJULoader:
    """
    Loads the TJU battery degradation dataset.

    The task stream is designed to stress both operating-condition transfer and
    chemistry transfer while keeping the CATL-style base/update pattern:
    A/B: NCA, 25C, 0.5C
    C/D: NCA, 45C, 0.5C        (condition shift)
    E/F: NCM, 25C, 0.5C        (chemistry + condition shift from D)
    """
    FEATURE_COLUMNS = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy', 'current mean', 'current std', 'current kurtosis', 'current skewness', 'CV Q', 'CV charge time', 'current slope', 'current entropy']
    LABEL_CAPACITY_COLUMN = 'capacity'

    def __init__(self, data_root, include_capacity_feature=False):
        self.data_root = data_root
        self.logger = logging.getLogger('TCBLS_RUL.DataLoader')
        self.include_capacity_feature = bool(include_capacity_feature)
        self.feature_columns = list(self.FEATURE_COLUMNS)
        if self.include_capacity_feature:
            self.feature_columns = self.feature_columns + [self.LABEL_CAPACITY_COLUMN]
        self.logger.info('[TJU] Input features: %d columns | capacity as input: %s (label uses capacity)', len(self.feature_columns), self.include_capacity_feature)
        self.clients = self._load_clients()
        self.grouped_clients = self._group_clients(self.clients)
        self.domain_label_stats = self._build_domain_label_stats()
        self.tasks = {'A': {'domain': ('NCA', 25, '0.5C'), 'train': [0, 1], 'inc': [], 'test': [2], 'update_type': 'initial', 'scaler_group': 'TJU_NCA_25C_05C'}, 'B': {'domain': ('NCA', 25, '0.5C'), 'train': [], 'inc': [3, 4], 'test': [5], 'update_type': 'same_condition', 'scaler_group': 'TJU_NCA_25C_05C'}, 'C': {'domain': ('NCA', 45, '0.5C'), 'train': [0, 1], 'inc': [], 'test': [2], 'update_type': 'new_condition', 'scaler_group': 'TJU_NCA_45C_05C'}, 'D': {'domain': ('NCA', 45, '0.5C'), 'train': [], 'inc': [3, 4], 'test': [5], 'update_type': 'same_condition', 'scaler_group': 'TJU_NCA_45C_05C'}, 'E': {'domain': ('NCM', 25, '0.5C'), 'train': [4, 7], 'inc': [], 'test': [15], 'update_type': 'new_condition', 'scaler_group': 'TJU_NCM_25C_05C'}, 'F': {'domain': ('NCM', 25, '0.5C'), 'train': [], 'inc': [14], 'test': [15], 'update_type': 'same_condition', 'scaler_group': 'TJU_NCM_25C_05C'}}

    def _load_clients(self):
        config_path = os.path.join(self.data_root, 'config_condition_based.json')
        if not os.path.exists(config_path):
            self.logger.warning(f'TJU config not found: {config_path}')
            return []
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        clients = []
        for item in config.get('clients', []):
            params = item.get('condition_params', {})
            raw_path = item.get('battery_file', '')
            path_parts = raw_path.replace('\\', os.sep).replace('/', os.sep).split(os.sep)
            if len(path_parts) < 2:
                continue
            local_path = os.path.join(self.data_root, path_parts[-2], path_parts[-1])
            clients.append({'path': local_path, 'chemistry': params.get('chemistry'), 'temperature': params.get('temperature'), 'charge_rate': params.get('charge_rate'), 'rated_capacity': params.get('capacity'), 'cell_id': params.get('cell_id'), 'subgroup_id': params.get('subgroup_id')})
        return clients

    def _group_clients(self, clients):
        groups = {}
        for client in clients:
            key = (client['chemistry'], client['temperature'], client['charge_rate'])
            if os.path.exists(client['path']):
                groups.setdefault(key, []).append(client)
        for key in groups:
            groups[key] = sorted(groups[key], key=lambda c: (int(c.get('subgroup_id') or 0), int(c.get('cell_id') or 0), os.path.basename(c['path'])))
        return groups

    def _read_capacity_series(self, path):
        try:
            df = pd.read_csv(path, usecols=['capacity'])
        except Exception:
            return None
        capacity = pd.to_numeric(df['capacity'], errors='coerce')
        capacity = capacity.interpolate(limit_direction='both').ffill().bfill()
        values = capacity.to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return None
        return values

    def _estimate_eol_index(self, capacity, eol_capacity, tolerance=0.0):
        if capacity is None or len(capacity) == 0 or (not np.isfinite(eol_capacity)):
            return None
        hits = np.where(capacity <= float(eol_capacity) + float(tolerance))[0]
        if len(hits) == 0:
            return None
        return int(hits[0])

    def _build_domain_label_stats(self):
        """
        Estimate a domain-level EOL threshold and full-life scale for TJU labels.

        TJU contains truncated cells whose CSV ends before capacity reaches EOL.
        A per-file linspace label would incorrectly mark those endpoints as RUL=0.
        We instead use a shared capacity threshold per chemistry/temperature/rate
        domain, and use complete cells in that domain to estimate the full-life
        cycle scale.
        """
        stats = {}
        for domain, clients in self.grouped_clients.items():
            summaries = []
            for client in clients:
                capacity = self._read_capacity_series(client['path'])
                if capacity is None:
                    continue
                summaries.append({'length': int(len(capacity)), 'first': float(capacity[0]), 'last': float(capacity[-1]), 'min': float(np.nanmin(capacity)), 'capacity': capacity})
            if not summaries:
                stats[domain] = {'eol_capacity': np.nan, 'full_life_cycles': None, 'eol_tolerance': 0.0, 'complete_cells': 0, 'total_cells': len(clients)}
                continue
            first_caps = np.array([s['first'] for s in summaries], dtype=np.float64)
            terminal_caps = np.array([min(s['last'], s['min']) for s in summaries], dtype=np.float64)
            eol_capacity = float(np.nanpercentile(terminal_caps, 10))
            start_capacity = float(np.nanmedian(first_caps))
            degradation_span = max(start_capacity - eol_capacity, 1e-06)
            eol_tolerance = max(0.02, 0.04 * degradation_span)
            complete_lives = []
            for summary in summaries:
                eol_idx = self._estimate_eol_index(summary['capacity'], eol_capacity, eol_tolerance)
                if eol_idx is not None:
                    complete_lives.append(eol_idx + 1)
            if complete_lives:
                full_life_cycles = int(np.nanmedian(complete_lives))
            else:
                lengths = np.array([s['length'] for s in summaries], dtype=np.float64)
                full_life_cycles = int(np.nanpercentile(lengths, 90))
            full_life_cycles = max(full_life_cycles, 2)
            stats[domain] = {'start_capacity': start_capacity, 'eol_capacity': eol_capacity, 'full_life_cycles': full_life_cycles, 'eol_tolerance': eol_tolerance, 'complete_cells': len(complete_lives), 'total_cells': len(summaries)}
            self.logger.info('[TJU] Label stats domain=%s | eol_capacity=%.4f | full_life_cycles=%d | complete=%d/%d', domain, eol_capacity, full_life_cycles, len(complete_lives), len(summaries))
        return stats

    def _make_rul_labels(self, capacity, domain):
        if capacity is None or len(capacity) < 2:
            return None
        n = len(capacity)
        stats = self.domain_label_stats.get(tuple(domain), {})
        domain_start_capacity = stats.get('start_capacity', np.nan)
        eol_capacity = stats.get('eol_capacity', np.nan)
        full_life_cycles = stats.get('full_life_cycles')
        if np.isfinite(eol_capacity):
            start_capacity = float(capacity[0])
            if not np.isfinite(start_capacity) or start_capacity <= eol_capacity:
                start_capacity = domain_start_capacity
            denom = float(start_capacity - eol_capacity)
            if np.isfinite(denom) and denom > 1e-06:
                labels = (capacity - eol_capacity) / denom
                labels = np.minimum.accumulate(labels)
                return np.clip(labels, 0.0, 1.0).astype(np.float32).reshape(-1, 1)
        if full_life_cycles is not None and full_life_cycles > 1:
            life_cycles = int(full_life_cycles)
        else:
            life_cycles = n
        cycle_idx = np.arange(n, dtype=np.float64)
        labels = 1.0 - cycle_idx / float(max(life_cycles - 1, 1))
        labels = np.clip(labels, 0.0, 1.0)
        return labels.astype(np.float32).reshape(-1, 1)

    def _process_battery_file(self, path, domain):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self.logger.warning(f'Failed to load TJU file {path}: {e}')
            return (None, None)
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        if not feature_cols:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if self.include_capacity_feature or col != self.LABEL_CAPACITY_COLUMN]
            feature_cols = numeric_cols
        if not feature_cols:
            return (None, None)
        features = df[feature_cols].apply(pd.to_numeric, errors='coerce').interpolate(limit_direction='both').ffill().bfill().fillna(0.0)
        X = features.values.astype(np.float32)
        if len(X) < 50:
            return (None, None)
        if self.LABEL_CAPACITY_COLUMN in df.columns:
            capacity = pd.to_numeric(df[self.LABEL_CAPACITY_COLUMN], errors='coerce').interpolate(limit_direction='both').ffill().bfill().to_numpy(dtype=np.float64)
        else:
            capacity = None
        Y = self._make_rul_labels(capacity, domain)
        if Y is None:
            Y = np.linspace(1.0, 0.0, len(X), dtype=np.float32).reshape(-1, 1)
        return (X, Y)

    def _select_clients(self, domain, indices):
        clients = self.grouped_clients.get(tuple(domain), [])
        selected = []
        for idx in indices:
            if idx < len(clients):
                selected.append(clients[idx])
        return selected

    def _load_units(self, domain, indices):
        Xs, Ys = ([], [])
        for client in self._select_clients(domain, indices):
            x, y = self._process_battery_file(client['path'], domain)
            if x is not None:
                Xs.append(x)
                Ys.append(y)
        if not Xs:
            return (None, None)
        return (Xs, Ys)

    def get_task_data(self, task_name):
        if task_name not in self.tasks:
            return ((None, None), (None, None), (None, None))
        config = self.tasks[task_name]
        domain = config['domain']
        train = self._load_units(domain, config['train'])
        inc = self._load_units(domain, config['inc'])
        test = self._load_units(domain, config['test'])
        return (train, inc, test)

    def get_task_update_type(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('update_type')

    def get_task_scaler_group(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('scaler_group')

class NCMAPSSLoader:
    """
    Loads Modified N-CMAPSS Dataset (pkl files).
    Train: train_df.pkl (Units 1-90)
    Test: test_df.pkl (Units 1-38, no RUL column)
    """

    def __init__(self, data_root):
        self.data_root = data_root
        self.logger = logging.getLogger('TCBLS_RUL.DataLoader')
        self.cache = {}
        try:
            self.logger.info(f'Loading N-CMAPSS data from {data_root}...')
            train_path = os.path.join(data_root, 'train_df.pkl')
            test_path = os.path.join(data_root, 'test_df.pkl')
            npz_path = os.path.join(data_root, 'DS01_resample_100.npz')
            if os.path.exists(train_path) and os.path.exists(test_path):
                self.train_df = pd.read_pickle(train_path)
                self.test_df = pd.read_pickle(test_path)
            elif os.path.exists(npz_path):
                self.logger.info(f'Pickle files not found. Loading from {npz_path}...')
                data = np.load(npz_path, allow_pickle=True)

                def to_df(data_arr, var_names):
                    df = pd.DataFrame(data_arr, columns=var_names)
                    if 'rul' in df.columns:
                        df.rename(columns={'rul': 'RUL'}, inplace=True)
                    return df
                self.train_df = to_df(data['train_data'], data['var'])
                self.test_df = to_df(data['test_data'], data['var'])
                self.logger.info(f'Loaded DS01. Train: {self.train_df.shape}, Test: {self.test_df.shape}')
            else:
                self.logger.error(f'File not found: {train_path} or {npz_path}')
                self.train_df = None
                self.test_df = None
        except Exception as e:
            self.logger.error(f'Error loading files: {e}')
            self.train_df = None
            self.test_df = None
        self.tasks = {'A': {'train_units': [1], 'test_units': [7], 'update_type': 'initial', 'scaler_group': 'NCMAPSS_Group1'}, 'B': {'train_units': [2], 'test_units': [7], 'update_type': 'same_condition', 'scaler_group': 'NCMAPSS_Group1'}, 'C': {'train_units': [3], 'test_units': [8], 'update_type': 'new_condition', 'scaler_group': 'NCMAPSS_Group2'}, 'D': {'train_units': [4], 'test_units': [8], 'update_type': 'same_condition', 'scaler_group': 'NCMAPSS_Group2'}, 'E': {'train_units': [5], 'test_units': [9], 'update_type': 'new_condition', 'scaler_group': 'NCMAPSS_Group3'}, 'F': {'train_units': [6], 'test_units': [10], 'update_type': 'same_condition', 'scaler_group': 'NCMAPSS_Group3'}}

    def _process_df(self, df, unit_ids=None):
        """
        Extract features and labels.
        Features: Cols 4-21 (T24...T2) -> 18 features
        Label: RUL (if exists) or Constructed from Cycle
        """
        if df is None:
            return (None, None)
        if unit_ids is not None:
            df = df[df['unit'].isin(unit_ids)]
        if df.empty:
            return (None, None)
        feature_cols = df.columns[4:22]
        X = df[feature_cols].values.astype(np.float32)
        if 'RUL' in df.columns:
            Y = df['RUL'].values.reshape(-1, 1).astype(np.float32)
        else:
            df_cycle = df[['unit', 'cycle']].astype(np.float32)
            max_cycles = df_cycle.groupby('unit')['cycle'].transform('max')
            Y = (max_cycles - df_cycle['cycle']).values.reshape(-1, 1)
        max_y = np.max(Y)
        if max_y > 0:
            Y = Y / max_y
        return (X, Y)

    def get_task_data(self, task_name):
        if task_name not in self.tasks:
            return ((None, None), (None, None), (None, None))
        config = self.tasks[task_name]
        X_train_data, Y_train_data = self._process_df(self.train_df, config['train_units'])
        X_test, Y_test = self._process_df(self.test_df, config['test_units'])
        if task_name == 'A':
            return ((X_train_data, Y_train_data), (None, None), (X_test, Y_test))
        else:
            return ((None, None), (X_train_data, Y_train_data), (X_test, Y_test))

    def get_task_update_type(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('update_type')

    def get_task_scaler_group(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('scaler_group')

class CMAPSSLoader:
    """
    Loads NASA C-MAPSS dataset.
    """
    '\n    Loads C-MAPSS Turbofan Engine Degradation Simulation Data Set.\n    Handles FD001 and FD004 sub-datasets.\n    '

    def __init__(self, data_root):
        self.data_root = data_root
        self.logger = logging.getLogger('TCBLS_RUL.DataLoader')
        self.cache = {}
        self.selected_features = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]
        self.max_rul = 135
        self.tasks = {'A': {'dataset': 'FD001', 'train': [1, 2, 3], 'inc': [], 'test': [10], 'update_type': 'initial', 'scaler_group': 'CMAPSS_FD001'}, 'B': {'dataset': 'FD001', 'train': [1, 2, 3], 'inc': [4, 5], 'test': [10], 'update_type': 'same_condition', 'scaler_group': 'CMAPSS_FD001'}, 'C': {'dataset': 'FD001', 'train': [1, 2, 3, 4, 5], 'inc': [6, 7], 'test': [10], 'update_type': 'same_condition', 'scaler_group': 'CMAPSS_FD001'}, 'D': {'dataset': 'FD001', 'train': [1, 2, 3, 4, 5, 6, 7], 'inc': [8, 9], 'test': [10], 'update_type': 'same_condition', 'scaler_group': 'CMAPSS_FD001'}, 'E': {'dataset': 'FD004', 'train': [1, 2, 3], 'inc': [], 'test': [10], 'update_type': 'new_condition', 'scaler_group': 'CMAPSS_FD004'}, 'F': {'dataset': 'FD004', 'train': [1, 2, 3], 'inc': [4, 5], 'test': [10], 'update_type': 'same_condition', 'scaler_group': 'CMAPSS_FD004'}, 'G': {'dataset': 'FD004', 'train': [1, 2, 3, 4, 5], 'inc': [6, 7], 'test': [10], 'update_type': 'same_condition', 'scaler_group': 'CMAPSS_FD004'}, 'H': {'dataset': 'FD004', 'train': [1, 2, 3, 4, 5, 6, 7], 'inc': [8, 9], 'test': [10], 'update_type': 'same_condition', 'scaler_group': 'CMAPSS_FD004'}}

    def _load_dataset_file(self, dataset_name, is_test=False):
        """Loads raw file."""
        prefix = 'test' if is_test else 'train'
        filename = f'{prefix}_{dataset_name}.txt'
        filepath = os.path.join(self.data_root, filename)
        if not os.path.exists(filepath):
            self.logger.error(f'File not found: {filepath}')
            return None
        cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        df = pd.read_csv(filepath, sep='\\s+', header=None, names=cols)
        return df

    def _load_rul_file(self, dataset_name):
        """Loads RUL ground truth for test set."""
        filename = f'RUL_{dataset_name}.txt'
        filepath = os.path.join(self.data_root, filename)
        if not os.path.exists(filepath):
            return None
        return pd.read_csv(filepath, sep='\\s+', header=None, names=['rul'])

    def _process_engine(self, df_engine, ground_truth_rul=None):
        """
        Process a single engine unit.
        Returns X (features) and Y (RUL).
        """
        sensor_cols = [f's{i}' for i in range(1, 22)]
        X_all = df_engine[sensor_cols].values
        X = X_all[:, self.selected_features]
        cycles = df_engine['cycle'].values
        max_cycle = cycles.max()
        if ground_truth_rul is not None:
            current_rul = ground_truth_rul + (max_cycle - cycles)
        else:
            current_rul = max_cycle - cycles
        Y = np.minimum(current_rul, self.max_rul).reshape(-1, 1)
        Y = Y / self.max_rul
        return (X, Y)

    def get_task_data(self, task_name):
        if task_name not in self.tasks:
            return ((None, None), (None, None), (None, None))
        config = self.tasks[task_name]
        dataset_name = config['dataset']
        df_train_all = self._load_dataset_file(dataset_name, is_test=False)
        df_test_all = self._load_dataset_file(dataset_name, is_test=True)
        df_rul_all = self._load_rul_file(dataset_name)
        if df_train_all is None:
            return ((None, None), (None, None), (None, None))

        def extract_units(df, unit_ids, is_test=False, rul_df=None):
            Xs, Ys = ([], [])
            for uid in unit_ids:
                df_unit = df[df['unit'] == uid]
                if df_unit.empty:
                    continue
                gt_rul = None
                if is_test and rul_df is not None:
                    if uid <= len(rul_df):
                        gt_rul = rul_df.iloc[uid - 1]['rul']
                x, y = self._process_engine(df_unit, gt_rul)
                Xs.append(x)
                Ys.append(y)
            if not Xs:
                return (None, None)
            return (Xs, Ys)
        X_train, Y_train = extract_units(df_train_all, config['train'])
        X_inc, Y_inc = extract_units(df_train_all, config['inc'])
        X_test, Y_test = extract_units(df_train_all, config['test'], is_test=False)
        return ((X_train, Y_train), (X_inc, Y_inc), (X_test, Y_test))

    def get_task_update_type(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('update_type')

    def get_task_scaler_group(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('scaler_group')

class PHM2012Loader:
    """
    Loads PHM 2012 Bearing Dataset.
    """
    OUTPUT_DIM = BEARING_TWO_CHANNEL_FEATURE_DIM * 5

    def __init__(self, data_root, phm_data_source='full'):
        self.data_root = data_root
        self.logger = logging.getLogger('TCBLS_RUL.DataLoader')
        self.cache = {}
        self.phm_data_source = str(phm_data_source).lower()
        self.tasks = {'A': {'train': ['Bearing1_1'], 'inc': [], 'test': ['Bearing1_6'], 'update_type': 'initial', 'scaler_group': 'PHM_Group1'}, 'B': {'train': ['Bearing1_1'], 'inc': ['Bearing1_2'], 'test': ['Bearing1_6'], 'update_type': 'same_condition', 'scaler_group': 'PHM_Group1'}, 'C': {'train': ['Bearing1_1', 'Bearing1_2'], 'inc': ['Bearing1_3'], 'test': ['Bearing1_6'], 'update_type': 'same_condition', 'scaler_group': 'PHM_Group1'}, 'D': {'train': ['Bearing1_1', 'Bearing1_2', 'Bearing1_3'], 'inc': ['Bearing1_5'], 'test': ['Bearing1_6'], 'update_type': 'same_condition', 'scaler_group': 'PHM_Group1'}, 'E': {'train': ['Bearing2_1'], 'inc': [], 'test': ['Bearing2_6'], 'update_type': 'new_condition', 'scaler_group': 'PHM_Group2'}, 'F': {'train': ['Bearing2_1'], 'inc': ['Bearing2_2'], 'test': ['Bearing2_6'], 'update_type': 'same_condition', 'scaler_group': 'PHM_Group2'}, 'G': {'train': ['Bearing2_1', 'Bearing2_2'], 'inc': ['Bearing2_4'], 'test': ['Bearing2_6'], 'update_type': 'same_condition', 'scaler_group': 'PHM_Group2'}, 'H': {'train': ['Bearing2_1', 'Bearing2_2', 'Bearing2_4'], 'inc': ['Bearing2_7'], 'test': ['Bearing2_6'], 'update_type': 'same_condition', 'scaler_group': 'PHM_Group2'}}

    def _load_bearing(self, name):
        if name in self.cache:
            return self.cache[name]
        paths = [os.path.join(self.data_root, 'Learning_set', name)]
        if self.phm_data_source == 'full':
            paths.extend([os.path.join(self.data_root, 'Full_Test_Set', name), os.path.join(self.data_root, 'Test_set', name)])
        else:
            paths.extend([os.path.join(self.data_root, 'Test_set', name), os.path.join(self.data_root, 'Full_Test_Set', name)])
        folder = None
        for p in paths:
            if os.path.exists(p):
                folder = p
                break
        if not folder:
            return (None, None)
        files = sorted(glob.glob(os.path.join(folder, 'acc_*.csv')))
        if not files:
            return (None, None)
        n_files_raw = len(files)
        self.logger.info(f'Bearing {name}: Found {n_files_raw} files in {folder}')
        step = PHM_DOWNSAMPLE_STEP
        if step is None:
            if len(files) > 1000:
                step = 10
            elif len(files) > 200:
                step = 2
            else:
                step = 1
        step = max(1, int(step))
        if step > 1:
            files = files[::step]
        X_list = []
        for f in files:
            try:
                df = pd.read_csv(f, header=None)
                sig_h = df.iloc[:, 4].values
                sig_v = df.iloc[:, 5].values
                signal_pair = np.column_stack([sig_h, sig_v])
                feat = _extract_bearing_multichannel_features(signal_pair, combine='concat')
                X_list.append(feat)
            except Exception as e:
                pass
        if not X_list:
            return (None, None)
        X = np.array(X_list, dtype=np.float32)
        Y = _build_bearing_fpt_hi_labels(X)
        X = _enhance_bearing_sequence(X)
        self.cache[name] = (X, Y)
        return (X, Y)

    def get_task_data(self, task_name):
        if task_name not in self.tasks:
            return ((None, None), (None, None), (None, None))
        config = self.tasks[task_name]

        def load_set(names):
            Xs, Ys = ([], [])
            for n in names:
                x, y = self._load_bearing(n)
                if x is not None:
                    Xs.append(x)
                    Ys.append(y)
            if not Xs:
                return (None, None)
            return (Xs, Ys)
        train = load_set(config['train'])
        inc = load_set(config['inc'])
        test = load_set(config['test'])
        return (train, inc, test)

    def get_task_update_type(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('update_type')

    def get_task_scaler_group(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('scaler_group')

class XJTULoader:

    @classmethod
    def get_output_dim(cls):
        base = BEARING_TWO_CHANNEL_FEATURE_DIM * 5
        if XJTU_USE_SEQUENCE_CONTEXT:
            base += XJTU_SEQUENCE_CONTEXT_DIM
        return base
    OUTPUT_DIM = BEARING_TWO_CHANNEL_FEATURE_DIM * 5

    def __init__(self, data_root):
        self.data_root = data_root
        self.logger = logging.getLogger('TCBLS_RUL.DataLoader')
        self.cache = {}
        self.conditions = {'Cond1': '35Hz12kN', 'Cond2': '37.5Hz11kN', 'Cond3': '40Hz10kN'}
        self.tasks = {'A': {'cond': 'Cond1', 'train': ['1_1', '1_2', '1_3'], 'inc': [], 'test': ['1_4'], 'update_type': 'initial', 'scaler_group': 'XJTU_Cond1'}, 'B': {'cond': 'Cond1', 'train': ['1_1', '1_2', '1_3'], 'inc': [], 'test': ['1_5'], 'update_type': 'same_condition', 'scaler_group': 'XJTU_Cond1'}, 'C': {'cond': 'Cond2', 'train': ['2_1', '2_2', '2_3'], 'inc': [], 'test': ['2_4'], 'update_type': 'new_condition', 'scaler_group': 'XJTU_Cond2'}, 'D': {'cond': 'Cond2', 'train': ['2_1', '2_2', '2_3'], 'inc': [], 'test': ['2_5'], 'update_type': 'same_condition', 'scaler_group': 'XJTU_Cond2'}, 'E': {'cond': 'Cond3', 'train': ['3_1', '3_2', '3_3'], 'inc': [], 'test': ['3_4'], 'update_type': 'new_condition', 'scaler_group': 'XJTU_Cond3'}, 'F': {'cond': 'Cond3', 'train': ['3_1', '3_2', '3_3'], 'inc': [], 'test': ['3_5'], 'update_type': 'same_condition', 'scaler_group': 'XJTU_Cond3'}}

    def _extract_features(self, signal):
        return _extract_bearing_multichannel_features(signal, combine='concat')

    def _load_bearing(self, condition_dir, bearing_name):
        folder_name = f'Bearing{bearing_name}'
        file_name = f'{folder_name}.npz'
        path = os.path.join(self.data_root, condition_dir, folder_name, file_name)
        if path in self.cache:
            return self.cache[path]
        if not os.path.exists(path):
            self.logger.warning(f'File not found: {path}')
            return (None, None)
        try:
            data = np.load(path, allow_pickle=True)
            keys = sorted(list(data.keys()), key=lambda x: int(x))
            X_list = []
            for k in keys:
                signal = data[k]
                feat = self._extract_features(signal)
                X_list.append(feat)
            X = np.array(X_list, dtype=np.float32)
            Y = _build_bearing_fpt_hi_labels(X)
            X = _enhance_xjtu_bearing_sequence(X)
            self.cache[path] = (X, Y)
            return (X, Y)
        except Exception as e:
            self.logger.error(f'Error loading {path}: {e}')
            return (None, None)

    def get_task_data(self, task_name):
        if task_name not in self.tasks:
            return ((None, None), (None, None), (None, None))
        config = self.tasks[task_name]
        cond_dir = self.conditions[config['cond']]

        def load_set(bearing_ids):
            Xs, Ys = ([], [])
            for bid in bearing_ids:
                x, y = self._load_bearing(cond_dir, bid)
                if x is not None:
                    Xs.append(x)
                    Ys.append(y)
            if not Xs:
                return (None, None)
            if len(Xs) == 1:
                return (Xs[0], Ys[0])
            return (Xs, Ys)
        train_data = load_set(config['train'])
        inc_data = load_set(config['inc'])
        test_data = load_set(config['test'])
        return (train_data, inc_data, test_data)

    def get_task_update_type(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('update_type')

    def get_task_scaler_group(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('scaler_group')

class IMSLoader:
    OUTPUT_DIM = BEARING_TWO_CHANNEL_MASKED_FEATURE_DIM * 5

    def __init__(self, data_root):
        self.data_root = data_root
        self.logger = logging.getLogger('TCBLS_RUL.DataLoader')
        self.cache = {}
        self.tasks = {'A': {'train': [('1st_test', 'bearing_1'), ('1st_test', 'bearing_2'), ('1st_test', 'bearing_3')], 'inc': [], 'test': [('1st_test', 'bearing_4')], 'update_type': 'initial', 'scaler_group': 'IMS_1st_test'}, 'B': {'train': [('1st_test', 'bearing_2'), ('1st_test', 'bearing_3'), ('1st_test', 'bearing_4')], 'inc': [], 'test': [('1st_test', 'bearing_1')], 'update_type': 'initial', 'scaler_group': 'IMS_1st_test'}, 'C': {'train': [('1st_test', 'bearing_3'), ('1st_test', 'bearing_4'), ('1st_test', 'bearing_1')], 'inc': [], 'test': [('1st_test', 'bearing_2')], 'update_type': 'initial', 'scaler_group': 'IMS_1st_test'}, 'D': {'train': [('1st_test', 'bearing_4'), ('1st_test', 'bearing_1'), ('1st_test', 'bearing_2')], 'inc': [], 'test': [('1st_test', 'bearing_3')], 'update_type': 'initial', 'scaler_group': 'IMS_1st_test'}}

    def _extract_features(self, signal_segment):
        return _extract_bearing_multichannel_features(signal_segment, combine='concat_masked')

    def _load_bearing(self, set_name, bearing_name):
        filename = f'{bearing_name}.npz'
        path = os.path.join(self.data_root, set_name, filename)
        if path in self.cache:
            return self.cache[path]
        if not os.path.exists(path):
            self.logger.warning(f'File not found: {path}')
            return (None, None)
        try:
            data_pkg = np.load(path, allow_pickle=True)
            raw_data = np.asarray(data_pkg['data'])
            timestamps = data_pkg['timestamps']
            if raw_data.ndim == 1:
                raw_data = raw_data[:, None]
            num_segments = len(timestamps)
            if IMS_SEGMENT_MODE == 'fixed_20480':
                segment_len = 20480
                num_segments = min(num_segments, raw_data.shape[0] // segment_len)
            else:
                segment_len = raw_data.shape[0] // max(num_segments, 1)
            X_list = []
            for i in range(num_segments):
                seg = raw_data[i * segment_len:(i + 1) * segment_len]
                feat = self._extract_features(seg)
                X_list.append(feat)
            X = np.array(X_list, dtype=np.float32)
            Y = _build_bearing_fpt_hi_labels(X)
            X = _enhance_bearing_sequence(X)
            self.cache[path] = (X, Y)
            return (X, Y)
        except Exception as e:
            self.logger.error(f'Error loading {path}: {e}')
            return (None, None)

    def get_task_data(self, task_name):
        if task_name not in self.tasks:
            return ((None, None), (None, None), (None, None))
        config = self.tasks[task_name]

        def load_set(bearing_list):
            Xs, Ys = ([], [])
            for set_name, b_name in bearing_list:
                x, y = self._load_bearing(set_name, b_name)
                if x is not None:
                    Xs.append(x)
                    Ys.append(y)
            if not Xs:
                return (None, None)
            if len(Xs) == 1:
                return (Xs[0], Ys[0])
            return (Xs, Ys)
        train_data = load_set(config['train'])
        inc_data = load_set(config['inc'])
        test_data = load_set(config['test'])
        return (train_data, inc_data, test_data)

    def get_task_update_type(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('update_type')

    def get_task_adapt_epochs(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('adapt_epochs')

    def get_task_scaler_group(self, task_name):
        config = self.tasks.get(task_name, {})
        return config.get('scaler_group')
