import numpy as np
import time
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from .tcbls import TCBLS
from .feature_extractor import LiquidTransformerFeatureExtractor
from .replay import ExperienceReplayBuffer

class LiquidTransBLS(TCBLS):
    """LiquidTransBLS with replay and condition-aware incremental learning."""

    def __init__(self, window_size, input_dim, d_model=64, n_heads=4, n_layers=2, n_enhancement_nodes=3000, reg_param=1e-08, dropout=0.1, replay_capacity=2000, **kwargs):
        self.base_feature_dim = 2 * d_model
        self.light_history_dim = 4
        n_mapping_nodes_in = self.base_feature_dim
        tcbls_mapping_groups = kwargs.get('tcbls_mapping_groups', None)
        tcbls_mapping_nodes = kwargs.get('tcbls_mapping_nodes', None)
        self.use_tcbls_mapping = tcbls_mapping_groups is not None or tcbls_mapping_nodes is not None
        if self.use_tcbls_mapping:
            n_mapping_groups_in = int(tcbls_mapping_groups if tcbls_mapping_groups is not None else 1)
            n_mapping_nodes_in = int(tcbls_mapping_nodes if tcbls_mapping_nodes is not None else n_mapping_nodes_in)
        else:
            n_mapping_groups_in = 1
        super().__init__(n_mapping_groups=n_mapping_groups_in, n_mapping_nodes=n_mapping_nodes_in, n_enhancement_nodes=n_enhancement_nodes, reg_param=reg_param)
        self.window_size = window_size
        self.input_dim_feat = input_dim
        self.d_model = d_model
        self.is_bearing_dataset = kwargs.get('is_bearing_dataset', False)
        self.history_scale = kwargs.get('history_scale', 0.12)
        self.lambda_distill = kwargs.get('lambda_distill', 0.35)
        self.lambda_stage = kwargs.get('lambda_stage', 0.55)
        self.lambda_global = kwargs.get('lambda_global', 0.15)
        self.lambda_trend = kwargs.get('lambda_trend', 0.05)
        self.lambda_order = kwargs.get('lambda_order', 0.0)
        self.incremental_lr_scale = kwargs.get('incremental_lr_scale', 0.35)
        self.replay_update_scale = kwargs.get('replay_update_scale', 0.18)
        self.replay_sample_ratio = kwargs.get('replay_sample_ratio', 0.4)
        self.same_condition_buffer_capacity = int(kwargs.get('same_condition_buffer_capacity', min(replay_capacity, 1200)))
        self.same_condition_recent_bias = kwargs.get('same_condition_recent_bias', 0.35)
        self.same_condition_stage_bias = kwargs.get('same_condition_stage_bias', 0.45)
        self.same_condition_hard_bias = kwargs.get('same_condition_hard_bias', 0.5)
        self.same_condition_similarity_bias = kwargs.get('same_condition_similarity_bias', 0.3)
        self.same_condition_drift_gain = kwargs.get('same_condition_drift_gain', 0.65)
        self.same_condition_order_bias = kwargs.get('same_condition_order_bias', 0.35)
        self.order_margin = kwargs.get('order_margin', 0.015)
        self.multi_scale_order_offsets = tuple((int(v) for v in kwargs.get('multi_scale_order_offsets', (1, 2, 4))))
        self.use_output_calibration = kwargs.get('use_output_calibration', False)
        self.calibration_min_samples = int(kwargs.get('calibration_min_samples', 48))
        self.calibration_max_points = int(kwargs.get('calibration_max_points', 2000))
        self.output_calibrator = None
        self.feature_extractor = LiquidTransformerFeatureExtractor(input_dim=input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.feature_extractor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.feature_extractor.eval()
        self.temp_head = torch.nn.Linear(self.base_feature_dim, 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_capacity, is_bearing=self.is_bearing_dataset)
        self.same_condition_anchor_X = None
        self.same_condition_anchor_Y = None
        self.supports_grouped_windows = True
        self.stateful_windows = False

    def _apply_output_calibration(self, preds):
        pred_arr = np.asarray(preds, dtype=np.float32)
        if not self.use_output_calibration or self.output_calibrator is None or pred_arr.size == 0:
            return pred_arr
        flat = pred_arr.reshape(-1)
        calibrated = self.output_calibrator.predict(flat)
        calibrated = np.clip(calibrated, 0.0, 1.0).astype(np.float32)
        return calibrated.reshape(pred_arr.shape)

    def _normalize_window_groups(self, X, default_stateful=False):
        if X is None:
            return []
        items = X if isinstance(X, list) else [X]
        groups = []
        for item in items:
            stateful = default_stateful
            group = item
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (bool, np.bool_)):
                group, stateful = item
            if group is None or len(group) == 0:
                continue
            group = np.asarray(group)
            if group.ndim == 3:
                group_reshaped = group
            elif group.ndim == 2 and group.shape[1] == self.window_size * self.input_dim_feat:
                group_reshaped = group.reshape(group.shape[0], self.window_size, self.input_dim_feat)
            else:
                raise ValueError(f'Unexpected window shape {group.shape}; expected (N, {self.window_size * self.input_dim_feat}) or (N, {self.window_size}, {self.input_dim_feat}).')
            groups.append((group_reshaped, bool(stateful)))
        return groups

    def _compute_light_history_features(self, feature_sequence, ordered=True):
        feature_sequence = np.asarray(feature_sequence, dtype=np.float32)
        n = feature_sequence.shape[0]
        if n == 0:
            return np.empty((0, self.light_history_dim), dtype=np.float32)
        current_energy = np.mean(np.abs(feature_sequence), axis=1, keepdims=True)
        current_disp = np.std(feature_sequence, axis=1, keepdims=True)
        prev_shift = np.zeros((n, 1), dtype=np.float32)
        prev_jump = np.zeros((n, 1), dtype=np.float32)
        if ordered and n > 1:
            delta = feature_sequence[1:] - feature_sequence[:-1]
            prev_shift[1:, 0] = np.mean(delta, axis=1)
            prev_jump[1:, 0] = np.mean(np.abs(delta), axis=1)
        cumulative_jump = np.cumsum(prev_jump[:, 0], dtype=np.float32) / np.arange(1, n + 1, dtype=np.float32)
        history_feats = np.concatenate([current_energy, current_disp, prev_shift, cumulative_jump.reshape(-1, 1)], axis=1)
        history_feats = np.nan_to_num(history_feats, nan=0.0, posinf=0.0, neginf=0.0)
        return (self.history_scale * np.clip(history_feats, -5.0, 5.0)).astype(np.float32)

    def _build_adaptation_features(self, base_features, ordered=True):
        if self.light_history_dim <= 0:
            return base_features
        history_feats = torch.tensor(self._compute_light_history_features(base_features.detach().cpu().numpy(), ordered=ordered), dtype=torch.float32, device=self.device)
        return torch.cat([base_features, history_feats], dim=1)

    def _flatten_window_groups(self, X):
        groups = self._normalize_window_groups(X, default_stateful=False)
        if not groups:
            return None
        return np.vstack([group.reshape(group.shape[0], -1) for group, _ in groups])

    def _ensure_column_targets(self, Y):
        if Y is None:
            return None
        Y_arr = np.asarray(Y, dtype=np.float32)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        return Y_arr

    def _safe_sampling_probs(self, weights):
        probs = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(probs) == 0:
            return probs
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = probs.sum()
        if total <= 0:
            return np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
        return probs / total

    def _trim_same_condition_memory(self, X_mem, Y_mem):
        if X_mem is None or Y_mem is None:
            return (None, None)
        X_arr = np.asarray(X_mem, dtype=np.float32)
        Y_arr = self._ensure_column_targets(Y_mem)
        if len(X_arr) <= self.same_condition_buffer_capacity:
            return (X_arr, Y_arr)
        y_flat = Y_arr.reshape(-1)
        recency = np.linspace(0.0, 1.0, len(y_flat), dtype=np.float32)
        stage_bonus = np.zeros(len(y_flat), dtype=np.float32)
        stage_bonus[(y_flat > 0.3) & (y_flat <= 0.7)] = 0.45
        stage_bonus[y_flat <= 0.3] = 1.0
        priority = 1.0 + self.same_condition_recent_bias * recency + self.same_condition_stage_bias * stage_bonus
        keep_idx = np.argsort(priority)[-self.same_condition_buffer_capacity:]
        keep_idx = np.sort(keep_idx)
        return (X_arr[keep_idx], Y_arr[keep_idx])

    def _extract_same_condition_fit_data(self, X, Y):
        Y_arr = self._ensure_column_targets(Y)
        if X is None or Y_arr is None:
            return (None, None)
        if isinstance(X, list):
            current_groups = []
            current_targets = []
            offset = 0
            for item in X:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (bool, np.bool_)):
                    break
                group = np.asarray(item)
                if group.ndim == 3:
                    group_flat = group.reshape(group.shape[0], -1)
                elif group.ndim == 2:
                    group_flat = group
                else:
                    continue
                end = offset + len(group_flat)
                if end > len(Y_arr):
                    break
                current_groups.append(group_flat.astype(np.float32, copy=False))
                current_targets.append(Y_arr[offset:end])
                offset = end
            if current_groups:
                return (np.vstack(current_groups), np.vstack(current_targets))
        return (self._flatten_window_groups(X), Y_arr)

    def _refresh_same_condition_memory(self, X, Y):
        anchor_X, anchor_Y = self._extract_same_condition_fit_data(X, Y)
        if anchor_X is None or anchor_Y is None or len(anchor_X) == 0:
            return
        self.same_condition_anchor_X, self.same_condition_anchor_Y = self._trim_same_condition_memory(anchor_X, anchor_Y)

    def _append_same_condition_memory(self, X_new_flat, Y_new):
        Y_arr = self._ensure_column_targets(Y_new)
        if X_new_flat is None or Y_arr is None or len(X_new_flat) == 0:
            return
        X_arr = np.asarray(X_new_flat, dtype=np.float32)
        if self.same_condition_anchor_X is None or self.same_condition_anchor_Y is None:
            mem_X, mem_Y = (X_arr, Y_arr)
        else:
            mem_X = np.vstack([self.same_condition_anchor_X, X_arr])
            mem_Y = np.vstack([self.same_condition_anchor_Y, Y_arr])
        self.same_condition_anchor_X, self.same_condition_anchor_Y = self._trim_same_condition_memory(mem_X, mem_Y)

    def _split_targets_by_groups(self, Y, groups):
        if Y is None:
            return []
        Y_arr = np.vstack(Y) if isinstance(Y, list) else np.asarray(Y)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        sizes = [group.shape[0] for group, _ in groups]
        total_size = sum(sizes)
        if len(Y_arr) != total_size:
            raise ValueError(f'Target size mismatch: got {len(Y_arr)}, expected {total_size}.')
        split_targets = []
        start = 0
        for size in sizes:
            split_targets.append(Y_arr[start:start + size])
            start += size
        return split_targets

    def _stage_labels_from_targets(self, Y):
        y = np.asarray(Y).reshape(-1)
        stages = np.zeros(len(y), dtype=np.int64)
        stages[(y <= 0.7) & (y > 0.3)] = 1
        stages[y <= 0.3] = 2
        return stages

    def _stage_labels_from_group_progress(self, group_length):
        if group_length <= 0:
            return np.empty((0,), dtype=np.int64)
        progress = np.linspace(0.0, 1.0, group_length, dtype=np.float32)
        stages = np.zeros(group_length, dtype=np.int64)
        stages[(progress >= 1.0 / 3.0) & (progress < 2.0 / 3.0)] = 1
        stages[progress >= 2.0 / 3.0] = 2
        return stages

    def _stat_align_loss(self, source_features, target_features):
        mu_s = source_features.mean(dim=0)
        std_s = source_features.std(dim=0, unbiased=False) + 1e-08
        mu_t = target_features.mean(dim=0)
        std_t = target_features.std(dim=0, unbiased=False) + 1e-08
        return torch.mean((mu_s - mu_t) ** 2) + torch.mean((std_s - std_t) ** 2)

    def _stage_aware_align_loss(self, source_features, target_features, source_stage_labels, target_stage_labels):
        source_stage_labels = torch.as_tensor(source_stage_labels, dtype=torch.long, device=self.device)
        target_stage_labels = torch.as_tensor(target_stage_labels, dtype=torch.long, device=self.device)
        matched_losses = []
        matched_weights = []
        stage_weights = [0.8, 1.0, 1.2]
        for stage_idx, stage_weight in enumerate(stage_weights):
            mask_s = source_stage_labels == stage_idx
            mask_t = target_stage_labels == stage_idx
            if mask_s.any() and mask_t.any():
                stage_loss = self._stat_align_loss(source_features[mask_s], target_features[mask_t])
                matched_losses.append(stage_loss * stage_weight)
                matched_weights.append(stage_weight)
        if matched_losses:
            return sum(matched_losses) / max(sum(matched_weights), 1e-08)
        return self._stat_align_loss(source_features, target_features)

    def _set_adaptation_trainable(self, incremental_mode=False):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        if not incremental_mode:
            for param in self.temp_head.parameters():
                param.requires_grad = True
            return
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for module in [self.feature_extractor.feature_attention, self.feature_extractor.temporal_attention]:
            for param in module.parameters():
                param.requires_grad = True
        if hasattr(self.feature_extractor, 'transformer_encoder') and len(self.feature_extractor.transformer_encoder.layers) > 0:
            for param in self.feature_extractor.transformer_encoder.layers[-1].parameters():
                param.requires_grad = True
        if hasattr(self.feature_extractor, 'embedding'):
            for param in self.feature_extractor.embedding.parameters():
                param.requires_grad = True
        if hasattr(self.feature_extractor, 'lnn_layer'):
            for param in self.feature_extractor.lnn_layer.parameters():
                param.requires_grad = True
        elif hasattr(self.feature_extractor, 'linear_layer'):
            for param in self.feature_extractor.linear_layer.parameters():
                param.requires_grad = True
        if hasattr(self.feature_extractor, 'smoothing_conv'):
            for param in self.feature_extractor.smoothing_conv.parameters():
                param.requires_grad = True
        for param in self.temp_head.parameters():
            param.requires_grad = True

    def _get_trainable_parameters(self):
        params = []
        params.extend([p for p in self.feature_extractor.parameters() if p.requires_grad])
        params.extend([p for p in self.temp_head.parameters() if p.requires_grad])
        return params

    def _predict_stage_from_features(self, features):
        if features.shape[1] != self.base_feature_dim:
            features = features[:, :self.base_feature_dim]
        with torch.no_grad():
            preds = self.temp_head(features).clamp(0.0, 1.0).cpu().numpy()
        return self._stage_labels_from_targets(preds)

    def _merge_stage_labels(self, pred_labels, prior_labels):
        pred_arr = np.asarray(pred_labels, dtype=np.float32)
        prior_arr = np.asarray(prior_labels, dtype=np.float32)
        merged = np.rint(0.3 * pred_arr + 0.7 * prior_arr)
        return np.clip(merged, 0, 2).astype(np.int64)

    def _weighted_mse_loss(self, preds, targets, target_np):
        weights_np = self._compute_sample_weights(target_np)
        weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        return torch.mean((preds - targets) ** 2 * weights)

    def _weighted_incremental_update(self, A_batch, Y_batch, sample_weight=1.0, sample_weights=None):
        if A_batch is None or len(A_batch) == 0:
            return
        if sample_weights is not None:
            row_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1)
            row_weights = np.clip(row_weights * float(sample_weight), 1e-06, None)
            row_scale = np.sqrt(row_weights)
            self._update_weights_data_increment(A_batch * row_scale, Y_batch * row_scale)
            return
        weight = float(sample_weight)
        if weight <= 0:
            return
        if abs(weight - 1.0) < 1e-08:
            self._update_weights_data_increment(A_batch, Y_batch)
            return
        scale = np.sqrt(weight)
        A_weighted = A_batch * scale
        Y_weighted = Y_batch * scale
        self._update_weights_data_increment(A_weighted, Y_weighted)

    def _local_trend_loss(self, preds, ordered=True):
        if not ordered or preds.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)
        diffs = preds[1:] - preds[:-1]
        return torch.mean(torch.relu(diffs))

    def _multi_scale_order_loss(self, preds, targets=None, ordered=True):
        if not ordered or preds.shape[0] < 3:
            return torch.tensor(0.0, device=self.device)
        pred_seq = preds.reshape(-1)
        target_seq = targets.reshape(-1) if targets is not None else None
        losses = []
        for offset in self.multi_scale_order_offsets:
            if pred_seq.shape[0] <= offset:
                continue
            pred_gap = pred_seq[offset:] - pred_seq[:-offset]
            if target_seq is not None:
                target_gap = torch.clamp(target_seq[:-offset] - target_seq[offset:], min=0.0)
                margin = self.order_margin + 0.2 * target_gap
            else:
                margin = self.order_margin
            losses.append(torch.mean(torch.relu(pred_gap + margin)))
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return sum(losses) / len(losses)

    def _sequence_violation_profile(self, preds, targets=None):
        pred_seq = np.asarray(preds, dtype=np.float32).reshape(-1)
        if len(pred_seq) < 2:
            return (np.ones(len(pred_seq), dtype=np.float32), 0.0)
        target_seq = None
        if targets is not None:
            target_arr = self._ensure_column_targets(targets)
            if target_arr is not None and len(target_arr) == len(pred_seq):
                target_seq = target_arr.reshape(-1)
        profile = np.zeros(len(pred_seq), dtype=np.float32)
        magnitudes = []
        for offset in self.multi_scale_order_offsets:
            if len(pred_seq) <= offset:
                continue
            pred_gap = pred_seq[offset:] - pred_seq[:-offset]
            if target_seq is not None:
                target_gap = np.clip(target_seq[:-offset] - target_seq[offset:], 0.0, 1.0)
                margin = self.order_margin + 0.2 * target_gap
            else:
                margin = self.order_margin
            violation = np.maximum(pred_gap + margin, 0.0)
            if len(violation) == 0:
                continue
            magnitudes.append(float(np.mean(violation)))
            profile[offset:] += violation
            profile[:-offset] += 0.35 * violation
        if not magnitudes:
            return (np.ones(len(pred_seq), dtype=np.float32), 0.0)
        normalized = profile / (np.mean(profile) + 1e-06)
        weights = 1.0 + self.same_condition_order_bias * np.clip(normalized, 0.0, 2.5)
        score = float(np.tanh(np.mean(magnitudes) / 0.05))
        return (weights.astype(np.float32), score)

    def _sample_group_chunk(self, groups, aux_arrays=None, batch_size=128):
        lengths = np.array([len(group) for group, _ in groups], dtype=np.float32)
        probs = lengths / lengths.sum()
        group_idx = np.random.choice(len(groups), p=probs)
        group, ordered = groups[group_idx]
        n_samples = len(group)
        if n_samples <= batch_size:
            start = 0
            end = n_samples
        else:
            start = np.random.randint(0, n_samples - batch_size + 1)
            end = start + batch_size
        chunk = group[start:end]
        aux_chunk = None
        if aux_arrays is not None:
            aux_chunk = aux_arrays[group_idx][start:end]
        return (chunk, aux_chunk, ordered)

    def _forward_group_train(self, group, batch_size=128, ordered=True):
        group_tensor = torch.tensor(group, dtype=torch.float32, device=self.device)
        z_chunks = []
        phys_chunks = []
        for start in range(0, len(group_tensor), batch_size):
            chunk = group_tensor[start:start + batch_size]
            z_chunk, phys_chunk, _ = self.feature_extractor(chunk, return_state=True)
            z_chunks.append(z_chunk)
            if phys_chunk is not None:
                phys_chunks.append(phys_chunk)
        z_all = torch.cat(z_chunks, dim=0)
        z_align = self._build_adaptation_features(z_all, ordered=ordered)
        phys_all = torch.cat(phys_chunks, dim=0) if phys_chunks else None
        return (z_all, z_align, phys_all)

    def _build_state_matrix(self, Z):
        input_dim_h = self.weights_enhancement.shape[0]
        H = self._activation(np.matmul(Z[:, :input_dim_h], self.weights_enhancement) + self.bias_enhancement)
        A = np.hstack([Z, H])
        return (H, A)

    def _build_mapping_features(self, Z_deep):
        if self.use_tcbls_mapping:
            return self._temporal_cascade_mapping(Z_deep)
        return Z_deep

    def _predict_raw_outputs(self, X, enable_dropout=False):
        Z_deep = self._get_deep_features(X, enable_dropout=enable_dropout)
        Z = self._build_mapping_features(Z_deep)
        H = self._activation(np.matmul(Z, self.weights_enhancement) + self.bias_enhancement)
        A = np.hstack([Z, H])
        return np.matmul(A, self.output_weights)

    def _refit_output_calibrator(self):
        if not self.use_output_calibration or self.output_weights is None:
            self.output_calibrator = None
            return
        X_mem = self.same_condition_anchor_X
        Y_mem = self.same_condition_anchor_Y
        if X_mem is None or Y_mem is None or len(X_mem) < self.calibration_min_samples:
            self.output_calibrator = None
            return
        raw_preds = self._predict_raw_outputs(X_mem).reshape(-1)
        targets = self._ensure_column_targets(Y_mem).reshape(-1)
        n = min(len(raw_preds), len(targets))
        if n < self.calibration_min_samples:
            self.output_calibrator = None
            return
        raw_preds = raw_preds[:n]
        targets = targets[:n]
        valid = np.isfinite(raw_preds) & np.isfinite(targets)
        raw_preds = raw_preds[valid]
        targets = targets[valid]
        if len(raw_preds) < self.calibration_min_samples or len(np.unique(np.round(raw_preds, 5))) < 8:
            self.output_calibrator = None
            return
        if len(raw_preds) > self.calibration_max_points:
            idx = np.linspace(0, len(raw_preds) - 1, self.calibration_max_points, dtype=int)
            raw_preds = raw_preds[idx]
            targets = targets[idx]
        sample_weights = self._compute_sample_weights(targets.reshape(-1, 1)).reshape(-1).astype(np.float32)
        sample_weights *= 1.0 + 0.2 * (targets <= 0.3).astype(np.float32)
        calibrator = IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0.0, y_max=1.0)
        calibrator.fit(raw_preds, targets, sample_weight=sample_weights)
        self.output_calibrator = calibrator

    def _estimate_same_condition_drift(self, Z_new, A_new, Y_new):
        Y_arr = self._ensure_column_targets(Y_new)
        pred_before = np.matmul(A_new, self.output_weights)
        residuals = np.abs(pred_before - Y_arr).reshape(-1)
        residual_score = float(np.tanh(np.mean(residuals) / 0.12))
        _, order_score = self._sequence_violation_profile(pred_before, Y_arr)
        feature_shift_score = 0.0
        if self.Z_latest is not None and len(self.Z_latest) > 0:
            ref_len = min(len(self.Z_latest), max(128, len(Z_new) * 4))
            ref_Z = self.Z_latest[-ref_len:]
            ref_std = np.std(ref_Z, axis=0) + 1e-06
            mean_gap = np.mean(np.abs(np.mean(Z_new, axis=0) - np.mean(ref_Z, axis=0)) / ref_std)
            ref_disp = np.mean(np.std(ref_Z, axis=0)) + 1e-06
            disp_gap = np.abs(np.mean(np.std(Z_new, axis=0)) - np.mean(np.std(ref_Z, axis=0))) / ref_disp
            feature_shift_score = float(np.tanh(0.2 * mean_gap + 0.8 * disp_gap))
        head_disagreement_score = 0.0
        if self.temp_head is not None and Z_new.shape[1] >= self.base_feature_dim:
            with torch.no_grad():
                z_tensor = torch.tensor(Z_new[:, :self.base_feature_dim], dtype=torch.float32, device=self.device)
                temp_pred = self.temp_head(z_tensor).clamp(0.0, 1.0).cpu().numpy()
            head_disagreement_score = float(np.tanh(np.mean(np.abs(temp_pred - pred_before)) / 0.12))
        history_score = 0.0
        if self.light_history_dim > 0 and len(Z_new) > 1:
            history_feats = self._compute_light_history_features(Z_new, ordered=True)
            history_jump = np.mean(np.abs(history_feats[:, 2:4])) / max(self.history_scale, 1e-06)
            history_score = float(np.tanh(history_jump / 1.5))
        y_flat = Y_arr.reshape(-1)
        late_ratio = float(np.mean(y_flat <= 0.3))
        transition_ratio = float(np.mean(y_flat <= 0.7))
        drift_score = np.clip(0.4 * residual_score + 0.18 * feature_shift_score + 0.17 * head_disagreement_score + 0.1 * history_score + 0.1 * order_score + 0.05 * late_ratio, 0.0, 1.5)
        data_weight = np.clip(0.9 + self.same_condition_drift_gain * drift_score + 0.12 * transition_ratio + 0.18 * late_ratio, 0.85, 1.85)
        replay_weight = np.clip(self.replay_update_scale * (1.15 - 0.5 * drift_score + 0.12 * (1.0 - late_ratio)), 0.05, max(self.replay_update_scale * 1.35, 0.08))
        return {'pred_before': pred_before, 'residuals': residuals, 'drift_score': float(drift_score), 'data_weight': float(data_weight), 'replay_weight': float(replay_weight), 'late_ratio': late_ratio, 'transition_ratio': transition_ratio, 'feature_shift_score': feature_shift_score, 'head_disagreement_score': head_disagreement_score, 'history_score': history_score, 'order_score': order_score}

    def _build_same_condition_update_weights(self, Y_batch, residuals=None, pred_before=None, base_scale=1.0):
        Y_arr = self._ensure_column_targets(Y_batch)
        base_weights = self._compute_sample_weights(Y_arr).astype(np.float32).reshape(-1)
        if residuals is None or len(residuals) == 0:
            hard_factor = np.ones_like(base_weights)
        else:
            residuals = np.asarray(residuals, dtype=np.float32).reshape(-1)
            residual_ratio = residuals / (np.mean(residuals) + 1e-06)
            hard_factor = 1.0 + self.same_condition_hard_bias * np.clip(residual_ratio - 0.7, 0.0, 2.5)
        y_flat = Y_arr.reshape(-1)
        transition_bonus = ((y_flat <= 0.55) & (y_flat > 0.2)).astype(np.float32)
        late_bonus = (y_flat <= 0.3).astype(np.float32)
        order_factor, _ = self._sequence_violation_profile(pred_before, Y_arr) if pred_before is not None else (np.ones_like(base_weights), 0.0)
        weights = base_weights * hard_factor
        weights *= 1.0 + 0.12 * transition_bonus + 0.15 * late_bonus
        weights *= order_factor
        weights *= float(base_scale)
        return np.clip(weights, 0.75, 4.0)

    def _sample_same_condition_replay(self, batch_size, Z_query=None, Y_query=None):
        if batch_size <= 0:
            return (None, None)
        X_mem = self.same_condition_anchor_X
        Y_mem = self.same_condition_anchor_Y
        if X_mem is None or Y_mem is None or len(X_mem) == 0:
            X_mem, Y_mem = self.replay_buffer.get_data()
        if X_mem is None or Y_mem is None or len(X_mem) == 0:
            return (None, None)
        X_mem = np.asarray(X_mem, dtype=np.float32)
        Y_mem = self._ensure_column_targets(Y_mem)
        n_samples = len(X_mem)
        if n_samples <= batch_size:
            return (X_mem, Y_mem)
        y_flat = Y_mem.reshape(-1)
        recency = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        stage_bonus = np.zeros(n_samples, dtype=np.float32)
        stage_bonus[(y_flat > 0.3) & (y_flat <= 0.7)] = 0.45
        stage_bonus[y_flat <= 0.3] = 1.0
        base_weights = 1.0 + self.same_condition_recent_bias * recency + self.same_condition_stage_bias * stage_bonus
        Y_query_arr = self._ensure_column_targets(Y_query)
        if Y_query_arr is not None and len(Y_query_arr) > 0:
            query_progress = float(np.mean(Y_query_arr))
            progress_match = np.exp(-np.abs(y_flat - query_progress) / 0.18).astype(np.float32)
            base_weights *= 1.0 + self.same_condition_similarity_bias * progress_match
        pool_size = min(n_samples, max(batch_size * 4, 128))
        base_probs = self._safe_sampling_probs(base_weights)
        candidate_idx = np.random.choice(n_samples, size=pool_size, replace=False, p=base_probs)
        final_weights = np.asarray(base_weights[candidate_idx], dtype=np.float32)
        X_pool = X_mem[candidate_idx]
        Y_pool = Y_mem[candidate_idx]
        if self.output_weights is not None and self.weights_enhancement is not None:
            Z_pool_raw = self._get_deep_features([(X_pool, False)])
            Z_pool = self._build_mapping_features(Z_pool_raw)
            _, A_pool = self._build_state_matrix(Z_pool)
            pred_pool = np.matmul(A_pool, self.output_weights)
            residual_pool = np.abs(pred_pool - Y_pool).reshape(-1)
            residual_ratio = residual_pool / (np.mean(residual_pool) + 1e-06)
            final_weights *= 1.0 + self.same_condition_hard_bias * np.clip(residual_ratio - 0.8, 0.0, 2.5)
            if Z_query is not None and len(Z_query) > 0:
                query_center = np.mean(Z_query, axis=0, keepdims=True)
                feature_dist = np.mean(np.abs(Z_pool - query_center), axis=1)
                feature_dist = feature_dist / (np.mean(feature_dist) + 1e-06)
                similarity = np.exp(-feature_dist).astype(np.float32)
                final_weights *= 1.0 + 0.2 * similarity
        select_probs = self._safe_sampling_probs(final_weights)
        selected_local = np.random.choice(pool_size, size=min(batch_size, pool_size), replace=False, p=select_probs)
        selected_idx = np.sort(candidate_idx[selected_local])
        return (X_mem[selected_idx], Y_mem[selected_idx])

    def _get_deep_features(self, X, batch_size=1024, enable_dropout=False):
        groups = self._normalize_window_groups(X, default_stateful=True)
        if not groups:
            return np.empty((0, self.base_feature_dim), dtype=np.float32)
        Z_list = []
        if enable_dropout:
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()
        for group, ordered in groups:
            group_z = []
            for start in range(0, len(group), batch_size):
                batch_X = group[start:start + batch_size]
                batch_tensor = torch.tensor(batch_X, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    batch_Z, _, _ = self.feature_extractor(batch_tensor, return_state=True)
                group_z.append(batch_Z.cpu().numpy())
            group_z = np.concatenate(group_z, axis=0)
            Z_list.append(group_z)
        Z = np.concatenate(Z_list, axis=0)
        if enable_dropout:
            self.feature_extractor.eval()
        return Z

    def adapt(self, X_source, Y_source, X_target=None, epochs=10, lr=0.001, batch_size=128, lambda_mmd=1.0, lambda_phy=0.1, replay_data=None, lambda_replay=0.5):
        source_groups = self._normalize_window_groups(X_source, default_stateful=True)
        if not source_groups:
            return
        source_targets = self._split_targets_by_groups(Y_source, source_groups)
        target_groups = self._normalize_window_groups(X_target, default_stateful=True)
        is_transductive = len(target_groups) > 0
        target_stages = [self._stage_labels_from_group_progress(len(group)) for group, _ in target_groups]
        incremental_mode = replay_data is not None
        teacher_extractor = None
        if incremental_mode:
            teacher_extractor = copy.deepcopy(self.feature_extractor).to(self.device)
            teacher_extractor.eval()
        self._set_adaptation_trainable(incremental_mode=incremental_mode)
        self.feature_extractor.train()
        self.temp_head.train()
        trainable_params = self._get_trainable_parameters()
        effective_lr = lr * (self.incremental_lr_scale if incremental_mode else 1.0)
        optimizer = optim.Adam(trainable_params, lr=effective_lr)
        align_weight = lambda_mmd * (0.35 if incremental_mode else 1.0)
        replay_groups = []
        replay_targets = []
        if replay_data is not None:
            X_r, Y_r = replay_data
            if X_r is not None and len(X_r) > 0:
                replay_groups = self._normalize_window_groups([(X_r, False)], default_stateful=False)
                replay_targets = self._split_targets_by_groups(Y_r, replay_groups)
        patience = 3
        best_loss = float('inf')
        patience_counter = 0
        min_epochs = min(3, epochs)
        for epoch in range(epochs):
            total_loss = 0
            n_source_steps = max(int(np.ceil(sum((len(group) for group, _ in source_groups)) / max(batch_size, 1))), 1)
            n_target_steps = int(np.ceil(sum((len(group) for group, _ in target_groups)) / max(batch_size, 1))) if is_transductive else 0
            n_replay_steps = int(np.ceil(sum((len(group) for group, _ in replay_groups)) / max(batch_size, 1))) if replay_groups else 0
            n_steps = max(n_source_steps, n_target_steps, n_replay_steps, 1)
            for step in range(n_steps):
                batch_xs, batch_ys_np, source_ordered = self._sample_group_chunk(source_groups, source_targets, batch_size=batch_size)
                batch_stage_s = self._stage_labels_from_targets(batch_ys_np)
                batch_ys = torch.tensor(batch_ys_np, dtype=torch.float32, device=self.device)
                optimizer.zero_grad()
                z_s_base, z_s_align, phys_s = self._forward_group_train(batch_xs, batch_size=batch_size, ordered=source_ordered)
                pred_s = self.temp_head(z_s_base)
                loss_task = self._weighted_mse_loss(pred_s, batch_ys, batch_ys_np)
                loss_trend = self._local_trend_loss(pred_s, ordered=source_ordered)
                loss_order = self._multi_scale_order_loss(pred_s, targets=batch_ys, ordered=source_ordered)
                loss = loss_task + self.lambda_trend * loss_trend + self.lambda_order * loss_order
                if phys_s is not None:
                    loss_phy_implicit = torch.mean(torch.relu(phys_s))
                else:
                    loss_phy_implicit = torch.tensor(0.0).to(self.device)
                if is_transductive:
                    batch_xt, target_stage_prior, target_ordered = self._sample_group_chunk(target_groups, target_stages, batch_size=batch_size)
                    z_t_base, z_t_align, phys_t = self._forward_group_train(batch_xt, batch_size=batch_size, ordered=target_ordered)
                    if incremental_mode:
                        if target_stage_prior is not None:
                            batch_stage_t = target_stage_prior
                        else:
                            batch_stage_t = self._predict_stage_from_features(z_t_base.detach())
                        stage_align = self._stage_aware_align_loss(z_s_align, z_t_align, batch_stage_s, batch_stage_t)
                        global_align = self._stat_align_loss(z_s_base, z_t_base)
                        loss_align = self.lambda_stage * stage_align + self.lambda_global * global_align
                    else:
                        loss_align = self._stat_align_loss(z_s_base, z_t_base)
                    if phys_t is not None:
                        loss_phy_implicit += torch.mean(torch.relu(phys_t))
                    loss += align_weight * loss_align
                loss += lambda_phy * loss_phy_implicit
                if replay_groups:
                    batch_xr, batch_yr_np, replay_ordered = self._sample_group_chunk(replay_groups, replay_targets, batch_size=batch_size)
                    batch_yr = torch.tensor(batch_yr_np, dtype=torch.float32, device=self.device)
                    z_r_base, _, _ = self._forward_group_train(batch_xr, batch_size=batch_size, ordered=replay_ordered)
                    pred_r = self.temp_head(z_r_base)
                    loss_replay = self._weighted_mse_loss(pred_r, batch_yr, batch_yr_np)
                    loss += lambda_replay * loss_replay
                    if teacher_extractor is not None:
                        teacher_extractor.eval()
                        with torch.no_grad():
                            teacher_z_r = []
                            replay_tensor = torch.tensor(batch_xr, dtype=torch.float32, device=self.device)
                            for start in range(0, len(replay_tensor), batch_size):
                                teacher_chunk = replay_tensor[start:start + batch_size]
                                z_teacher, _, _ = teacher_extractor(teacher_chunk, return_state=True)
                                teacher_z_r.append(z_teacher.cpu().numpy())
                            teacher_z_r = np.concatenate(teacher_z_r, axis=0)
                            teacher_full = torch.tensor(teacher_z_r, dtype=torch.float32, device=self.device)
                        loss_distill = F.mse_loss(z_r_base, teacher_full)
                        loss += self.lambda_distill * loss_distill
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / max(n_steps, 1)
            self.logger.info(f'  Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}')
            if epoch_loss < best_loss - 1e-05:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if epoch + 1 >= min_epochs and patience_counter >= patience:
                self.logger.info(f'  [Early Stop] Loss stabilized after {epoch + 1} epochs. (Best: {best_loss:.6f})')
                break
        self.feature_extractor.eval()

    def _compute_sample_weights(self, Y):
        Y_flat = Y.flatten()
        alpha = 1.5
        weights = 1.0 + alpha * (1.0 - Y_flat) ** 2
        return weights

    def fit(self, X, Y):
        start_time = time.time()
        self._refresh_same_condition_memory(X, Y)
        X_flat = self._flatten_window_groups(X)
        if X_flat is None:
            raise ValueError('fit() received no valid windows.')
        self.X_latest = X_flat.shape[1]
        self.replay_buffer.add(X_flat, Y)
        Z_deep = self._get_deep_features(X)
        Z = self._build_mapping_features(Z_deep)
        self.Z_latest = Z
        H = self._enhancement_nodes(Z)
        self.H_latest = H
        A = np.hstack([Z, H])
        self.A_latest = A
        if self.reg_param > 0:
            weights = self._compute_sample_weights(Y)
            sqrt_weights = np.sqrt(weights).reshape(-1, 1)
            A_weighted = A * sqrt_weights
            Y_weighted = Y * sqrt_weights
            ATA = np.matmul(A_weighted.T, A_weighted)
            self.inv_ATA = np.linalg.inv(ATA + self.reg_param * np.eye(ATA.shape[0]))
            self.output_weights = np.matmul(np.matmul(self.inv_ATA, A_weighted.T), Y_weighted)
        else:
            self.output_weights = np.matmul(np.linalg.pinv(A), Y)
        self.output_calibrator = None

    def predict(self, X):
        raw_pred = self._predict_raw_outputs(X)
        return self._apply_output_calibration(raw_pred)

    def predict_with_uncertainty(self, X, n_iter=20):
        preds = []
        for i in range(n_iter):
            pred = self._predict_raw_outputs(X, enable_dropout=True)
            pred = self._apply_output_calibration(pred)
            preds.append(pred)
        preds = np.array(preds)
        return (np.mean(preds, axis=0), np.std(preds, axis=0))

    def update_data(self, X_new, Y_new):
        start_time = time.time()
        Y_new = self._ensure_column_targets(Y_new)
        X_new_flat = self._flatten_window_groups(X_new)
        if X_new_flat is None:
            raise ValueError('update_data() received no valid windows.')
        Z_new = self._build_mapping_features(self._get_deep_features(X_new))
        H_new, A_new = self._build_state_matrix(Z_new)
        drift_info = self._estimate_same_condition_drift(Z_new, A_new, Y_new)
        X_replay = None
        Y_replay = None
        replay_batch_size = int(max(32, round(len(X_new_flat) * self.replay_sample_ratio)))
        X_replay, Y_replay = self._sample_same_condition_replay(replay_batch_size, Z_query=Z_new, Y_query=Y_new)
        new_sample_weights = self._build_same_condition_update_weights(Y_new, residuals=drift_info['residuals'], pred_before=drift_info['pred_before'], base_scale=drift_info['data_weight'])
        self._weighted_incremental_update(A_new, Y_new, sample_weights=new_sample_weights)
        self.replay_buffer.add(X_new_flat, Y_new)
        self._append_same_condition_memory(X_new_flat, Y_new)
        if X_replay is not None and len(X_replay) > 0:
            Z_replay = self._build_mapping_features(self._get_deep_features([(X_replay, False)]))
            _, A_replay = self._build_state_matrix(Z_replay)
            replay_pred_before = np.matmul(A_replay, self.output_weights)
            replay_residuals = np.abs(replay_pred_before - Y_replay).reshape(-1)
            replay_sample_weights = self._build_same_condition_update_weights(Y_replay, residuals=replay_residuals, pred_before=replay_pred_before, base_scale=drift_info['replay_weight'])
            y_query_center = float(np.mean(Y_new))
            y_replay_flat = np.asarray(Y_replay).reshape(-1)
            progress_match = np.exp(-np.abs(y_replay_flat - y_query_center) / 0.2).astype(np.float32)
            replay_sample_weights = np.clip(replay_sample_weights * (1.0 + 0.15 * progress_match), 0.05, 2.5)
            self._weighted_incremental_update(A_replay, Y_replay, sample_weights=replay_sample_weights)
        if self.A_latest is not None:
            self.A_latest = np.vstack([self.A_latest, A_new])
            self.Z_latest = np.vstack([self.Z_latest, Z_new])
            self.H_latest = np.vstack([self.H_latest, H_new])
        self._refit_output_calibrator()
        end_time = time.time()
        self.logger.info('[Same-Condition Update] drift=%.4f | new_w=%.4f | replay_w=%.4f | late=%.2f | feat_shift=%.4f | order=%.4f | time=%.4fs', drift_info['drift_score'], drift_info['data_weight'], drift_info['replay_weight'], drift_info['late_ratio'], drift_info['feature_shift_score'], drift_info['order_score'], end_time - start_time)
