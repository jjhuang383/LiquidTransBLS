import numpy as np
import logging
import time

class TCBLS:
    """
    Temporal Cascade Broad Learning System (TCBLS) for RUL Prediction.
    """

    def __init__(self, n_mapping_groups=50, n_mapping_nodes=10, n_enhancement_nodes=3000, reg_param=1e-08):
        """
        Args:
            n_mapping_groups (n): Number of groups in the mapping layer. (Paper: 40-80)
            n_mapping_nodes (p): Number of nodes per mapping group.
            n_enhancement_nodes (m): Number of enhancement nodes. (Paper: 2000-6000)
            reg_param (lambda): Regularization parameter for Ridge Regression. (Paper: 1e-8)
        """
        self.n = n_mapping_groups
        self.p = n_mapping_nodes
        self.m = n_enhancement_nodes
        self.reg_param = reg_param
        self.scale_factor = 0.1
        self.weights_mapping = {}
        self.bias_mapping = {}
        self.weights_enhancement = None
        self.bias_enhancement = None
        self.output_weights = None
        self.inv_ATA = None
        self.logger = logging.getLogger('TCBLS_RUL.Core')
        self.Z_latest = None
        self.H_latest = None
        self.A_latest = None
        self.X_latest = None

    def _generate_random_weights(self, input_dim, output_dim):
        W = np.random.randn(input_dim, output_dim) * self.scale_factor
        b = np.random.randn(1, output_dim) * self.scale_factor
        return (W, b)

    def _linear_mapping(self, X, W, b):
        return np.matmul(X, W) + b

    def _activation(self, X):
        return np.where(X > 0, X, X * 0.01)

    def _temporal_cascade_mapping(self, X):
        """
        Generates Temporal Cascade Mapping Features (Z).
        """
        Z_list = []
        if 0 not in self.weights_mapping:
            self.weights_mapping[0], self.bias_mapping[0] = self._generate_random_weights(X.shape[1], self.p)
        Z1 = self._linear_mapping(X, self.weights_mapping[0], self.bias_mapping[0])
        Z1 = self._activation(Z1)
        Z_list.append(Z1)
        for k in range(1, self.n):
            if k not in self.weights_mapping:
                self.weights_mapping[k], self.bias_mapping[k] = self._generate_random_weights(Z_list[-1].shape[1], self.p)
            Zk = self._linear_mapping(Z_list[-1], self.weights_mapping[k], self.bias_mapping[k])
            Zk = self._activation(Zk)
            Z_list.append(Zk)
        Z_final = np.concatenate(Z_list, axis=1)
        return Z_final

    def _enhancement_nodes(self, Z):
        """
        Generates Enhancement Features (H).
        """
        if self.weights_enhancement is None:
            self.weights_enhancement, self.bias_enhancement = self._generate_random_weights(Z.shape[1], self.m)
        H = self._activation(np.matmul(Z, self.weights_enhancement) + self.bias_enhancement)
        return H

    def fit(self, X, Y):
        """
        Offline Training.
        """
        start_time = time.time()
        self.X_latest = X.shape[1]
        Z = self._temporal_cascade_mapping(X)
        self.Z_latest = Z
        H = self._enhancement_nodes(Z)
        self.H_latest = H
        A = np.hstack([Z, H])
        self.A_latest = A
        if self.reg_param > 0:
            ATA = np.matmul(A.T, A)
            n_samples, n_features = A.shape
            adaptive_reg = self.reg_param * max(1.0, n_features / max(n_samples, 1))
            self.inv_ATA = np.linalg.inv(ATA + adaptive_reg * np.eye(ATA.shape[0]))
            self.output_weights = np.matmul(np.matmul(self.inv_ATA, A.T), Y)
        else:
            self.output_weights = np.matmul(np.linalg.pinv(A), Y)
        end_time = time.time()
        self.logger.info(f'Offline training finished in {end_time - start_time:.4f}s')

    def predict(self, X):
        """
        Predicts RUL.
        """
        Z_list = []
        Z1 = self._linear_mapping(X, self.weights_mapping[0], self.bias_mapping[0])
        Z1 = self._activation(Z1)
        Z_list.append(Z1)
        for k in range(1, self.n):
            Zk = self._linear_mapping(Z_list[-1], self.weights_mapping[k], self.bias_mapping[k])
            Zk = self._activation(Zk)
            Z_list.append(Zk)
        Z = np.concatenate(Z_list, axis=1)
        H = self._activation(np.matmul(Z, self.weights_enhancement) + self.bias_enhancement)
        A = np.hstack([Z, H])
        return np.matmul(A, self.output_weights)

    def _update_weights_data_increment(self, A_new, Y_new):
        """
        Helper for incremental weight update (Data Increment).
        Uses Sherman-Morrison / Woodbury Identity.
        """
        P_old = self.inv_ATA
        A_new_T = A_new.T
        C = np.matmul(A_new, P_old)
        D = np.eye(A_new.shape[0]) + np.matmul(C, A_new_T) + 1e-06 * np.eye(A_new.shape[0])
        inv_D = np.linalg.inv(D)
        K = np.matmul(P_old, np.matmul(A_new_T, inv_D))
        self.inv_ATA = P_old - np.matmul(K, C)
        error = Y_new - np.matmul(A_new, self.output_weights)
        self.output_weights = self.output_weights + np.matmul(K, error)

    def update_data(self, X_new, Y_new):
        """
        Incremental Learning for New Input Data.
        """
        start_time = time.time()
        Z_list = []
        Z1 = self._linear_mapping(X_new, self.weights_mapping[0], self.bias_mapping[0])
        Z1 = self._activation(Z1)
        Z_list.append(Z1)
        for k in range(1, self.n):
            Zk = self._linear_mapping(Z_list[-1], self.weights_mapping[k], self.bias_mapping[k])
            Zk = self._activation(Zk)
            Z_list.append(Zk)
        Z_new = np.concatenate(Z_list, axis=1)
        input_dim_h = self.weights_enhancement.shape[0]
        H_new = self._activation(np.matmul(Z_new[:, :input_dim_h], self.weights_enhancement) + self.bias_enhancement)
        A_new = np.hstack([Z_new, H_new])
        if self.A_latest is not None:
            self.A_latest = np.vstack([self.A_latest, A_new])
            self.Z_latest = np.vstack([self.Z_latest, Z_new])
            self.H_latest = np.vstack([self.H_latest, H_new])
        self._update_weights_data_increment(A_new, Y_new)
        end_time = time.time()
        self.logger.info(f'Incremental update (data) finished in {end_time - start_time:.4f}s')
