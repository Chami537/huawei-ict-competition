"""
纯 numpy MLP 模型 + MSE 损失 + SGD momentum + 学习率衰减
"""
import numpy as np
import math


class MLP:
    """3 层全连接神经网络，纯 numpy 实现"""

    def __init__(self, input_dim, hidden_dims, output_dim, lr=0.001, momentum=0.9, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.momentum = momentum

        dims = [input_dim] + hidden_dims + [output_dim]
        self.weights = []
        self.biases = []
        self.v_w = []
        self.v_b = []

        for i in range(len(dims) - 1):
            scale = math.sqrt(2.0 / dims[i])
            self.weights.append(np.random.randn(dims[i], dims[i + 1]).astype(np.float32) * scale)
            self.biases.append(np.zeros((1, dims[i + 1]), dtype=np.float32))
            self.v_w.append(np.zeros((dims[i], dims[i + 1]), dtype=np.float32))
            self.v_b.append(np.zeros((1, dims[i + 1]), dtype=np.float32))

    def _relu(self, x):
        return np.maximum(x, 0)

    def _relu_deriv(self, x):
        return (x > 0).astype(np.float32)

    def forward(self, X):
        activations = [X]
        pre_acts = []

        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_acts.append(z)
            activations.append(self._relu(z))

        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_acts.append(z)
        activations.append(z)

        return activations, pre_acts

    def backward(self, X, Y, activations, pre_acts):
        batch_size = X.shape[0]
        grad_w = []
        grad_b = []

        # MSE gradient: 2*(pred - Y) / batch_size
        delta = 2.0 * (activations[-1] - Y) / batch_size

        for i in range(len(self.weights) - 1, -1, -1):
            grad_w.insert(0, np.dot(activations[i].T, delta))
            grad_b.insert(0, np.sum(delta, axis=0, keepdims=True))

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._relu_deriv(pre_acts[i - 1])

        return grad_w, grad_b

    def update(self, grad_w, grad_b, clip_norm=1.0):
        for i in range(len(self.weights)):
            gw_norm = np.linalg.norm(grad_w[i])
            if gw_norm > clip_norm:
                grad_w[i] = grad_w[i] * (clip_norm / gw_norm)
            gb_norm = np.linalg.norm(grad_b[i])
            if gb_norm > clip_norm:
                grad_b[i] = grad_b[i] * (clip_norm / gb_norm)

            self.v_w[i] = self.momentum * self.v_w[i] - self.lr * grad_w[i]
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * grad_b[i]
            self.weights[i] += self.v_w[i]
            self.biases[i] += self.v_b[i]

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def compute_mse(self, Y_pred, Y_true):
        diff = Y_pred - Y_true
        return np.mean(diff * diff)

    def save(self, path):
        np.savez(path, *self.weights, *self.biases)

    def load(self, path):
        data = np.load(path)
        n = len(self.weights)
        self.weights = [data[f'arr_{i}'] for i in range(n)]
        self.biases = [data[f'arr_{i + n}'] for i in range(n)]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]


def compute_mape_auc(Y_pred, Y_true):
    """
    计算 MAPE-AUC（评分指标）。
    Y_pred, Y_true: 原始尺度 (未归一化), shape (N, 96)
    返回: mape_auc, 各阈值比例
    """
    Y_pred = np.maximum(Y_pred, 1e-8)
    Y_true = np.maximum(Y_true, 1e-8)
    abs_pct = np.abs(Y_pred - Y_true) / Y_true
    # Reshape to (N, 24, 4): 24 hours × 4 metrics
    abs_pct = abs_pct.reshape(-1, 24, 4)
    # MAPE per sample-hour: average over 4 metrics
    mape_per_hour = np.mean(abs_pct, axis=2).flatten()

    ratios = []
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        ratios.append(np.mean(mape_per_hour < threshold))
    mape_auc = 0.25 * sum(ratios)
    return mape_auc, ratios


def train_model(X_train, Y_train, X_val=None, Y_val=None,
                Y_train_raw=None, Y_val_raw=None,
                Y_mean=None, Y_std=None,
                hidden_dims=None, epochs=200, batch_size=128,
                lr=0.001, momentum=0.9, patience=20, lr_decay=0.5,
                lr_decay_epochs=50, model_path=None, seed=42):
    """训练 MLP 模型"""
    if hidden_dims is None:
        hidden_dims = [512, 256]

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    print(f"Model: input={input_dim}, hidden={hidden_dims}, output={output_dim}")
    print(f"Training: samples={X_train.shape[0]}, epochs={epochs}, batch_size={batch_size}")

    model = MLP(input_dim, hidden_dims, output_dim, lr=lr, momentum=momentum, seed=seed)

    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    best_val_mape_auc = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        epoch_mse = 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            activations, pre_acts = model.forward(X_batch)
            grad_w, grad_b = model.backward(X_batch, Y_batch, activations, pre_acts)
            model.update(grad_w, grad_b)

            epoch_mse += model.compute_mse(activations[-1], Y_batch) * (end - start)

        epoch_mse /= n_samples

        # Learning rate decay
        if (epoch + 1) % lr_decay_epochs == 0:
            model.lr *= lr_decay
            print(f"  LR decay to {model.lr:.6f}")

        # Validation
        if X_val is not None and Y_val is not None:
            val_pred = model.predict(X_val)
            val_mse = model.compute_mse(val_pred, Y_val)

            # Compute MAPE-AUC on raw (unnormalized) values if available
            if Y_val_raw is not None and Y_mean is not None and Y_std is not None:
                val_pred_denorm = val_pred * Y_std + Y_mean
                val_mape_auc, ratios = compute_mape_auc(val_pred_denorm, Y_val_raw)
            else:
                val_mape_auc = -val_mse  # fallback: use negative MSE
                ratios = []

            if val_mape_auc > best_val_mape_auc:
                best_val_mape_auc = val_mape_auc
                best_epoch = epoch
                patience_counter = 0
                if model_path:
                    model.save(model_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                ratio_str = f" R=[{ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f},{ratios[3]:.3f}]" if ratios else ""
                print(f"Epoch {epoch:4d}: train_mse={epoch_mse:.4f}, val_mse={val_mse:.4f}, "
                      f"val_mape_auc={val_mape_auc:.4f}{ratio_str}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
                break
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d}: train_mse={epoch_mse:.4f}")

    if model_path and X_val is not None:
        model.load(model_path)
        print(f"Loaded best model from epoch {best_epoch} (val_mape_auc={best_val_mape_auc:.4f})")

    return model


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import prepare_data

    base = 'E:/华为比赛/1780886490950118786/线上阶段数据集/AI数据集'
    X_train, Y_train, X_test, norm_params, test_meta = prepare_data(
        f'{base}/train_data.csv', f'{base}/test_data.csv',
        f'{base}/weather.csv', f'{base}/parameter.csv',
    )

    np.random.seed(42)
    indices = np.random.permutation(len(X_train))
    split = int(len(indices) * 0.9)
    train_idx = indices[:split]
    val_idx = indices[split:]

    model = train_model(
        X_train[train_idx], Y_train[train_idx],
        X_val=X_train[val_idx], Y_val=Y_train[val_idx],
        epochs=200, batch_size=128, lr=0.001, patience=25,
        model_path='Model/best_model.npz',
    )
