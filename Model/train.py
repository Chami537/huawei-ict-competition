"""
训练脚本：数据预处理 + 训练 + 保存模型和归一化参数
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import prepare_data
from model import train_model, compute_mape_auc


def main():
    base = 'E:/华为比赛/1780886490950118786/线上阶段数据集/AI数据集'
    model_dir = os.path.dirname(__file__)

    X_train, Y_train_norm, X_test, norm_params, test_meta = prepare_data(
        f'{base}/train_data.csv',
        f'{base}/test_data.csv',
        f'{base}/weather.csv',
        f'{base}/parameter.csv',
    )

    # Reconstruct raw Y for MAPE computation
    Y_train_raw = Y_train_norm * norm_params['Y_std'] + norm_params['Y_mean']

    # Save normalization params
    np.savez(
        os.path.join(model_dir, 'norm_params.npz'),
        feat_mean=norm_params['feat_mean'],
        feat_std=norm_params['feat_std'],
        Y_mean=norm_params['Y_mean'],
        Y_std=norm_params['Y_std'],
    )

    # Train/val split: random 90/10
    np.random.seed(42)
    indices = np.random.permutation(len(X_train))
    split = int(len(indices) * 0.9)
    train_idx = indices[:split]
    val_idx = indices[split:]

    input_dim = X_train.shape[1]
    print(f"\n=== Training ({split} train, {len(indices) - split} val, {input_dim} features) ===")
    model = train_model(
        X_train[train_idx], Y_train_norm[train_idx],
        X_val=X_train[val_idx], Y_val=Y_train_norm[val_idx],
        Y_val_raw=Y_train_raw[val_idx],
        Y_mean=norm_params['Y_mean'],
        Y_std=norm_params['Y_std'],
        hidden_dims=[512, 256],
        epochs=300,
        batch_size=128,
        lr=0.001,
        momentum=0.9,
        patience=30,
        lr_decay=0.5,
        lr_decay_epochs=60,
        model_path=os.path.join(model_dir, 'best_model.npz'),
        seed=42,
    )

    print("\n=== Final Evaluation ===")
    val_pred_norm = model.predict(X_train[val_idx])
    val_pred_raw = val_pred_norm * norm_params['Y_std'] + norm_params['Y_mean']
    mape_auc, ratios = compute_mape_auc(val_pred_raw, Y_train_raw[val_idx])

    print(f"  R_0.2: {ratios[0]:.4f}")
    print(f"  R_0.3: {ratios[1]:.4f}")
    print(f"  R_0.4: {ratios[2]:.4f}")
    print(f"  R_0.5: {ratios[3]:.4f}")
    print(f"  MAPE_AUC: {mape_auc:.4f}")
    print(f"  Estimated Score: {max(100, 5000 * mape_auc):.0f} / 5000")


if __name__ == '__main__':
    main()
