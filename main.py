"""
main.py — 华为智联杯入口
- 有 stdin 输入: 通信资源联合分配（Part B）
- 无 stdin 输入: AI 话务预测推理（Part A），输出 results.csv
"""
import sys
import os


def run_scheduling():
    from scheduler import parse_input, allocate_beams, greedy_allocate
    from scheduler import local_improve, format_output

    data = sys.stdin.read().strip().split('\n')
    if not data or not data[0].strip():
        return

    P, N, K, T, beamMaxNum, M, MU, SU, CAP, buffer, SINR, RES_SUB = parse_input(data)
    beam_alloc = allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB)
    user_alloc = greedy_allocate(N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc)
    user_alloc = local_improve(user_alloc, beam_alloc, CAP, SINR, buffer, RES_SUB,
                               N, P, T, MU, SU)
    print(format_output(beam_alloc, user_alloc, T, N))


def run_prediction():
    import csv
    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Model'))
    from model import MLP
    from preprocess import (
        load_weather, load_parameters, load_test_rows,
        build_test_windows, interpolate_nils, parse_datetime
    )

    base = os.path.join(os.path.dirname(__file__),
                        '1780886490950118786', '线上阶段数据集')
    model_dir = os.path.join(os.path.dirname(__file__), 'Model')

    test_rows = load_test_rows(os.path.join(base, 'AI数据集', 'test_data.csv'))
    weather = load_weather(os.path.join(base, 'AI数据集', 'weather.csv'))
    param_features, _ = load_parameters(os.path.join(base, 'AI数据集', 'parameter.csv'))

    test_data_dict = {}
    for ts, cell, metrics in test_rows:
        if cell not in test_data_dict:
            test_data_dict[cell] = []
        test_data_dict[cell].append((ts, metrics))
    interpolate_nils(test_data_dict)
    test_rows_clean = []
    for ts, cell, _ in test_rows:
        for ts2, metrics in test_data_dict.get(cell, []):
            if ts2 == ts:
                test_rows_clean.append((ts, cell, metrics))
                break

    X_test, test_meta = build_test_windows(test_rows_clean, weather, param_features)

    norm_data = np.load(os.path.join(model_dir, 'norm_params.npz'))
    X_test_norm = (X_test - norm_data['feat_mean']) / (norm_data['feat_std'] + 1e-8)
    X_test_norm = np.clip(X_test_norm, -10, 10)

    model = MLP(X_test_norm.shape[1], [256, 128, 64], 96)
    model.load(os.path.join(model_dir, 'best_model.npz'))

    Y_pred_norm = model.predict(X_test_norm)
    Y_pred = np.maximum(Y_pred_norm * norm_data['Y_std'] + norm_data['Y_mean'], 0)

    output_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['时间', '小区名称', '小区上行平均激活用户数',
                         '小区下行平均激活用户数', '下行平均使用的PRB个数',
                         '上行平均使用的PRB个数'])

        for idx, (cell, last_ts) in enumerate(test_meta):
            pred = Y_pred[idx].reshape(24, 4)
            ts = last_ts
            for h in range(24):
                if h == 0:
                    ts = _next_hour(last_ts)
                else:
                    ts = _next_hour(ts)
                writer.writerow([
                    ts, cell,
                    f'{pred[h, 0]:.4f}', f'{pred[h, 1]:.4f}',
                    f'{pred[h, 2]:.4f}', f'{pred[h, 3]:.4f}',
                ])

    print(f"Wrote {len(test_meta) * 24} rows to results.csv")


def _next_hour(ts_str):
    y, m, d, h = ts_str.split()[0].split('/') + [ts_str.split()[1].split(':')[0]]
    y, m, d, h = int(y), int(m), int(d), int(h)
    h += 1
    if h >= 24:
        h = 0
        d += 1
        if m == 7 and d > 31:
            d, m = 1, 8
        elif m == 8 and d > 31:
            d, m = 1, 9
    return f'{y}/{m}/{d} {h}:00'


if __name__ == '__main__':
    if sys.stdin.isatty():
        run_prediction()
    else:
        run_scheduling()
