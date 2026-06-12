"""
main.py — 华为智联杯入口
- 有 stdin 输入: 通信资源联合分配（Part B）
- 无 stdin 输入: AI 话务预测推理（Part A），输出 results.csv
"""
import sys
import os


def run_scheduling():
    from scheduler import (parse_input, allocate_beams, greedy_allocate,
                           fast_improve, make_user_orders, compute_total_T,
                           format_output)

    data = sys.stdin.read().strip().split('\n')
    if not data or not data[0].strip():
        return

    P, N, K, T, beamMaxNum, M, MU, SU, CAP, buffer, SINR, RES_SUB = parse_input(data)
    beam_alloc = allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB)
    orders = make_user_orders(N, buffer, SINR, CAP, MU, SU)

    # Greedy on all, keep top 2
    top = []
    for order in orders:
        ua, resources, rtu, su = greedy_allocate(
            N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc, order)
        T_val = compute_total_T(ua, rtu, beam_alloc, CAP, SINR, buffer, N)
        top.append((T_val, ua, rtu, su))
        top.sort(key=lambda x: x[0], reverse=True)
        if len(top) > 2:
            top.pop()

    # Fast improve with swap on each top result
    best_T = -1
    best_ua = None
    for _, ua, rtu, su in top:
        fast_improve(ua, rtu, su, beam_alloc, CAP, SINR, buffer, RES_SUB, N, T, MU, SU,
                     max_iter=3000, sa_mode=False, enable_swap=True)
        T_val = compute_total_T(ua, rtu, beam_alloc, CAP, SINR, buffer, N)
        if T_val > best_T:
            best_T = T_val
            best_ua = ua

    print(format_output(beam_alloc, best_ua, N))


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


def _days_in_month(y, m):
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif m in (4, 6, 9, 11):
        return 30
    elif m == 2:
        return 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
    return 30  # fallback


def _next_hour(ts_str):
    date_part, time_part = ts_str.split()
    y, m, d = map(int, date_part.split('/'))
    h = int(time_part.split(':')[0])
    h += 1
    if h >= 24:
        h = 0
        d += 1
        if d > _days_in_month(y, m):
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
    return f'{y}/{m}/{d} {h}:00'


if __name__ == '__main__':
    if sys.stdin.isatty():
        run_prediction()
    else:
        run_scheduling()
