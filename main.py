"""
main.py — 华为智联杯入口
- 有 stdin 输入: 通信资源联合分配（Part B）
- 无 stdin 输入: AI 话务预测推理（Part A），输出 results.csv
"""
import sys
import os


def run_scheduling():
    import random
    random.seed(42)  # 可复现 + ε-Greedy 内部探索
    from scheduler import (parse_input, allocate_beams, greedy_allocate,
                           fast_improve, make_user_orders, compute_total_T,
                           format_output)

    data = sys.stdin.read().strip().split('\n')
    if not data or not data[0].strip():
        return

    P, N, K, T, beamMaxNum, M, MU, SU, CAP, buffer, SINR, RES_SUB = parse_input(data)
    beam_alloc = allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB)

    # Resource-driven greedy (single pass with RU sharing, no orders needed)
    ua, _, rtu, su = greedy_allocate(
        N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc)

    # Fast improve on the single result
    fast_improve(ua, rtu, su, beam_alloc, CAP, SINR, buffer, RES_SUB, N, T, MU, SU,
                 max_iter=3000, sa_mode=False, enable_swap=False)

    print(format_output(beam_alloc, ua, N))


def run_prediction():
    import csv
    import numpy as np
    from datetime import datetime, timedelta
    from collections import defaultdict

    base = os.path.join(os.path.dirname(__file__),
                        '1780886490950118786', '线上阶段数据集')
    train_path = os.path.join(base, 'AI数据集', 'train_data.csv')
    test_path = os.path.join(base, 'AI数据集', 'test_data.csv')

    TARGETS = ['小区上行平均激活用户数', '小区下行平均激活用户数',
               '下行平均使用的PRB个数', '上行平均使用的PRB个数']

    def to_float(s):
        s = s.strip()
        if s == '' or s.upper() == 'NIL':
            return None
        try:
            v = float(s)
            return v if v == v else None
        except ValueError:
            return None

    def median(vals):
        if not vals:
            return 0.0
        xs = sorted(vals)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    # ---- 1. Statistical Model Path ----
    print("Building global stats...")
    global_hour = defaultdict(list)
    with open(train_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            try:
                dt = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M")
            except ValueError:
                continue
            for i, tgt in enumerate(TARGETS):
                v = to_float(row[2 + i])
                if v is not None:
                    global_hour[(dt.hour, tgt)].append(v)
    global_hour = {k: median(vals) for k, vals in global_hour.items()}
    TRAIN_P99 = [7.89, 16.17, 116.31, 41.39]  # training data p99 global caps

    rows = []
    with open(test_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and len(row) >= 6:
                rows.append(row)

    GROUP = 336
    assert len(rows) % GROUP == 0
    N_GROUPS = len(rows) // GROUP

    # Build predictions with KNN anchor (finds similar days, can't extrapolate)
    print("Running KNN-anchored prediction...")
    final_pred = np.zeros((N_GROUPS, 24, 4), dtype=np.float64)
    meta = []  # (cell, last_dt) per group

    for g in range(N_GROUPS):
        block = rows[g * GROUP:(g + 1) * GROUP]
        cell = block[0][1]
        last_dt = datetime.strptime(block[-1][0].strip(), "%Y/%m/%d %H:%M")
        meta.append((cell, last_dt))

        hist = {}
        for r in block:
            t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M")
            for i, tgt in enumerate(TARGETS):
                v = to_float(r[2 + i])
                if v is not None:
                    hist[(t, tgt)] = v

        for h in range(24):
            target_time = last_dt + timedelta(hours=h + 1)
            # Build per-day features for this hour (shared across metrics)
            day_feats = []  # list of (dow, is_wknd, z_vals_for_4metrics)
            day_vals = [[] for _ in range(4)]  # per-metric values for 14 days
            for d in range(1, 15):
                dt = target_time - timedelta(days=d)
                dow = dt.weekday()
                is_wknd = 1.0 if dow >= 5 else 0.0
                dow_sin = np.sin(2 * np.pi * dow / 7.0)
                dow_cos = np.cos(2 * np.pi * dow / 7.0)
                vals = []
                for m in range(4):
                    v = hist.get((dt, TARGETS[m]))
                    day_vals[m].append(v)
                    vals.append(v if v is not None else np.nan)
                day_feats.append((dow, is_wknd, dow_sin, dow_cos, vals))

            # Target day features
            tdow = target_time.weekday()
            t_is_wknd = 1.0 if tdow >= 5 else 0.0
            t_dow_sin = np.sin(2 * np.pi * tdow / 7.0)
            t_dow_cos = np.cos(2 * np.pi * tdow / 7.0)

            for m in range(4):
                tgt = TARGETS[m]
                prev_day = hist.get((target_time - timedelta(days=1), tgt))
                prev_week = hist.get((target_time - timedelta(days=7), tgt))

                # Collect same-hour values + build KNN features
                same_hour = []
                hist_feats = []  # (n_valid, 4): [is_wknd, dow_sin, dow_cos, z_val]
                for d in range(1, 15):
                    v = day_vals[m][d - 1]
                    if v is not None:
                        same_hour.append(v)
                        df = day_feats[d - 1]
                        hist_feats.append([df[1], df[2], df[3], 0.0])  # z_val filled below

                n_sh = len(same_hour)
                if n_sh < 2:
                    anchor = same_hour[0] if n_sh == 1 else 0.0
                else:
                    sh_arr = np.array(same_hour, dtype=np.float64)
                    mu, sig = sh_arr.mean(), sh_arr.std()
                    if sig < 1e-6:
                        sig = 1e-6
                    # Fill z-score column
                    for i in range(n_sh):
                        hist_feats[i][3] = (sh_arr[i] - mu) / sig

                    # Target: yesterday's z-score captures recent trend
                    yv = prev_day if prev_day is not None else (same_hour[-1] if same_hour else mu)
                    t_z = (yv - mu) / sig
                    t_feat = np.array([t_is_wknd, t_dow_sin, t_dow_cos, t_z])
                    hf = np.array(hist_feats, dtype=np.float64)

                    # Normalize features (per-column std)
                    hf_scale = hf.std(axis=0) + 1e-6
                    dists = np.sum(((hf - t_feat) / hf_scale) ** 2, axis=1)

                    K = min(7, n_sh)
                    k_idx = np.argpartition(dists, K - 1)[:K]
                    # Interpolate between p25 and p35 for fine-grained control
                    knn_vals = np.sort(sh_arr[k_idx])
                    nk = len(knn_vals)
                    p25_val = knn_vals[max(0, int(nk * 0.25))]
                    p35_val = knn_vals[max(0, int(nk * 0.35))]
                    anchor = float(0.3 * p25_val + 0.7 * p35_val)

                # Cap: min(local_p70, train_p99)
                p70 = sorted(same_hour)[int(n_sh * 0.70)] if n_sh >= 4 else (anchor * 1.2 if anchor > 0 else 1.0)
                cap = min(p70, TRAIN_P99[m]) if anchor > 0 else TRAIN_P99[m]

                # Trend signal
                is_wknd = 1 if tdow >= 5 else 0
                is_peak = 1 if target_time.hour in (8, 9, 10, 17, 18, 19) else 0
                if is_wknd:
                    w_pd, w_pw = (0.55, 0.45) if is_peak else (0.45, 0.55)
                else:
                    w_pd, w_pw = (0.65, 0.35) if is_peak else (0.50, 0.50)

                pieces = []
                if prev_day is not None:
                    pieces.append((w_pd, prev_day))
                if prev_week is not None:
                    pieces.append((w_pw, prev_week))
                trend = anchor
                if pieces:
                    wsum = sum(w for w, _ in pieces)
                    trend = sum(w * v for w, v in pieces) / wsum

                pred = 0.80 * anchor + 0.20 * trend
                pred = max(0.0001, min(pred, cap))
                final_pred[g, h, m] = pred

    print(f"Final mean={final_pred.mean():.4f}")

    # ---- 4. Write results.csv ----
    output_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['时间', '小区名称'] + TARGETS)
        for g in range(N_GROUPS):
            cell, last_dt = meta[g]
            for h in range(24):
                dt = last_dt + timedelta(hours=h + 1)
                ts = "%d/%d/%d %d:00" % (dt.year, dt.month, dt.day, dt.hour)
                writer.writerow([ts, cell] + ["%.4f" % final_pred[g, h, m] for m in range(4)])

    print(f"Wrote {N_GROUPS * 24} rows to results.csv")


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
