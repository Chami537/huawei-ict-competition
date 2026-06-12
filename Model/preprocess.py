"""
数据预处理：加载、清洗、特征工程、滑动窗口构造
纯 Python csv + numpy 实现
"""
import csv
import math
import numpy as np


def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    return header, rows


def parse_datetime(ts):
    y, m, d = map(int, ts.split()[0].split('/'))
    h = int(ts.split()[1].split(':')[0])
    return y, m, d, h


def ts_to_date_str(ts):
    y, m, d, _ = parse_datetime(ts)
    return f'{y}{m:02d}{d:02d}'


def _days_in_month_static(y, m):
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif m in (4, 6, 9, 11):
        return 30
    elif m == 2:
        return 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
    return 30


def date_to_days(y, m, d):
    if m <= 2:
        y -= 1
        m += 12
    return 365 * y + y // 4 - y // 100 + y // 400 + (306 * (m + 1)) // 10 + d - 429


def load_traffic_data(path):
    header, rows = load_csv(path)
    data = {}
    for row in rows:
        ts = row[0].strip()
        cell = row[1].strip()
        metrics = []
        for v in row[2:6]:
            v = v.strip()
            metrics.append(None if v in ('NIL', '') else float(v))
        if cell not in data:
            data[cell] = []
        data[cell].append((ts, metrics))
    for cell in data:
        data[cell].sort(key=lambda x: x[0])
    return data


def load_weather(path):
    weather_types = ['晴', '雨', '雾']
    header, rows = load_csv(path)
    weather = {}
    for row in rows:
        date_str = row[0].strip()
        if not date_str:
            continue
        wt = row[1].strip()
        wt_onehot = [1.0 if wt == w else 0.0 for w in weather_types]
        numeric = []
        for v in row[2:]:
            try:
                numeric.append(float(v))
            except (ValueError, TypeError):
                numeric.append(0.0)
        weather[date_str] = np.array(wt_onehot + numeric, dtype=np.float32)
    return weather


def load_parameters(path):
    scenarios = [
        '高校', '城区道路', '风景区', '工业园区', '高层居民区', '低层居民区',
        '商业中心', '写字楼', '企事业单位', '星级酒店', '乡镇', '村庄',
        '高速公路', '中小学', '城中村', '集贸市场', '休闲娱乐场所',
        '党政军机关', '其他', '会展中心', '机场', '火车站', '郊区道路',
        '体育场馆', '公墓'
    ]
    header, rows = load_csv(path)
    params = {}
    xs, ys = [], []
    for row in rows:
        if not row[0].strip():
            continue
        cell = row[0].strip()
        azimuth = float(row[1])
        scenario = row[2].strip()
        x = float(row[3])
        y = float(row[4])
        xs.append(x)
        ys.append(y)
        params[cell] = {'azimuth': azimuth, 'scenario': scenario, 'x': x, 'y': y}
    x_mean, x_std = np.mean(xs), np.std(xs)
    y_mean, y_std = np.mean(ys), np.std(ys)

    features = {}
    for cell, p in params.items():
        scenario_oh = [1.0 if p['scenario'] == s else 0.0 for s in scenarios]
        azimuth_sin = math.sin(math.radians(p['azimuth']))
        azimuth_cos = math.cos(math.radians(p['azimuth']))
        x_norm = (p['x'] - x_mean) / (x_std + 1e-8)
        y_norm = (p['y'] - y_mean) / (y_std + 1e-8)
        features[cell] = np.array(
            scenario_oh + [azimuth_sin, azimuth_cos, x_norm, y_norm], dtype=np.float32)
    return features, scenarios


def interpolate_nils(data):
    for cell in data:
        records = data[cell]
        n = len(records)
        i = 0
        while i < n:
            if records[i][1][0] is None:
                start = i
                while i < n and records[i][1][0] is None:
                    i += 1
                end = i
                before = records[start - 1][1] if start > 0 else None
                after = records[end][1] if end < n else None
                for k in range(4):
                    if before is not None and after is not None:
                        bv, av = before[k], after[k]
                        if bv is not None and av is not None:
                            gap = end - start + 1
                            for j, pos in enumerate(range(start, end)):
                                records[pos][1][k] = bv + (j + 1) / gap * (av - bv)
                    elif before is not None:
                        for pos in range(start, end):
                            records[pos][1][k] = before[k] if before[k] is not None else 0.0
                    elif after is not None:
                        for pos in range(start, end):
                            records[pos][1][k] = after[k] if after[k] is not None else 0.0
                    else:
                        for pos in range(start, end):
                            records[pos][1][k] = 0.0
            else:
                i += 1


def extract_features(input_records, weather, cell_params, output_hours=24):
    """
    从输入记录中提取特征向量。
    只提取在训练集中有方差的特征（无 time_encoding/pred_weather）。
    返回特征向量 (numpy array)。
    """
    metrics_matrix = np.array([[r[1][k] if r[1][k] is not None else 0.0
                                for k in range(4)] for r in input_records], dtype=np.float32)

    # 1. Global stats
    global_mean = np.mean(metrics_matrix, axis=0)
    global_std = np.std(metrics_matrix, axis=0)
    global_min = np.min(metrics_matrix, axis=0)
    global_max = np.max(metrics_matrix, axis=0)
    global_p25 = np.percentile(metrics_matrix, 25, axis=0)
    global_p75 = np.percentile(metrics_matrix, 75, axis=0)

    # 2. Recent window stats
    recent_stats = []
    for window_size in [24, 48, 72, 168]:
        recent = metrics_matrix[-window_size:]
        recent_stats.append(np.mean(recent, axis=0))
        recent_stats.append(np.std(recent, axis=0))
    recent_stats = np.concatenate(recent_stats)

    # 3. Hour-of-day pattern (14 days → 24h × 4 metrics)
    hod_mean = np.zeros((24, 4), dtype=np.float32)
    hod_counts = np.zeros(24, dtype=np.float32)
    for r in input_records:
        _, _, _, h = parse_datetime(r[0])
        for k in range(4):
            v = r[1][k]
            if v is not None:
                hod_mean[h][k] += v
        hod_counts[h] += 1
    for h in range(24):
        if hod_counts[h] > 0:
            hod_mean[h] /= hod_counts[h]
    hod_flat = hod_mean.flatten()

    # 4. Day-of-week pattern
    dow_mean = np.zeros((7, 4), dtype=np.float32)
    dow_counts = np.zeros(7, dtype=np.float32)
    for r in input_records:
        y, m, d, _ = parse_datetime(r[0])
        days_since = date_to_days(y, m, d) - date_to_days(2024, 7, 20)
        dow = days_since % 7
        for k in range(4):
            v = r[1][k]
            if v is not None:
                dow_mean[dow][k] += v
        dow_counts[dow] += 1
    for d in range(7):
        if dow_counts[d] > 0:
            dow_mean[d] /= dow_counts[d]
    dow_flat = dow_mean.flatten()

    # 5. Trend slopes
    x_range = np.arange(len(metrics_matrix), dtype=np.float32)
    mid = (len(metrics_matrix) - 1) / 2.0
    trends = []
    for k in range(4):
        y_vals = metrics_matrix[:, k]
        denom = np.sum((x_range - mid) ** 2) + 1e-8
        slope = np.sum((x_range - mid) * (y_vals - global_mean[k])) / denom
        trends.append(slope)
    trends = np.array(trends, dtype=np.float32)

    # 6. Recent raw values (last 48 hours for more signal)
    recent_raw = metrics_matrix[-48:].flatten()

    # 7. Weather stats over input period
    weather_vecs = []
    for r in input_records:
        date_str = ts_to_date_str(r[0])
        w = weather.get(date_str)
        if w is not None:
            weather_vecs.append(w)
    if weather_vecs:
        avg_weather = np.mean(weather_vecs, axis=0)
    else:
        avg_weather = np.zeros(11, dtype=np.float32)

    # 8. Last 24h stats (mean + std per metric)
    last24 = metrics_matrix[-24:]
    last24_mean = np.mean(last24, axis=0)
    last24_std = np.std(last24, axis=0)

    # Combine base features
    X_base = np.concatenate([
        global_mean, global_std, global_min, global_max,   # 16
        global_p25, global_p75,                             # 8
        recent_stats,                                        # 32
        hod_flat,                                            # 96
        dow_flat,                                            # 28
        trends,                                              # 4
        recent_raw,                                          # 192 (48h×4)
        last24_mean, last24_std,                            # 8
        avg_weather,                                         # 11
        cell_params,                                         # 29
    ]).astype(np.float32)
    # Total base: 16+8+32+96+28+4+192+8+11+29 = 424 features

    # ── Prediction-time encoding ──
    # Compute first prediction hour from the last input record
    last_ts = input_records[-1][0]
    y, m, d, last_h = parse_datetime(last_ts)

    pred_h = last_h + 1
    pred_d = d
    pred_m = m
    pred_y = y
    if pred_h >= 24:
        pred_h = 0
        pred_d += 1
        dim = _days_in_month_static(pred_y, pred_m)
        if pred_d > dim:
            pred_d = 1
            pred_m += 1
            if pred_m > 12:
                pred_m = 1
                pred_y += 1

    # Hour-of-day sin/cos
    pred_hour_sin = math.sin(2 * math.pi * pred_h / 24.0)
    pred_hour_cos = math.cos(2 * math.pi * pred_h / 24.0)

    # Day-of-week sin/cos
    days_since_epoch = date_to_days(pred_y, pred_m, pred_d) - date_to_days(2024, 7, 20)
    pred_dow = days_since_epoch % 7
    pred_dow_sin = math.sin(2 * math.pi * pred_dow / 7.0)
    pred_dow_cos = math.cos(2 * math.pi * pred_dow / 7.0)

    time_enc = np.array([pred_hour_sin, pred_hour_cos, pred_dow_sin, pred_dow_cos], dtype=np.float32)

    # ── Prediction-day weather ──
    pred_date1 = f'{pred_y}{pred_m:02d}{pred_d:02d}'
    pred_w1 = weather.get(pred_date1)
    if pred_w1 is None:
        pred_w1 = np.zeros(11, dtype=np.float32)

    # If prediction starts after hour 0, it spans into the next calendar day
    if pred_h > 0:
        next_d = pred_d + 1
        next_m = pred_m
        next_y = pred_y
        dim = _days_in_month_static(next_y, next_m)
        if next_d > dim:
            next_d = 1
            next_m += 1
            if next_m > 12:
                next_m = 1
                next_y += 1
        pred_date2 = f'{next_y}{next_m:02d}{next_d:02d}'
        pred_w2 = weather.get(pred_date2)
        if pred_w2 is None:
            pred_w2 = np.zeros(11, dtype=np.float32)
    else:
        pred_w2 = pred_w1.copy()

    X = np.concatenate([X_base, time_enc, pred_w1, pred_w2]).astype(np.float32)
    # Total: 424 + 4(time_enc) + 11(day1_weather) + 11(day2_weather) = 450

    return X


def build_windows(data, weather, param_features, input_hours=336, output_hours=24):
    """训练集滑动窗口构造"""
    samples_x = []
    samples_y = []

    for cell, records in data.items():
        if len(records) < input_hours + output_hours:
            continue
        cell_params = param_features.get(cell)
        if cell_params is None:
            cell_params = np.zeros(29, dtype=np.float32)

        for start in range(0, len(records) - input_hours - output_hours + 1, 24):
            input_records = records[start:start + input_hours]
            output_records = records[start + input_hours:start + input_hours + output_hours]

            X = extract_features(input_records, weather, cell_params)

            Y = np.array([[r[1][k] if r[1][k] is not None else 0.0
                           for k in range(4)] for r in output_records],
                         dtype=np.float32).flatten()

            samples_x.append(X)
            samples_y.append(Y)

    return np.array(samples_x, dtype=np.float32), np.array(samples_y, dtype=np.float32)


def build_test_windows(rows, weather, param_features, input_hours=336):
    """测试集按固定 336 行分组构造窗口（不滑动）"""
    samples_x = []
    test_meta = []

    pos = 0
    while pos + input_hours <= len(rows):
        group = rows[pos:pos + input_hours]
        cell = group[0][1].strip()
        last_ts = group[-1][0]

        cell_params = param_features.get(cell)
        if cell_params is None:
            cell_params = np.zeros(29, dtype=np.float32)

        # Convert group to (ts, metrics) format for extract_features
        input_records = [(r[0], r[2]) for r in group]

        X = extract_features(input_records, weather, cell_params)

        samples_x.append(X)
        test_meta.append((cell, last_ts))
        pos += input_hours

    return np.array(samples_x, dtype=np.float32), test_meta


def load_test_rows(path):
    header, rows = load_csv(path)
    result = []
    for row in rows:
        ts = row[0].strip()
        cell = row[1].strip()
        metrics = []
        for v in row[2:6]:
            v = v.strip()
            metrics.append(None if v in ('NIL', '') else float(v))
        result.append((ts, cell, metrics))
    return result


def normalize_features(X_train, X_test=None):
    """Z-score normalize features"""
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    std = np.maximum(std, 0.01)
    if X_test is not None:
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        X_test_norm = np.clip(X_test_norm, -10, 10)
        return X_train_norm, X_test_norm, mean, std
    return (X_train - mean) / std, mean, std


def prepare_data(train_path, test_path, weather_path, param_path):
    """主入口"""
    print("Loading traffic data...")
    train_data = load_traffic_data(train_path)
    test_rows = load_test_rows(test_path)
    print(f"  Train: {len(train_data)} cells")
    print(f"  Test: {len(test_rows)} rows")

    print("Loading weather...")
    weather = load_weather(weather_path)
    print(f"  Weather: {len(weather)} days")

    print("Loading parameters...")
    param_features, scenarios = load_parameters(param_path)
    print(f"  Parameters: {len(param_features)} cells")

    print("Interpolating NIL values...")
    interpolate_nils(train_data)

    test_data_dict = {}
    for ts, cell, metrics in test_rows:
        if cell not in test_data_dict:
            test_data_dict[cell] = []
        test_data_dict[cell].append((ts, metrics))
    interpolate_nils(test_data_dict)
    test_rows_clean = []
    for ts, cell, _ in test_rows:
        for ts2, metrics2 in test_data_dict.get(cell, []):
            if ts2 == ts:
                test_rows_clean.append((ts, cell, metrics2))
                break

    print("Building training windows...")
    X_train, Y_train = build_windows(train_data, weather, param_features)
    print(f"  Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")

    print("Building test windows...")
    X_test, test_meta = build_test_windows(test_rows_clean, weather, param_features)
    print(f"  Test groups: {X_test.shape[0]}")

    print("Normalizing...")
    X_train_norm, X_test_norm, feat_mean, feat_std = normalize_features(X_train, X_test)

    Y_mean = np.mean(Y_train, axis=0, keepdims=True)
    Y_std = np.std(Y_train, axis=0, keepdims=True)
    Y_std = np.maximum(Y_std, 0.01)
    Y_train_norm = (Y_train - Y_mean) / Y_std

    print(f"Final: X_train={X_train_norm.shape}, Y_train={Y_train_norm.shape}, X_test={X_test_norm.shape}")

    norm_params = {
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'Y_mean': Y_mean,
        'Y_std': Y_std,
    }

    return X_train_norm, Y_train_norm, X_test_norm, norm_params, test_meta


if __name__ == '__main__':
    base = 'E:/华为比赛/1780886490950118786/线上阶段数据集/AI数据集'
    prepare_data(
        f'{base}/train_data.csv', f'{base}/test_data.csv',
        f'{base}/weather.csv', f'{base}/parameter.csv',
    )
