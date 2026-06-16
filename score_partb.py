"""Score Part B output: compute total throughput T from (input + output)."""
import sys
import math

def lin2db(x):
    if x <= 1e-12: return -1000.0
    return 10.0 * math.log10(x)

CAP_TABLE = [(-1000, -10, 0), (-10, 0, 8), (0, 3, 24), (3, 10, 90),
             (10, 15, 120), (15, 20, 162), (20, 1e9, 222)]

def cap_lookup(fse):
    for lo, hi, val in CAP_TABLE:
        if lo < fse <= hi: return val
    return 0

def score_one(in_path, out_text):
    """Return throughput for one test case. out_text is the stdout of the solver."""
    with open(in_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    idx = 0
    P, N, K, T, beamMaxNum = map(int, lines[idx].split())
    idx += 1; M = int(lines[idx]); idx += 1
    MU = []
    for _ in range(M):
        parts = list(map(int, lines[idx].split()))
        MU.append(parts[1:]); idx += 1
    parts = list(map(int, lines[idx].split()))
    SU = set(parts[1:]); idx += 1
    CAP = {}
    for i in range(1, N + 1):
        CAP[i] = list(map(float, lines[idx].split())); idx += 1
    buffer = {}; SINR = {}
    for i in range(1, N + 1):
        parts = lines[idx].split()
        buffer[i] = int(parts[0]); SINR[i] = float(parts[1]); idx += 1
    RES_SUB = {}
    all_res = set()
    for t in range(1, T + 1):
        parts = list(map(int, lines[idx].split()))
        RES_SUB[t] = parts[1:]; idx += 1
        for r in RES_SUB[t]:
            all_res.add(r)

    out_lines = [l.strip() for l in out_text.strip().split('\n') if l.strip()]
    out_idx = 0

    # Read beam allocations
    beam_alloc = {}
    total_beams = 0
    for t in range(1, T + 1):
        parts = list(map(int, out_lines[out_idx].split()))
        out_idx += 1
        beam_alloc[t] = parts[1:]
        total_beams += parts[0]

    # Basic constraint: total beams <= beamMaxNum
    if total_beams > beamMaxNum:
        print(f"  VIOLATION: total_beams={total_beams} > beamMaxNum={beamMaxNum}")
        return 0

    # Basic constraint: each subband with allocated users must have at least 1 beam
    user_alloc = {}
    for i in range(1, N + 1):
        parts = list(map(int, out_lines[out_idx].split()))
        out_idx += 1
        if parts[0] == 0:
            user_alloc[i] = []
            continue
        cnt = parts[0]
        user_alloc[i] = []
        for j in range(cnt):
            sub = parts[1 + j * 2]
            rid = parts[2 + j * 2]
            user_alloc[i].append((sub, rid))

    # Check: subbands with allocated users must have beams
    used_subs = set()
    for i in range(1, N + 1):
        for sub, rid in user_alloc[i]:
            used_subs.add(sub)
    for t in used_subs:
        if not beam_alloc.get(t):
            return 0  # illegal

    # Check no duplicate resource per user
    for i in range(1, N + 1):
        seen = set()
        for sub, rid in user_alloc[i]:
            if rid in seen:
                return 0  # duplicate
            seen.add(rid)

    # Compute throughput
    res_users = {}
    for i in range(1, N + 1):
        for sub, rid in user_alloc[i]:
            key = (sub, rid)
            if key not in res_users:
                res_users[key] = set()
            res_users[key].add(i)

    total = 0
    for i in range(1, N + 1):
        if not user_alloc[i]:
            continue
        buf_cap = 0
        for sub, rid in user_alloc[i]:
            n = len(res_users.get((sub, rid), {i}))
            beams = beam_alloc.get(sub, [])
            cap_with = sum(CAP[i][b - 1] for b in beams)
            cap_all = sum(CAP[i])
            fse = SINR[i] + lin2db(1.0 / max(1, n)) + lin2db(cap_with) - lin2db(cap_all)
            buf_cap += cap_lookup(fse)
        total += min(buffer[i], buf_cap)

    return total


if __name__ == '__main__':
    base = r'E:\华为比赛\1780886490950118786\线上阶段数据集\调度开放示例'
    import subprocess, os

    exe = r'E:\华为比赛\main_fast.exe'
    for case in ['0', '1', '2']:
        in_path = os.path.join(base, f'{case}.in')
        result = subprocess.run([exe], input=open(in_path, 'rb').read(),
                               capture_output=True, timeout=5)
        out_text = result.stdout.decode()
        score = score_one(in_path, out_text)
        print(f"Case {case}.in: T = {score}")
