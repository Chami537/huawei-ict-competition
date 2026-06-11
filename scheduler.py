"""
通信资源联合分配调度器 — 优化版（<200ms）
"""
import sys
import math


def lin2db(x):
    if x <= 1e-12:
        return -1000.0
    return 10.0 * math.log10(x)


CAP_TABLE = [(-1000, -10, 0), (-10, 0, 8), (0, 3, 24), (3, 10, 90),
             (10, 15, 120), (15, 20, 162), (20, 1e9, 222)]


def cap_lookup(fse):
    for lo, hi, val in CAP_TABLE:
        if lo < fse <= hi:
            return val
    return 0


def parse_input(lines):
    idx = 0
    P, N, K, T, beamMaxNum = map(int, lines[idx].strip().split())
    idx += 1
    M = int(lines[idx].strip())
    idx += 1
    MU = []
    for _ in range(M):
        parts = list(map(int, lines[idx].strip().split()))
        MU.append(parts[1:])
        idx += 1
    parts = list(map(int, lines[idx].strip().split()))
    SU = parts[1:]
    idx += 1
    CAP = {}
    for i in range(1, N + 1):
        CAP[i] = list(map(float, lines[idx].strip().split()))
        idx += 1
    buffer = {}
    SINR = {}
    for i in range(1, N + 1):
        parts = lines[idx].strip().split()
        buffer[i] = int(parts[0])
        SINR[i] = float(parts[1])
        idx += 1
    RES_SUB = {}
    for t in range(1, T + 1):
        parts = list(map(int, lines[idx].strip().split()))
        RES_SUB[t] = parts[1:]
        idx += 1
    return P, N, K, T, beamMaxNum, M, MU, SU, CAP, buffer, SINR, RES_SUB


def allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB):
    beam_alloc = {t: [] for t in range(1, T + 1)}
    res_counts = {t: len(RES_SUB[t]) for t in range(1, T + 1)}
    beam_scores = []
    for b in range(P):
        beam_scores.append((sum(CAP[i][b] for i in range(1, N + 1)), b + 1))
    beam_scores.sort(reverse=True)
    allocated = 0
    for t in range(1, T + 1):
        if allocated < beamMaxNum:
            beam_alloc[t].append(beam_scores[allocated % len(beam_scores)][1])
            allocated += 1
    while allocated < beamMaxNum:
        best_t = max(range(1, T + 1), key=lambda t: (res_counts[t], -len(beam_alloc[t])))
        if len(beam_alloc[best_t]) >= P:
            break
        used = set(beam_alloc[best_t])
        for _, beam_id in beam_scores:
            if beam_id not in used:
                beam_alloc[best_t].append(beam_id)
                allocated += 1
                break
        else:
            break
    return beam_alloc


def compute_fse(user, n_users, sub_band, beam_alloc, CAP, SINR):
    """与旧代码完全一致的 fse 计算"""
    sinr = SINR.get(user, 0)
    share_penalty = lin2db(1.0 / max(1, n_users))
    beams = beam_alloc.get(sub_band, [])
    cap_with = sum(CAP[user][b - 1] for b in beams)
    cap_all = sum(CAP[user])
    beam_gain = lin2db(cap_with) - lin2db(cap_all)
    return sinr + share_penalty + beam_gain


def greedy_allocate(N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc):
    """贪心分配，与旧代码一致"""
    mu_to_group = {}
    for g_idx, group in enumerate(MU):
        for u in group:
            mu_to_group[u] = g_idx
    su_set = set(SU)

    resources = {}
    for t in range(1, T + 1):
        for rid in RES_SUB[t]:
            resources[(t, rid)] = [None, set()]  # [mu_group, users_set]

    user_priority = []
    for u in range(1, N + 1):
        if buffer[u] <= 0:
            continue
        tc = sum(CAP[u])
        sinr_lin = 10 ** (max(-10, min(30, SINR[u])) / 10)
        user_priority.append((buffer[u] * tc * sinr_lin, u))
    user_priority.sort(reverse=True)

    assignments = {u: [] for u in range(1, N + 1)}
    su_used = set()

    for _, user in user_priority:
        best_fse = -1e9
        best_res = None
        for (t, rid), (group, users) in resources.items():
            if (t, rid) in su_used:
                if user not in su_set:
                    if group is not None and mu_to_group.get(user) != group:
                        continue
                else:
                    continue
            candidate = users | {user}
            n_after = len(candidate)

            if user in su_set:
                if n_after > 1:
                    continue
            elif user in mu_to_group:
                g = mu_to_group[user]
                if any(cu in su_set for cu in candidate):
                    continue
                if any(mu_to_group.get(cu, g) != g for cu in candidate if cu in mu_to_group):
                    continue

            fse = compute_fse(user, n_after, t, beam_alloc, CAP, SINR)
            if cap_lookup(fse) > 0 and fse > best_fse:
                best_fse = fse
                best_res = (t, rid)

        if best_res is not None:
            t, rid = best_res
            assignments[user].append(best_res)
            resources[(t, rid)][1].add(user)
            if user in su_set:
                su_used.add((t, rid))
            elif user in mu_to_group:
                resources[(t, rid)][0] = mu_to_group[user]

    # Compute initial T
    res_to_users = {}
    for u, res_list in assignments.items():
        for sub, rid in res_list:
            key = (sub, rid)
            if key not in res_to_users:
                res_to_users[key] = set()
            res_to_users[key].add(u)

    return assignments, resources, res_to_users, su_used


def fast_improve(assignments, resources, res_to_users, su_used, beam_alloc,
                 CAP, SINR, buffer, RES_SUB, N, T, MU, SU, max_iter=3000):
    """快速局部改进：增量维护 user_tran_cache + res_to_users"""
    mu_to_group = {}
    for g_idx, group in enumerate(MU):
        for u in group:
            mu_to_group[u] = g_idx
    su_set = set(SU)

    user_tran_cache = {}
    for u in range(1, N + 1):
        if not assignments[u]:
            user_tran_cache[u] = 0
        else:
            bc = 0
            for sub, rid in assignments[u]:
                n = len(res_to_users.get((sub, rid), {u}))
                bc += cap_lookup(compute_fse(u, n, sub, beam_alloc, CAP, SINR))
            user_tran_cache[u] = min(buffer[u], bc)

    for _ in range(max_iter):
        best_delta = 0
        best_action = None

        for u in range(1, N + 1):
            if buffer.get(u, 0) <= 0:
                continue
            existing = set(assignments[u])
            old_u_tran = user_tran_cache[u]
            if old_u_tran >= buffer[u]:
                continue  # already saturated

            for t in range(1, T + 1):
                if not beam_alloc.get(t):
                    continue
                for rid in RES_SUB[t]:
                    if (t, rid) in existing:
                        continue
                    key = (t, rid)
                    cur_users = res_to_users.get(key, set())
                    n_before = len(cur_users)
                    n_after = n_before + 1

                    # Constraint check
                    if u in su_set:
                        if n_after > 1:
                            continue
                    elif u in mu_to_group:
                        g = mu_to_group[u]
                        skip = False
                        for cu in cur_users:
                            if cu in su_set:
                                skip = True; break
                            if cu in mu_to_group and mu_to_group[cu] != g:
                                skip = True; break
                        if skip:
                            continue

                    # Delta for u
                    fse_new = compute_fse(u, n_after, t, beam_alloc, CAP, SINR)
                    delta = min(buffer[u], old_u_tran + cap_lookup(fse_new)) - old_u_tran

                    # Delta for existing users on this resource
                    if n_before > 0:
                        for cu in cur_users:
                            old_c = cap_lookup(compute_fse(cu, n_before, t, beam_alloc, CAP, SINR))
                            new_c = cap_lookup(compute_fse(cu, n_after, t, beam_alloc, CAP, SINR))
                            if old_c != new_c:
                                old_ct = user_tran_cache[cu]
                                delta += min(buffer[cu], old_ct - old_c + new_c) - old_ct

                    if delta > best_delta:
                        best_delta = delta
                        best_action = (u, t, rid)

        if best_action is None:
            break

        u, t, rid = best_action
        key = (t, rid)
        n_before = len(res_to_users.get(key, set()))

        # Apply
        if key not in res_to_users:
            res_to_users[key] = set()
        res_to_users[key].add(u)
        assignments[u].append((t, rid))

        # Update cache
        cur_users = res_to_users[key]
        n_after = len(cur_users)
        fse_u = compute_fse(u, n_after, t, beam_alloc, CAP, SINR)
        user_tran_cache[u] = min(buffer[u], user_tran_cache[u] + cap_lookup(fse_u))

        for cu in cur_users:
            if cu == u:
                continue
            old_c = cap_lookup(compute_fse(cu, n_before, t, beam_alloc, CAP, SINR))
            new_c = cap_lookup(compute_fse(cu, n_after, t, beam_alloc, CAP, SINR))
            if old_c != new_c:
                user_tran_cache[cu] = min(buffer[cu], user_tran_cache[cu] - old_c + new_c)

        # Update resource tracking
        if u in su_set:
            su_used.add(key)
        elif u in mu_to_group:
            resources[key][0] = mu_to_group[u]
        resources[key][1] = res_to_users[key]

    return assignments


def format_output(beam_alloc, user_alloc, N):
    lines = []
    for t in sorted(beam_alloc.keys()):
        beams = beam_alloc[t]
        lines.append(f"{len(beams)}" + ''.join(f" {b}" for b in beams))
    for i in range(1, N + 1):
        alloc = user_alloc.get(i, [])
        if not alloc:
            lines.append("0")
        else:
            parts = [str(len(alloc))]
            for sub, res_id in alloc:
                parts.append(str(sub))
                parts.append(str(res_id))
            lines.append(' '.join(parts))
    return '\n'.join(lines)


def solve():
    data = sys.stdin.read().strip().split('\n')
    if not data or not data[0].strip():
        return
    P, N, K, T, beamMaxNum, M, MU, SU, CAP, buffer, SINR, RES_SUB = parse_input(data)
    beam_alloc = allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB)
    assignments, resources, res_to_users, su_used = greedy_allocate(
        N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc)
    assignments = fast_improve(assignments, resources, res_to_users, su_used,
                               beam_alloc, CAP, SINR, buffer, RES_SUB,
                               N, T, MU, SU)
    print(format_output(beam_alloc, assignments, N))


if __name__ == '__main__':
    solve()
