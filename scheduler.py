"""
通信资源联合分配调度器（<200ms, 多策略优化）
"""
import sys
import math
import time
import random


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
    beam_scores = [(sum(CAP[i][b] for i in range(1, N + 1)), b + 1) for b in range(P)]
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
    sinr = SINR.get(user, 0)
    share_penalty = lin2db(1.0 / max(1, n_users))
    beams = beam_alloc.get(sub_band, [])
    cap_with = sum(CAP[user][b - 1] for b in beams)
    cap_all = sum(CAP[user])
    return sinr + share_penalty + lin2db(cap_with) - lin2db(cap_all)


def greedy_allocate(N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc,
                    user_order):
    """贪心分配：按给定顺序处理用户，每个用户选最佳资源"""
    mu_to_group = {}
    for g_idx, group in enumerate(MU):
        for u in group:
            mu_to_group[u] = g_idx
    su_set = set(SU)

    resources = {}
    for t in range(1, T + 1):
        for rid in RES_SUB[t]:
            resources[(t, rid)] = [None, set()]

    assignments = {u: [] for u in range(1, N + 1)}
    su_used = set()

    for user in user_order:
        if buffer.get(user, 0) <= 0:
            continue
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

    res_to_users = {}
    for u, res_list in assignments.items():
        for sub, rid in res_list:
            key = (sub, rid)
            if key not in res_to_users:
                res_to_users[key] = set()
            res_to_users[key].add(u)
    return assignments, resources, res_to_users, su_used


def fast_improve(assignments, res_to_users, su_used, beam_alloc,
                 CAP, SINR, buffer, RES_SUB, N, T, MU, SU,
                 max_iter=3000, sa_mode=False, T_init=50.0,
                 time_budget_ms=float('inf'), enable_swap=True):
    """增量改进（Add + Swap），可选 SA 模式，预计算 base_fse 加速"""
    mu_to_group = {}
    for g_idx, group in enumerate(MU):
        for u in group:
            mu_to_group[u] = g_idx
    su_set = set(SU)

    # Precompute base_fse[u][t] = SINR + beam_gain (without sharing penalty)
    base_fse = {}
    for u in range(1, N + 1):
        base_fse[u] = {}
        cap_all = sum(CAP[u])
        for t in range(1, T + 1):
            beams = beam_alloc.get(t, [])
            cap_with = sum(CAP[u][b - 1] for b in beams)
            base_fse[u][t] = SINR[u] + lin2db(cap_with) - lin2db(cap_all)

    def fse_fast(u, t, n_users):
        return base_fse[u].get(t, -1000) + lin2db(1.0 / max(1, n_users))

    # Precompute cap_lookup table for faster access
    cap_cache = {}
    def cap_fast(fse_val):
        if fse_val not in cap_cache:
            cap_cache[fse_val] = cap_lookup(fse_val)
        return cap_cache[fse_val]

    user_tran_cache = {}
    for u in range(1, N + 1):
        if not assignments[u]:
            user_tran_cache[u] = 0
        else:
            bc = 0
            for sub, rid in assignments[u]:
                n = len(res_to_users.get((sub, rid), {u}))
                bc += cap_fast(fse_fast(u, sub, n))
            user_tran_cache[u] = min(buffer[u], bc)

    # SA state
    current_total = sum(user_tran_cache.values())
    best_total = current_total
    best_snapshot = None
    T_sa = T_init
    no_improve = 0
    t_start = time.perf_counter()

    def _make_undo_add(u, t, rid):
        return ('undo_add', u, t, rid)

    def _make_undo_swap(u, t_old, rid_old, t_new, rid_new):
        return ('undo_swap', u, t_old, rid_old, t_new, rid_new)

    undo_action = None  # filled before each apply

    # Precompute flat resource list for swap enumeration
    all_res = [(t, rid) for t in range(1, T + 1) if beam_alloc.get(t)
               for rid in RES_SUB[t]]

    is_su = {u: u in su_set for u in range(1, N + 1)}

    for it in range(max_iter):
        # Granular time check (before heavy eval)
        if sa_mode and it % 2 == 0:
            if (time.perf_counter() - t_start) * 1000 > time_budget_ms:
                break

        best_delta = -1e-9
        best_action = None  # ('add', u, t, rid) or ('swap', u, t_old, rid_old, t_new, rid_new)

        for u in range(1, N + 1):
            # Time check inside eval loop (SA can be slow)
            if sa_mode and u % 3 == 0:
                if (time.perf_counter() - t_start) * 1000 > time_budget_ms:
                    break
            if buffer.get(u, 0) <= 0:
                continue
            existing = set(assignments[u])
            old_u_tran = user_tran_cache[u]

            # --- Add ---
            if old_u_tran < buffer[u]:
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
                        if is_su[u]:
                            if n_after > 1:
                                continue
                        elif u in mu_to_group:
                            g = mu_to_group[u]
                            skip = False
                            for cu in cur_users:
                                if is_su[cu]:
                                    skip = True; break
                                if cu in mu_to_group and mu_to_group[cu] != g:
                                    skip = True; break
                            if skip:
                                continue
                        new_cap = cap_fast(fse_fast(u, t, n_after))
                        delta = min(buffer[u], old_u_tran + new_cap) - old_u_tran
                        if n_before > 0:
                            for cu in cur_users:
                                old_c = cap_fast(fse_fast(cu, t, n_before))
                                new_c = cap_fast(fse_fast(cu, t, n_after))
                                if old_c != new_c:
                                    old_ct = user_tran_cache[cu]
                                    delta += min(buffer[cu], old_ct - old_c + new_c) - old_ct
                        if delta > best_delta:
                            best_delta = delta
                            best_action = (0, u, t, rid)

            # --- Swap (sampled to limit cost) ---
            if enable_swap and existing:
                # Pick one random existing resource
                t_old, rid_old = random.choice(list(existing))
                old_key = (t_old, rid_old)
                old_cur = res_to_users.get(old_key, set())
                n_old_before = len(old_cur)
                old_u_cap = cap_fast(fse_fast(u, t_old, n_old_before))

                # Sample candidate resources (full scan too expensive)
                candidates = [r for r in all_res if r not in existing
                              and r[0] != t_old]  # exclude same timeslot
                if len(candidates) > 8:
                    candidates = random.sample(candidates, 8)
                for t_new, rid_new in candidates:
                        new_key = (t_new, rid_new)
                        new_cur = res_to_users.get(new_key, set())
                        n_new_before = len(new_cur)
                        n_new_after = n_new_before + 1

                        # Validate new resource (same rules as add)
                        if is_su[u]:
                            if n_new_after > 1:
                                continue
                        elif u in mu_to_group:
                            g = mu_to_group[u]
                            skip = False
                            for cu in new_cur:
                                if is_su[cu]:
                                    skip = True; break
                                if cu in mu_to_group and mu_to_group[cu] != g:
                                    skip = True; break
                            if skip:
                                continue

                        new_u_cap = cap_fast(fse_fast(u, t_new, n_new_after))

                        # Delta for user u
                        delta = (min(buffer[u], old_u_tran - old_u_cap + new_u_cap)
                                 - old_u_tran)

                        # Co-users on OLD resource gain capacity (u leaves)
                        n_old_after = n_old_before - 1
                        if n_old_after > 0:
                            for cu in old_cur:
                                if cu == u:
                                    continue
                                old_c = cap_fast(fse_fast(cu, t_old, n_old_before))
                                new_c = cap_fast(fse_fast(cu, t_old, n_old_after))
                                if old_c != new_c:
                                    old_ct = user_tran_cache[cu]
                                    delta += (min(buffer[cu], old_ct - old_c + new_c)
                                              - old_ct)

                        # Co-users on NEW resource lose capacity (u joins)
                        if n_new_before > 0:
                            for cu in new_cur:
                                if cu == u:
                                    continue
                                old_c = cap_fast(fse_fast(cu, t_new, n_new_before))
                                new_c = cap_fast(fse_fast(cu, t_new, n_new_after))
                                if old_c != new_c:
                                    old_ct = user_tran_cache[cu]
                                    delta += (min(buffer[cu], old_ct - old_c + new_c)
                                              - old_ct)

                        if delta > best_delta:
                            best_delta = delta
                            best_action = (1, u, t_old, rid_old, t_new, rid_new)

        if best_action is None:
            break

        # --- Apply action ---
        old_cache = {}  # for undo: {u: old_val}
        if best_action[0] == 0:  # Add
            _, u, t, rid = best_action
            key = (t, rid)
            n_before = len(res_to_users.get(key, set()))
            if key not in res_to_users:
                res_to_users[key] = set()
            res_to_users[key].add(u)
            assignments[u].append((t, rid))
            cur_users = res_to_users[key]
            n_after = len(cur_users)
            old_cache[u] = user_tran_cache[u]
            user_tran_cache[u] = min(buffer[u], user_tran_cache[u] +
                                     cap_fast(fse_fast(u, t, n_after)))
            for cu in cur_users:
                if cu == u:
                    continue
                old_c = cap_fast(fse_fast(cu, t, n_before))
                new_c = cap_fast(fse_fast(cu, t, n_after))
                if old_c != new_c:
                    old_cache[cu] = user_tran_cache[cu]
                    user_tran_cache[cu] = min(buffer[cu],
                                              user_tran_cache[cu] - old_c + new_c)

            def _undo_add():
                res_to_users[key].discard(u)
                if not res_to_users[key]:
                    del res_to_users[key]
                assignments[u].remove((t, rid))
                for cu, old_v in old_cache.items():
                    user_tran_cache[cu] = old_v

            undo_func = _undo_add

        else:  # Swap
            _, u, t_old, rid_old, t_new, rid_new = best_action
            old_key = (t_old, rid_old)
            new_key = (t_new, rid_new)

            n_old_before = len(res_to_users.get(old_key, set()))
            old_u_cap = cap_fast(fse_fast(u, t_old, n_old_before))

            # Remove old
            assignments[u].remove((t_old, rid_old))
            if old_key in res_to_users:
                res_to_users[old_key].discard(u)
                if not res_to_users[old_key]:
                    del res_to_users[old_key]

            old_affected = {}
            for cu in res_to_users.get(old_key, set()):
                old_c = cap_fast(fse_fast(cu, t_old, n_old_before))
                new_c = cap_fast(fse_fast(cu, t_old, n_old_before - 1))
                if old_c != new_c:
                    old_affected[cu] = user_tran_cache[cu]
                    user_tran_cache[cu] = min(buffer[cu],
                                              user_tran_cache[cu] - old_c + new_c)

            # Add new
            n_new_before = len(res_to_users.get(new_key, set()))
            if new_key not in res_to_users:
                res_to_users[new_key] = set()
            res_to_users[new_key].add(u)
            assignments[u].append((t_new, rid_new))

            old_cache[u] = user_tran_cache[u]  # capture pre-swap value
            new_u_cap = cap_fast(fse_fast(u, t_new, n_new_before + 1))
            user_tran_cache[u] = min(buffer[u],
                                     user_tran_cache[u] - old_u_cap + new_u_cap)

            new_affected = {}
            for cu in res_to_users[new_key]:
                if cu == u:
                    continue
                old_c = cap_fast(fse_fast(cu, t_new, n_new_before))
                new_c = cap_fast(fse_fast(cu, t_new, n_new_before + 1))
                if old_c != new_c:
                    new_affected[cu] = user_tran_cache[cu]
                    user_tran_cache[cu] = min(buffer[cu],
                                              user_tran_cache[cu] - old_c + new_c)

            def _undo_swap():
                # Remove new
                assignments[u].remove((t_new, rid_new))
                if new_key in res_to_users:
                    res_to_users[new_key].discard(u)
                    if not res_to_users[new_key]:
                        del res_to_users[new_key]
                # Restore old
                assignments[u].append((t_old, rid_old))
                if old_key not in res_to_users:
                    res_to_users[old_key] = set()
                res_to_users[old_key].add(u)
                # Restore all affected caches
                user_tran_cache[u] = old_cache[u]
                for cu, old_v in old_affected.items():
                    user_tran_cache[cu] = old_v
                for cu, old_v in new_affected.items():
                    user_tran_cache[cu] = old_v

            undo_func = _undo_swap

        new_total = sum(user_tran_cache.values())

        # --- SA accept/reject ---
        if sa_mode and best_delta <= 0:
            accept_prob = math.exp(best_delta / max(T_sa, 1e-4))
            if random.random() >= accept_prob:
                undo_func()
                no_improve += 1
            else:
                current_total = new_total
                no_improve = 0
                if current_total > best_total:
                    best_total = current_total
        else:
            current_total = new_total
            no_improve = 0
            if sa_mode and current_total > best_total:
                best_total = current_total
        # Note: greedy mode (non-SA) doesn't need best tracking — last state is best

        # --- Temperature decay + time check ---
        if sa_mode:
            T_sa = T_init * (1.0 - (it + 1) / max(max_iter, 1))
            elapsed = (time.perf_counter() - t_start) * 1000
            if elapsed > time_budget_ms:
                break
            if T_sa < 0.1 and no_improve > 500:
                break


def compute_total_T(assignments, res_to_users, beam_alloc, CAP, SINR, buffer, N):
    total = 0
    for u in range(1, N + 1):
        if not assignments[u]:
            continue
        bc = 0
        for sub, rid in assignments[u]:
            n = len(res_to_users.get((sub, rid), {u}))
            bc += cap_lookup(compute_fse(u, n, sub, beam_alloc, CAP, SINR))
        total += min(buffer[u], bc)
    return total


def compute_congestion(assignments, buffer, N, T):
    """按时隙统计用户需求密度（拥挤度）"""
    t_demand = {t: 0.0 for t in range(1, T + 1)}
    t_count = {t: 0 for t in range(1, T + 1)}
    for u in range(1, N + 1):
        if not assignments[u] or buffer.get(u, 0) <= 0:
            continue
        seen_t = set()
        for t, rid in assignments[u]:
            if t not in seen_t:
                seen_t.add(t)
                t_demand[t] += buffer[u]
                t_count[t] += 1
    congestion = {}
    for t in range(1, T + 1):
        congestion[t] = t_demand[t] / t_count[t] if t_count[t] > 0 else 0.0
    return congestion


def reallocate_beams(beam_alloc, congestion, P, T, beamMaxNum):
    """从最不拥挤时隙移一个波束到最拥挤时隙"""
    if not congestion:
        return
    sorted_t = sorted(congestion.keys(),
                      key=lambda t: (congestion[t], t), reverse=True)
    most_t = sorted_t[0]
    least_t = sorted_t[-1]
    if congestion[most_t] <= congestion[least_t] * 1.1:
        return  # gap too small
    if len(beam_alloc[most_t]) >= P:
        return
    if len(beam_alloc[least_t]) <= 1:
        return
    beam = beam_alloc[least_t].pop()
    beam_alloc[most_t].append(beam)


def format_output(beam_alloc, user_alloc, N):
    lines = []
    for t in sorted(beam_alloc.keys()):
        lines.append(f"{len(beam_alloc[t])}" + ''.join(f" {b}" for b in beam_alloc[t]))
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


def make_user_orders(N, buffer, SINR, CAP, MU, SU, seed=42):
    """生成多种用户优先级排序（按不同评分函数 + 随机）"""
    rng = random.Random(seed)
    tc = {u: sum(CAP[u]) for u in range(1, N + 1)}
    users = [u for u in range(1, N + 1) if buffer[u] > 0]
    orders = []

    # 1: buffer × total_cap × sinr_linear (default)
    orders.append(sorted(users, key=lambda u: buffer[u] * tc[u] *
                         (10 ** (max(-10, min(30, SINR[u])) / 10)), reverse=True))
    # 2: buffer only
    orders.append(sorted(users, key=lambda u: buffer[u], reverse=True))
    # 3: buffer × total_cap
    orders.append(sorted(users, key=lambda u: buffer[u] * tc[u], reverse=True))
    # 4: SINR-weighted buffer
    orders.append(sorted(users, key=lambda u: buffer[u] * (SINR[u] + 30) * tc[u], reverse=True))
    # 5: SU first, then MU by buffer
    su_set = set(SU)
    su_users = sorted([u for u in users if u in su_set], key=lambda u: buffer[u], reverse=True)
    mu_users = sorted([u for u in users if u not in su_set], key=lambda u: buffer[u] * tc[u], reverse=True)
    orders.append(su_users + mu_users)
    # 6: MU by group, then SU
    mu_list = []
    for g in MU:
        mu_list.extend(sorted(g, key=lambda u: buffer[u] * tc[u], reverse=True))
    orders.append(mu_list + su_users)
    # 7-8: reverse of first two
    orders.append(list(reversed(orders[0])))
    orders.append(list(reversed(orders[1])))
    # 9-38: random shuffles (30)
    for _ in range(30):
        shuffled = list(users)
        rng.shuffle(shuffled)
        orders.append(shuffled)

    return orders


def solve():
    t_start = time.perf_counter()
    data = sys.stdin.read().strip().split('\n')
    if not data or not data[0].strip():
        return

    P, N, K, T, beamMaxNum, M, MU, SU, CAP, buffer, SINR, RES_SUB = parse_input(data)
    beam_alloc = allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB)
    user_orders = make_user_orders(N, buffer, SINR, CAP, MU, SU)

    # Phase 1: greedy on all strategies, keep top 2
    top = []
    for order in user_orders:
        ua, resources, rtu, su_used = greedy_allocate(
            N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc, order)
        T_val = compute_total_T(ua, rtu, beam_alloc, CAP, SINR, buffer, N)
        top.append((T_val, ua, rtu, su_used))
        top.sort(key=lambda x: x[0], reverse=True)
        if len(top) > 2:
            top.pop()

    # Phase 2: fast_improve with swap on each top result, pick best
    best_T = -1
    best_ua = None
    for _, ua, rtu, su_used in top:
        fast_improve(ua, rtu, su_used, beam_alloc, CAP, SINR, buffer,
                     RES_SUB, N, T, MU, SU, max_iter=3000,
                     sa_mode=False, enable_swap=True)
        T_val = compute_total_T(ua, rtu, beam_alloc, CAP, SINR, buffer, N)
        if T_val > best_T:
            best_T = T_val
            best_ua = ua

    print(format_output(beam_alloc, best_ua, N))


if __name__ == '__main__':
    solve()
