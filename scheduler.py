"""
通信资源联合分配调度器
用法: python scheduler.py < input.in
"""
import sys
import math


def lin2db(x):
    if x <= 1e-12:
        return -1000.0
    return 10.0 * math.log10(x)


def cap_lookup(fse):
    if fse <= -10:
        return 0
    elif fse <= 0:
        return 8
    elif fse <= 3:
        return 24
    elif fse <= 10:
        return 90
    elif fse <= 15:
        return 120
    elif fse <= 20:
        return 162
    else:
        return 222


def parse_input(lines):
    idx = 0
    P, N, K, T, beamMaxNum = map(int, lines[idx].strip().split())
    idx += 1
    M = int(lines[idx].strip())
    idx += 1

    MU = []
    for _ in range(M):
        parts = list(map(int, lines[idx].strip().split()))
        MU.append(parts[1:])  # skip count
        idx += 1

    parts = list(map(int, lines[idx].strip().split()))
    SU = parts[1:]  # skip count
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
        RES_SUB[t] = parts[1:]  # skip count
        idx += 1

    return P, N, K, T, beamMaxNum, M, MU, SU, CAP, buffer, SINR, RES_SUB


def allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB):
    """
    波束分配：确保不同子带使用不同波束以覆盖更多用户。
    策略：
    1. 每个子带至少 1 个波束，按轮询分配不同波束
    2. 剩余波束按每个子带的资源数比例分配
    """
    beam_alloc = {t: [] for t in range(1, T + 1)}
    res_counts = {t: len(RES_SUB[t]) for t in range(1, T + 1)}

    # Score each beam for each user (total cap per beam)
    beam_scores = []
    for b in range(P):
        score = sum(CAP[i][b] for i in range(1, N + 1))
        beam_scores.append((score, b + 1))  # (score, 1-indexed beam)
    beam_scores.sort(reverse=True)

    # Phase 1: each sub-band gets 1 beam, cycling through best beams
    allocated_count = 0
    for t in range(1, T + 1):
        if allocated_count < beamMaxNum:
            beam_idx = allocated_count % len(beam_scores)
            beam_alloc[t].append(beam_scores[beam_idx][1])
            allocated_count += 1

    # Phase 2: distribute remaining beams to sub-bands with most resources
    while allocated_count < beamMaxNum:
        # Pick sub-band with most resources and fewest beams
        best_t = max(range(1, T + 1),
                     key=lambda t: (res_counts[t], -len(beam_alloc[t])))
        if len(beam_alloc[best_t]) >= P:
            break
        # Pick next best beam not already in this sub-band
        used = set(beam_alloc[best_t])
        for _, beam_id in beam_scores:
            if beam_id not in used:
                beam_alloc[best_t].append(beam_id)
                allocated_count += 1
                break
        else:
            break

    return beam_alloc


def compute_fse(user, resource_users, sub_band, beam_alloc, CAP, SINR):
    sinr = SINR.get(user, 0)
    n_users = max(1, len(resource_users))
    share_penalty = lin2db(1.0 / n_users)
    beams = beam_alloc.get(sub_band, [])
    cap_with_beams = sum(CAP[user][b - 1] for b in beams)
    cap_all = sum(CAP[user])
    beam_gain = lin2db(cap_with_beams) - lin2db(cap_all)
    return sinr + share_penalty + beam_gain


def compute_user_transmission(user, res_assignments, beam_alloc, CAP, SINR, buffer,
                              all_assignments):
    """
    all_assignments: dict[user] → [(sub_band, res_id)]
    """
    if not res_assignments:
        return 0
    # Build resource → users mapping
    res_to_users = {}
    for u, res_list in all_assignments.items():
        for sub, res_id in res_list:
            key = (sub, res_id)
            if key not in res_to_users:
                res_to_users[key] = set()
            res_to_users[key].add(u)

    buf_cap = 0
    for sub, res_id in res_assignments:
        key = (sub, res_id)
        users_on_res = res_to_users.get(key, {user})
        fse = compute_fse(user, users_on_res, sub, beam_alloc, CAP, SINR)
        buf_cap += cap_lookup(fse)

    return min(buffer.get(user, 0), buf_cap)


def compute_total_T(all_assignments, beam_alloc, CAP, SINR, buffer):
    total = 0
    for user, res_list in all_assignments.items():
        total += compute_user_transmission(user, res_list, beam_alloc, CAP, SINR,
                                           buffer, all_assignments)
    return total


def greedy_allocate(N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc):
    """
    贪心用户-资源分配。
    """
    mu_to_group = {}
    for g_idx, group in enumerate(MU):
        for u in group:
            mu_to_group[u] = g_idx
    su_set = set(SU)

    # resources[(sub, res_id)] → (group_idx_or_None, set_of_users)
    resources = {}
    for t in range(1, T + 1):
        for res_id in RES_SUB[t]:
            resources[(t, res_id)] = (None, set())  # (mu_group, users)

    # User priority: by buffer * effective_cap
    user_priority = []
    for u in range(1, N + 1):
        if buffer[u] <= 0:
            continue
        total_cap = sum(CAP[u])
        sinr_linear = 10 ** (max(-10, min(30, SINR[u])) / 10)
        score = buffer[u] * total_cap * sinr_linear
        user_priority.append((score, u))
    user_priority.sort(reverse=True)

    # Track assignments
    assignments = {u: [] for u in range(1, N + 1)}
    res_used = set()  # (sub, res_id) already assigned to a SU user

    for _, user in user_priority:
        best_fse = -1e9
        best_res = None

        for (t, res_id), (group, users) in resources.items():
            if (t, res_id) in res_used:
                # SU's exclusive resource
                if user not in su_set:
                    # MU user can't use SU's exclusive resource unless same MU group
                    if group is not None and mu_to_group.get(user) != group:
                        continue
                else:
                    continue  # SU can't share

            # Build candidate users for this resource
            candidate_users = users | {user}

            # Check constraints
            if user in su_set:
                # SU needs exclusive access
                if len(candidate_users) > 1:
                    continue
            elif user in mu_to_group:
                # MU user: all co-users must be from same MU group, no SU mixing
                g = mu_to_group[user]
                if any(cu in su_set for cu in candidate_users):
                    continue
                if any(cu in mu_to_group and mu_to_group[cu] != g
                       for cu in candidate_users):
                    continue

            fse = compute_fse(user, candidate_users, t, beam_alloc, CAP, SINR)
            capacity = cap_lookup(fse)
            if capacity > 0 and fse > best_fse:
                best_fse = fse
                best_res = (t, res_id)

        if best_res is not None:
            t, res_id = best_res
            assignments[user].append(best_res)
            # Update resource tracking
            resources[(t, res_id)][1].add(user)
            if user in su_set:
                res_used.add((t, res_id))
            elif user in mu_to_group:
                g = mu_to_group[user]
                resources[(t, res_id)] = (g, resources[(t, res_id)][1])

    return assignments


def local_improve(assignments, beam_alloc, CAP, SINR, buffer, RES_SUB, N, P, T,
                  MU, SU, max_iter=200):
    """尝试给每个用户额外分配未使用的资源"""
    mu_to_group = {}
    for g_idx, group in enumerate(MU):
        for u in group:
            mu_to_group[u] = g_idx
    su_set = set(SU)

    # Track used resources
    used_res = set()
    for u, res_list in assignments.items():
        for r in res_list:
            used_res.add(r)

    best_score = compute_total_T(assignments, beam_alloc, CAP, SINR, buffer)

    for _ in range(max_iter):
        improved = False
        for user in range(1, N + 1):
            if buffer.get(user, 0) <= 0:
                continue
            existing = set(assignments[user])
            for t in range(1, T + 1):
                if not beam_alloc.get(t):
                    continue
                for res_id in RES_SUB[t]:
                    if (t, res_id) in existing:
                        continue

                    # Quick check: can this user use this resource?
                    is_used = (t, res_id) in used_res
                    if is_used:
                        if user in su_set:
                            continue  # SU can't share

                    test_assignments = {u: list(v) for u, v in assignments.items()}
                    test_assignments[user].append((t, res_id))
                    new_score = compute_total_T(test_assignments, beam_alloc, CAP, SINR, buffer)

                    if new_score > best_score:
                        best_score = new_score
                        assignments = test_assignments
                        used_res.add((t, res_id))
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return assignments


def format_output(beam_alloc, user_alloc, T, N):
    lines = []
    for t in range(1, T + 1):
        beams = beam_alloc.get(t, [])
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

    # Beam allocation
    beam_alloc = allocate_beams(P, T, beamMaxNum, CAP, N, RES_SUB)

    # User-resource allocation
    user_alloc = greedy_allocate(N, K, T, MU, SU, CAP, buffer, SINR, RES_SUB, beam_alloc)

    # Local improvement
    user_alloc = local_improve(user_alloc, beam_alloc, CAP, SINR, buffer, RES_SUB,
                               N, P, T, MU, SU)

    # Output
    print(format_output(beam_alloc, user_alloc, T, N))


if __name__ == '__main__':
    solve()
