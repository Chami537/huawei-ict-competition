import sys
from typing import List

class SubBandBeamAlloc:
    def __init__(self, sub_band_idx: int, beam_count: int, beams: List[int]):
        self.sub_band_idx = sub_band_idx
        self.beam_count = beam_count
        self.beams = beams

class UserResAlloc:
    def __init__(self, user_idx: int, res_count: int, sub_bands: List[int], res_ids: List[int]):
        self.user_idx = user_idx
        self.res_count = res_count
        self.sub_bands = sub_bands
        self.res_ids = res_ids

def solve():
    lines = sys.stdin.readlines()

    idx = 0
    B, U, K, T, beamMaxNum = map(int, lines[idx].strip().split())
    idx += 1

    M = int(lines[idx].strip())
    idx += 1

    MU = []
    for mu_idx in range(M):
        parts = list(map(int, lines[idx].strip().split()))
        user_count = parts[0]
        users = parts[1:]
        MU.append(users)
        idx += 1

    parts = list(map(int, lines[idx].strip().split()))
    su_count = parts[0]
    SU = parts[1:]
    idx += 1

    CAP = {}
    for i in range(1, U + 1):
        caps = list(map(float, lines[idx].strip().split()))
        CAP[i] = caps
        idx += 1

    buffer = {}
    SINR = {}
    for i in range(1, U + 1):
        parts = lines[idx].strip().split()
        buffer[i] = int(parts[0])
        SINR[i] = float(parts[1])
        idx += 1

    RES_SUB = {}
    for i in range(1, T + 1):
        parts = list(map(int, lines[idx].strip().split()))
        res_count = parts[0]
        res_list = parts[1:]
        RES_SUB[i] = res_list
        idx += 1

    sub_band_allocs = []
    for i in range(1, T + 1):
        beam_idx = (i - 1) % B + 1
        sub_band_allocs.append(SubBandBeamAlloc(i, 1, [beam_idx]))

    for i in range(T):
        alloc = sub_band_allocs[i]
        print(f"{alloc.beam_count}", end='')
        for beam in alloc.beams:
            print(f" {beam}", end='')
        print()

    user_allocs = []
    res_idx = 1
    for user_idx in range(1, U + 1):
        if user_idx <= U and res_idx <= K:
            for sub_idx, res_list in RES_SUB.items():
                if res_idx in res_list:
                    user_allocs.append(UserResAlloc(user_idx, 1, [sub_idx], [res_idx]))
                    res_idx += 1
                    break
        else:
            user_allocs.append(UserResAlloc(user_idx, 0, [], []))

    for i in range(U):
        alloc = user_allocs[i]
        print(f"{alloc.res_count}", end='')
        for j in range(alloc.res_count):
            print(f" {alloc.sub_bands[j]} {alloc.res_ids[j]}", end='')
        print()

if __name__ == '__main__':
    solve()
