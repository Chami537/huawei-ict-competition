/*
 * 2026 华为智联杯·无线程序设计大赛
 * 赛题二：通信资源联合分配 —— 主程序 (C, 适配 gcc 11.4.0 / -O2 / x86_64 Linux)
 *
 * 思路（与 Model/ 同源的 Python 原型经开放用例验证后移植，无随机数）：
 *   1) 波束选择：按全局流行度 sum_i cap[i][p] 选 top-k 波束；自适应扫描“启用子带数 A”
 *      (1..T)，把 beamMaxNum 预算均摊到 res 最多的前 A 个子带，分别求解后取传输量 T 最大者。
 *   2) 资源/用户分配：逐资源块贪心。对每个 res(所属子带 t)，比较
 *        - 单用户独占；
 *        - 同一 RU 内 s 个用户共享(s=2..min(|RU|,10))，取边际传输增益最高的 s 个用户；
 *      选边际增益最大的配置，更新各用户剩余 buffer。
 *
 * 不使用任何第三方库；自带 log10 实现以避免对 libm 的链接依赖。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple xorshift RNG for deterministic reproducible mutations */
static unsigned int rng_state = 2463534242u;
static unsigned int xorshift(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

#define MAXP 33
#define MAXN 52
#define MAXK 40
#define MAXT 20
#define MAXM 10

/* ---------- 自带 log/log10（仅用 + - * / 与循环，避免链接 libm） ---------- */
static double my_ln(double x) {
    if (x <= 0.0) return -1e18;
    int e = 0;
    while (x >= 2.0) { x *= 0.5; e++; }
    while (x < 1.0)  { x *= 2.0; e--; }
    /* x in [1,2): ln(x)=2*(y+y^3/3+y^5/5+...), y=(x-1)/(x+1) */
    double y = (x - 1.0) / (x + 1.0);
    double y2 = y * y, term = y, sum = 0.0;
    for (int k = 1; k <= 25; k += 2) { sum += term / k; term *= y2; }
    static const double LN2 = 0.6931471805599453;
    return 2.0 * sum + e * LN2;
}
static const double INV_LN10 = 0.43429448190325176; /* 1/ln(10) */
static double lin2db(double x) { return 10.0 * my_ln(x) * INV_LN10; }

/* ---------- 速率台阶函数 ---------- */
static int cap_rate(double fse) {
    if (fse <= -10.0) return 0;
    if (fse <= 0.0)   return 8;
    if (fse <= 3.0)   return 24;
    if (fse <= 10.0)  return 90;
    if (fse <= 15.0)  return 120;
    if (fse <= 20.0)  return 162;
    return 222;
}

/* ---------- 全局输入 ---------- */
static int P, N, K, T, beamMax, M;
static int ru[MAXM][12], ru_sz[MAXM];
static double CAP[MAXN][MAXP];   /* CAP[i][p], 1..N, 1..P */
static double Sall[MAXN];        /* sum_p CAP[i][p] */
static int buf[MAXN];
static double sinr[MAXN];
static int rsub[MAXT][MAXK], rsub_sz[MAXT];
static int res2sub[MAXK];

static int pop_order[MAXP];      /* 波束按流行度降序 */

/* 当前方案 */
static int beams_cnt[MAXT], beams_list[MAXT][MAXP];
static double G[MAXN][MAXT];     /* 波束增益项 */
static double rem[MAXN];         /* 剩余 buffer */
static int al_cnt[MAXN], al_sub[MAXN][MAXK], al_res[MAXN][MAXK];

/* 最优方案备份 */
static int best_beams_cnt[MAXT], best_beams_list[MAXT][MAXP];
static int best_al_cnt[MAXN], best_al_sub[MAXN][MAXK], best_al_res[MAXN][MAXK];

static double db_share[12];      /* lin2db(1/s), s=1..10 */

static void sort_beams_by_score(double *score) {
    for (int p = 1; p <= P; p++) pop_order[p - 1] = p;
    for (int a = 0; a < P; a++)
        for (int b = a + 1; b < P; b++)
            if (score[pop_order[b]] > score[pop_order[a]]) {
                int t = pop_order[a]; pop_order[a] = pop_order[b]; pop_order[b] = t;
            }
}

/* Strategy 0: popularity — sum of raw CAP across all users */
static void compute_pop(void) {
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int i = 1; i <= N; i++) score[p] += CAP[i][p];
    }
    sort_beams_by_score(score);
}

/* Strategy 1: marginal urgency — buffer-weighted relative capacity */
static void compute_marginal(void) {
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int i = 1; i <= N; i++)
            if (Sall[i] > 1e-9 && buf[i] > 0)
                score[p] += buf[i] * CAP[i][p] / Sall[i];
    }
    sort_beams_by_score(score);
}

/* Strategy 2: balanced — cap per-user contribution to spread beams */
static void compute_balanced(void) {
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int i = 1; i <= N; i++) {
            double rel = (Sall[i] > 1e-9) ? CAP[i][p] / Sall[i] : 0.0;
            score[p] += rel < 0.3 ? rel : 0.3;  /* cap at 0.3 */
        }
    }
    sort_beams_by_score(score);
}

/* Strategy 3: MU sharing potential — reward beams with many high-cap users across RUs */
static void compute_mupot(void) {
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int m = 0; m < M; m++) {
            double best = 0.0;
            for (int j = 0; j < ru_sz[m]; j++) {
                int u = ru[m][j];
                double rel = (Sall[u] > 1e-9 && buf[u] > 0) ? CAP[u][p] / Sall[u] : 0.0;
                if (rel > best) best = rel;
            }
            score[p] += best * best;  /* square to reward strong per-group signal */
        }
    }
    sort_beams_by_score(score);
}

/* Save current allocation state */
static void save_best(double Tval, double *bestT) {
    if (Tval > *bestT) {
        *bestT = Tval;
        for (int t = 1; t <= T; t++) {
            best_beams_cnt[t] = beams_cnt[t];
            for (int j = 0; j < beams_cnt[t]; j++) best_beams_list[t][j] = beams_list[t][j];
        }
        for (int i = 1; i <= N; i++) {
            best_al_cnt[i] = al_cnt[i];
            for (int j = 0; j < al_cnt[i]; j++) { best_al_sub[i][j] = al_sub[i][j]; best_al_res[i][j] = al_res[i][j]; }
        }
    }
}

/* 按 res 数降序给出子带顺序 */
static void subband_order(int *order) {
    for (int t = 0; t < T; t++) order[t] = t + 1;
    for (int a = 0; a < T; a++)
        for (int b = a + 1; b < T; b++)
            if (rsub_sz[order[b]] > rsub_sz[order[a]]) {
                int tmp = order[a]; order[a] = order[b]; order[b] = tmp;
            }
}

/* 分配波束：启用前 A 个(res 最多)子带，均摊预算 */
static void alloc_beams(int A) {
    for (int t = 1; t <= T; t++) beams_cnt[t] = 0;
    if (A < 1) A = 1;
    if (A > T) A = T;
    if (A > beamMax) A = beamMax;
    int order[MAXT]; subband_order(order);
    int base = beamMax / A; if (base > P) base = P; if (base < 1) base = 1;
    int rcnt = beamMax - base * A; if (rcnt < 0) rcnt = 0;
    for (int idx = 0; idx < A; idx++) {
        int t = order[idx];
        int k = base + (idx < rcnt ? 1 : 0); if (k > P) k = P;
        beams_cnt[t] = k;
        for (int j = 0; j < k; j++) beams_list[t][j] = pop_order[j];
    }
}

static void compute_gain(void) {
    for (int i = 1; i <= N; i++)
        for (int t = 1; t <= T; t++) {
            if (beams_cnt[t] == 0 || Sall[i] <= 0.0) { G[i][t] = -1e18; continue; }
            double s = 0.0;
            for (int j = 0; j < beams_cnt[t]; j++) s += CAP[i][beams_list[t][j]];
            G[i][t] = (s > 0.0) ? (lin2db(s) - lin2db(Sall[i])) : -1e18;
        }
}

static double rate_of(int i, int t, int s) {
    if (G[i][t] <= -1e17) return 0.0;
    return (double)cap_rate(sinr[i] + db_share[s] + G[i][t]);
}

/* 在子带 t 用一个 res，求最佳配置；返回边际增益，chosen[]/cr[] 填充用户与其获得速率 */
static double best_config(int t, int *chosen, double *cr, int *ncho) {
    double bestv = -1.0; int bn = 0; int bu[12]; double bc[12];
    /* 单用户 */
    for (int i = 1; i <= N; i++) {
        if (rem[i] <= 1e-9) continue;
        double r = rate_of(i, t, 1);
        double v = rem[i] < r ? rem[i] : r;
        if (v > bestv) { bestv = v; bn = 1; bu[0] = i; bc[0] = r; }
    }
    /* RU 内共享 */
    for (int m = 0; m < M; m++) {
        int av[12], na = 0;
        for (int j = 0; j < ru_sz[m]; j++) { int u = ru[m][j]; if (rem[u] > 1e-9) av[na++] = u; }
        if (na < 2) continue;
        int smax = na < 10 ? na : 10;
        for (int s = 2; s <= smax; s++) {
            double contrib[12]; int uid[12];
            for (int j = 0; j < na; j++) {
                int u = av[j]; double r = rate_of(u, t, s);
                contrib[j] = rem[u] < r ? rem[u] : r; uid[j] = u;
            }
            /* 取 contrib 最大的 s 个（选择排序前 s 个） */
            for (int a = 0; a < s; a++)
                for (int b = a + 1; b < na; b++)
                    if (contrib[b] > contrib[a]) {
                        double tc = contrib[a]; contrib[a] = contrib[b]; contrib[b] = tc;
                        int tu = uid[a]; uid[a] = uid[b]; uid[b] = tu;
                    }
            double v = 0.0; for (int a = 0; a < s; a++) v += contrib[a];
            if (v > bestv) {
                bestv = v; bn = s;
                for (int a = 0; a < s; a++) { bu[a] = uid[a]; bc[a] = rate_of(uid[a], t, s); }
            }
        }
    }
    for (int a = 0; a < bn; a++) { chosen[a] = bu[a]; cr[a] = bc[a]; }
    *ncho = bn;
    return bestv;
}

/* 用当前 beams 跑一遍贪心，返回总传输量 T，并填好 al_* */
static double run_greedy(void) {
    for (int i = 1; i <= N; i++) { rem[i] = (double)buf[i]; al_cnt[i] = 0; }
    compute_gain();
    /* res 列表，按所属子带波束数降序处理 */
    int reslist[MAXK], rcnt = 0;
    for (int t = 1; t <= T; t++)
        for (int j = 0; j < rsub_sz[t]; j++) reslist[rcnt++] = rsub[t][j];
    for (int a = 0; a < rcnt; a++)
        for (int b = a + 1; b < rcnt; b++)
            if (beams_cnt[res2sub[reslist[b]]] > beams_cnt[res2sub[reslist[a]]]) {
                int tmp = reslist[a]; reslist[a] = reslist[b]; reslist[b] = tmp;
            }
    for (int a = 0; a < rcnt; a++) {
        int res = reslist[a], t = res2sub[res];
        if (beams_cnt[t] == 0) continue;
        int chosen[12]; double cr[12]; int nch;
        double v = best_config(t, chosen, cr, &nch);
        if (nch > 0 && v > 1e-9) {
            for (int x = 0; x < nch; x++) {
                int u = chosen[x];
                al_sub[u][al_cnt[u]] = t; al_res[u][al_cnt[u]] = res; al_cnt[u]++;
                double got = rem[u] < cr[x] ? rem[u] : cr[x];
                rem[u] -= got;
            }
        }
    }
    double Tval = 0.0;
    for (int i = 1; i <= N; i++) Tval += (double)buf[i] - rem[i];
    return Tval;
}

int main(void) {
    if (scanf("%d %d %d %d %d", &P, &N, &K, &T, &beamMax) != 5) return 0;
    scanf("%d", &M);
    for (int m = 0; m < M; m++) {
        scanf("%d", &ru_sz[m]);
        for (int j = 0; j < ru_sz[m]; j++) scanf("%d", &ru[m][j]);
    }
    int su_sz; scanf("%d", &su_sz);
    for (int j = 0; j < su_sz; j++) { int x; scanf("%d", &x); }  /* SU 仅独占，算法中按单用户处理，读入跳过 */
    for (int i = 1; i <= N; i++) {
        Sall[i] = 0.0;
        for (int p = 1; p <= P; p++) { scanf("%lf", &CAP[i][p]); Sall[i] += CAP[i][p]; }
    }
    for (int i = 1; i <= N; i++) scanf("%d %lf", &buf[i], &sinr[i]);
    for (int t = 1; t <= T; t++) {
        scanf("%d", &rsub_sz[t]);
        for (int j = 0; j < rsub_sz[t]; j++) { scanf("%d", &rsub[t][j]); res2sub[rsub[t][j]] = t; }
    }

    db_share[1] = 0.0;
    for (int s = 2; s <= 10; s++) db_share[s] = lin2db(1.0 / s);

    int Amax = T < beamMax ? T : beamMax; if (Amax < 1) Amax = 1;
    double bestT = -1.0;

    /* 4 beam ranking strategies */
    void (*strategies[4])(void) = {compute_pop, compute_marginal, compute_balanced, compute_mupot};
    for (int strat = 0; strat < 4; strat++) {
        strategies[strat]();

        for (int A = 1; A <= Amax; A++) {
            alloc_beams(A);
            double Tval = run_greedy();
            save_best(Tval, &bestT);
        }
    }

    /* Directed local search: swap worst user's worst resource to a better one */
    {
        for (int t = 1; t <= T; t++) {
            beams_cnt[t] = best_beams_cnt[t];
            for (int j = 0; j < best_beams_cnt[t]; j++) beams_list[t][j] = best_beams_list[t][j];
        }
        for (int i = 1; i <= N; i++) {
            al_cnt[i] = best_al_cnt[i];
            for (int j = 0; j < best_al_cnt[i]; j++) {
                al_sub[i][j] = best_al_sub[i][j];
                al_res[i][j] = best_al_res[i][j];
            }
        }
        compute_gain();

        for (int iter = 0; iter < 100; iter++) {
            /* Find worst user by utilization ratio */
            int worst = -1;
            double worst_ratio = 2.0;
            for (int u = 1; u <= N; u++) {
                if (buf[u] <= 0 || al_cnt[u] == 0) continue;
                double tran = 0.0;
                for (int j = 0; j < al_cnt[u]; j++) {
                    int t = al_sub[u][j], rid = al_res[u][j];
                    int n = 0;
                    for (int u2 = 1; u2 <= N; u2++)
                        for (int k = 0; k < al_cnt[u2]; k++)
                            if (al_sub[u2][k] == t && al_res[u2][k] == rid) n++;
                    double r = rate_of(u, t, n);
                    tran += r < (buf[u] - tran) ? r : (buf[u] - tran);
                }
                double ratio = tran / (double)buf[u];
                if (ratio < worst_ratio) { worst_ratio = ratio; worst = u; }
            }
            if (worst < 0) break;

            /* Compute user's throughput helper */
            #define user_tput(uu, alist, nt_old, nrid_old, nt_new, nrid_new) ({ \
                double _tran = 0.0; \
                for (int _j = 0; _j < al_cnt[uu]; _j++) { \
                    int _t = alist##_sub[uu][_j], _rid = alist##_res[uu][_j]; \
                    if (_t == nt_old && _rid == nrid_old && uu == worst) continue; \
                    int _n = 0; \
                    for (int _u2 = 1; _u2 <= N; _u2++) \
                        for (int _k = 0; _k < al_cnt[_u2]; _k++) \
                            if (alist##_sub[_u2][_k] == _t && alist##_res[_u2][_k] == _rid) _n++; \
                    if (_t == nt_old && _rid == nrid_old) _n--; \
                    if (_t == nt_new && _rid == nrid_new) _n++; \
                    double _r = rate_of(uu, _t, _n); \
                    _tran += _r < (buf[uu] - _tran) ? _r : (buf[uu] - _tran); \
                } \
                _tran < buf[uu] ? _tran : buf[uu]; \
            })

            /* Try swapping each of worst user's resources to a better one */
            int best_ot = 0, best_orid = 0, best_nt = 0, best_nrid = 0;
            double best_gain = 0.0;
            for (int j = 0; j < al_cnt[worst]; j++) {
                int ot = al_sub[worst][j], orid = al_res[worst][j];
                for (int t = 1; t <= T; t++) {
                    if (beams_cnt[t] == 0) continue;
                    for (int j2 = 0; j2 < rsub_sz[t]; j2++) {
                        int nrid = rsub[t][j2];
                        int has = 0;
                        for (int k = 0; k < al_cnt[worst]; k++)
                            if (al_sub[worst][k] == t && al_res[worst][k] == nrid) { has = 1; break; }
                        if (has) continue;

                        /* Delta: affected = worst + co-users on ot/orid + co-users on t/nrid */
                        double old_sum = 0.0, new_sum = 0.0;
                        /* worst user */
                        old_sum += user_tput(worst, al, 0,0,0,0);
                        new_sum += user_tput(worst, al, ot,orid,t,nrid);
                        /* co-users on old resource */
                        for (int cu = 1; cu <= N; cu++) {
                            if (cu == worst) continue;
                            int on_old = 0, on_new = 0;
                            for (int k = 0; k < al_cnt[cu]; k++) {
                                if (al_sub[cu][k] == ot && al_res[cu][k] == orid) on_old = 1;
                                if (al_sub[cu][k] == t && al_res[cu][k] == nrid) on_new = 1;
                            }
                            if (on_old || on_new) {
                                old_sum += user_tput(cu, al, 0,0,0,0);
                                new_sum += user_tput(cu, al, ot,orid,t,nrid);
                            }
                        }
                        double delta = new_sum - old_sum;
                        if (delta > best_gain) {
                            best_gain = delta;
                            best_ot = ot; best_orid = orid;
                            best_nt = t; best_nrid = nrid;
                        }
                    }
                }
            }
            if (best_gain > 1e-9) {
                /* Apply swap */
                for (int j = 0; j < al_cnt[worst]; j++) {
                    if (al_sub[worst][j] == best_ot && al_res[worst][j] == best_orid) {
                        al_sub[worst][j] = best_nt;
                        al_res[worst][j] = best_nrid;
                        break;
                    }
                }
                /* Update best if improved (full recalc) */
                double newT = 0.0;
                for (int i = 1; i <= N; i++) {
                    double tran = 0.0;
                    for (int j = 0; j < al_cnt[i]; j++) {
                        int ti = al_sub[i][j], ri = al_res[i][j];
                        int n = 0;
                        for (int u2 = 1; u2 <= N; u2++)
                            for (int k = 0; k < al_cnt[u2]; k++)
                                if (al_sub[u2][k] == ti && al_res[u2][k] == ri) n++;
                        double r = rate_of(i, ti, n);
                        tran += r < (buf[i] - tran) ? r : (buf[i] - tran);
                    }
                    newT += tran < buf[i] ? tran : buf[i];
                }
                if (newT > bestT) { bestT = newT; save_best(newT, &bestT); }
            } else break;
        }

    }

    /* 输出：T 行子带波束 + N 行用户资源 */
    for (int t = 1; t <= T; t++) {
        printf("%d", best_beams_cnt[t]);
        for (int j = 0; j < best_beams_cnt[t]; j++) printf(" %d", best_beams_list[t][j]);
        printf("\n");
    }
    for (int i = 1; i <= N; i++) {
        printf("%d", best_al_cnt[i]);
        for (int j = 0; j < best_al_cnt[i]; j++) printf(" %d %d", best_al_sub[i][j], best_al_res[i][j]);
        printf("\n");
    }
    return 0;
}
