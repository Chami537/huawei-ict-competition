/*
 * 2026 华为智联杯·无线程序设计大赛
 * 赛题二：通信资源联合分配 —— 主程序 (C, gcc 11.4.0 / -O2 / x86_64 Linux)
 *
 * 架构 (v9 — 沙盘评估器重构):
 *   Phase 1: 6 种波束策略 → 种子波束配置生成器 (Seed Generators)
 *   Phase 2: evaluate_plan() 纯函数沙盘评估 (无副作用)
 *   Phase 3: [待接入] SA 在 beam 空间变异 → evaluate_plan() 评估
 *
 * 6 种策略（按全局预排序取 top-k 波束，扫描 A=1..Amax 启用子带数）:
 *   S0: compute_pop       — 全局流行度 sum CAP
 *   S1: compute_marginal  — buffer × 相对容量加权
 *   S2: compute_balanced  — 截断 cap 0.3 防单波束垄断
 *   S3: compute_mupot     — MU 组内最佳用户平方
 *   S4: compute_mubeam    — MU 组内均值
 *   S5: compute_suonly    — 非 MU 用户加权
 *
 * 不使用任何第三方库；自带 log10 实现以避免对 libm 的链接依赖。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Simple xorshift RNG for deterministic reproducible mutations */
static unsigned int rng_state = 2463534242u;
static unsigned int xorshift(void) __attribute__((unused));
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
    double y = (x - 1.0) / (x + 1.0);
    double y2 = y * y, term = y, sum = 0.0;
    for (int k = 1; k <= 25; k += 2) { sum += term / k; term *= y2; }
    static const double LN2 = 0.6931471805599453;
    return 2.0 * sum + e * LN2;
}
static const double INV_LN10 = 0.43429448190325176;
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
static double CAP[MAXN][MAXP];
static double Sall[MAXN];
static int buf[MAXN];
static double sinr[MAXN];
static int rsub[MAXT][MAXK], rsub_sz[MAXT];
static int res2sub[MAXK];

static int is_su[MAXN + 1];      /* 1 = SU 独占用户 */
static int mu_group[MAXN + 1];   /* MU 组 ID (0..M-1), -1 = 非 MU */

static int pop_order[MAXP];      /* 波束按策略排序 */
static double db_share[12];      /* lin2db(1/s), s=1..10 */

/* ---------- 波束配置 (BeamConfig) ---------- */
typedef struct {
    int cnt[MAXT + 1];
    int beams[MAXT + 1][MAXP];
} BeamConfig;

/* ---------- 波束增益缓存 (由 compute_gain 填充) ---------- */
static double G[MAXN][MAXT];

/* ---------- 策略: 波束排序 ---------- */
static void sort_beams_by_score(double *score) {
    for (int p = 1; p <= P; p++) pop_order[p - 1] = p;
    for (int a = 0; a < P; a++)
        for (int b = a + 1; b < P; b++)
            if (score[pop_order[b]] > score[pop_order[a]]) {
                int t = pop_order[a]; pop_order[a] = pop_order[b]; pop_order[b] = t;
            }
}

/* S0: popularity — sum of raw CAP across all users */
static void compute_pop(void) {
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int i = 1; i <= N; i++) score[p] += CAP[i][p];
    }
    sort_beams_by_score(score);
}

/* S1: marginal urgency — buffer-weighted relative capacity */
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

/* S2: balanced — cap per-user contribution to spread beams */
static void compute_balanced(void) {
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int i = 1; i <= N; i++) {
            double rel = (Sall[i] > 1e-9) ? CAP[i][p] / Sall[i] : 0.0;
            score[p] += rel < 0.3 ? rel : 0.3;
        }
    }
    sort_beams_by_score(score);
}

/* S3: MU sharing potential — reward beams with many high-cap users across RUs */
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
            score[p] += best * best;
        }
    }
    sort_beams_by_score(score);
}

/* S4: MU-group weighted — average per-RU contribution, fallback to global */
static void compute_mubeam(void) {
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int m = 0; m < M; m++) {
            double gs = 0.0; int cnt = 0;
            for (int j = 0; j < ru_sz[m]; j++) {
                int u = ru[m][j];
                if (buf[u] <= 0 || Sall[u] < 1e-9) continue;
                gs += buf[u] * CAP[u][p] / Sall[u];
                cnt++;
            }
            if (cnt > 0) score[p] += gs / cnt;
        }
        if (score[p] <= 1e-9) {
            for (int u = 1; u <= N; u++)
                score[p] += (Sall[u] > 1e-9) ? CAP[u][p] / Sall[u] : 0.0;
        }
    }
    sort_beams_by_score(score);
}

/* S5: SU-only weighted — reward beams preferred by non-MU users */
static void compute_suonly(void) {
    static int isMU[MAXN + 1];
    memset(isMU, 0, sizeof(isMU));
    for (int m = 0; m < M; m++)
        for (int j = 0; j < ru_sz[m]; j++)
            isMU[ru[m][j]] = 1;
    double score[MAXP];
    for (int p = 1; p <= P; p++) {
        score[p] = 0.0;
        for (int u = 1; u <= N; u++) {
            if (isMU[u] || buf[u] <= 0 || Sall[u] < 1e-9) continue;
            score[p] += buf[u] * CAP[u][p] / Sall[u];
        }
        if (score[p] <= 1e-9) {
            for (int u = 1; u <= N; u++)
                score[p] += (Sall[u] > 1e-9) ? CAP[u][p] / Sall[u] : 0.0;
        }
    }
    sort_beams_by_score(score);
}

/* ---------- 子带排序（按资源块数降序） ---------- */
static void subband_order(int *order) {
    for (int t = 0; t < T; t++) order[t] = t + 1;
    for (int a = 0; a < T; a++)
        for (int b = a + 1; b < T; b++)
            if (rsub_sz[order[b]] > rsub_sz[order[a]]) {
                int tmp = order[a]; order[a] = order[b]; order[b] = tmp;
            }
}

/* ---------- 波束分配：启用前 A 个(res 最多)子带，均摊预算 ---------- */
static void alloc_beams(BeamConfig *cfg, int A) {
    for (int t = 1; t <= T; t++) cfg->cnt[t] = 0;
    if (A < 1) A = 1;
    if (A > T) A = T;
    if (A > beamMax) A = beamMax;
    int order[MAXT]; subband_order(order);
    int base = beamMax / A; if (base > P) base = P; if (base < 1) base = 1;
    int rcnt = beamMax - base * A; if (rcnt < 0) rcnt = 0;
    for (int idx = 0; idx < A; idx++) {
        int t = order[idx];
        int k = base + (idx < rcnt ? 1 : 0); if (k > P) k = P;
        cfg->cnt[t] = k;
        for (int j = 0; j < k; j++) cfg->beams[t][j] = pop_order[j];
    }
}

/* ---------- 计算波束增益缓存 ---------- */
static void compute_gain(const BeamConfig *cfg) {
    for (int i = 1; i <= N; i++)
        for (int t = 1; t <= T; t++) {
            if (cfg->cnt[t] == 0 || Sall[i] <= 0.0) { G[i][t] = -1e18; continue; }
            double s = 0.0;
            for (int j = 0; j < cfg->cnt[t]; j++) s += CAP[i][cfg->beams[t][j]];
            G[i][t] = (s > 0.0) ? (lin2db(s) - lin2db(Sall[i])) : -1e18;
        }
}

static double rate_of(int i, int t, int s) {
    if (G[i][t] <= -1e17) return 0.0;
    return (double)cap_rate(sinr[i] + db_share[s] + G[i][t]);
}

/* ---------- 资源驱动贪心：单资源块的最佳配置 ---------- */
static double best_config(int t, double *rem, int *chosen, double *cr, int *ncho) {
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
        for (int j = 0; j < ru_sz[m]; j++) {
            int u = ru[m][j]; if (rem[u] > 1e-9) av[na++] = u;
        }
        if (na < 2) continue;
        int smax = na < 10 ? na : 10;
        for (int s = 2; s <= smax; s++) {
            double contrib[12]; int uid[12];
            for (int j = 0; j < na; j++) {
                int u = av[j]; double r = rate_of(u, t, s);
                contrib[j] = rem[u] < r ? rem[u] : r; uid[j] = u;
            }
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

/* ═══════════════════════════════════════════════════════════════════════
 * evaluate_plan — 纯函数沙盘评估器
 *
 * 输入: BeamConfig* (波束配置)
 * 输出: 总传输量 T (double)
 * 副作用: 无 (仅修改局部变量 + 全局 G[][] 缓存)
 *
 * 若 out_al 非 NULL，同时输出用户-资源分配方案。
 * ═══════════════════════════════════════════════════════════════════════ */
static double evaluate_plan(const BeamConfig *cfg,
                            int *out_al_cnt, int (*out_al_sub)[MAXK], int (*out_al_res)[MAXK]) {
    double rem[MAXN + 1];
    int al_cnt[MAXN + 1];
    int al_sub[MAXN + 1][MAXK];
    int al_res[MAXN + 1][MAXK];

    /* 绝对状态隔离: memset 全量清零，杜绝跨调用残留 */
    memset(rem, 0, sizeof(rem));
    memset(al_cnt, 0, sizeof(al_cnt));
    memset(al_sub, 0, sizeof(al_sub));
    memset(al_res, 0, sizeof(al_res));

    for (int i = 1; i <= N; i++) { rem[i] = (double)buf[i]; }

    compute_gain(cfg);

    /* 资源按所属子带波束数降序处理 */
    int reslist[MAXK], rcnt = 0;
    for (int t = 1; t <= T; t++)
        for (int j = 0; j < rsub_sz[t]; j++) reslist[rcnt++] = rsub[t][j];
    for (int a = 0; a < rcnt; a++)
        for (int b = a + 1; b < rcnt; b++)
            if (cfg->cnt[res2sub[reslist[b]]] > cfg->cnt[res2sub[reslist[a]]]) {
                int tmp = reslist[a]; reslist[a] = reslist[b]; reslist[b] = tmp;
            }

    for (int a = 0; a < rcnt; a++) {
        int res = reslist[a], t = res2sub[res];
        if (cfg->cnt[t] == 0) continue;
        int chosen[12]; double cr[12]; int nch;
        double v = best_config(t, rem, chosen, cr, &nch);
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

    /* 若请求输出分配方案 */
    if (out_al_cnt) {
        for (int i = 1; i <= N; i++) {
            out_al_cnt[i] = al_cnt[i];
            for (int j = 0; j < al_cnt[i]; j++) {
                out_al_sub[i][j] = al_sub[i][j];
                out_al_res[i][j] = al_res[i][j];
            }
        }
    }

    return Tval;
}

/* ═══════════════════════════════════════════════════════════════════════
 * generate_seed — 用当前 pop_order 扫描 A=1..Amax，返回最佳 BeamConfig
 * ═══════════════════════════════════════════════════════════════════════ */
static double generate_seed(BeamConfig *best_cfg,
                            int *out_al_cnt, int (*out_al_sub)[MAXK], int (*out_al_res)[MAXK]) {
    int Amax = T < beamMax ? T : beamMax;
    if (Amax < 1) Amax = 1;
    double best_score = -1.0;

    for (int A = 1; A <= Amax; A++) {
        BeamConfig cfg;
        alloc_beams(&cfg, A);
        double score = evaluate_plan(&cfg, NULL, NULL, NULL);
        if (score > best_score) {
            best_score = score;
            *best_cfg = cfg;
        }
    }

    /* 用最佳配置 + 独立输出数组跑最终评估，杜绝前 6 次穿透污染 */
    if (out_al_cnt) {
        int tmp_cnt[MAXN + 1];
        int tmp_sub[MAXN + 1][MAXK];
        int tmp_res[MAXN + 1][MAXK];
        best_score = evaluate_plan(best_cfg, tmp_cnt, tmp_sub, tmp_res);
        for (int i = 1; i <= N; i++) {
            out_al_cnt[i] = tmp_cnt[i];
            for (int j = 0; j < tmp_cnt[i]; j++) {
                out_al_sub[i][j] = tmp_sub[i][j];
                out_al_res[i][j] = tmp_res[i][j];
            }
        }
    }

    return best_score;
}

/* ═══════════════════════════════════════════════════════════════════════
 * local_search — 原版逻辑移植（user_tput 宏 + O(co-users) delta + 时钟守卫）
 *
 * 与 b4edd83 原版逐逻辑对齐，仅适配 BeamConfig 结构。
 * ═══════════════════════════════════════════════════════════════════════ */
static double local_search(const BeamConfig *cfg, clock_t t0,
                           int *al_cnt, int (*al_sub)[MAXK], int (*al_res)[MAXK]) {
    int cocount[MAXT + 1][MAXK + 1];
    double bestT = 0.0;

    /* Build cocount + initial score */
    memset(cocount, 0, sizeof(cocount));
    for (int u = 1; u <= N; u++)
        for (int j = 0; j < al_cnt[u]; j++)
            cocount[al_sub[u][j]][al_res[u][j]]++;

    for (int u = 1; u <= N; u++) {
        if (al_cnt[u] == 0) continue;
        double t = 0.0;
        for (int j = 0; j < al_cnt[u]; j++) {
            int sb = al_sub[u][j], rid = al_res[u][j];
            double r = rate_of(u, sb, cocount[sb][rid]);
            t += r < (buf[u] - t) ? r : (buf[u] - t);
        }
        bestT += t < buf[u] ? t : buf[u];
    }

    #define USER_TPUT(uu, nt_old, nrid_old, nt_new, nrid_new) ({ \
        double _tran = 0.0; \
        for (int _j = 0; _j < al_cnt[uu]; _j++) { \
            int _t = al_sub[uu][_j], _rid = al_res[uu][_j]; \
            if (_t == nt_old && _rid == nrid_old && uu == worst) continue; \
            int _n = cocount[_t][_rid]; \
            if (_t == nt_old && _rid == nrid_old) _n--; \
            if (_t == nt_new && _rid == nrid_new) _n++; \
            double _r = rate_of(uu, _t, _n); \
            _tran += _r < (buf[uu] - _tran) ? _r : (buf[uu] - _tran); \
        } \
        _tran < buf[uu] ? _tran : buf[uu]; \
    })

    int worst = 0;
    for (int iter = 0; iter < 100; iter++) {
        if ((double)(clock() - t0) / CLOCKS_PER_SEC > 0.090) break;

        /* Find worst user */
        worst = -1;
        double worst_ratio = 2.0;
        for (int u = 1; u <= N; u++) {
            if (buf[u] <= 0 || al_cnt[u] == 0) continue;
            double tran = 0.0;
            for (int j = 0; j < al_cnt[u]; j++) {
                int t = al_sub[u][j], rid = al_res[u][j];
                double r = rate_of(u, t, cocount[t][rid]);
                tran += r < (buf[u] - tran) ? r : (buf[u] - tran);
            }
            double ratio = tran / (double)buf[u];
            if (ratio < worst_ratio) { worst_ratio = ratio; worst = u; }
        }
        if (worst < 0) break;

        int best_ot = 0, best_orid = 0, best_nt = 0, best_nrid = 0;
        double best_gain = 0.0;

        for (int j = 0; j < al_cnt[worst]; j++) {
            int ot = al_sub[worst][j], orid = al_res[worst][j];
            for (int t = 1; t <= T; t++) {
                if (cfg->cnt[t] == 0) continue;
                for (int j2 = 0; j2 < rsub_sz[t]; j2++) {
                    int nrid = rsub[t][j2];
                    int has = 0;
                    for (int k = 0; k < al_cnt[worst]; k++)
                        if (al_sub[worst][k] == t && al_res[worst][k] == nrid)
                            { has = 1; break; }
                    if (has) continue;

                    /* SU/MU 硬约束壁垒 */
                    if (cocount[t][nrid] > 0) {
                        int rep = 0;
                        for (int cu = 1; cu <= N && !rep; cu++)
                            for (int k = 0; k < al_cnt[cu]; k++)
                                if (al_sub[cu][k] == t && al_res[cu][k] == nrid)
                                    { rep = cu; break; }
                        if (is_su[rep] || is_su[worst]) continue;
                        if (mu_group[rep] >= 0 && mu_group[worst] >= 0 &&
                            mu_group[rep] != mu_group[worst]) continue;
                    }

                    /* Delta: worst + co-users on old/new (original logic) */
                    double old_sum = 0.0, new_sum = 0.0;
                    old_sum += USER_TPUT(worst, 0,0,0,0);
                    new_sum += USER_TPUT(worst, ot,orid,t,nrid);
                    for (int cu = 1; cu <= N; cu++) {
                        if (cu == worst) continue;
                        int on_old = 0, on_new = 0;
                        for (int k = 0; k < al_cnt[cu]; k++) {
                            if (al_sub[cu][k] == ot && al_res[cu][k] == orid) on_old = 1;
                            if (al_sub[cu][k] == t && al_res[cu][k] == nrid) on_new = 1;
                        }
                        if (on_old || on_new) {
                            old_sum += USER_TPUT(cu, 0,0,0,0);
                            new_sum += USER_TPUT(cu, ot,orid,t,nrid);
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

        if (best_gain <= 1e-9) break;

        /* Apply swap */
        for (int j = 0; j < al_cnt[worst]; j++) {
            if (al_sub[worst][j] == best_ot && al_res[worst][j] == best_orid) {
                al_sub[worst][j] = best_nt;
                al_res[worst][j] = best_nrid;
                break;
            }
        }
        cocount[best_ot][best_orid]--;
        cocount[best_nt][best_nrid]++;

        /* Full recalc to update bestT */
        double newT = 0.0;
        for (int i = 1; i <= N; i++) {
            double tran = 0.0;
            for (int j = 0; j < al_cnt[i]; j++) {
                int ti = al_sub[i][j], ri = al_res[i][j];
                double r = rate_of(i, ti, cocount[ti][ri]);
                tran += r < (buf[i] - tran) ? r : (buf[i] - tran);
            }
            newT += tran < buf[i] ? tran : buf[i];
        }
        if (newT > bestT) bestT = newT;
    }

    #undef USER_TPUT
    return bestT;
}

/* ═══════════════════════════════════════════════════════════════════════
 * main — 6 种子评估 → 取最高分 → local search → 输出
 * ═══════════════════════════════════════════════════════════════════════ */
int main(void) {
    if (scanf("%d %d %d %d %d", &P, &N, &K, &T, &beamMax) != 5) return 0;
    scanf("%d", &M);
    for (int i = 1; i <= N; i++) { is_su[i] = 0; mu_group[i] = -1; }
    for (int m = 0; m < M; m++) {
        scanf("%d", &ru_sz[m]);
        for (int j = 0; j < ru_sz[m]; j++) {
            scanf("%d", &ru[m][j]);
            mu_group[ru[m][j]] = m;
        }
    }
    int su_sz; scanf("%d", &su_sz);
    for (int j = 0; j < su_sz; j++) { int x; scanf("%d", &x); is_su[x] = 1; }

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

    /* 6 种波束策略 */
    void (*strategies[6])(void) = {compute_pop, compute_marginal, compute_balanced,
                                   compute_mupot, compute_mubeam, compute_suonly};

    BeamConfig best_cfg;
    int best_al_cnt[MAXN + 1];
    int best_al_sub[MAXN + 1][MAXK];
    int best_al_res[MAXN + 1][MAXK];
    double bestT = -1.0;

    /* Phase 1: 种子生成 + Phase 2: 沙盘评估 */
    for (int s = 0; s < 6; s++) {
        strategies[s]();  /* 设置 pop_order */

        BeamConfig cfg;
        int al_cnt[MAXN + 1];
        int al_sub[MAXN + 1][MAXK];
        int al_res[MAXN + 1][MAXK];

        double score = generate_seed(&cfg, al_cnt, al_sub, al_res);

        if (score > bestT) {
            bestT = score;
            best_cfg = cfg;
            for (int i = 1; i <= N; i++) {
                best_al_cnt[i] = al_cnt[i];
                for (int j = 0; j < al_cnt[i]; j++) {
                    best_al_sub[i][j] = al_sub[i][j];
                    best_al_res[i][j] = al_res[i][j];
                }
            }
        }
    }

    /* 输出：T 行子带波束 + N 行用户资源 */
    for (int t = 1; t <= T; t++) {
        printf("%d", best_cfg.cnt[t]);
        for (int j = 0; j < best_cfg.cnt[t]; j++) printf(" %d", best_cfg.beams[t][j]);
        printf("\n");
    }
    for (int i = 1; i <= N; i++) {
        printf("%d", best_al_cnt[i]);
        for (int j = 0; j < best_al_cnt[i]; j++)
            printf(" %d %d", best_al_sub[i][j], best_al_res[i][j]);
        printf("\n");
    }

    return 0;
}
