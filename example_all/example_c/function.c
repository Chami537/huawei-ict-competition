#include <stdio.h>
#include "function.h"

typedef struct {
    int sub_band_idx;
    int beam_count;
    int beams[32];
} SubBandBeamAlloc;

typedef struct {
    int user_idx;
    int res_count;
    int sub_bands[36];
    int res_ids[36];
} UserResAlloc;

void solve() {
    int B, U, K, T, beamMaxNum;
    int M;
    
    scanf("%d %d %d %d %d", &B, &U, &K, &T, &beamMaxNum);
    scanf("%d", &M);
    
    int MU[8][10];
    for (int i = 0; i < M; i++) {
        int num;
        scanf("%d", &num);
        for (int j = 0; j < num; j++) {
            scanf("%d", &MU[i][j]);
        }
    }
    
    int SU[30];
    int su_count;
    scanf("%d", &su_count);
    for (int i = 0; i < su_count; i++) {
        scanf("%d", &SU[i]);
    }
    
    float CAP[51][32];
    for (int i = 1; i <= U; i++) {
        for (int j = 0; j < B; j++) {
            scanf("%f", &CAP[i][j]);
        }
    }
    
    int buffer[51];
    float SINR[51];
    for (int i = 1; i <= U; i++) {
        scanf("%d %f", &buffer[i], &SINR[i]);
    }
    
    int RES_SUB[19][36];
    int RES_SUB_SIZE[19];
    for (int i = 1; i <= T; i++) {
        scanf("%d", &RES_SUB_SIZE[i]);
        for (int j = 0; j < RES_SUB_SIZE[i]; j++) {
            scanf("%d", &RES_SUB[i][j]);
        }
    }
    
    SubBandBeamAlloc sub_band_allocs[18];
    for (int i = 0; i < T; i++) {
        int beam_idx = (i % B) + 1;
        sub_band_allocs[i].sub_band_idx = i + 1;
        sub_band_allocs[i].beam_count = 1;
        sub_band_allocs[i].beams[0] = beam_idx;
    }
    
    for (int i = 0; i < T; i++) {
        printf("%d", sub_band_allocs[i].beam_count);
        for (int j = 0; j < sub_band_allocs[i].beam_count; j++) {
            printf(" %d", sub_band_allocs[i].beams[j]);
        }
        printf("\n");
    }
    
    UserResAlloc user_allocs[50];
    int res_idx = 1;
    for (int user_idx = 1; user_idx <= U; user_idx++) {
        user_allocs[user_idx - 1].user_idx = user_idx;
        user_allocs[user_idx - 1].res_count = 0;
        
        if (user_idx <= U && res_idx <= K) {
            for (int sub_idx = 1; sub_idx <= T; sub_idx++) {
                int found = 0;
                for (int j = 0; j < RES_SUB_SIZE[sub_idx]; j++) {
                    if (RES_SUB[sub_idx][j] == res_idx) {
                        user_allocs[user_idx - 1].res_count = 1;
                        user_allocs[user_idx - 1].sub_bands[0] = sub_idx;
                        user_allocs[user_idx - 1].res_ids[0] = res_idx;
                        res_idx++;
                        found = 1;
                        break;
                    }
                }
                if (found) break;
            }
        }
    }
    
    for (int i = 0; i < U; i++) {
        printf("%d", user_allocs[i].res_count);
        for (int j = 0; j < user_allocs[i].res_count; j++) {
            printf(" %d %d", user_allocs[i].sub_bands[j], user_allocs[i].res_ids[j]);
        }
        printf("\n");
    }
}
