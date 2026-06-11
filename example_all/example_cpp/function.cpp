#include <iostream>
#include <vector>
#include "function.h"

struct SubBandBeamAlloc {
    int sub_band_idx;
    int beam_count;
    std::vector<int> beams;
};

struct UserResAlloc {
    int user_idx;
    int res_count;
    std::vector<int> sub_bands;
    std::vector<int> res_ids;
};

void solve() {
    int B, U, K, T, beamMaxNum;
    int M;
    
    std::cin >> B >> U >> K >> T >> beamMaxNum;
    std::cin >> M;
    
    std::vector<std::vector<int>> MU(M);
    for (int i = 0; i < M; i++) {
        int num;
        std::cin >> num;
        MU[i].resize(num);
        for (int j = 0; j < num; j++) {
            std::cin >> MU[i][j];
        }
    }
    
    int su_count;
    std::cin >> su_count;
    std::vector<int> SU(su_count);
    for (int i = 0; i < su_count; i++) {
        std::cin >> SU[i];
    }
    
    std::vector<std::vector<float>> CAP(U + 1, std::vector<float>(B));
    for (int i = 1; i <= U; i++) {
        for (int j = 0; j < B; j++) {
            std::cin >> CAP[i][j];
        }
    }
    
    std::vector<int> buffer(U + 1);
    std::vector<float> SINR(U + 1);
    for (int i = 1; i <= U; i++) {
        std::cin >> buffer[i] >> SINR[i];
    }
    
    std::vector<std::vector<int>> RES_SUB(T + 1);
    for (int i = 1; i <= T; i++) {
        int res_count;
        std::cin >> res_count;
        RES_SUB[i].resize(res_count);
        for (int j = 0; j < res_count; j++) {
            std::cin >> RES_SUB[i][j];
        }
    }
    
    std::vector<SubBandBeamAlloc> sub_band_allocs;
    for (int i = 0; i < T; i++) {
        int beam_idx = (i % B) + 1;
        SubBandBeamAlloc alloc;
        alloc.sub_band_idx = i + 1;
        alloc.beam_count = 1;
        alloc.beams.push_back(beam_idx);
        sub_band_allocs.push_back(alloc);
    }
    
    for (int i = 0; i < T; i++) {
        std::cout << sub_band_allocs[i].beam_count;
        for (int j = 0; j < sub_band_allocs[i].beam_count; j++) {
            std::cout << " " << sub_band_allocs[i].beams[j];
        }
        std::cout << std::endl;
    }
    
    std::vector<UserResAlloc> user_allocs;
    int res_idx = 1;
    for (int user_idx = 1; user_idx <= U; user_idx++) {
        UserResAlloc alloc;
        alloc.user_idx = user_idx;
        alloc.res_count = 0;
        
        if (user_idx <= U && res_idx <= K) {
            for (int sub_idx = 1; sub_idx <= T; sub_idx++) {
                bool found = false;
                for (int j = 0; j < RES_SUB[sub_idx].size(); j++) {
                    if (RES_SUB[sub_idx][j] == res_idx) {
                        alloc.res_count = 1;
                        alloc.sub_bands.push_back(sub_idx);
                        alloc.res_ids.push_back(res_idx);
                        res_idx++;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }
        user_allocs.push_back(alloc);
    }
    
    for (int i = 0; i < U; i++) {
        std::cout << user_allocs[i].res_count;
        for (int j = 0; j < user_allocs[i].res_count; j++) {
            std::cout << " " << user_allocs[i].sub_bands[j] << " " << user_allocs[i].res_ids[j];
        }
        std::cout << std::endl;
    }
}
