import java.util.*;

public class Function {
    static class SubBandBeamAlloc {
        int subBandIdx;
        int beamCount;
        int[] beams;
        
        SubBandBeamAlloc(int subBandIdx, int beamCount, int[] beams) {
            this.subBandIdx = subBandIdx;
            this.beamCount = beamCount;
            this.beams = beams;
        }
    }
    
    static class UserResAlloc {
        int userIdx;
        int resCount;
        int[] subBands;
        int[] resIds;
        
        UserResAlloc(int userIdx, int resCount, int[] subBands, int[] resIds) {
            this.userIdx = userIdx;
            this.resCount = resCount;
            this.subBands = subBands;
            this.resIds = resIds;
        }
    }
    
    public static void solve() {
        Scanner sc = new Scanner(System.in);
        
        int B = sc.nextInt();
        int U = sc.nextInt();
        int K = sc.nextInt();
        int T = sc.nextInt();
        int beamMaxNum = sc.nextInt();
        
        int M = sc.nextInt();
        
        int[][] MU = new int[M][];
        for (int i = 0; i < M; i++) {
            int num = sc.nextInt();
            MU[i] = new int[num];
            for (int j = 0; j < num; j++) {
                MU[i][j] = sc.nextInt();
            }
        }
        
        int suCount = sc.nextInt();
        int[] SU = new int[suCount];
        for (int i = 0; i < suCount; i++) {
            SU[i] = sc.nextInt();
        }
        
        float[][] CAP = new float[U + 1][B];
        for (int i = 1; i <= U; i++) {
            for (int j = 0; j < B; j++) {
                CAP[i][j] = sc.nextFloat();
            }
        }
        
        int[] buffer = new int[U + 1];
        float[] SINR = new float[U + 1];
        for (int i = 1; i <= U; i++) {
            buffer[i] = sc.nextInt();
            SINR[i] = sc.nextFloat();
        }
        
        int[][] RES_SUB = new int[T + 1][];
        for (int i = 1; i <= T; i++) {
            int resCount = sc.nextInt();
            RES_SUB[i] = new int[resCount];
            for (int j = 0; j < resCount; j++) {
                RES_SUB[i][j] = sc.nextInt();
            }
        }
        
        List<SubBandBeamAlloc> subBandAllocs = new ArrayList<>();
        for (int i = 0; i < T; i++) {
            int beamIdx = (i % B) + 1;
            SubBandBeamAlloc alloc = new SubBandBeamAlloc(i + 1, 1, new int[]{beamIdx});
            subBandAllocs.add(alloc);
        }
        
        for (int i = 0; i < T; i++) {
            SubBandBeamAlloc alloc = subBandAllocs.get(i);
            System.out.print(alloc.beamCount);
            for (int j = 0; j < alloc.beamCount; j++) {
                System.out.print(" " + alloc.beams[j]);
            }
            System.out.println();
        }
        
        List<UserResAlloc> userAllocs = new ArrayList<>();
        int resIdx = 1;
        for (int userIdx = 1; userIdx <= U; userIdx++) {
            int resCount = 0;
            int[] subBands = new int[0];
            int[] resIds = new int[0];
            
            if (userIdx <= U && resIdx <= K) {
                for (int subIdx = 1; subIdx <= T; subIdx++) {
                    boolean found = false;
                    for (int j = 0; j < RES_SUB[subIdx].length; j++) {
                        if (RES_SUB[subIdx][j] == resIdx) {
                            resCount = 1;
                            subBands = new int[]{subIdx};
                            resIds = new int[]{resIdx};
                            resIdx++;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
            }
            UserResAlloc alloc = new UserResAlloc(userIdx, resCount, subBands, resIds);
            userAllocs.add(alloc);
        }
        
        for (int i = 0; i < U; i++) {
            UserResAlloc alloc = userAllocs.get(i);
            System.out.print(alloc.resCount);
            for (int j = 0; j < alloc.resCount; j++) {
                System.out.print(" " + alloc.subBands[j] + " " + alloc.resIds[j]);
            }
            System.out.println();
        }
    }
}
