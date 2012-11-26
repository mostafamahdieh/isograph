import java.io.File;
import java.io.PrintStream;
import java.util.Random;
import java.util.Scanner;


public class GnnNoExtraPoints {
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		String basePath = "gnn";
		String dataset = "MNIST";
		
		int pc = 35;
		int kSmall = 4; 
		int kBig = 20;
		int points = 1;
		boolean globalPCA = false;
		int subsetN = 1000; 

		Scanner scanner;
		String subsetFilePath = basePath + "\\" + subsetN + "\\" + dataset + "_subset_" + subsetN + ".txt";
		scanner = new Scanner(new File(subsetFilePath));
		int R = scanner.nextInt();
		int M = scanner.nextInt();
		assert(M == subsetN);
		
		int[][] subset = new int[R][M];
		for (int j = 0; j < R; ++j) {
			for (int i = 0; i < M; ++i) {
				subset[j][i] = scanner.nextInt()-1;
			}
		}
		scanner.close();


		for (int r = 0; r < R; ++r) {
			if (globalPCA)
				scanner = new Scanner(new File(basePath + "\\" + dataset + "_PCA_Dt_" + pc + ".txt"));
			else
				scanner = new Scanner(new File(basePath + "\\" + subsetN + "\\" + dataset + "_PCA_Dt_" + pc + "_" + (r+1) + ".txt"));

			int N = scanner.nextInt();
			int DI = scanner.nextInt();			
			
			float[][] Dt = new float[N][];
			for (int i = 0; i < N; ++i) {
				Dt[i] = new float[DI];
				for (int j = 0; j < DI; ++j) {
					Dt[i][j] = scanner.nextFloat();
				}
			}
			scanner.close();
						
			Random rg = new Random(1390);
			int perm[] = new int[N];
			
			for (int i = 0; i < N; ++i) {
				int v = rg.nextInt(i+1);
				if (v == i)
					perm[i] = i;
				else {
					perm[i] = perm[v];
					perm[v] = i;
				}
			}
			
			boolean[] has = new boolean[N];
			for (int i = 0; i < N; ++i) {
				assert (!has[perm[i]]);
				has[perm[i]] = true;
			}
			
			int pos = 0;			
			for (int i = M; i < N; ++i) {
				while (inSubset[perm[pos]])
					pos++;
				Dt[i] = DtCopy[perm[pos]];
				pos++;
			}
			while (pos < N && inSubset[perm[pos]])
				pos++;
//			System.out.println("pos: " + pos);
			assert(pos == N);
						
			for (int ip = 1; ip <= points; ++ip) {
				int n = N*ip/points;
				KNN kNN = new KNN(Dt, n);
				Graph G = kNN.createGraph(kSmall);
				System.out.println("r: " + r + " ip: " + ip);
				
				String outputPath;
				if (points > 1)
					outputPath = basePath + "\\" + subsetN + "\\" + dataset + "_" + M + "_" + String.valueOf(kBig) + "_" + (r+1) + "_" + ip + ".txt";
				else
					outputPath = basePath + "\\" + subsetN + "\\" + dataset + "_" + M + "_" + String.valueOf(kBig) + "_" + (r+1) + ".txt";

				PrintStream outputPS = new PrintStream(new File(outputPath));

				
				Dijkstra dij = new Dijkstra(G);
							
				Graph GM = new Graph(M);
				for (int iu = 0; iu < M; ++iu) {
 					dij.run(iu, kBig, M);
					for (int i = 0; i < kBig; ++i) {
						GM.addEdge(iu, dij.seen.get(i), dij.dist[dij.seen.get(i)]);
						GM.addEdge(dij.seen.get(i), iu, dij.dist[dij.seen.get(i)]);
					}
				}
			
				for (int iu = 0; iu < M; ++iu) {
					for (int i = 0; i < GM.negh.get(iu).size(); ++i) {
						outputPS.println((iu+1) + " " + (GM.negh.get(iu).get(i)+1) + " " + GM.distance.get(iu).get(i));
					}
				}
				outputPS.close();
			}
		}
	}
}