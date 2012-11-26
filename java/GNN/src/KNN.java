import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;


public class KNN {
	float[][] Dt;
	int N;
	
	public KNN(float[][] Dt, int N) {
		this.Dt = Dt;
		this.N = N;
	}
	public float dist(int u, int v) {
		float sum = 0.0f;
		for (int i = 0; i < Dt[u].length; ++i) {
			float d = Dt[u][i]-Dt[v][i];
			sum += d*d;
		}
		return (float) Math.sqrt(sum);
	}
	
	public Graph createGraph(int k) {
		Graph G = new Graph(N);
		float[] dists = new float[N];
		float[] distsCopy = new float[N];
		for (int v = 0; v < N; ++v) {
			for (int u = 0; u < N; ++u) {
				float d = dist(u, v);
				dists[u] = d;
				distsCopy[u] = d;
				//G.addEdge(src, dest, sim);
			}
			Arrays.sort(dists);
			int pos = 0;
			float distK = dists[k-1];
			for (int u = 0; u < N; ++u) {
				if (u != v && distsCopy[u] <= distK) {
					pos++;
					G.addEdge(u, v, distsCopy[u]);
					G.addEdge(v, u, distsCopy[u]);
//					System.out.println(distsCopy[u]);
				}
				if (pos == k)
					break;
			}
			
		}
		
		dfs(G);
		
		int cs = comps.size();
		System.out.println(cs + " components");
		
		
		if (cs == 1)
			return G;
		
		float dist[][] = new float[cs][cs];
		int dU[][] = new int[cs][cs], dV[][] = new int[cs][cs];
		
		for (int c1 = 0; c1 < comps.size(); ++c1) {
			ArrayList<Integer> comp1 = comps.get(c1);
			for (int c2 = c1+1; c2 < comps.size(); ++c2) {
				ArrayList<Integer> comp2 = comps.get(c2);
				float minDist = Float.POSITIVE_INFINITY;
				int minDistU = 0, minDistV = 0;
				for (Integer u: comp1) {
					for (Integer v: comp2) {
						float d = dist(u, v);
						if (d < minDist) {
							minDist = d;
							minDistU = u;
							minDistV = v;
						}
					}
				}
				dist[c1][c2] = dist[c2][c1] = minDist;
				dU[c1][c2] = dU[c2][c1] = minDistU;
				dV[c1][c2] = dV[c2][c1] = minDistV;
			}
		}
		
		int[] parMST = new int[N];
		boolean[] markMST = new boolean[N];
		float[] distMST = new float[N];
		
		for (int v = 0; v < cs; ++v)
			distMST[v] = Float.POSITIVE_INFINITY;
		
		distMST[0] = 0;
		
		int edgesAdded = 0;
			
		for (int t = 0; t < cs; ++t) {
			//System.out.println("t: " + t);
			float leastDist = Float.POSITIVE_INFINITY;
			int b = -1;
			for (int u = 0; u < cs; ++u) {
				if (distMST[u] <= leastDist && !markMST[u]) {
					leastDist = distMST[u];
					b = u;
				}
			}
			//System.out.println("b: " + b);
			if (b != 0) {
				int u = dU[b][parMST[b]], v = dV[b][parMST[b]];
				float d = dist[b][parMST[b]];
				G.addEdge(u, v, d);
				G.addEdge(v, u, d);
				System.out.println("Adding edge " + u + " " + v + " to graph");

				edgesAdded ++;
			}
			markMST[b] = true;
			for (int u = 0; u < cs; ++u) {
				if (!markMST[u] && distMST[b] + dist[b][u] <= distMST[u]) {
					distMST[u] = distMST[b] + dist[b][u];
					parMST[u] = b;
				}
			}
		}
		
		System.out.println("Added " + edgesAdded + " edges to kNN graph");
		
		return G;
	}
	
	boolean mark[];
	
	void dfs(Graph G) {
		mark = new boolean[N];
		comps = new ArrayList<ArrayList<Integer>>();
		Stack<Integer> S = new Stack<Integer>();
		for (int u = 0; u < N; ++u) {
			if (!mark[u]) {
				currComp = new ArrayList<Integer>();
				S.push(u);
				while (!S.empty()) {
					int u0 = S.pop();
					currComp.add(u0);
					mark[u0] = true;
					for (int i = 0; i < G.negh.get(u0).size(); ++i) {
						int v = G.negh.get(u0).get(i);
						if (!mark[v]) {
							S.push(v);
						}
					}
				}
				comps.add(currComp);
			}
		}
	}
	
	ArrayList<ArrayList<Integer>> comps;
	ArrayList<Integer> currComp;
	
}
