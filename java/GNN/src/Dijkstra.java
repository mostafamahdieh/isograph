import java.util.ArrayList;


public class Dijkstra {
	private Graph G;
	public ArrayList<Integer> seen = new ArrayList<Integer>();
	public boolean[] mark;
	public float[] dist;
	public float[][] Dt; 
	
	public Dijkstra(Graph G) {
		this.G = G;
		mark = new boolean[G.size()];
		dist = new float[G.size()];
	}
	
	public void run(int source, int neededNum, int M) {
		for (int i = 0; i < G.size(); ++i) {
			mark[i] = false;
			dist[i] = Float.POSITIVE_INFINITY;
		}
		seen.clear();
		dist[source] = 0.0f;
		for (int t = 0; t < neededNum; ) {
			float minDist = Float.POSITIVE_INFINITY;
			int minDistPos = -1;
			for (int u = 0; u < G.size(); ++u) {
				if (!mark[u] && dist[u] < minDist) {
					minDist = dist[u];
					minDistPos = u;
				}
			}
			if (minDistPos == -1) {
				throw new RuntimeException("Graph does not have enough vertices in one of components");
			}
			mark[minDistPos] = true;
			if (minDistPos != source && minDistPos < M) {				
				seen.add(minDistPos);
				t++;
			}
			for (int i = 0; i < G.negh.get(minDistPos).size(); ++i) {
				int u = G.negh.get(minDistPos).get(i);
				float w = G.distance.get(minDistPos).get(i); 
				if (!mark[u] && minDist+w < dist[u]) {
					dist[u] = minDist+w;
//					System.out.println("update" + u + " to " + w);
				}
			}
		}
	}
}
