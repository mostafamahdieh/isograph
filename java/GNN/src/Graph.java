import java.util.ArrayList;


public class Graph {
	private int size;
	public ArrayList<ArrayList<Integer>> negh;
	public ArrayList<ArrayList<Float>> distance;
	public Graph(int size) {
		this.size = size;
		negh = new ArrayList<ArrayList<Integer>>(size);
		distance = new ArrayList<ArrayList<Float>>(size); 
		for (int i = 0; i < size; ++i) {
			negh.add(i, new ArrayList<Integer>());
			distance.add(i, new ArrayList<Float>());
		}
	}
	public int size() {
		return size;
	}
	public void addEdge(int src, int dest, float dist) {
		for (int i =0;i < negh.get(src).size(); ++i) {
			if (negh.get(src).get(i) == dest)
				return;
		}
		negh.get(src).add(dest);
		distance.get(src).add(dist);
	}
}
