import java.util.ArrayList;

public class LabelPropagation {

	float[][] labels;
	float[][] temp;
	int[] correctLabelsIndices;
	float[][] correctLabels;
	int size;
	int labelsSize;
	Graph G;
	
	public LabelPropagation(Graph G, float[][] correctLabels, int[] correctLabelsIndices)
	{
		this.G = G;
		size = G.size();
		this.correctLabels = correctLabels;
		this.correctLabelsIndices = correctLabelsIndices;
		labelsSize = correctLabels.length;
		labels = new float[size][size];
		temp = new float[size][size];
	}

	private float propagateSingle() {
		float sumChange = 0.0f;
		for (int v = 0; v < size; ++v) {
			ArrayList<Integer> neghV = G.negh.get(v);
			ArrayList<Float> similarityV = G.distance.get(v);
			for (int l = 0; l < labelsSize; ++l) {
				float sumS = 0;
				float sum = 0;
				for (int j = 0; j < neghV.size(); ++j) {
					int u = neghV.get(j);
					float s = similarityV.get(j);
					sum += labels[l][u] * s;
					sumS += s;
				}
				temp[v][l] = sum / sumS;
			}
		}
		for (int l = 0; l < labelsSize; ++l)
			for (int i = 0; i < correctLabelsIndices.length; ++i)
				temp[l][correctLabelsIndices[i]] = correctLabels[l][i];
		
		for (int l = 0; l < labelsSize; ++l)
			for (int v = 0; v < size; ++v) {
				float val = labels[l][v] - temp[l][v];
				sumChange += val*val;
				labels[l][v] = temp[l][v];
			}
		
		return (float)Math.sqrt((double)(sumChange / (labelsSize * size)));
	}
	
	public void propagateMultiple(int times) {
		for (int i = 0; i < times; ++i)
			propagateSingle();
	}
	
	public void propagateUntilConvergance(float meanSquaredChangeThreshold) {
		while (true) {
			float msc = propagateSingle();
			if (msc < meanSquaredChangeThreshold)
				return;
		}
	}
}
