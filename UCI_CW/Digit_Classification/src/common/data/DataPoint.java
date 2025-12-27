package common.data;

/**
 * Represents a single handwritten digit data point from UCI dataset
 * Each data point consists of 64 features (8x8 grid) and a digit label (0-9)
 */
public class DataPoint {
    private double[] features;  // 64 feature values (0-16 pixel intensities)
    private int actualLabel; // True digit label (0-9)
    private int predictedLabel; // Model-predicted digit label (0-9)
    
    /**
     * Creates a new data point with features and true label
     */
    public DataPoint(double[] features, int label) {
        this.features = features;
        this.actualLabel = label;
    }
    
    // Accessor methods
    public double[] getFeatures() { return features; }
    public int getActualLabel() { return actualLabel; }
    public int getPredictedLabel() { return predictedLabel; }
    
    /**
     * Sets the model's prediction for this data point
     */
    public void setPredictedLabel(int predictedLabel) { 
        this.predictedLabel = predictedLabel; 
    }
}