package knnModel;

import common.data.DataPoint;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Comparator;

/**
 * k-Nearest Neighbors classifier for  handwritten digit recognition
 * Implements the baseline algorithm from UCI repository using Euclidean distance
 */
public class KnnClassifier {
	
	// Training dataset used for classification
    private List<DataPoint> trainingData;
    private boolean normalizeFeatures;
    private boolean useWeightedVoting;
    private double[] minVals;
    private double[] maxVals;
    
    // Distance metric to use (EUCLIDEAN_DISTANCE_METRIC or MANHATTAN_DISTANCE_METRIC)
    private String distanceMetric;
    
    /**
     * Constructor with training data only, uses default Euclidean distance metric
     * @param trainingData the dataset to train the classifier
     * @param distanceMetric the distance metric to use **/
    
    public KnnClassifier(List<DataPoint> trainingData) {
        this.trainingData = trainingData; //  trainingData the dataset to train the classifier
        this.distanceMetric = "euclidean"; //default Euclidean distance metric
        this.normalizeFeatures = false;
        this.useWeightedVoting = false;
    }
    
    //Constructor with training data and specified distance metric
    public KnnClassifier(List<DataPoint> trainingData, String distanceMetric) {
    	 this.trainingData = trainingData;
         this.distanceMetric = distanceMetric;
         this.normalizeFeatures = false;
         this.useWeightedVoting = false;
    }
    
    /**
     * Enhanced constructor with all configuration options
     * @param normalizeFeatures whether to normalize feature values
     * @param useWeightedVoting whether to use weighted voting
     */
    public KnnClassifier(List<DataPoint> trainingData, String distanceMetric, 
                         boolean normalizeFeatures, boolean useWeightedVoting) {
        this.trainingData = trainingData;
        this.distanceMetric = distanceMetric;
        this.normalizeFeatures = normalizeFeatures;
        this.useWeightedVoting = useWeightedVoting;
        
        if (normalizeFeatures && !trainingData.isEmpty()) {
            computeNormalizationParams();
        }
    }
    
    /**
     * Compute min and max for each feature for normalization
     */
    private void computeNormalizationParams() {
        int numFeatures = trainingData.get(0).getFeatures().length;
        minVals = new double[numFeatures];
        maxVals = new double[numFeatures];
        
        // Initialize with extreme values
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            minVals[featureIndex] = Double.MAX_VALUE;
            maxVals[featureIndex] = Double.MIN_VALUE;
        }
        
        // Find min and max for each feature
        for (DataPoint trainingPoint : trainingData) {
            double[] features = trainingPoint.getFeatures();
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                if (features[featureIndex] < minVals[featureIndex]) minVals[featureIndex] = features[featureIndex];
                if (features[featureIndex] > maxVals[featureIndex]) maxVals[featureIndex] = features[featureIndex];
            }
        }
    }
    
    /**
     * Normalize a feature vector using pre-computed min/max values
     * @param features the feature vector to normalize
     */
    private double[] normalizeFeatures(double[] features) {
        double[] normalizedFeatures = new double[features.length];
        for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
            double range = maxVals[featureIndex] - minVals[featureIndex];
            if (range == 0) range = 1; // Avoid division by zero for constant features
            normalizedFeatures[featureIndex] = (features[featureIndex] - minVals[featureIndex]) / range;
        }
        return normalizedFeatures; //the normalized feature vector
    }
    
    /**
     * Classifies a test point using k-Nearest Neighbors algorithm
     * @param testPoint the data point to classify
     * @param numOfNeighbors the number of nearest neighbors to consider
     * @return the predicted digit label (0-9)
     */
    public int classify(DataPoint testPoint, int numOfNeighbors) {
    	
    	// List to store all neighbors with their distances
        List<Neighbor> neighbors = new ArrayList<>();
        
        // Calculate distance to every point in training set
        for (DataPoint trainPoint : trainingData) {
            double distance = calculateDistance(
                testPoint.getFeatures(), 
                trainPoint.getFeatures()
            );
            neighbors.add(new Neighbor(distance, trainPoint.getActualLabel()));
        }
        // Sort neighbors by distance (ascending - nearest first)
        Collections.sort(neighbors, new DistanceComparator());
        
        // Count votes from k nearest neighbors
        int[] votes = new int[10];
        for (int neighborIndex = 0; neighborIndex < numOfNeighbors && neighborIndex < neighbors.size(); neighborIndex++) {
            int label = neighbors.get(neighborIndex).label;
            votes[label]++;
        }
        
        return getMajorityVote(votes); // Return the label with majority votes
    }
    
	/**
	 * Calculates distance between two feature vectors
	 * @return the calculated distance
	 */
    private double calculateDistance(double[] features1, double[] features2) {
        switch (distanceMetric.toLowerCase()) {
            case "manhattan":
                return DistanceMetric.manhattanDistance(features1, features2);
            case "euclidean":
            default:
                return DistanceMetric.euclideanDistance(features1, features2);
        }
    }
    
    /**
     * Finds the label with the most votes using simple majority
     * @param votes array containing vote counts for each digit
     * @param neighbors list of sorted neighbors
     * @param numberOfNeighbors number of neighbors to consider
     * @return the digit label with the most votes
     */
    private int getMajorityVote(int[] votes) {
        int maxVotes = -1;
        int predictedLabel = 0;
        // Count votes from k nearest neighbors
        for (int digitLabel = 0; digitLabel < votes.length; digitLabel++) {
            if (votes[digitLabel] > maxVotes) {
                maxVotes = votes[digitLabel];
                predictedLabel = digitLabel;
            }
        }
        return predictedLabel; // the digit label with highest votes
    }
    
    /**
     * Weighted voting where closer neighbors have more influence
     */
    private int getWeightedMajorityVote(List<Neighbor> neighbors, int k) {
        double[] weightedVotes = new double[10];
        for (int neighbourIndex = 0; neighbourIndex < k && neighbourIndex < neighbors.size(); neighbourIndex++) {
            Neighbor n = neighbors.get(neighbourIndex);
            double weight = 1.0 / (n.distance + 0.0001);
            weightedVotes[n.label] += weight;
        }
        
        // Find label with maximum votes
        int predictedLabel = 0;
        double maxWeight = -1;
        for (int digitLabel = 0; digitLabel < weightedVotes.length; digitLabel++) {
            if (weightedVotes[digitLabel] > maxWeight) {
                maxWeight = weightedVotes[digitLabel];
                predictedLabel = digitLabel;
            }
        }
        return predictedLabel;
    }
    
    /**
     * Inner class to store neighbor information (distance and label)
     */
    private class Neighbor {
        double distance; // Distance to test point
        int label; // Digit label of this neighbor
        Neighbor(double distance, int label) {
            this.distance = distance; // Distance to test point
            this.label = label; // Digit label of this neighbor
        }
    }
    
    /**
     * Comparator to sort neighbors by distance (nearest first)
     */
    private class DistanceComparator implements Comparator<Neighbor> {
        @Override
        public int compare(Neighbor n1, Neighbor n2) {
            return Double.compare(n1.distance, n2.distance);
        }
    }
}


