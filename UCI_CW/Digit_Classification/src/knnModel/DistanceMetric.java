package knnModel;

/**
 * Provides distance metric calculations for k-Nearest Neighbors algorithm
 * @param features1 first feature vector (64 dimensions for digit data)
 * @param features2 second feature vector (64 dimensions for digit data)
 */
public class DistanceMetric {
	/**
     * Calculates Euclidean distance between two feature vectors
     * @return Euclidean distance between the two vectors
     */
    public static double euclideanDistance(double[] features1, double[] features2) { 	
        double sqDifferenceSum = 0.0;
        
        // Sum squared differences for each feature dimension
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
            double featureDiff = features1[featureIndex] - features2[featureIndex];
            sqDifferenceSum += featureDiff * featureDiff;
        }
        // Apply Pythagorean theorem for n-dimensional space
        return Math.sqrt(sqDifferenceSum);
    }
    
    /**
     * Calculates Manhattan distance between two feature vectors
     * @return Manhattan distance between the two vectors
     */
    public static double manhattanDistance(double[] features1, double[] features2) {
        double absoluteDiffsum = 0.0;
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
        	absoluteDiffsum += Math.abs(features1[featureIndex] - features2[featureIndex]);
        }
        return absoluteDiffsum;
    }
    
    /**
     * Calculates weighted Euclidean distance: Each feature is multiplied by a weight indicating its importance
     * @param weights array of weights for each feature dimension
     * @return weighted Euclidean distance
     */
    public static double weightedEuclideanDistance(double[] features1, double[] features2, double[] weights) {
    	 validateArrayLengths(features1, features2, weights);
    	 
        double weightedSqSum = 0.0;
        
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
            double featureDiff = features1[featureIndex] - features2[featureIndex];
            weightedSqSum += weights[featureIndex] * featureDiff * featureDiff;
        }
        return Math.sqrt(weightedSqSum);
    }
    
    /**
     * Calculates weighted Manhattan distance
     * @param weights array of weights for each feature dimension
     * @return weighted Manhattan distance
     */
    public static double weightedManhattanDistance(double[] features1, double[] features2, double[] weights) {
    	 validateArrayLengths(features1, features2, weights);
        
    	 double weightedAbsoluteSum = 0.0;
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
        	weightedAbsoluteSum += weights[featureIndex] * Math.abs(features1[featureIndex] - features2[featureIndex]);
        }
        return weightedAbsoluteSum;
    }
    
    /**
     * Validates that all input arrays have the same length
     * 
     * @param array1 first array to check
     * @param array2 second array to check
     * @param array3 third array to check
     * @throws IllegalArgumentException if arrays have mismatched lengths
     */
    private static void validateArrayLengths(double[] array1, double[] array2, double[] array3) {
        if (array1.length != array2.length || array1.length != array3.length) {
            throw new IllegalArgumentException(
                "Array length mismatch. Expected all arrays to have length " + array1.length +
                ". Got lengths: " + array1.length + ", " + array2.length + ", " + array3.length
            );
        }
    }
}