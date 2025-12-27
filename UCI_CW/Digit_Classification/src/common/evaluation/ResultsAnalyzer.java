package common.evaluation;
import common.data.DataPoint;

import java.util.List;

/**
 * Handles result analysis and performance metrics for digit classification
 * Calculates accuracy and generates confusion matrices for model evaluation
 */
public class ResultsAnalyzer {
    
	/**
     * Analyzes classification results and displays performance metrics
     * @param testData list of test data points with actual and predicted labels
     * @param algorithm name/description of the algorithm being evaluated
     */
    public void analyzeResults(List<DataPoint> testData, String algorithm) {
    	// Initialize confusion matrix for 10-digit classification
        ConfusionMatrix cm = new ConfusionMatrix(10); 
        
        // Populate confusion matrix with all predictions
        for (DataPoint point : testData) {
            cm.addPrediction(point.getActualLabel(), point.getPredictedLabel());
        }
        
        // Calculate performance metrics
        int correct = cm.getCorrectPredictions();
        int total = cm.getTotalPredictions();
        double accuracy = (double) correct / total * 100;
        
        // Display confusion matrix
        cm.printMatrix();
    }

}
