package knnModel;

import common.data.DataPoint;
import common.data.DatasetLoader;
import common.evaluation.ResultsAnalyzer;
import common.evaluation.ConfusionMatrix;
import java.util.List;

/**
 * Main class for k-NN HANDWRITTEN DIGIT CLASSIFICATION 
 * Comprehensive analysis with normalization capabilities  
 */
public class MainKnn {
	
	// Variables to store best configuration results
    private static int bestOverallK = 0;  // 0 means not set yet
    private static String bestOverallMetric = null;  // null means not set yet
    private static boolean bestOverallNormalization = false;
    private static double bestOverallAccuracy = -1.0;  // -1 means not set yet
    
    public static void main(String[] args) {
        System.out.println("kNN - Handwritten Digit Classification with Normalization");
        System.out.println("===========================================================\n");
        
        try {
            // Load datasets
            DatasetLoader loader = new DatasetLoader();
            List<DataPoint> dataset1 = loader.loadDataset("datasets/dataSet1.csv");
            List<DataPoint> dataset2 = loader.loadDataset("datasets/dataSet2.csv");
            
            // Perform comprehensive analysis with normalization
            performNormalizedAnalysis(dataset1, dataset2);
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Main analysis method with normalization
     */
    private static void performNormalizedAnalysis(List<DataPoint> dataset1, 
                                                              List<DataPoint> dataset2) {
        System.out.println("=== COMPREHENSIVE k-NN ANALYSIS WITH NORMALIZATION ===");
        
        // Test both distance metrics with normalization
        System.out.println("\n1. EUCLIDEAN DISTANCE WITH NORMALIZATION");
        analyzeWithNormalization(dataset1, dataset2, "euclidean");
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. MANHATTAN DISTANCE WITH NORMALIZATION");
        System.out.println("=".repeat(60));
        analyzeWithNormalization(dataset1, dataset2, "manhattan");
        
        // Find and display best configuration
        System.out.println("\n" + "=".repeat(60));
        System.out.println("BEST OVERALL CONFIGURATION");
        System.out.println("=".repeat(60));
        displayBestConfiguration(dataset1, dataset2);
    }
    
    /**
     * Analyzes a distance metric with normalization
     */
    private static void analyzeWithNormalization(List<DataPoint> dataset1,
                                                List<DataPoint> dataset2,
                                                String distanceMetric) {
        // Find optimal k with normalization using two-fold average
        int optimalK = findOptimalKTwoFold(dataset1, dataset2, distanceMetric, true);
        
        // Perform two-fold test with optimal k
        performTwoFoldTest(dataset1, dataset2, optimalK, distanceMetric, true);
    }
    
    /**
     * Finds optimal k using two-fold average
     */
    private static int findOptimalKTwoFold(List<DataPoint> dataset1,
                                         List<DataPoint> dataset2,
                                         String distanceMetric,
                                         boolean normalize) {
        System.out.println("\n--- Finding Optimal k (Two-Fold Average) ---");
        
        int bestK = 1;
        double bestAvgAccuracy = 0;
        int[] kValues = {1, 3, 5, 7}; // odd k-values
        
        for (int currentK : kValues) {
            double fold1Accuracy = testConfigurationSingleFold(dataset1, dataset2, currentK, distanceMetric, normalize);
            double fold2Accuracy = testConfigurationSingleFold(dataset2, dataset1, currentK, distanceMetric, normalize);
            double avgAccuracy = (fold1Accuracy + fold2Accuracy) / 2;
            
            // Track overall best configuration (initialize if first time or better)
            if (bestOverallAccuracy < 0 || avgAccuracy > bestOverallAccuracy) {
                bestOverallAccuracy = avgAccuracy;
                bestOverallK = currentK;
                bestOverallMetric = distanceMetric;
                bestOverallNormalization = normalize;
            }
            
            System.out.printf("  k=%2d: Fold1=%.4f%%, Fold2=%.4f%%, Avg=%.4f%%", 
            		currentK, fold1Accuracy, fold2Accuracy, avgAccuracy);
            
            if (avgAccuracy > bestAvgAccuracy) {
            	bestAvgAccuracy = avgAccuracy;
                bestK = currentK;
                System.out.print(" ← NEW BEST");
                
                if (avgAccuracy >= 98.25) {
                    System.out.print(" ✓ SURPASSES BASELINE!");
                }
            }
            System.out.println();
        }
        
        System.out.printf("\nOptimal k: %d (Average Accuracy: %.4f%%)\n", 
                         bestK, bestAvgAccuracy);
        return bestK;
    }
    
    /**
     * Performs two-fold test
     */
    private static void performTwoFoldTest(List<DataPoint> dataset1,
                                         List<DataPoint> dataset2,
                                         int currentK,
                                         String distanceMetric,
                                         boolean normalize) {
        System.out.println("\n--- Two-Fold Test with k=" + currentK + " ---");
        
        ResultsAnalyzer analyzer = new ResultsAnalyzer();
        
        // Fold 1
        System.out.println("\nFOLD 1: Train on dataset1, Test on dataset2");
        double accuracy1 = performFoldWithAnalysis(dataset1, dataset2, currentK, distanceMetric, normalize, analyzer);
        
        // Fold 2
        System.out.println("\nFOLD 2: Train on dataset2, Test on dataset1");
        double accuracy2 = performFoldWithAnalysis(dataset2, dataset1, currentK, distanceMetric, normalize, analyzer);
        
        // Calculate average
        double avgAccuracy = (accuracy1 + accuracy2) / 2;
        printAccuracySummary(avgAccuracy, "Average Accuracy");
    }
    
    /**
     * Performs a fold with full analysis and confusion matrix
     */
    private static double performFoldWithAnalysis(List<DataPoint> trainingData,
                                                List<DataPoint> testData,
                                                int currentK,
                                                String distanceMetric,
                                                boolean normalize,
                                                ResultsAnalyzer analyzer) {
        KnnClassifier classifier = new KnnClassifier(trainingData, distanceMetric, normalize, false);
        
        for (DataPoint testPoint : testData) {
            int predictedLabel = classifier.classify(testPoint, currentK);
            testPoint.setPredictedLabel(predictedLabel);
        }
        
        String algorithmDesc = String.format("k-NN (k=%d, %s, norm=%s)", currentK, distanceMetric, normalize);
        analyzer.analyzeResults(testData, algorithmDesc);
        
        return calculateAccuracy(testData);
    }
    
    /**
     * Displays the best configuration
     */
    private static void displayBestConfiguration(List<DataPoint> dataset1,
                                               List<DataPoint> dataset2) {
    	 // Check if we found a best configuration
        if (bestOverallAccuracy < 0) {
            System.out.println("Error: No configuration analyzed yet. Running default analysis...");
            // Run a quick analysis to get initial values
            findOptimalKTwoFold(dataset1, dataset2, "euclidean", false);
        }
        
        // Use the stored best configuration results
        int bestK = bestOverallK;
        String bestMetric = bestOverallMetric;
        boolean bestNorm = bestOverallNormalization;
        
        
        // Test the best configuration
        double fold1Acc = testConfigurationSingleFold(dataset1, dataset2, bestK, bestMetric, bestNorm);
        double fold2Acc = testConfigurationSingleFold(dataset2, dataset1, bestK, bestMetric, bestNorm);
        double bestAccuracy = (fold1Acc + fold2Acc) / 2;
        
        // Calculate total correct predictions
        int totalCorrect = (int)((fold1Acc / 100 * dataset2.size()) + (fold2Acc / 100 * dataset1.size()));
        int totalPoints = dataset1.size() + dataset2.size();
        
        // Display best configuration
        System.out.printf("Algorithm: k-NN (k=%d, %s, norm=%s)\n", bestK, bestMetric, bestNorm);
        System.out.printf("Average Correct: %d/%d\n", totalCorrect / 2, totalPoints / 2);
        System.out.printf("Average Accuracy: %.4f%%\n", bestAccuracy);
        System.out.printf("UCI Baseline: 98.2500%%\n");
        
        if (bestAccuracy >= 98.25) {
            System.out.printf("✓ SUCCESS! Surpassed baseline by +%.4f%%\n\n", bestAccuracy - 98.25);
            
            // Display confusion matrix from the better performing fold
            System.out.println("Confusion Matrix of best performing fold:");
            displayBestFoldConfusionMatrix(dataset1, dataset2, bestK, bestMetric, bestNorm);
        }
    }
    
    /**
     * Displays confusion matrix from the best performing fold
     */
    private static void displayBestFoldConfusionMatrix(List<DataPoint> dataset1,
                                                     List<DataPoint> dataset2,
                                                     int bestk,
                                                     String distanceMetric,
                                                     boolean normalize) {
        // Test both folds to see which one performs better
        KnnClassifier classifier1 = new KnnClassifier(dataset1, distanceMetric, normalize, false);
        int correct1 = 0;
        for (DataPoint testPoint : dataset2) {
            int predicted = classifier1.classify(testPoint, bestk);
            if (predicted == testPoint.getActualLabel()) {
                correct1++;
            }
        }
        double acc1 = (double) correct1 / dataset2.size() * 100;
        
        KnnClassifier classifier2 = new KnnClassifier(dataset2, distanceMetric, normalize, false);
        int correct2 = 0;
        for (DataPoint testPoint : dataset1) {
            int predicted = classifier2.classify(testPoint, bestk);
            if (predicted == testPoint.getActualLabel()) {
                correct2++;
            }
        }
        double acc2 = (double) correct2 / dataset1.size() * 100;
        
        // Use the better performing fold (Fold 2 in your case)
        if (acc2 >= acc1) {
            // Use Fold 2 confusion matrix (dataset2 → dataset1)
            classifier2 = new KnnClassifier(dataset2, distanceMetric, normalize, false);
            for (DataPoint testPoint : dataset1) {
                int predicted = classifier2.classify(testPoint, bestk);
                testPoint.setPredictedLabel(predicted);
            }
            printConfusionMatrix(dataset1);
        } else {
            // Use Fold 1 confusion matrix (dataset1 → dataset2)
            classifier1 = new KnnClassifier(dataset1, distanceMetric, normalize, false);
            for (DataPoint testPoint : dataset2) {
                int predicted = classifier1.classify(testPoint, bestk);
                testPoint.setPredictedLabel(predicted);
            }
            printConfusionMatrix(dataset2);
        }
    }
    
    /**
     * Prints a confusion matrix
     */
    private static void printConfusionMatrix(List<DataPoint> testData) {
        ConfusionMatrix cm = new ConfusionMatrix(10);
        
        for (DataPoint point : testData) {
            cm.addPrediction(point.getActualLabel(), point.getPredictedLabel());
        }
        
        cm.printMatrix();
    }
    
    /**
     * Tests a single fold (without analysis)
     */
    private static double testConfigurationSingleFold(List<DataPoint> trainingData,
                                                    List<DataPoint> testData,
                                                    int k,
                                                    String distanceMetric,
                                                    boolean normalize) {
        KnnClassifier classifier = new KnnClassifier(trainingData, distanceMetric, normalize, false);
        int correct = 0;
        for (DataPoint testPoint : testData) {
            int predicted = classifier.classify(testPoint, k);
            if (predicted == testPoint.getActualLabel()) {
                correct++;
            }
        }
        return (double) correct / testData.size() * 100;
    }
    
    /**
     * Calculates accuracy for a dataset
     */
    private static double calculateAccuracy(List<DataPoint> testData) {
        int correct = 0;
        for (DataPoint testPoint : testData) {
            if (testPoint.getPredictedLabel() == testPoint.getActualLabel()) {
                correct++;
            }
        }
        return (double) correct / testData.size() * 100;
    }
    
    /**
     * Prints accuracy summary
     */
    private static void printAccuracySummary(double accuracy, String label) {
        System.out.printf("\n%s: %.4f%%\n", label, accuracy);
        System.out.printf("UCI Baseline: 98.2500%%\n");
        
        if (accuracy >= 98.25) {
            System.out.printf("✓ SURPASSES BASELINE by +%.4f%%\n", accuracy - 98.25);
        } else {
            System.out.printf("Below baseline by %.4f%%\n", 98.25 - accuracy);
        }
    }
}