package svmModel;

import common.data.DataPoint;
import common.data.DatasetLoader;
import common.evaluation.ResultsAnalyzer;
import common.evaluation.ConfusionMatrix;
import java.util.List;

/**
 * Main class for SVM Handwritten Digit Classification
 * Targets UCI SVM baseline: 96.884% - 98.275%
 */
public class MainSvm {
    /**
     * Main entry point for SVM handwritten digit classification system
     */
    public static void main(String[] args) {
        System.out.println("SVM - Handwritten Digit Classification");
        System.out.println("Target: UCI SVM Baseline (96.884% - 98.275%)");
        System.out.println("===========================================\n");
        
        try {
            // Load datasets
            DatasetLoader loader = new DatasetLoader();
            List<DataPoint> dataset1 = loader.loadDataset("datasets/dataSet1.csv");
            List<DataPoint> dataset2 = loader.loadDataset("datasets/dataSet2.csv");

            
            // Comprehensive SVM testing
            testComprehensiveSVM(dataset1, dataset2);
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Comprehensive SVM testing across multiple kernel types and hyperparameters
     */
    private static void testComprehensiveSVM(List<DataPoint> dataset1, 
                                           List<DataPoint> dataset2) {
        System.out.println("\n===========================================");
        System.out.println("Finding Optimal Configuration (Two-fold Average)");
        System.out.println("-".repeat(60));
        
        // Test different configurations 
        SVMConfig[] configs = {
            //linear kernel configurations
            new SVMConfig("linear", 1.0, 0.01, 2),
            new SVMConfig("linear", 10.0, 0.01, 2),
            // Sigmoid kernel configurations
            new SVMConfig("sigmoid", 1.0, 0.01, 2),
            new SVMConfig("sigmoid", 10.0, 0.01, 2),
            new SVMConfig("sigmoid", 100.0, 0.01, 2),
            // RBF kernel configurations
            new SVMConfig("rbf", 1.0, 0.010, 2),
            new SVMConfig("rbf", 10.0, 0.001, 2),
            new SVMConfig("rbf", 1.0, 0.015, 2),   // Higher gamma
            new SVMConfig("rbf", 1.0, 0.020, 2),   // Even higher gamma
            // Polynomial kernel configurations
            new SVMConfig("poly", 1.0, 0.01, 2),
            new SVMConfig("poly", 10.0, 0.01, 3),
            
            
        };
        
        String bestConfig = "";
        double bestAccuracy = 0;
        SVMConfig bestConfigObj = null;
        int configNumber = 1;
        
        // Iterate through all configurations
        for (SVMConfig config : configs) {
            // Test current configuration
            double[] results = testSVMConfigurationSummarized(dataset1, dataset2, config);
            double accuracy1 = results[0];
            double accuracy2 = results[1];
            double averageAccuracy = results[2];
            
            // Display summarized results in the exact format
            System.out.printf("Kernel: %-7s, C=%-4.1f", config.kernel, config.C);
            
            // Add kernel-specific parameters with proper spacing
            if (config.kernel.equals("rbf")) {
                System.out.printf(", gamma=%.3f,  ", config.gamma);
            } else if (config.kernel.equals("poly")) {
                System.out.printf(", degree=%d, ", config.degree);
            } else {
                System.out.print(", ".repeat(2)); // Padding for alignment
            }
            
            System.out.printf("Fold1: %6.2f%%, ", accuracy1);
            System.out.printf("Fold2: %6.2f%%, ", accuracy2);
            System.out.printf("Avg: %6.2f%%", averageAccuracy);
            
            // Track best performing configuration
            if (averageAccuracy > bestAccuracy) {
                bestAccuracy = averageAccuracy;
                bestConfig = String.format("%s kernel, C=%.1f", config.kernel, config.C);
                if (config.kernel.equals("rbf")) {
                    bestConfig += String.format(", gamma=%.3f", config.gamma);
                } else if (config.kernel.equals("poly")) {
                    bestConfig += String.format(", degree=%d", config.degree);
                }
                bestConfigObj = config;
                System.out.print(" ← NEW BEST");
                
                // Check against baseline
                if (averageAccuracy >= 96.884) {
                    System.out.print(" ✓");
                }
            }
            
            System.out.println();
        }
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("BEST CONFIGURATION FOUND");
        System.out.println("=".repeat(60));
        System.out.printf("Configuration: %s\n", bestConfig);
        System.out.printf("Average Accuracy: %.2f%%\n", bestAccuracy);
        System.out.printf("UCI SVM Baseline: 96.884%% - 98.275%%\n");
        
        if (bestAccuracy >= 96.884) {
            if (bestAccuracy >= 98.275) {
                System.out.println("Exceeds UCI SVM baseline!");
            } else {
                System.out.println("Within UCI SVM baseline range!");
            }
        } else {
            System.out.printf("Below baseline by %.2f%%\n", 96.884 - bestAccuracy);
        }
        
        // Show detailed analysis for best configuration
        if (bestConfigObj != null && bestAccuracy > 50.0) {
            System.out.println("\n" + "=".repeat(60));
            System.out.println("CONFUSION MATRICES FOR BEST CONFIGURATION:");
            System.out.println("=".repeat(60));
            
            showDetailedAnalysis(dataset1, dataset2, bestConfigObj, bestAccuracy);
        }
    }
    
    /**
     * Tests a specific SVM configuration and returns results
     * @return array [accuracy1, accuracy2, averageAccuracy]
     */
    private static double[] testSVMConfigurationSummarized(List<DataPoint> dataset1,
                                                          List<DataPoint> dataset2,
                                                          SVMConfig config) {
        // Fold 1
        double accuracy1 = performSingleFoldQuiet(dataset1, dataset2, config);
        
        // Fold 2
        double accuracy2 = performSingleFoldQuiet(dataset2, dataset1, config);
       
        // Calculate average
        double averageAccuracy = (accuracy1 + accuracy2) / 2;
        
        return new double[]{accuracy1, accuracy2, averageAccuracy};
    }
    
    /**
     * Performs single fold quietly (no console output)
     */
    private static double performSingleFoldQuiet(List<DataPoint> trainingData,
                                               List<DataPoint> testData,
                                               SVMConfig config) {
        // Create and train SVM classifier
        SvmClassifier classifier = new SvmClassifier(
            trainingData, config.kernel, config.C, config.gamma, config.degree
        );
        
        // Classify all test points
        int correct = 0;
        for (DataPoint testPoint : testData) {
            int predictedLabel = classifier.classify(testPoint);
            if (predictedLabel == testPoint.getActualLabel()) {
                correct++;
            }
        }
        
        return (double) correct / testData.size() * 100;
    }
    
    /**
     * Shows detailed analysis for the best configuration
     */
    private static void showDetailedAnalysis(List<DataPoint> dataset1,
                                           List<DataPoint> dataset2,
                                           SVMConfig config,
                                           double bestAccuracy) {
        // Fold 1 detailed analysis
        System.out.println("\nFOLD 1: Train on dataset1, Test on dataset2");
        double acc1 = performSingleFoldWithMatrix(dataset1, dataset2, config);
        
        // Fold 2 detailed analysis
        System.out.println("\nFOLD 2: Train on dataset2, Test on dataset1");
        double acc2 = performSingleFoldWithMatrix(dataset2, dataset1, config);
        
        /*
        System.out.println("\n" + "-".repeat(60));
        System.out.println("COMBINED PERFORMANCE SUMMARY");
        System.out.println("-".repeat(60));
        System.out.printf("Fold 1 Accuracy: %.2f%%\n", acc1);
        System.out.printf("Fold 2 Accuracy: %.2f%%\n", acc2);
        System.out.printf("Average Accuracy: %.2f%%\n", bestAccuracy); */
 
    }
    
    /**
     * Performs single fold with confusion matrix display
     */
    private static double performSingleFoldWithMatrix(List<DataPoint> trainingData,
                                                     List<DataPoint> testData,
                                                     SVMConfig config) {
        // Create and train SVM classifier
        SvmClassifier classifier = new SvmClassifier(
            trainingData, config.kernel, config.C, config.gamma, config.degree
        );
        
        // Classify all test points
        int correct = 0;
        for (DataPoint testPoint : testData) {
            int predictedLabel = classifier.classify(testPoint);
            testPoint.setPredictedLabel(predictedLabel);
            if (predictedLabel == testPoint.getActualLabel()) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / testData.size() * 100;
        System.out.printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, testData.size());
        
        // Show confusion matrix

        ConfusionMatrix cm = new ConfusionMatrix(10);
        for (DataPoint point : testData) {
            cm.addPrediction(point.getActualLabel(), point.getPredictedLabel());
        }
        cm.printMatrix();
        
        return accuracy;
    }
    
    /**
     * Configuration container class for SVM hyperparameters
     */
    private static class SVMConfig {
        String kernel;
        double C;
        double gamma;
        int degree;
        
        SVMConfig(String kernel, double C, double gamma, int degree) {
            this.kernel = kernel;
            this.C = C;
            this.gamma = gamma;
            this.degree = degree;
        }
    }
}