package mlpModel;

import common.data.DataPoint;
import common.data.DatasetLoader;
import common.evaluation.ResultsAnalyzer;
import java.util.List;

/**
 * Main class for MLP Handwritten Digit Classification
 * Uses the fixed MlpClassifier with proper softmax + cross-entropy
 */
public class MainMlp {
    
    public static void main(String[] args) {
        System.out.println("MLP - Handwritten Digit Classification (Fixed)");
        
        try {
            DatasetLoader loader = new DatasetLoader();
            List<DataPoint> dataset1 = loader.loadDataset("datasets/dataSet1.csv");
            List<DataPoint> dataset2 = loader.loadDataset("datasets/dataSet2.csv");
            
            // Test different MLP configurations
            testMlpConfigurations(dataset1, dataset2);
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testMlpConfigurations(List<DataPoint> dataset1, 
                                            List<DataPoint> dataset2) {
        System.out.println("=== MLP CONFIGURATION TESTING ===");
        
        // Test different architectures and learning rates
        MlpConfig[] configs = {
            new MlpConfig("64-32-10", 0.01, 100),   // Simple 1 hidden layer
            new MlpConfig("64-64-10", 0.01, 100),   // Medium hidden layer
            new MlpConfig("64-128-10", 0.01, 100),  // Large hidden layer
            new MlpConfig("64-128-64-10", 0.01, 100), // 2 hidden layers
            new MlpConfig("64-32-10", 0.05, 100),   // Higher learning rate
            new MlpConfig("64-64-10", 0.05, 100),   // Higher learning rate
        };
        
        String bestConfig = "";
        double bestAccuracy = 0;
        
        for (MlpConfig config : configs) {
            System.out.println("\n" + "=".repeat(50));
            System.out.printf("Testing: %s, LR=%.3f, Epochs=%d\n", 
                            config.architecture, config.learningRate, config.epochs);
            System.out.println("=".repeat(50));
            
            double accuracy = testMlpConfiguration(dataset1, dataset2, config);
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestConfig = config.architecture + " (LR=" + config.learningRate + ")";
            }
        }
        
        printFinalSummary(bestConfig, bestAccuracy);
    }
    
    private static double testMlpConfiguration(List<DataPoint> dataset1,
                                             List<DataPoint> dataset2,
                                             MlpConfig config) {
        // Fold 1
        System.out.println("\n--- FOLD 1: Train on dataset1, Test on dataset2 ---");
        double accuracy1 = performSingleFold(dataset1, dataset2, config);
        
        // Fold 2
        System.out.println("\n--- FOLD 2: Train on dataset2, Test on dataset1 ---");
        double accuracy2 = performSingleFold(dataset2, dataset1, config);
        
        double averageAccuracy = (accuracy1 + accuracy2) / 2;
        
        System.out.printf("\nAverage Accuracy: %.2f%%\n", averageAccuracy);
        System.out.printf("UCI k-NN Baseline: 98.25%%, SVM Baseline: 96.884%% - 98.275%%\n");
        
        // Compare with other model baselines
        if (averageAccuracy >= 90.0) {
            System.out.println("EXCELLENT: MLP performing well!");
        } else if (averageAccuracy >= 80.0) {
            System.out.println("GOOD: Solid neural network performance");
        } else if (averageAccuracy >= 50.0) {
            System.out.println("BASIC LEARNING: Network is learning patterns");
        } else {
            System.out.println("NEEDS WORK: Still debugging learning");
        }
        
        return averageAccuracy;
    }
    
    private static double performSingleFold(List<DataPoint> trainingData,
                                          List<DataPoint> testData,
                                          MlpConfig config) {
        long startTime = System.currentTimeMillis();
        
        // Use the fixed MlpClassifier with architecture parameter
        MlpClassifier classifier = new MlpClassifier(
            config.architecture, config.learningRate, config.epochs
        );
        
        classifier.train(trainingData);
        
        int correct = 0;
        for (DataPoint testPoint : testData) {
            int predictedLabel = classifier.classify(testPoint);
            testPoint.setPredictedLabel(predictedLabel);
            if (predictedLabel == testPoint.getActualLabel()) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / testData.size() * 100;
        long endTime = System.currentTimeMillis();
        
        System.out.printf("Accuracy: %.2f%% (%d/%d) - Time: %.1fs\n", 
                         accuracy, correct, testData.size(), (endTime - startTime) / 1000.0);
        
        return accuracy;
    }
    
    private static void printFinalSummary(String bestConfig, double bestAccuracy) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("FINAL MLP SUMMARY");
        System.out.println("=".repeat(60));
        System.out.printf("Best Configuration: %s\n", bestConfig);
        System.out.printf("Best Accuracy: %.2f%%\n", bestAccuracy);
        System.out.printf("Model Comparison:\n");
        System.out.printf("  k-NN: 98.11%% (your implementation)\n");
        System.out.printf("  SVM: 98.24%% (your implementation)\n");
        System.out.printf("  MLP: %.2f%% (this implementation)\n", bestAccuracy);
        
        if (bestAccuracy >= 90.0) {
            System.out.println("EXCELLENT: MLP performs competitively!");
        } else if (bestAccuracy >= 80.0) {
            System.out.println("GOOD: MLP learning complex patterns");
        } else if (bestAccuracy >= 50.0) {
            System.out.println("MODERATE: Basic learning achieved - room for improvement");
        } else {
            System.out.println("DEBUGGING NEEDED: Focus on getting basic learning first");
        }
        
    }
    
    private static class MlpConfig {
        String architecture;
        double learningRate;
        int epochs;
        
        MlpConfig(String architecture, double learningRate, int epochs) {
            this.architecture = architecture;
            this.learningRate = learningRate;
            this.epochs = epochs;
        }
    }
}