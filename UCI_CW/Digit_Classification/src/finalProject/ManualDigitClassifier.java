package finalProject;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
/**
 * Combined main method for running both k-NN and SVM handwritten digit classification
 * Performs two-fold testing and displays confusion matrices for each model
 */
public class ManualDigitClassifier {
    
	//========================MAIN METHOD=======================//
    // Tracking variables for k-NN best configuration
    private static int bestOverallK = 0;
    private static String bestOverallMetric = null;
    private static boolean bestOverallNormalization = false;
    private static double bestOverallAccuracy = -1.0;
    
    // Constants for SVM evaluation
    private static final double KNN_UCI_BASELINE = 98.25;
    private static final double SVM_UCI_BASELINE_MIN = 96.884;
    private static final double SVM_UCI_BASELINE_MAX = 98.275;
    private static final int NUMBER_OF_DIGIT_CLASSES = 10;
    private static final int MINIMUM_K_VALUE = 1;
    
    /*MAIN METHOD*/
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("COMBINED HANDWRITTEN DIGIT CLASSIFICATION: k-NN & SVM");
        System.out.println("=".repeat(70));
        
        try {
            // Load datasets
            //DatasetLoader dataLoader = new DatasetLoader();
            //List<DataPoint> firstDataset = dataLoader.loadDataset("datasets/dataSet1.csv");
            //List<DataPoint> secondDataset = dataLoader.loadDataset("datasets/dataSet2.csv");
            
            System.out.println("\n" + "=".repeat(70));
            System.out.println("PART 1: k-NEAREST NEIGHBORS CLASSIFICATION");
            System.out.println("=".repeat(70));
            
            // Run k-NN classification with optimal configuration finding
            //runKnnClassification(firstDataset, secondDataset);
            
            System.out.println("\n\n" + "=".repeat(70));
            System.out.println("PART 2: SUPPORT VECTOR MACHINE CLASSIFICATION");
            System.out.println("=".repeat(70));
            
            // Run SVM classification with optimal configuration finding
            //runSvmClassification(firstDataset, secondDataset);
            
            System.out.println("\n" + "=".repeat(70));
            System.out.println("CLASSIFICATION COMPLETE");
            System.out.println("=".repeat(70));
            
        } catch (Exception exception) {
            System.err.println("Error: " + exception.getMessage());
            exception.printStackTrace();
        }
    }
    
    /**
     * Runs k-NN classification with optimal configuration finding
     */
   
    
}
 