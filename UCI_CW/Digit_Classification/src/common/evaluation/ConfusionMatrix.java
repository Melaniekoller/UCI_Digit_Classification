package common.evaluation;

/**
 * Confusion matrix for digit classification
 * Tracks actual vs predicted classifications for digits 0-9
 *  Rows represent actual classes, columns represent predicted classes
 */

public class ConfusionMatrix {
    private int[][] matrix;  // matrix[actual][predicted] = count
    private int numClasses; // Number of digit classes (10 for digits 0-9)
    
    /**
     * Creates a new confusion matrix for specified number of classes
     */
    public ConfusionMatrix(int numClasses) {
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];  // Initialize all counts to 0
    }
    
    /**
     * Records a single prediction in the confusion matrix
     * @param actual true class label (0-9)
     * @param predicted predicted class label (0-9)
     */
    public void addPrediction(int actual, int predicted) {
        
     // Validate indices are within bounds
        if (actual >= 0 && actual < numClasses && predicted >= 0 && predicted < numClasses) {
            matrix[actual][predicted]++; // Increment count for this (actual, predicted) pair
        }
    }
    
    /**
     * Prints formatted confusion matrix to console
     * Shows actual vs predicted counts with row totals
    */ 
    
    public void printMatrix() {
        System.out.println("\nConfusion Matrix(Actual vs Predicted):");
        System.out.print("A\\P\t"); // Header: Actual vs Predicted
        
        // Print column headers (predicted classes)
        for (int classIndex = 0; classIndex < numClasses; classIndex++) {
            System.out.print(classIndex + "\t");
        }
        System.out.println("Total");  // Row total header
        
        // Print matrix rows
        for (int actualClass = 0; actualClass < numClasses; actualClass++) {
            System.out.print(actualClass + "\t"); // Row header (actual class)
            int rowTotal = 0;
            
            // Print counts for each predicted class
            for (int predictedClass = 0; predictedClass < numClasses; predictedClass++) {
                System.out.print(matrix[actualClass][predictedClass] + "\t");
                rowTotal += matrix[actualClass][predictedClass];
            }
            System.out.println(rowTotal); // Print row total
        }
    } 
    
    /**
     * Calculates total number of correct predictions (diagonal elements)
     * @return sum of diagonal entries (true positives for each class)
     */
    public int getCorrectPredictions() {
        int correct = 0;
        for (int classIndex = 0; classIndex < numClasses; classIndex++) {
            correct += matrix[classIndex][classIndex]; // Sum diagonal
        }
        return correct;
    }
    
    /**
     * Calculates total number of predictions made
     * @return sum of all matrix entries
     */
    public int getTotalPredictions() {
        int total = 0;
        for (int actualClass = 0; actualClass < numClasses; actualClass++) {
            for (int predictedClass = 0; predictedClass < numClasses; predictedClass++) {
                total += matrix[actualClass][predictedClass];
            }
        }
        return total;
    }
}