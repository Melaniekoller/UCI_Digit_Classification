package combinedMain;

/**
 * COMBINED HANDWRITTEN DIGIT CLASSIFICATION SYSTEM
 * This single file contains all classes of my UCI Digit Classification project
 * Includes: k-NN, SVM, MLP classifiers and all supporting utilities
 * 
 * PREREQUISITES:
 * 1. Java JDK 8 or higher must be installed
 * 2. Two CSV dataset files must be placed in a 'datasets' folder:
 *    - datasets/dataSet1.csv
 *    - datasets/dataSet2.csv
 * 
 * SETUP INSTRUCTIONS:
 * 1. Save this file as DigitClassifier.java
 * 2. Create a java project "JavaProject"
 * 3. Inside src of the "JavaProject" create a package called "combinedMain"
 * 4. Paste this "DigitClassifier.java" file inside the combinedMain folder
 * 5. The dataset folder should be in the same directory as the src folder
 *    Project structure should look like:
 *    - JavaProject/
 *        ├── datasets/
 *        │   ├── dataSet1.csv
 *        │   └── dataSet2.csv
 *        └── src/
 *            └── combinedMain/
 *                └── DigitClassifier.java
 * 6. Compile and run from JavaProject directory  
 */
 

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

// ============================================================================
// 1. DATA POINT CLASS
// ============================================================================

/**
 * Represents a single handwritten digit data point from UCI dataset
 * Each data point consists of 64 features (8x8 grid) and a digit label (0-9)
 */
class DataPoint {
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

// ============================================================================
// 2. DATASET LOADER
// ============================================================================

/**
 * Handles loading and parsing of UCI Handwritten Digits datasets
 */
class DatasetLoader {
    private static final int NUM_FEATURES = 64;  // 8x8 pixel grid features
    private static final int NUM_CLASSES = 10; // Digits 0 through 9
    private static final int EXPECTED_COLUMNS = NUM_FEATURES + 1;  // Features + label
    
    
    /**
     * Loads and parses a dataset from CSV file
     * @param filePath path to the dataset CSV file
     */
    public List<DataPoint> loadDataset(String filePath) throws IOException {
        List<DataPoint> dataset = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        int lineCount = 0;
        
        System.out.println("Loading dataset from: " + filePath);
        
        // Read file line by line
        while ((line = reader.readLine()) != null) {
            lineCount++;
            String[] values = line.split(",");
            
            // Validate line format - must have exactly 65 values (64 features + 1 label)
            if (values.length != EXPECTED_COLUMNS) {
                System.err.printf("Warning: Line %d has %d values, expected %d\n", 
                                lineCount, values.length, EXPECTED_COLUMNS);
                continue;  // Skip malformed line but continue processing
            }
            
            try {
            	// Parse label (last column) and features (first 64 columns)
                int label = Integer.parseInt(values[NUM_FEATURES].trim());
                double[] features = new double[NUM_FEATURES];
                
                // Parse each feature value
                for (int featureIndex = 0; featureIndex < NUM_FEATURES; featureIndex++) {
                    features[featureIndex] = Double.parseDouble(values[featureIndex].trim());
                }
                
                // Create and store data point
                dataset.add(new DataPoint(features, label));
                
            } catch (NumberFormatException e) {
            	// Handle lines with non-numeric values
                System.err.printf("Warning: Number format error on line %d\n", lineCount);
            }
        }
        
        reader.close();
        System.out.printf("Loaded %d data points\n", dataset.size());
        return dataset; // List of DataPoint objects containing features and labels
    }
}

// ============================================================================
// 3. CONFUSION MATRIX
// ============================================================================

/**
 * Confusion matrix for digit classification
 * Tracks actual vs predicted classifications for digits 0-9
 *  Rows represent actual classes, columns represent predicted classes
 */
class ConfusionMatrix {
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
     */
    public int getCorrectPredictions() {
        int correct = 0;
        for (int classIndex = 0; classIndex < numClasses; classIndex++) {
            correct += matrix[classIndex][classIndex]; // Sum diagonal
        }
        return correct; //sum of diagonal entries (true positives for each class)
    }
    
    /**
     * Calculates total number of predictions made
     */
    public int getTotalPredictions() {
        int total = 0;
        for (int actualClass = 0; actualClass < numClasses; actualClass++) {
            for (int predictedClass = 0; predictedClass < numClasses; predictedClass++) {
                total += matrix[actualClass][predictedClass];
            }
        }
        return total; // sum of all matrix entries
    }
}

// ============================================================================
// 4. RESULTS ANALYZER
// ============================================================================

/**
 * Handles result analysis and performance metrics for digit classification
 * Calculates accuracy and generates confusion matrices for model evaluation
 */
class ResultsAnalyzer {
    
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

// ============================================================================
// 5. STATISTICS UTILITY
// ============================================================================

/**
 * Utility class for statistical calculations
 */
class Statistics {
	public static double calculateAccuracy(int correct, int total) {
        return (double) correct / total * 100;
    }
}
												/******************** KNN***************************/

// ============================================================================
// 6. DISTANCE METRICS (for k-NN)
// ============================================================================

/**
 * Provides distance metric calculations for k-Nearest Neighbors algorithm
 */
class DistanceMetric {
	/**
     * Calculates Euclidean distance between two feature vectors
     * @param features1 first feature vector (64 dimensions for digit data)
     * @param features2 second feature vector (64 dimensions for digit data)
     */
    public static double euclideanDistance(double[] features1, double[] features2) { 	
        double sqDifferenceSum = 0.0;
        
        // Sum squared differences for each feature dimension
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
            double featureDiff = features1[featureIndex] - features2[featureIndex];
            sqDifferenceSum += featureDiff * featureDiff;
        }
        // Apply Pythagorean theorem for n-dimensional space
        return Math.sqrt(sqDifferenceSum); // Euclidean distance between the two vectors
    }
    
    /**
     * Calculates Manhattan distance between two feature vectors
     * @param features1 first feature vector (64 dimensions for digit data)
     * @param features2 second feature vector (64 dimensions for digit data)
     */
    public static double manhattanDistance(double[] features1, double[] features2) {
        double absoluteDiffsum = 0.0;
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
        	absoluteDiffsum += Math.abs(features1[featureIndex] - features2[featureIndex]);
        }
        return absoluteDiffsum; // Manhattan distance between the two vectors
    }
    
    /**
     * Calculates weighted Euclidean distance: Each feature is multiplied by a weight indicating its importance
     * @param features1 first feature vector (64 dimensions for digit data)
     * @param features2 second feature vector (64 dimensions for digit data)
     * @param weights array of weights for each feature dimension
     */
    public static double weightedEuclideanDistance(double[] features1, double[] features2, double[] weights) {
    	 validateArrayLengths(features1, features2, weights);
    	 
        double weightedSqSum = 0.0;
        
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
            double featureDiff = features1[featureIndex] - features2[featureIndex];
            weightedSqSum += weights[featureIndex] * featureDiff * featureDiff;
        }
        return Math.sqrt(weightedSqSum); // weighted Euclidean distance
    }
    
    /**
     * Calculates weighted Manhattan distance
     * @param features1 first feature vector (64 dimensions for digit data)
     * @param features2 second feature vector (64 dimensions for digit data)
     * @param weights array of weights for each feature dimension
     */
    public static double weightedManhattanDistance(double[] features1, double[] features2, double[] weights) {
    	 validateArrayLengths(features1, features2, weights);
        
    	 double weightedAbsoluteSum = 0.0;
        for (int featureIndex = 0; featureIndex < features1.length; featureIndex++) {
        	weightedAbsoluteSum += weights[featureIndex] * Math.abs(features1[featureIndex] - features2[featureIndex]);
        }
        return weightedAbsoluteSum; // weighted Manhattan distance
    }
    
    /**
     * Validates that all input arrays have the same length
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

// ============================================================================
// 7. k-NN CLASSIFIER
// ============================================================================

/**
 * k-Nearest Neighbors classifier for handwritten digit recognition
 * Implements the baseline algorithm from UCI repository using Euclidean distance
 */
class KnnClassifier {
	
	// Training dataset used for classification
    private List<DataPoint> trainingData;
    private boolean normalizeFeatures;
    private boolean useWeightedVoting;
    private double[] minVals;
    private double[] maxVals;
    
    // Distance metric to use ("euclidean" or "manhattan")
    private String distanceMetric;
    
    /**
     * Constructor with training data only, uses default Euclidean distance metric
     * @param trainingData the dataset to train the classifier
     */
    public KnnClassifier(List<DataPoint> trainingData) {
        this.trainingData = trainingData;
        this.distanceMetric = "euclidean"; //default Euclidean distance metric
        this.normalizeFeatures = false;
        this.useWeightedVoting = false;
    }
    
    /**
     * Constructor with training data and specified distance metric
     */
    public KnnClassifier(List<DataPoint> trainingData, String distanceMetric) {
    	 this.trainingData = trainingData; // the dataset to train the classifier
         this.distanceMetric = distanceMetric; // the distance metric to use
         this.normalizeFeatures = false; // whether to normalize feature values
         this.useWeightedVoting = false; // whether to use weighted voting
    }
    
    /**
     * Enhanced constructor with all configuration options
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
        return normalizedFeatures; 
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
	 * @param features1 first feature vector
	 * @param features2 second feature vector
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
     * @param neighbors list of sorted neighbors
     * @param k number of neighbors to consider
     * @return the predicted label with weighted voting
     */
    private int getWeightedMajorityVote(List<Neighbor> neighbors, int numOfNeighbors) {
        double[] weightedVotes = new double[10];
        for (int neighbourIndex = 0; neighbourIndex < numOfNeighbors && neighbourIndex < neighbors.size(); neighbourIndex++) {
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
            this.distance = distance;
            this.label = label;
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

												/******************** MLP***************************/

// ============================================================================
// 8. ACTIVATION FUNCTIONS (for MLP)
// ============================================================================

/**
 * Provides activation functions for the Multi-Layer Perceptron
 */
class Activation {
    
    /**
     * ACTIVATION FUNCTION OPTION 1: SIGMOID 
     * @param x input value
     */
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x)); // sigmoid activation output between 0 and 1
    }
    
    /**
     * Derivative of sigmoid function - used in backpropagation
     * @param x input value (sigmoid output)
     */
    public static double sigmoidDerivative(double x) {
        return x * (1 - x); // derivative of sigmoid at x
    }
    
    
    /**
     * ACTIVATION FUNCTION OPTION 2: RECTIFIED LINEAR UNIT (ReLU) 
     * @param x input value
     */
    public static double relu(double x) {
        return Math.max(0, x); // ReLU activation output
    }
    
    /**
     * Derivative of ReLU function - used in backpropagation
     * @param x input value
     */
    public static double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0; // derivative of ReLU at x
    }
    
    /**
     * ACTIVATION FUNCTION OPTION 3: SOFTMAX FUNCTION
     * @param inputs array of raw scores for each class
     */
    public static double[] softmax(double[] inputs) {
        double[] outputs = new double[inputs.length];
        double sum = 0.0;
        
        // Subtract max for numerical stability (prevents overflow)
        double max = inputs[0];
        for (double value : inputs) {
            if (value > max) max = value;
        }
        
        // Compute exponentials and sum
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = Math.exp(inputs[i] - max);
            sum += outputs[i];
        }
        
        // Normalize to probabilities
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] /= sum;
        }
        
        return outputs; // returns probability distribution summing to 1.0
    }
}

// ============================================================================
// 9. NEURAL NETWORK LAYER (for MLP)
// ============================================================================

/**
 * Represents a single layer in the Multi-Layer Perceptron
 * Each layer contains neurons with weights, biases, and activation functions
 */
class Layer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;  // weights[output][input]
    private double[] biases;     // biases[output]
    private double[] outputs;    // outputs[output]
    private double[] inputs;     // inputs[input] - stored for backpropagation
    private String activation;
    
    /**
     * Constructor for neural network layer
     */
    public Layer(int inputSize, int outputSize, String activation) {
        this.inputSize = inputSize; // number of neurons in previous layer
        this.outputSize = outputSize; // number of neurons in this current layer
        this.activation = activation; // function type ("sigmoid", "relu" , "softmax")
        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];
        this.outputs = new double[outputSize];
        
        initializeWeights();
    }
    
    /**
     * Initialize weights using Xavier/Glorot initialization
     */
    private void initializeWeights() {
        Random rand = new Random();
        double variance = 2.0 / (inputSize + outputSize);
        double stddev = Math.sqrt(variance);
        
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = rand.nextGaussian() * stddev;
            }
            biases[i] = 0.1; // Small positive bias to avoid dead neurons
        }
    }
    
    /**
     * Forward pass through the layer
     * Computes: output = activation(weights * inputs + bias)
     * @param inputs input values from previous layer
     */
    public double[] forward(double[] inputs) {
        this.inputs = inputs; // Store for backpropagation
        
        for (int i = 0; i < outputSize; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * inputs[j];
            }
            
            // Apply activation function
            switch (activation) {
                case "sigmoid":
                    outputs[i] = Activation.sigmoid(sum);
                    break;
                case "relu":
                    outputs[i] = Activation.relu(sum);
                    break;
                default:
                    outputs[i] = Activation.sigmoid(sum);
            }
        }
        
        return outputs; // activated outputs of this layer
    }
    
    /**
     * Backward pass - computes gradients and updates weights
     * Implements the core of backpropagation algorithm
     * @param errors errors from next layer
     * @param learningRate step size for weight updates
     */
    public double[] backward(double[] errors, double learningRate) {
        double[] previousErrors = new double[inputSize];
        
        for (int i = 0; i < outputSize; i++) {
            // Compute gradient including activation derivative
            double gradient = errors[i];
            if (activation.equals("sigmoid")) {
                gradient *= Activation.sigmoidDerivative(outputs[i]);
            } else if (activation.equals("relu")) {
                gradient *= Activation.reluDerivative(outputs[i]);
            } else if (activation.equals("softmax")) {
                // gradient *= Activation.softmaxDerivative(outputs[i]);
            }
            
            // Update weights and biases
            for (int j = 0; j < inputSize; j++) {
                double weightUpdate = learningRate * gradient * inputs[j];
                weights[i][j] -= weightUpdate;
                previousErrors[j] += gradient * weights[i][j];
            }
            biases[i] -= learningRate * gradient;
        }
        
        return previousErrors; // errors to propagate to previous layer
    }
    
    // Getters for access during training and testing
    public double[] getOutputs() { return outputs; }
    public double[][] getWeights() { return weights; }
    public double[] getBiases() { return biases; }
}

// ============================================================================
// 10. MLP CLASSIFIER
// ============================================================================

/**
 * Fixed Multi-Layer Perceptron (MLP) neural network for handwritten digit classification
 * Uses proper softmax output and cross-entropy loss for multi-class classification
 */
class MlpClassifier {
    private List<Layer> layers;
    private double learningRate;
    private int epochs;
    private Random random;
    
    /**
     * Constructor with default architecture
     */
    public MlpClassifier(double learningRate, int epochs) {
        this.learningRate = learningRate; // learning rate for gradient descent
        this.epochs = epochs; // number of training epochs
        this.random = new Random();
        this.layers = new ArrayList<>();
        
        // Fixed architecture with proper activations
        layers.add(new Layer(64, 128, "relu"));    // Hidden layer 1
        layers.add(new Layer(128, 64, "relu"));     // Hidden layer 2  
        layers.add(new Layer(64, 10, "linear"));    // Output layer (raw scores for softmax)
        
        System.out.println("MLP initialized: 64-128-64-10 architecture");
        System.out.println("Learning rate: " + learningRate + ", Epochs: " + epochs);
    }
    
    /**
     * Alternative constructor for different architectures
     * @param architecture string like "64-32-10" or "64-128-64-10"
     */
    public MlpClassifier(String architecture, double learningRate, int epochs) {
        this.learningRate = learningRate; // learning rate for gradient descent
        this.epochs = epochs; // number of training epochs
        this.random = new Random();
        this.layers = new ArrayList<>();
        
        initializeArchitecture(architecture);
        System.out.println("MLP initialized: " + architecture + " architecture");
        System.out.println("Learning rate: " + learningRate + ", Epochs: " + epochs);
    }
    
    /**
     * Parse architecture string like "64-32-10" or "64-128-64-10"
     * @param arch architecture string
     */
    private void initializeArchitecture(String arch) {
        String[] parts = arch.split("-");
        if (parts.length < 2) {
            throw new IllegalArgumentException("Invalid architecture: " + arch);
        }
        
        // Create layers based on architecture string
        for (int i = 0; i < parts.length - 1; i++) {
            int inputSize = Integer.parseInt(parts[i]);
            int outputSize = Integer.parseInt(parts[i + 1]);
            String activation;
            
            if (i == parts.length - 2) {
                // Output layer - linear activation (softmax applied separately)
                activation = "linear";
            } else {
                // Hidden layers - ReLU activation
                activation = "relu";
            }
            
            layers.add(new Layer(inputSize, outputSize, activation));
        }
    }
    
    /**
     * Train the MLP on the provided dataset
     * @param trainingData list of training data points
     */
    public void train(List<DataPoint> trainingData) {
        System.out.println("Training MLP on " + trainingData.size() + " examples...");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            int correct = 0;
            
            // Shuffle data for each epoch
            List<DataPoint> shuffledData = new ArrayList<>(trainingData);
            java.util.Collections.shuffle(shuffledData, random);
            
            for (DataPoint dataPoint : shuffledData) {
                // Forward pass
                double[] outputs = forward(dataPoint.getFeatures());
                
                // Apply softmax to get probabilities
                double[] probabilities = Activation.softmax(outputs);
                
                // Convert actual label to one-hot encoding
                double[] targets = oneHotEncode(dataPoint.getActualLabel());
                
                // Calculate cross-entropy loss and accuracy
                totalLoss += crossEntropyLoss(probabilities, targets);
                if (predict(probabilities) == dataPoint.getActualLabel()) {
                    correct++;
                }
                
                // Backward pass with cross-entropy + softmax gradient
                backward(probabilities, targets);
            }
            
            // Print progress every 10 epochs
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                double accuracy = (double) correct / trainingData.size() * 100;
                double avgLoss = totalLoss / trainingData.size();
                System.out.printf("Epoch %3d: Loss=%.4f, Accuracy=%.2f%%\n", 
                                epoch + 1, avgLoss, accuracy);
            }
        }
        System.out.println("MLP training completed!");
    }
    
    /**
     * Forward pass through the network
     * @param inputs input feature vector
     * @return raw scores (logits) from output layer
     */
    public double[] forward(double[] inputs) {
        double[] currentOutputs = inputs;
        
        for (Layer layer : layers) {
            currentOutputs = layer.forward(currentOutputs);
        }
        
        return currentOutputs; // Returns raw scores (logits)
    }
    
    /**
     * Fixed backward pass with proper cross-entropy + softmax gradient
     * The gradient simplifies to: ∂L/∂z = probabilities - targets
     * @param probabilities softmax probabilities
     * @param targets one-hot encoded targets
     */
    private void backward(double[] probabilities, double[] targets) {
        // For softmax + cross-entropy: gradient = probabilities - targets
        double[] errors = new double[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            errors[i] = probabilities[i] - targets[i];
        }
        
        // Backpropagate errors through layers
        for (int i = layers.size() - 1; i >= 0; i--) {
            errors = layers.get(i).backward(errors, learningRate);
        }
    }
    
    /**
     * Classify a data point
     * @param dataPoint the data point to classify
     * @return predicted digit label (0-9)
     */
    public int classify(DataPoint dataPoint) {
        double[] outputs = forward(dataPoint.getFeatures());
        double[] probabilities = Activation.softmax(outputs);
        return predict(probabilities);
    }
    
    /**
     * Predict class from probabilities
     * @param probabilities class probabilities
     */
    private int predict(double[] probabilities) {
        int maxIndex = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex; // index of highest probability
    }
    
    /**
     * Convert label to one-hot encoding
     */
    private double[] oneHotEncode(int label) {
        double[] encoded = new double[10]; //  digit label (0-9)
        encoded[label] = 1.0;
        return encoded; // one-hot encoded array
    }
    
    /**
     * Cross-entropy loss for multi-class classification
     * Better suited for classification than mean squared error
     * @param probabilities predicted probabilities
     * @param targets one-hot encoded targets
     */
    private double crossEntropyLoss(double[] probabilities, double[] targets) {
        double loss = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            if (targets[i] == 1.0) {
                // Add small epsilon to avoid log(0)
                loss += -Math.log(probabilities[i] + 1e-8);
            }
        }
        return loss; // cross-entropy loss
    }
}

									        /******************** SVM***************************/
// ============================================================================
// 11. KERNEL FUNCTIONS (for SVM)
// ============================================================================

/**
 * Provides kernel functions for SVM
 * Kernels transform data into higher-dimensional space for linear separation
 */
class Kernel {
    
    // Named constants for common kernel parameter values
    private static final double DEFAULT_POLYNOMIAL_CONSTANT = 1.0;
    private static final int DEFAULT_POLYNOMIAL_DEGREE = 2;
    private static final double DEFAULT_RBF_GAMMA = 0.1;
    private static final double DEFAULT_SIGMOID_ALPHA = 0.01;
    private static final double DEFAULT_SIGMOID_CONSTANT = 0.0;
    private static final int MINIMUM_POLYNOMIAL_DEGREE = 1;
    		
    /**
     * Linear kernel - simplest kernel, works well for linearly separable data
     * @param firstVector first feature vector (64 dimensions for digit data)
     * @param secondVector second feature vector (64 dimensions for digit data)
     */
    public static double linear(double[] firstVector, double[] secondVector) {
        double dotProductResult = 0.0;
        for (int featureIndex = 0; featureIndex < firstVector.length; featureIndex++) {
        	dotProductResult += firstVector[featureIndex] * secondVector[featureIndex];
        }
        return dotProductResult; // dot product of the two vectors
    }
    
    /**
     * Polynomial kernel with default - can capture nonlinear relationships
     * Maps data to feature space of all polynomial combinations up to degree d
     * @return polynomial kernel value
     */
    public static double polynomial(double[] firstVector, double[] secondVector) {
        return polynomial(firstVector, secondVector, DEFAULT_POLYNOMIAL_CONSTANT, DEFAULT_POLYNOMIAL_DEGREE);
    }
    
    /**
     * Polynomial kernel with custom parameters
     * @param polyConstant constant term (bias/offset; usually 0 or 1)
     * @param polyDegree polynomial degree (complexity of relationships)
     * @return polynomial kernel value
     */
    public static double polynomial(double[] firstVector, double[] secondVector, 
                                  double polyConstant, int polyDegree) {
        if (polyDegree < MINIMUM_POLYNOMIAL_DEGREE) {
            throw new IllegalArgumentException("Polynomial degree must be at least " + MINIMUM_POLYNOMIAL_DEGREE);
        }
        
        double dotProductResult = linear(firstVector, secondVector);
        return Math.pow(dotProductResult + polyConstant, polyDegree);
    }
    
    /**
     * Radial Basis Function (RBF) kernel with default gamma- most popular for nonlinear problems   
     * @return RBF kernel value between 0 and 1
     */
    public static double rbf(double[] firstVector, double[] secondVector) {
        return rbf(firstVector, secondVector, DEFAULT_RBF_GAMMA);
    }
    
    /**
     * Radial Basis Function (RBF) kernel with custom gamma
     * @param gammaParameter kernel width parameter (γ)
     * @return RBF kernel value between 0 and 1
     */
    public static double rbf(double[] firstVector, double[] secondVector, double gammaParameter) {
        if (gammaParameter <= 0) {
            throw new IllegalArgumentException("Gamma must be positive");
        }
        
        double squaredEuclideanDistance = 0.0;
        for (int featureIndex = 0; featureIndex < firstVector.length; featureIndex++) {
            double featureDifference = firstVector[featureIndex] - secondVector[featureIndex];
            squaredEuclideanDistance += featureDifference * featureDifference;
        }
        return Math.exp(-gammaParameter * squaredEuclideanDistance);
    }
    
    
    /**
     * Sigmoid kernel with default parameters - neural network inspired
     * Similar to two-layer perceptron neural network
     * @return sigmoid kernel value
     */
    public static double sigmoid(double[] firstVector, double[] secondVector) {
        return sigmoid(firstVector, secondVector, DEFAULT_SIGMOID_ALPHA, DEFAULT_SIGMOID_CONSTANT);
    }
    
    /**
     * Sigmoid kernel with custom parameters
     * @param alphaParameter controls the scaling
     * @param biasConstant bias term
     * @return sigmoid kernel value
     */
    public static double sigmoid(double[] firstVector, double[] secondVector, 
                               double alphaParameter, double biasConstant) {
        double dotProductResult = linear(firstVector, secondVector);
        return Math.tanh(alphaParameter * dotProductResult + biasConstant);
    }
}

// ============================================================================
// 12. SVM CLASSIFIER
// ============================================================================

/**
 * Enhanced SVM classifier for handwritten digit recognition
 * Uses One-vs-Rest multi-class classification with improved optimization
 */
class SvmClassifier {
    // Named constants for configuration values
    private static final int NUMBER_OF_DIGIT_CLASSES = 10;
    private static final double DEFAULT_REGULARIZATION_PARAMETER = 1.0;
    private static final double DEFAULT_GAMMA_VALUE = 0.01;
    private static final int DEFAULT_POLYNOMIAL_DEGREE = 2;
    private static final int MAXIMUM_TRAINING_ITERATIONS = 500;
    private static final double CONVERGENCE_TOLERANCE = 1e-3;
    private static final long FIXED_RANDOM_SEED = 12345L;
    private static final int EARLY_STOPPING_THRESHOLD = 50;
    private static final double SUPPORT_VECTOR_THRESHOLD = 1e-5;
    private static final double DEFAULT_SIGMOID_ALPHA = 0.01;
    private static final double DEFAULT_SIGMOID_BIAS = 0.0;
    private static final int DEFAULT_INDEX_VALUE = 0;
    
    // Kernel type constants
    private static final String LINEAR_KERNEL_TYPE = "linear";
    private static final String POLYNOMIAL_KERNEL_TYPE = "poly";
    private static final String RBF_KERNEL_TYPE = "rbf";
    private static final String SIGMOID_KERNEL_TYPE = "sigmoid";
    private static final double POSITIVE_CLASS_LABEL = 1.0;
    private static final double NEGATIVE_CLASS_LABEL = -1.0;
    private static final int MINIMUM_POLYNOMIAL_DEGREE = 1;
    
    private List<DataPoint> trainingData;
    private String kernelType;
    private double regularizationParameter; // C parameter
    private double gammaParameter; // For RBF kernel
    private int polynomialDegree; // For polynomial kernel
    private int numberOfClasses;
    
    // For multiclass classification (One-vs-Rest)
    private List<BinarySvmClassifier> binaryClassifiers; 
    
    /**
     * Constructor with kernel type and regularization parameter only
     * @param trainingData the training dataset
     * @param kernelType the type of kernel to use
     * @param regularizationParameter the regularization parameter C
     */
    public SvmClassifier(List<DataPoint> trainingData, String kernelType, double regularizationParameter) {
        this(trainingData, kernelType, regularizationParameter, 
             DEFAULT_GAMMA_VALUE, DEFAULT_POLYNOMIAL_DEGREE);
    }
    
    /**
     * Constructor with all parameters
     * @param gammaParameter the gamma parameter for RBF kernel
     * @param polynomialDegree the degree for polynomial kernel
     */
    public SvmClassifier(List<DataPoint> trainingData, String kernelType, 
                        double regularizationParameter, double gammaParameter, int polynomialDegree) {
        this.trainingData = trainingData;
        this.kernelType = kernelType;
        this.regularizationParameter = regularizationParameter;
        this.gammaParameter = gammaParameter;
        this.polynomialDegree = polynomialDegree;
        this.numberOfClasses = NUMBER_OF_DIGIT_CLASSES;
        this.binaryClassifiers = new ArrayList<>();
        
        trainMulticlass();
    }
    
    /**
     * Trains multiple binary SVM classifiers for One-vs-Rest multiclass classification
     */
    private void trainMulticlass() {
        // Train one binary classifier for each digit class
        for (int currentClassLabel = 0; currentClassLabel < numberOfClasses; currentClassLabel++) {
            BinarySvmClassifier binarySvm = new BinarySvmClassifier(trainingData, currentClassLabel, 
                                                                  kernelType, regularizationParameter, 
                                                                  gammaParameter, polynomialDegree);
            binarySvm.train();
            binaryClassifiers.add(binarySvm);
        }
    }
    
    /**
     * Classifies a test point using One-vs-Rest strategy
     * @param testPoint the data point to classify
     * @return the predicted digit label (0-9)
     */
    public int classify(DataPoint testPoint) {
        double[] allDecisionValues = new double[numberOfClasses];
        
        // Get decision value from each binary classifier
        for (int classifierIndex = 0; classifierIndex < numberOfClasses; classifierIndex++) {
            allDecisionValues[classifierIndex] = binaryClassifiers.get(classifierIndex)
                .decisionFunction(testPoint.getFeatures());
        }
        
        // Return class with highest decision value
        return findIndexOfMaximumValue(allDecisionValues);
    }
    
    /**
     * Utility method to find the index of the maximum value in an array
     * @param valueArray the array of values
     * @return the index of the maximum value
     */
    private int findIndexOfMaximumValue(double[] valueArray) {
        int maximumValueIndex = DEFAULT_INDEX_VALUE;
        for (int valueIndex = 1; valueIndex < valueArray.length; valueIndex++) {
            if (valueArray[valueIndex] > valueArray[maximumValueIndex]) {
                maximumValueIndex = valueIndex;
            }
        }
        return maximumValueIndex;
    }
    
    /**
     * Inner class representing a binary SVM classifier for One-vs-Rest
     */
    private class BinarySvmClassifier {
        private List<DataPoint> trainingData;
        private int positiveClassLabel;
        private String kernelType;
        private double regularizationParameter;
        private double gammaParameter;
        private int polynomialDegree;
        private double[] lagrangeMultipliers; // alphas
        private double biasTerm;
        private List<SupportVector> supportVectors;
        
        /**
         * Constructor for binary SVM classifier
         * @param positiveClassLabel the positive class for this binary classifier
         */
        public BinarySvmClassifier(List<DataPoint> trainingData, int positiveClassLabel, 
                                 String kernelType, double regularizationParameter, 
                                 double gammaParameter, int polynomialDegree) {
            this.trainingData = trainingData;
            this.positiveClassLabel = positiveClassLabel;
            this.kernelType = kernelType;
            this.regularizationParameter = regularizationParameter;
            this.gammaParameter = gammaParameter;
            this.polynomialDegree = polynomialDegree;
            this.lagrangeMultipliers = new double[trainingData.size()];
            this.supportVectors = new ArrayList<>();
        }
        
        /**
         * Enhanced training using simplified SMO-like algorithm with fixed random seed
         */
        public void train() {
            Random randomNumberGenerator = new Random(FIXED_RANDOM_SEED);
            
            for (int iterationCounter = 0; iterationCounter < MAXIMUM_TRAINING_ITERATIONS; iterationCounter++) {
                int numberOfChangedMultipliers = DEFAULT_INDEX_VALUE;
                
                // Examine each training example for possible optimization
                for (int firstExampleIndex = 0; firstExampleIndex < trainingData.size(); firstExampleIndex++) {
                    double firstExampleError = calculatePredictionError(firstExampleIndex);
                    double firstExampleLabel = getBinaryLabel(trainingData.get(firstExampleIndex)) ? 
                                              POSITIVE_CLASS_LABEL : NEGATIVE_CLASS_LABEL;
                    
                    // Check KKT violation conditions
                    if ((firstExampleLabel * firstExampleError < -CONVERGENCE_TOLERANCE && 
                         lagrangeMultipliers[firstExampleIndex] < regularizationParameter) ||
                        (firstExampleLabel * firstExampleError > CONVERGENCE_TOLERANCE && 
                         lagrangeMultipliers[firstExampleIndex] > DEFAULT_INDEX_VALUE)) {
                        
                        // Choose random second example different from first example
                        int secondExampleIndex = firstExampleIndex;
                        while (secondExampleIndex == firstExampleIndex) {
                            secondExampleIndex = randomNumberGenerator.nextInt(trainingData.size());
                        }
                        
                        double secondExampleError = calculatePredictionError(secondExampleIndex);
                        double secondExampleLabel = getBinaryLabel(trainingData.get(secondExampleIndex)) ? 
                                                   POSITIVE_CLASS_LABEL : NEGATIVE_CLASS_LABEL;
                        
                        // Store old values for bias update
                        double firstMultiplierOld = lagrangeMultipliers[firstExampleIndex];
                        double secondMultiplierOld = lagrangeMultipliers[secondExampleIndex];
                        
                        // Compute L and H bounds for second Lagrange multiplier
                        double lowerBound, upperBound;
                        if (firstExampleLabel != secondExampleLabel) {
                            lowerBound = Math.max(DEFAULT_INDEX_VALUE, 
                                                lagrangeMultipliers[secondExampleIndex] - 
                                                lagrangeMultipliers[firstExampleIndex]);
                            upperBound = Math.min(regularizationParameter, 
                                                regularizationParameter + 
                                                lagrangeMultipliers[secondExampleIndex] - 
                                                lagrangeMultipliers[firstExampleIndex]);
                        } else {
                            lowerBound = Math.max(DEFAULT_INDEX_VALUE, 
                                                lagrangeMultipliers[firstExampleIndex] + 
                                                lagrangeMultipliers[secondExampleIndex] - 
                                                regularizationParameter);
                            upperBound = Math.min(regularizationParameter, 
                                                lagrangeMultipliers[firstExampleIndex] + 
                                                lagrangeMultipliers[secondExampleIndex]);
                        }
                        
                        if (lowerBound == upperBound) continue;
                        
                        // Compute kernel values
                        double etaValue = 2 * computeKernelBetweenExamples(firstExampleIndex, secondExampleIndex) - 
                                        computeKernelBetweenExamples(firstExampleIndex, firstExampleIndex) - 
                                        computeKernelBetweenExamples(secondExampleIndex, secondExampleIndex);
                        if (etaValue >= DEFAULT_INDEX_VALUE) continue;
                        
                        // Update second Lagrange multiplier
                        lagrangeMultipliers[secondExampleIndex] -= secondExampleLabel * 
                                                                  (firstExampleError - secondExampleError) / 
                                                                  etaValue;
                        
                        // Clip second Lagrange multiplier to bounds
                        if (lagrangeMultipliers[secondExampleIndex] > upperBound) {
                            lagrangeMultipliers[secondExampleIndex] = upperBound;
                        } else if (lagrangeMultipliers[secondExampleIndex] < lowerBound) {
                            lagrangeMultipliers[secondExampleIndex] = lowerBound;
                        }
                        
                        // Skip if change is too small
                        if (Math.abs(lagrangeMultipliers[secondExampleIndex] - secondMultiplierOld) < CONVERGENCE_TOLERANCE) {
                            continue;
                        }
                        
                        // Update first Lagrange multiplier
                        lagrangeMultipliers[firstExampleIndex] += firstExampleLabel * secondExampleLabel * 
                                                                  (secondMultiplierOld - lagrangeMultipliers[secondExampleIndex]);
                        
                        // Update bias term
                        updateBiasTerm(firstExampleIndex, secondExampleIndex, firstExampleError, 
                                     secondExampleError, firstExampleLabel, secondExampleLabel, 
                                     firstMultiplierOld, secondMultiplierOld);
                        
                        numberOfChangedMultipliers++;
                    }
                }
                
                // Early stopping if no changes for several iterations
                if (numberOfChangedMultipliers == DEFAULT_INDEX_VALUE && iterationCounter > EARLY_STOPPING_THRESHOLD) {
                    break;
                }
            }
            
            // Extract support vectors
            extractSupportVectors();
        }
        
        /**
         * Calculates prediction error for training example
         * @param exampleIndex the index of the training example
         * @return the prediction error
         */
        private double calculatePredictionError(int exampleIndex) {
            return decisionFunction(trainingData.get(exampleIndex).getFeatures()) - 
                   (getBinaryLabel(trainingData.get(exampleIndex)) ? 
                    POSITIVE_CLASS_LABEL : NEGATIVE_CLASS_LABEL);
        }
        
        /**
         * Updates bias term using updated Lagrange multipliers
         * @param firstExampleIndex index of first example
         * @param secondExampleIndex index of second example
         * @param firstError error of first example
         * @param secondError error of second example
         * @param firstLabel label of first example
         * @param secondLabel label of second example
         * @param firstMultiplierOld old Lagrange multiplier of first example
         * @param secondMultiplierOld old Lagrange multiplier of second example
         */
        private void updateBiasTerm(int firstExampleIndex, int secondExampleIndex, double firstError, 
                                  double secondError, double firstLabel, double secondLabel, 
                                  double firstMultiplierOld, double secondMultiplierOld) {
            double firstBiasCandidate = biasTerm - firstError - 
                                      firstLabel * (lagrangeMultipliers[firstExampleIndex] - firstMultiplierOld) * 
                                      computeKernelBetweenExamples(firstExampleIndex, firstExampleIndex) -
                                      secondLabel * (lagrangeMultipliers[secondExampleIndex] - secondMultiplierOld) * 
                                      computeKernelBetweenExamples(firstExampleIndex, secondExampleIndex);
            
            double secondBiasCandidate = biasTerm - secondError - 
                                       firstLabel * (lagrangeMultipliers[firstExampleIndex] - firstMultiplierOld) * 
                                       computeKernelBetweenExamples(firstExampleIndex, secondExampleIndex) -
                                       secondLabel * (lagrangeMultipliers[secondExampleIndex] - secondMultiplierOld) * 
                                       computeKernelBetweenExamples(secondExampleIndex, secondExampleIndex);
            
            if (lagrangeMultipliers[firstExampleIndex] > DEFAULT_INDEX_VALUE && 
                lagrangeMultipliers[firstExampleIndex] < regularizationParameter) {
                biasTerm = firstBiasCandidate;
            } else if (lagrangeMultipliers[secondExampleIndex] > DEFAULT_INDEX_VALUE && 
                     lagrangeMultipliers[secondExampleIndex] < regularizationParameter) {
                biasTerm = secondBiasCandidate;
            } else {
                biasTerm = (firstBiasCandidate + secondBiasCandidate) / 2.0;
            }
        }
        
        /**
         * Extracts support vectors from training data 
         */
        private void extractSupportVectors() {
            for (int exampleIndex = 0; exampleIndex < trainingData.size(); exampleIndex++) {
                if (lagrangeMultipliers[exampleIndex] > SUPPORT_VECTOR_THRESHOLD) {
                    supportVectors.add(new SupportVector(
                        trainingData.get(exampleIndex).getFeatures(),
                        getBinaryLabel(trainingData.get(exampleIndex)) ? 
                        POSITIVE_CLASS_LABEL : NEGATIVE_CLASS_LABEL,
                        lagrangeMultipliers[exampleIndex]
                    ));
                }
            }
        }
        
        /**
         * Computes kernel value between two training examples
         * @return kernel value
         */
        private double computeKernelBetweenExamples(int firstExampleIndex, int secondExampleIndex) {
            return computeKernelFunction(
                trainingData.get(firstExampleIndex).getFeatures(),
                trainingData.get(secondExampleIndex).getFeatures()
            );
        }
        
        /**
         * Computes kernel function between two feature vectors
         * @return kernel value
         */
        private double computeKernelFunction(double[] firstFeatureVector, double[] secondFeatureVector) {
            switch (kernelType) {
                case LINEAR_KERNEL_TYPE:
                    return Kernel.linear(firstFeatureVector, secondFeatureVector);
                case POLYNOMIAL_KERNEL_TYPE:
                    return Kernel.polynomial(firstFeatureVector, secondFeatureVector, 
                                           DEFAULT_REGULARIZATION_PARAMETER, polynomialDegree);
                case RBF_KERNEL_TYPE:
                    return Kernel.rbf(firstFeatureVector, secondFeatureVector, gammaParameter);
                case SIGMOID_KERNEL_TYPE:
                    return Kernel.sigmoid(firstFeatureVector, secondFeatureVector, 
                                        DEFAULT_SIGMOID_ALPHA, DEFAULT_SIGMOID_BIAS);
                default:
                    return Kernel.linear(firstFeatureVector, secondFeatureVector);
            }
        }
        
        /**
         * Decision function for binary classification
         * @param featureVector the feature vector to classify
         * @return the decision function value
         */
        public double decisionFunction(double[] featureVector) {
            double decisionSum = 0.0;
            for (SupportVector currentSupportVector : supportVectors) {
                decisionSum += currentSupportVector.lagrangeMultiplier * 
                              currentSupportVector.classLabel * 
                              computeKernelFunction(currentSupportVector.features, featureVector);
            }
            return decisionSum - biasTerm;
        }
        
        /**
         * Converts multiclass label to binary label for this classifier
         * @param dataPoint the data point
         * @return true if data point belongs to positive class, false otherwise
         */
        private boolean getBinaryLabel(DataPoint dataPoint) {
            return dataPoint.getActualLabel() == positiveClassLabel;
        }
        
        /**
         * Inner class representing a support vector
         */
        private class SupportVector {
            double[] features;
            double classLabel;
            double lagrangeMultiplier;
            
            SupportVector(double[] features, double classLabel, double lagrangeMultiplier) {
                this.features = features;
                this.classLabel = classLabel;
                this.lagrangeMultiplier = lagrangeMultiplier;
            }
        }
    }
}

// ============================================================================
// 13. COMBINED MAIN CLASS
// ============================================================================

/**
 * Combined main method for running both k-NN and SVM handwritten digit classification
 * Performs two-fold testing and displays confusion matrices for each model
 */
public class DigitClassifier {
    
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
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("COMBINED HANDWRITTEN DIGIT CLASSIFICATION: k-NN & SVM");
        System.out.println("=".repeat(70));
        
        try {
            // Load datasets
            DatasetLoader dataLoader = new DatasetLoader();
            List<DataPoint> firstDataset = dataLoader.loadDataset("datasets/dataSet1.csv");
            List<DataPoint> secondDataset = dataLoader.loadDataset("datasets/dataSet2.csv");
            
            System.out.println("\n" + "=".repeat(70));
            System.out.println("PART 1: k-NEAREST NEIGHBORS CLASSIFICATION");
            System.out.println("=".repeat(70));
            
            // Run k-NN classification with optimal configuration finding
            runKnnClassification(firstDataset, secondDataset);
            
            System.out.println("\n\n" + "=".repeat(70));
            System.out.println("PART 2: SUPPORT VECTOR MACHINE CLASSIFICATION");
            System.out.println("=".repeat(70));
            
            // Run SVM classification with optimal configuration finding
            runSvmClassification(firstDataset, secondDataset);
            
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
     * @param firstDataset first dataset for cross-validation
     * @param secondDataset second dataset for cross-validation
     */
    private static void runKnnClassification(List<DataPoint> firstDataset, 
                                           List<DataPoint> secondDataset) {
        System.out.println("\n--- Finding Optimal k-NN Configuration ---");
        
        // Test Euclidean distance with normalization
        System.out.println("\n1. EUCLIDEAN DISTANCE WITH NORMALIZATION:");
        int euclideanOptimalK = findOptimalKTwoFold(firstDataset, secondDataset, "euclidean", true);
        System.out.printf("Optimal k for Euclidean: %d\n", euclideanOptimalK);
        
        // Test Manhattan distance with normalization  
        System.out.println("\n2. MANHATTAN DISTANCE WITH NORMALIZATION:");
        int manhattanOptimalK = findOptimalKTwoFold(firstDataset, secondDataset, "manhattan", true);
        System.out.printf("Optimal k for Manhattan: %d\n", manhattanOptimalK);
        
        System.out.println("\n" + "-".repeat(60));
        System.out.println("TWO-FOLD TESTING WITH BEST CONFIGURATIONS");
        System.out.println("-".repeat(60));
        
        // Perform two-fold test for Euclidean with its optimal k
        System.out.println("\nEUCLIDEAN DISTANCE (Two-Fold Test):");
        performKnnTwoFoldTest(firstDataset, secondDataset, euclideanOptimalK, "euclidean", true);
        
        // Perform two-fold test for Manhattan with its optimal k
        System.out.println("\nMANHATTAN DISTANCE (Two-Fold Test):");
        performKnnTwoFoldTest(firstDataset, secondDataset, manhattanOptimalK, "manhattan", true);
        
        // Determine and display best performer
        displayKnnBestPerformer();
    }
    
    /**
     * Runs SVM classification with optimal configuration finding
     * @param firstDataset first dataset for cross-validation
     * @param secondDataset second dataset for cross-validation
     */
    private static void runSvmClassification(List<DataPoint> firstDataset, 
                                           List<DataPoint> secondDataset) {
        System.out.println("\n--- Finding Optimal SVM Configuration ---");
        
        // Test different SVM configurations
        SvmConfiguration bestSvmConfig = findOptimalSvmConfiguration(firstDataset, secondDataset);
        
        System.out.println("\n" + "-".repeat(60));
        System.out.println("BEST SVM CONFIGURATION FOUND:");
        System.out.println("-".repeat(60));
        System.out.printf("Kernel: %s, C=%.1f", bestSvmConfig.kernelType, bestSvmConfig.regularization);
        if (bestSvmConfig.kernelType.equals("rbf")) {
            System.out.printf(", gamma=%.3f", bestSvmConfig.gamma);
        } else if (bestSvmConfig.kernelType.equals("poly")) {
            System.out.printf(", degree=%d", bestSvmConfig.degree);
        }
        System.out.printf(" (Accuracy: %.4f%%)\n", bestSvmConfig.accuracy);
        
        System.out.println("\n" + "-".repeat(60));
        System.out.println("TWO-FOLD TESTING WITH BEST SVM CONFIGURATION");
        System.out.println("-".repeat(60));
        
        // Perform two-fold test with best SVM configuration
        performSvmTwoFoldTest(firstDataset, secondDataset, bestSvmConfig);
    }
    
    /**
     * Finds optimal k using two-fold average for k-NN
     * @param dataset1 first dataset
     * @param dataset2 second dataset
     * @param distanceMetric distance metric to use
     * @param normalize whether to normalize features
     * @return optimal k value
     */
    private static int findOptimalKTwoFold(List<DataPoint> dataset1,
                                         List<DataPoint> dataset2,
                                         String distanceMetric,
                                         boolean normalize) {
        int bestKValue = MINIMUM_K_VALUE;
        double bestAverageAccuracy = 0;
        int[] kValuesToTest = {1, 3, 5, 7};
        
        for (int currentKValue : kValuesToTest) {
            double firstFoldAccuracy = testKnnSingleFold(dataset1, dataset2, currentKValue, 
                                                       distanceMetric, normalize);
            double secondFoldAccuracy = testKnnSingleFold(dataset2, dataset1, currentKValue, 
                                                        distanceMetric, normalize);
            double averageAccuracy = (firstFoldAccuracy + secondFoldAccuracy) / 2;
            
            // Track overall best configuration
            if (bestOverallAccuracy < 0 || averageAccuracy > bestOverallAccuracy) {
                bestOverallAccuracy = averageAccuracy;
                bestOverallK = currentKValue;
                bestOverallMetric = distanceMetric;
                bestOverallNormalization = normalize;
            }
            
            System.out.printf("  k=%2d: Fold1=%.4f%%, Fold2=%.4f%%, Avg=%.4f%%", 
                            currentKValue, firstFoldAccuracy, secondFoldAccuracy, averageAccuracy);
            
            if (averageAccuracy > bestAverageAccuracy) {
                bestAverageAccuracy = averageAccuracy;
                bestKValue = currentKValue;
                System.out.print(" ← BEST");
            }
            System.out.println();
        }
        
        return bestKValue;
    }
    
    /**
     * Finds optimal SVM configuration
     * @return best SVM configuration
     */
    private static SvmConfiguration findOptimalSvmConfiguration(List<DataPoint> dataset1,
                                                              List<DataPoint> dataset2) {
        SvmConfiguration[] configurations = {
            // Linear kernel configurations
            new SvmConfiguration("linear", 1.0, 0.01, 2),
            new SvmConfiguration("linear", 10.0, 0.01, 2),
            // Sigmoid kernel configurations
            new SvmConfiguration("sigmoid", 1.0, 0.01, 2),
            new SvmConfiguration("sigmoid", 10.0, 0.01, 2),
            new SvmConfiguration("sigmoid", 100.0, 0.01, 2),
            // RBF kernel configurations
            new SvmConfiguration("rbf", 1.0, 0.010, 2),
            new SvmConfiguration("rbf", 10.0, 0.001, 2),
            new SvmConfiguration("rbf", 1.0, 0.015, 2),
            new SvmConfiguration("rbf", 1.0, 0.020, 2),
            // Polynomial kernel configurations
            new SvmConfiguration("poly", 1.0, 0.01, 2),
            new SvmConfiguration("poly", 10.0, 0.01, 3),
        };
        
        SvmConfiguration bestConfig = configurations[0];
        double bestAccuracy = 0;
        
        for (SvmConfiguration config : configurations) {
            double[] results = testSvmConfigurationSummarized(dataset1, dataset2, config);
            double averageAccuracy = results[2];
            
            System.out.printf("Kernel: %-7s, C=%-4.1f", config.kernelType, config.regularization);
            
            if (config.kernelType.equals("rbf")) {
                System.out.printf(", gamma=%.3f,  ", config.gamma);
            } else if (config.kernelType.equals("poly")) {
                System.out.printf(", degree=%d, ", config.degree);
            } else {
                System.out.print(", ".repeat(2));
            }
            
            System.out.printf("Fold1: %6.2f%%, ", results[0]);
            System.out.printf("Fold2: %6.2f%%, ", results[1]);
            System.out.printf("Avg: %6.2f%%", averageAccuracy);
            
            if (averageAccuracy > bestAccuracy) {
                bestAccuracy = averageAccuracy;
                bestConfig = config;
                bestConfig.accuracy = averageAccuracy;
                System.out.print(" ← NEW BEST");
            }
            
            System.out.println();
        }
        
        return bestConfig;
    }
    
    /**
     * Performs k-NN two-fold test with confusion matrices
     * @param kValue k value to use
     * @param distanceMetric distance metric to use
     * @param normalize whether to normalize features
     */
    private static void performKnnTwoFoldTest(List<DataPoint> dataset1,
                                            List<DataPoint> dataset2,
                                            int kValue,
                                            String distanceMetric,
                                            boolean normalize) {
        System.out.println("\nFOLD 1: Train on dataset1, Test on dataset2");
        double firstFoldAccuracy = performKnnFoldWithConfusionMatrix(dataset1, dataset2, 
                                                                   kValue, distanceMetric, normalize, 1);
        
        System.out.println("\nFOLD 2: Train on dataset2, Test on dataset1");
        double secondFoldAccuracy = performKnnFoldWithConfusionMatrix(dataset2, dataset1, 
                                                                    kValue, distanceMetric, normalize, 2);
        
        double averageAccuracy = (firstFoldAccuracy + secondFoldAccuracy) / 2;
        System.out.printf("\nAverage Accuracy: %.4f%%\n", averageAccuracy);
        printKnnBaselineComparison(averageAccuracy);
    }
    
    /**
     * Performs SVM two-fold test with confusion matrices
     * @param config SVM configuration to use
     */
    private static void performSvmTwoFoldTest(List<DataPoint> dataset1,
                                            List<DataPoint> dataset2,
                                            SvmConfiguration config) {
        System.out.println("\nFOLD 1: Train on dataset1, Test on dataset2");
        double firstFoldAccuracy = performSvmFoldWithConfusionMatrix(dataset1, dataset2, 
                                                                   config, 1);
        
        System.out.println("\nFOLD 2: Train on dataset2, Test on dataset1");
        double secondFoldAccuracy = performSvmFoldWithConfusionMatrix(dataset2, dataset1, 
                                                                    config, 2);
        
        double averageAccuracy = (firstFoldAccuracy + secondFoldAccuracy) / 2;
        System.out.printf("\nAverage Accuracy: %.4f%%\n", averageAccuracy);
        printSvmBaselineComparison(averageAccuracy);
    }
    
    /**
     * Performs a single k-NN fold with confusion matrix
     * @param trainingData training dataset
     * @param testData test dataset
     * @param kValue k value to use
     * @param distanceMetric distance metric to use
     * @param normalize whether to normalize features
     * @param foldNumber fold number (for display)
     * @return fold accuracy
     */
    private static double performKnnFoldWithConfusionMatrix(List<DataPoint> trainingData,
                                                          List<DataPoint> testData,
                                                          int kValue,
                                                          String distanceMetric,
                                                          boolean normalize,
                                                          int foldNumber) {
        KnnClassifier knnClassifier = new KnnClassifier(trainingData, distanceMetric, normalize, false);
        int correctPredictions = 0;
        
        for (DataPoint testPoint : testData) {
            int predictedLabel = knnClassifier.classify(testPoint, kValue);
            testPoint.setPredictedLabel(predictedLabel);
            if (predictedLabel == testPoint.getActualLabel()) {
                correctPredictions++;
            }
        }
        
        double foldAccuracy = (double) correctPredictions / testData.size() * 100;
        System.out.printf("Accuracy: %.4f%% (%d/%d)\n", foldAccuracy, correctPredictions, testData.size());
        
        // Display confusion matrix
        System.out.println("Confusion Matrix:");
        displayConfusionMatrix(testData);
        
        return foldAccuracy;
    }
    
    /**
     * Performs a single SVM fold with confusion matrix
     */
    private static double performSvmFoldWithConfusionMatrix(List<DataPoint> trainingData,
                                                          List<DataPoint> testData,
                                                          SvmConfiguration config,
                                                          int foldNumber) {
        SvmClassifier svmClassifier = new SvmClassifier(trainingData, config.kernelType, 
                                                       config.regularization, config.gamma, 
                                                       config.degree);
        int correctPredictions = 0;
        
        for (DataPoint testPoint : testData) {
            int predictedLabel = svmClassifier.classify(testPoint);
            testPoint.setPredictedLabel(predictedLabel);
            if (predictedLabel == testPoint.getActualLabel()) {
                correctPredictions++;
            }
        }
        
        double foldAccuracy = (double) correctPredictions / testData.size() * 100;
        System.out.printf("Accuracy: %.4f%% (%d/%d)\n", foldAccuracy, correctPredictions, testData.size());
        
        // Display confusion matrix
        System.out.println("Confusion Matrix:");
        displayConfusionMatrix(testData);
        
        return foldAccuracy;
    }
    
    /**
     * Tests SVM configuration and returns results
     * @return array [accuracy1, accuracy2, averageAccuracy]
     */
    private static double[] testSvmConfigurationSummarized(List<DataPoint> dataset1,
                                                         List<DataPoint> dataset2,
                                                         SvmConfiguration config) {
        double accuracy1 = testSvmSingleFold(dataset1, dataset2, config);
        double accuracy2 = testSvmSingleFold(dataset2, dataset1, config);
        double averageAccuracy = (accuracy1 + accuracy2) / 2;
        
        return new double[]{accuracy1, accuracy2, averageAccuracy};
    }
    
    /**
     * Tests a single k-NN fold without analysis
     * @param normalize whether to normalize features
     * @return accuracy percentage
     */
    private static double testKnnSingleFold(List<DataPoint> trainingData,
                                          List<DataPoint> testData,
                                          int kValue,
                                          String distanceMetric,
                                          boolean normalize) {
        KnnClassifier knnClassifier = new KnnClassifier(trainingData, distanceMetric, normalize, false);
        int correctPredictions = 0;
        
        for (DataPoint testPoint : testData) {
            int predictedLabel = knnClassifier.classify(testPoint, kValue);
            if (predictedLabel == testPoint.getActualLabel()) {
                correctPredictions++;
            }
        }
        
        return (double) correctPredictions / testData.size() * 100;
    }
    
    /**
     * Tests a single SVM fold without analysis
     */
    private static double testSvmSingleFold(List<DataPoint> trainingData,
                                          List<DataPoint> testData,
                                          SvmConfiguration config) {
        SvmClassifier svmClassifier = new SvmClassifier(trainingData, config.kernelType, 
                                                       config.regularization, config.gamma, 
                                                       config.degree);
        int correctPredictions = 0;
        
        for (DataPoint testPoint : testData) {
            int predictedLabel = svmClassifier.classify(testPoint);
            if (predictedLabel == testPoint.getActualLabel()) {
                correctPredictions++;
            }
        }
        
        return (double) correctPredictions / testData.size() * 100;
    }
    
    /**
     * Displays confusion matrix for a test dataset
     * @param testData test dataset with predicted labels
     */
    private static void displayConfusionMatrix(List<DataPoint> testData) {
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(NUMBER_OF_DIGIT_CLASSES);
        
        for (DataPoint dataPoint : testData) {
            confusionMatrix.addPrediction(dataPoint.getActualLabel(), dataPoint.getPredictedLabel());
        }
        
        confusionMatrix.printMatrix();
    }
    
    /**
     * Displays k-NN best performer summary
     */
    private static void displayKnnBestPerformer() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("k-NN BEST OVERALL PERFORMER");
        System.out.println("=".repeat(60));
        
        if (bestOverallAccuracy >= 0) {
            System.out.printf("Best Configuration: k=%d, metric=%s, normalization=%s\n",
                             bestOverallK, bestOverallMetric, bestOverallNormalization);
            System.out.printf("Best Average Accuracy: %.4f%%\n", bestOverallAccuracy);
            printKnnBaselineComparison(bestOverallAccuracy);
        }
    }
    
    /**
     * Prints k-NN baseline comparison
     * @param accuracy achieved accuracy
     */
    private static void printKnnBaselineComparison(double accuracy) {
        System.out.printf("UCI k-NN Baseline: %.4f%%\n", KNN_UCI_BASELINE);
        
        if (accuracy >= KNN_UCI_BASELINE) {
            System.out.printf("✓ SURPASSES BASELINE by +%.4f%%\n", accuracy - KNN_UCI_BASELINE);
        } else {
            System.out.printf("Below baseline by %.4f%%\n", KNN_UCI_BASELINE - accuracy);
        }
    }
    
    /**
     * Prints SVM baseline comparison
     * @param accuracy achieved accuracy
     */
    private static void printSvmBaselineComparison(double accuracy) {
        System.out.printf("UCI SVM Baseline: %.3f%% - %.3f%%\n", 
                         SVM_UCI_BASELINE_MIN, SVM_UCI_BASELINE_MAX);
        
        if (accuracy >= SVM_UCI_BASELINE_MIN) {
            if (accuracy >= SVM_UCI_BASELINE_MAX) {
                System.out.printf("✓ EXCEEDS UCI SVM baseline by +%.4f%%\n", 
                                accuracy - SVM_UCI_BASELINE_MAX);
            } else {
                System.out.println("✓ WITHIN UCI SVM baseline range");
            }
        } else {
            System.out.printf("Below baseline by %.4f%%\n", SVM_UCI_BASELINE_MIN - accuracy);
        }
    }
    
    /**
     * Configuration container class for SVM hyperparameters
     */
    private static class SvmConfiguration {
        String kernelType;
        double regularization;
        double gamma;
        int degree;
        double accuracy;
        
        SvmConfiguration(String kernelType, double regularization, double gamma, int degree) {
            this.kernelType = kernelType;
            this.regularization = regularization;
            this.gamma = gamma;
            this.degree = degree;
            this.accuracy = 0.0;
        }
    }
}
