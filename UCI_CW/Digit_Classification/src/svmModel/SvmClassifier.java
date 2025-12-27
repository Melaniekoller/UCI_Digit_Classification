package svmModel;

import common.data.DataPoint;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Enhanced SVM classifier for handwritten digit recognition
 * Uses One-vs-Rest multi-class classification with improved optimization
 */
public class SvmClassifier {
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
         */
        private double calculatePredictionError(int exampleIndex) {
            return decisionFunction(trainingData.get(exampleIndex).getFeatures()) - 
                   (getBinaryLabel(trainingData.get(exampleIndex)) ? 
                    POSITIVE_CLASS_LABEL : NEGATIVE_CLASS_LABEL); // returns the prediction error
        }
        
        /**
         * Updates bias term using updated Lagrange multipliers
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
         * @param firstExampleIndex index of first training example
         * @param secondExampleIndex index of second training example
         */
        private double computeKernelBetweenExamples(int firstExampleIndex, int secondExampleIndex) {
            return computeKernelFunction(
                trainingData.get(firstExampleIndex).getFeatures(),
                trainingData.get(secondExampleIndex).getFeatures()
            ); // returns kernel value
        }
        
        /**
         * Computes kernel function between two feature vectors
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
         */
        public double decisionFunction(double[] featureVector) {
            double decisionSum = 0.0;
            for (SupportVector currentSupportVector : supportVectors) {
                decisionSum += currentSupportVector.lagrangeMultiplier * 
                              currentSupportVector.classLabel * 
                              computeKernelFunction(currentSupportVector.features, featureVector);
            }
            return decisionSum - biasTerm; // returns the decision function value
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