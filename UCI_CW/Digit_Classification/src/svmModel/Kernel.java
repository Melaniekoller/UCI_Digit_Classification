package svmModel;

/**
 * Provides kernel functions for SVM
 * Kernels transform data into higher-dimensional space for linear separation
 */
public class Kernel {
    
    // Named constants for common kernel parameter values
    private static final double DEFAULT_POLYNOMIAL_CONSTANT = 1.0;
    private static final int DEFAULT_POLYNOMIAL_DEGREE = 2;
    private static final double DEFAULT_RBF_GAMMA = 0.1;
    private static final double DEFAULT_SIGMOID_ALPHA = 0.01;
    private static final double DEFAULT_SIGMOID_CONSTANT = 0.0;
    private static final int MINIMUM_POLYNOMIAL_DEGREE = 1;
    		
    /**
     * Linear kernel - simplest kernel, works well for linearly separable data
     * @param x first feature vector (64 dimensions for digit data)
     * @param y second feature vector (64 dimensions for digit data)
     */
    public static double linear(double[] x, double[] y) {
        double dotProductResult = 0.0;
        for (int featureIndex = 0; featureIndex < x.length; featureIndex++) {
        	dotProductResult += x[featureIndex] * y[featureIndex];
        }
        return dotProductResult; // returns dot product of the two vectors
    }
    
    /**
     * Polynomial kernel with default - can capture nonlinear relationships
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
     * @param gamma kernel width parameter (γ)
     */
    
    public static double rbf(double[] firstVector, double[] secondVector) {
        return rbf(firstVector, secondVector, DEFAULT_RBF_GAMMA); //RBF kernel value between 0 and 1
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