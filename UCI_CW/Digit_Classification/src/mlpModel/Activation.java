package mlpModel;

/**
 * Provides activation functions for the Multi-Layer Perceptron
 */
public class Activation {
    
    /**
     * ACTIVATION FUNCTION OPTION 1: SIGMOID 
     */
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x)); // output between 0 and 1
    }
    
    /**
     * Derivative of sigmoid function - used in backpropagation
     */
    public static double sigmoidDerivative(double x) {
    	// x input value (sigmoid output)
        return x * (1 - x);
    }
    
    
    /**
     * ACTIVATION FUNCTION OPTION 2: RECTIFIED LINEAR UNIT (ReLU) 
     */
    public static double relu(double x) {
        return Math.max(0, x);
    }
    
    /**
     * Derivative of ReLU function - used in backpropagation
     */
    public static double reluDerivative(double x) {
    	// x input value (sigmoid output)
        return x > 0 ? 1.0 : 0.0;
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