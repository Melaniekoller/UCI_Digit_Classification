package mlpModel;
import java.util.Random;
/**
 * Represents a single layer in the Multi-Layer Perceptron
 * Each layer contains neurons with weights, biases, and activation functions
 */
public class Layer {
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
        this.activation = activation; //  activation function type ("sigmoid", "relu" , "softmax")
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
            	//gradient *= Activation.softmaxDerivative(outputs[i]);
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

