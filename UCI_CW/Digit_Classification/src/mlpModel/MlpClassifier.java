package mlpModel;

import common.data.DataPoint;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Fixed Multi-Layer Perceptron (MLP) neural network for handwritten digit classification
 * Uses proper softmax output and cross-entropy loss for multi-class classification
 */
public class MlpClassifier {
    private List<Layer> layers;
    private double learningRate;
    private int epochs;
    private Random random;
    
    public MlpClassifier(double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
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
     */
    public MlpClassifier(String architecture, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.random = new Random();
        this.layers = new ArrayList<>();
        
        initializeArchitecture(architecture);
        System.out.println("MLP initialized: " + architecture + " architecture");
        System.out.println("Learning rate: " + learningRate + ", Epochs: " + epochs);
    }
    
    /**
     * Parse architecture string like "64-32-10" or "64-128-64-10"
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
    
    public int classify(DataPoint dataPoint) {
        double[] outputs = forward(dataPoint.getFeatures());
        double[] probabilities = Activation.softmax(outputs);
        return predict(probabilities);
    }
    
    private int predict(double[] probabilities) {
        int maxIndex = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    private double[] oneHotEncode(int label) {
        double[] encoded = new double[10];
        encoded[label] = 1.0;
        return encoded;
    }
    
    /**
     * Cross-entropy loss for multi-class classification
     * Better suited for classification than mean squared error
     */
    private double crossEntropyLoss(double[] probabilities, double[] targets) {
        double loss = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            if (targets[i] == 1.0) {
                // Add small epsilon to avoid log(0)
                loss += -Math.log(probabilities[i] + 1e-8);
            }
        }
        return loss;
    }
}