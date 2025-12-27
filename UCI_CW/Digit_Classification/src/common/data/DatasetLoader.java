package common.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles loading and parsing of UCI Handwritten Digits datasets
 */
public class DatasetLoader {
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