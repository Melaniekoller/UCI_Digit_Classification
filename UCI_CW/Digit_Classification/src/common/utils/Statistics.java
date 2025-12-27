package common.utils;

/**
 * Utility class for statistical calculations
 */

public class Statistics {
	public static double calculateAccuracy(int correct, int total) {
        return (double) correct / total * 100;
    }
} 


