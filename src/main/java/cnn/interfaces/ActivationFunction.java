package cnn.interfaces;

public interface ActivationFunction {
    double activate(double x);
    double derivative(double x);
    
    default double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = activate(input[i]);
        }
        return output;
    }
}