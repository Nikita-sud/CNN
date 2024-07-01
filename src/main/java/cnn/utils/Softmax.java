package cnn.utils;

import cnn.interfaces.ActivationFunction;
// import java.util.Arrays;

public class Softmax implements ActivationFunction {
    @Override
    public double activate(double x) {
        // Softmax не применяется по одному элементу, поэтому этот метод не используется
        throw new UnsupportedOperationException("Use the activate(double[] input) method for Softmax.");
    }

    @Override
    public double derivative(double x) {
        // Производная Softmax не применяется по одному элементу, поэтому этот метод не используется
        throw new UnsupportedOperationException("Use the derivative(double[] output, double[] target) method for Softmax.");
    }

    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        double max = Double.NEGATIVE_INFINITY;
        
        for (double v : input) {
            if (v > max) {
                max = v;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        // System.out.println("Softmax output: " + Arrays.toString(output));

        return output;
    }

    public double[] derivative(double[] output, double[] target) {
        double[] gradient = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[i] = output[i] - target[i];
        }
        return gradient;
    }
}