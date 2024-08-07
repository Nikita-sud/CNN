package cnn.utils;

import cnn.interfaces.ActivationFunction;

/**
 * Sigmoid activation function.
 * The Sigmoid function is defined as:
 * f(x) = 1 / (1 + exp(-x))
 * This activation function is used to introduce non-linearity in the network.
 */
public class Sigmoid implements ActivationFunction {

    /**
     * Applies the Sigmoid activation function to a single input value.
     *
     * @param x the input value to be activated
     * @return the activated output value, which is 1 / (1 + exp(-x))
     */
    @Override
    public double activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Computes the derivative of the Sigmoid activation function for a given input value.
     *
     * @param x the input value for which the derivative is calculated
     * @return the derivative of the Sigmoid function, which is sigmoid(x) * (1 - sigmoid(x))
     */
    @Override
    public double derivative(double x) {
        double sigmoid = activate(x);
        return sigmoid * (1 - sigmoid);
    }
}