package cnn.utils;

import cnn.interfaces.ActivationFunction;

/**
 * Hyperbolic Tangent (Tanh) activation function.
 * The Tanh function is defined as:
 * f(x) = tanh(x)
 * This activation function is used to introduce non-linearity in the network.
 */
public class Tanh implements ActivationFunction {

    /**
     * Applies the Tanh activation function to a single input value.
     *
     * @param x the input value to be activated
     * @return the activated output value, which is tanh(x)
     */
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    /**
     * Computes the derivative of the Tanh activation function for a given input value.
     *
     * @param x the input value for which the derivative is calculated
     * @return the derivative of the Tanh function, which is 1 - tanh^2(x)
     */
    @Override
    public double derivative(double x) {
        double tanh = activate(x);
        return 1 - tanh * tanh;
    }
}