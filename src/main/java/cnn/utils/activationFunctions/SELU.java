package cnn.utils.activationFunctions;

import java.io.Serializable;
import cnn.interfaces.ActivationFunction;

/**
 * Scaled Exponential Linear Unit (SELU) activation function.
 * SELU is defined as:
 * f(x) = lambda * x if x > 0, lambda * alpha * (exp(x) - 1) if x <= 0
 * SELU automatically normalizes the mean and variance of the activations
 * when used in a neural network with appropriate weight initialization.
 */
public class SELU implements ActivationFunction, Serializable {
    private static final double ALPHA = 1.6732632423543772848170429916717;
    private static final double LAMBDA = 1.0507009873554804934193349852946;

    /**
     * Applies the SELU activation function to a single input value.
     *
     * @param x the input value to be activated
     * @return the activated output value
     */
    @Override
    public double activate(double x) {
        return x > 0 ? LAMBDA * x : LAMBDA * ALPHA * (Math.exp(x) - 1);
    }

    /**
     * Computes the derivative of the SELU activation function for a given input value.
     *
     * @param x the input value for which the derivative is calculated
     * @return the derivative of the SELU function at the given input value
     */
    @Override
    public double derivative(double x) {
        return x > 0 ? LAMBDA : LAMBDA * ALPHA * Math.exp(x);
    }
}