package cnn.utils.activationFunctions;

import java.io.Serializable;

import cnn.interfaces.ActivationFunction;

/**
 * Leaky Rectified Linear Unit (LeakyReLU) activation function.
 * LeakyReLU is defined as:
 * f(x) = x if x > 0, alpha * x if x <= 0
 * This activation function is used to introduce non-linearity in the network while allowing a small gradient when the unit is not active.
 */
public class LeakyReLU implements ActivationFunction, Serializable {
    private static final long serialVersionUID = 1L;
    private double alpha;

    /**
     * Constructs a LeakyReLU activation function with the specified alpha parameter.
     *
     * @param alpha the alpha parameter, which controls the slope of the function for negative inputs
     */
    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Applies the LeakyReLU activation function to a single input value.
     *
     * @param x the input value to be activated
     * @return the activated output value, which is x if x > 0, otherwise alpha * x
     */
    @Override
    public double activate(double x) {
        return x > 0 ? x : alpha * x;
    }

    /**
     * Computes the derivative of the LeakyReLU activation function for a given input value.
     *
     * @param x the input value for which the derivative is calculated
     * @return the derivative of the LeakyReLU function, which is 1 if x > 0, otherwise alpha
     */
    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : alpha;
    }
}