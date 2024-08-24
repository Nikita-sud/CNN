package cnn.utils.activationFunctions;

import java.io.Serializable;

import cnn.interfaces.ActivationFunction;

/**
 * Exponential Linear Unit (ELU) activation function.
 * ELU is defined as:
 * f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
 * This activation function is used to introduce non-linearity in the network.
 */
public class ELU implements ActivationFunction, Serializable{
    private double alpha;

    /**
     * Constructs an ELU activation function with the specified alpha parameter.
     *
     * @param alpha the alpha parameter, which controls the value to which an ELU saturates for negative net inputs
     */
    public ELU(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Applies the ELU activation function to a single input value.
     *
     * @param x the input value to be activated
     * @return the activated output value
     */
    @Override
    public double activate(double x) {
        return x > 0 ? x : alpha * (Math.exp(x) - 1);
    }

    /**
     * Computes the derivative of the ELU activation function for a given input value.
     *
     * @param x the input value for which the derivative is calculated
     * @return the derivative of the ELU function at the given input value
     */
    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : alpha * Math.exp(x);
    }
}