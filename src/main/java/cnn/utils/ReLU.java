package cnn.utils;

import java.io.Serializable;

import cnn.interfaces.ActivationFunction;

/**
 * Rectified Linear Unit (ReLU) activation function.
 * ReLU is defined as:
 * f(x) = max(0, x)
 * This activation function is used to introduce non-linearity in the network.
 */
public class ReLU implements ActivationFunction, Serializable{

    /**
     * Applies the ReLU activation function to a single input value.
     *
     * @param x the input value to be activated
     * @return the activated output value, which is max(0, x)
     */
    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }

    /**
     * Computes the derivative of the ReLU activation function for a given input value.
     *
     * @param x the input value for which the derivative is calculated
     * @return the derivative of the ReLU function, which is 1 if x > 0, otherwise 0
     */
    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
}