package cnn.utils;

import cnn.interfaces.ActivationFunction;

public class ReLU implements ActivationFunction {
    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
}