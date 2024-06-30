package cnn.utils;

import cnn.interfaces.ActivationFunction;

public class Tanh implements ActivationFunction {
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        double tanh = activate(x);
        return 1 - tanh * tanh;
    }
}