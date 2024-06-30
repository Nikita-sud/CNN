package cnn.utils;

import cnn.interfaces.ActivationFunction;

public class Sigmoid implements ActivationFunction {
    @Override
    public double activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        double sigmoid = activate(x);
        return sigmoid * (1 - sigmoid);
    }
}