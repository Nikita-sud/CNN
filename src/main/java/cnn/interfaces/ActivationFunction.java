package cnn.interfaces;

public interface ActivationFunction {
    double activate(double x);
    double derivative(double x);
}
