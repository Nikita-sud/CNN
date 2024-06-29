package cnn.interfaces;

public interface Layer {
    double[][][] forward(double[][][] input);
    double[][][] backward(double[][][] gradient);
}