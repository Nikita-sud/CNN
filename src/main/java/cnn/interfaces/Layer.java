package cnn.interfaces;

public interface Layer {
    double[][][] forward(double[][][] input);
    double[][][] backward(double[][][] gradient);
    void updateParameters(double learningRate, int miniBatchSize);
    void resetGradients();
}