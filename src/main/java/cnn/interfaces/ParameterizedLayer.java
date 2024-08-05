package cnn.interfaces;

public interface ParameterizedLayer extends Layer {
    void updateParameters(double learningRate, int miniBatchSize);
    void resetGradients();
}
