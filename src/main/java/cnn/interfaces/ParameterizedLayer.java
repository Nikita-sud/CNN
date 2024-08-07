package cnn.interfaces;

/**
 * An interface representing a parameterized layer in a neural network.
 * A parameterized layer has learnable parameters that can be updated during training.
 */
public interface ParameterizedLayer extends Layer {

    /**
     * Updates the parameters of the layer using the accumulated gradients.
     *
     * @param learningRate the learning rate to use for updating the parameters
     * @param miniBatchSize the size of the mini-batch used for averaging the gradients
     */
    void updateParameters(double learningRate, int miniBatchSize);

    /**
     * Resets the accumulated gradients to zero.
     */
    void resetGradients();
}