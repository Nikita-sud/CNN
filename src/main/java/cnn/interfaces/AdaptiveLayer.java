package cnn.interfaces;

/**
 * An interface representing an adaptive layer in a neural network.
 * An adaptive layer can be initialized with a specific input shape.
 */
public interface AdaptiveLayer extends Layer {

    /**
     * Initializes the layer with the given input shape.
     *
     * @param inputShape an array of integers representing the dimensions of the input tensor
     * @throws IllegalArgumentException if the input shape is invalid
     */
    void initialize(int... inputShape);
}