package cnn.interfaces;

/**
 * An interface representing a layer in a neural network.
 * A layer can perform forward and backward propagation, and determine the output shape given an input shape.
 */
public interface Layer {

    /**
     * Performs the forward pass through the layer.
     *
     * @param input a 3D array representing the input tensor
     * @return a 3D array representing the output tensor after processing by the layer
     */
    double[][][] forward(double[][][] input);

    /**
     * Performs the backward pass through the layer, calculating the gradient of the loss with respect to the input.
     *
     * @param gradient a 3D array representing the gradient of the loss with respect to the output
     * @return a 3D array representing the gradient of the loss with respect to the input
     */
    double[][][] backward(double[][][] gradient);

    /**
     * Computes the output shape of the layer given the input shape.
     *
     * @param inputShape an array of integers representing the dimensions of the input tensor
     * @return an array of integers representing the dimensions of the output tensor
     */
    int[] getOutputShape(int... inputShape);
}