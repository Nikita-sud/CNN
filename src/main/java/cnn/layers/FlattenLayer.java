package cnn.layers;

import cnn.interfaces.AdaptiveLayer;

/**
 * A layer that flattens a 3D input tensor (depth, height, width) into a 1D output tensor.
 * This layer is commonly used in the transition from convolutional to fully connected layers in a neural network.
 */
public class FlattenLayer implements AdaptiveLayer {

    private int depth;
    private int height;
    private int width;

    /**
     * Initializes the layer with the shape of the input tensor.
     *
     * @param inputShape an array of three integers representing the depth, height, and width of the input tensor.
     * @throws IllegalArgumentException if the input shape does not have exactly three dimensions.
     */
    @Override
    public void initialize(int... inputShape) {
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Expected input shape with 3 dimensions (depth, height, width).");
        }
        this.depth = inputShape[0];
        this.height = inputShape[1];
        this.width = inputShape[2];
    }

    /**
     * Performs the forward pass by flattening the input 3D tensor into a 1D tensor.
     *
     * @param input a 3D array representing the input tensor.
     * @return a 3D array with shape [1][1][depth * height * width] representing the flattened output tensor.
     * @throws IllegalArgumentException if the input dimensions do not match the initialized shape.
     */
    @Override
    public double[][][] forward(double[][][] input) {
        if (input.length != depth || input[0].length != height || input[0][0].length != width) {
            throw new IllegalArgumentException("Input dimensions do not match the initialized shape.");
        }

        double[][][] output = new double[1][1][depth * height * width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    output[0][0][d * height * width + i * width + j] = input[d][i][j];
                }
            }
        }

        return output;
    }

    /**
     * Performs the backward pass by reshaping the gradient from a 1D tensor back to the original 3D tensor shape.
     *
     * @param gradient a 3D array with shape [1][1][depth * height * width] representing the gradient.
     * @return a 3D array with the original input shape [depth][height][width] representing the reshaped gradient.
     * @throws IllegalArgumentException if the gradient dimensions do not match the expected flattened shape.
     */
    @Override
    public double[][][] backward(double[][][] gradient) {
        if (gradient[0][0].length != depth * height * width) {
            throw new IllegalArgumentException("Gradient dimensions do not match the expected shape.");
        }

        double[][][] reshapedGradient = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    reshapedGradient[d][i][j] = gradient[0][0][d * height * width + i * width + j];
                }
            }
        }

        return reshapedGradient;
    }

    /**
     * Computes the output shape of the layer given the input shape.
     *
     * @param inputShape an array of three integers representing the depth, height, and width of the input tensor.
     * @return an array with one integer representing the flattened size of the output tensor.
     * @throws IllegalArgumentException if the input shape does not have exactly three dimensions.
     */
    @Override
    public int[] getOutputShape(int... inputShape) {
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Expected input shape with 3 dimensions (depth, height, width).");
        }
        int flatSize = inputShape[0] * inputShape[1] * inputShape[2];
        return new int[]{flatSize};
    }
}