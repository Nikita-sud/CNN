package cnn.layers;

import cnn.interfaces.Layer;

import java.io.Serializable;
import java.util.Random;

/**
 * A dropout layer in a neural network, which randomly sets a fraction of input units to zero during training.
 * This layer helps prevent overfitting by introducing noise during training.
 */
public class DropoutLayer implements Layer, Serializable {
    private double rate;
    private double[][][] mask;
    private boolean isTraining;

    /**
     * Constructs a DropoutLayer with the specified dropout rate.
     *
     * @param rate the probability of dropping out a unit, between 0.0 and 1.0
     */
    public DropoutLayer(double rate) {
        this.rate = rate;
        this.isTraining = true;
    }

    /**
     * Performs the forward pass through the dropout layer. During training, randomly sets a fraction of input units to zero.
     *
     * @param input a 3D array representing the input tensor
     * @return a 3D array representing the output tensor after applying dropout
     */
    @Override
    public double[][][] forward(double[][][] input) {
        if (!isTraining) {
            return input;
        }

        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        mask = new double[depth][height][width];
        Random random = new Random();

        double[][][] output = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    mask[d][i][j] = random.nextDouble() > rate ? 1.0 : 0.0;
                    output[d][i][j] = input[d][i][j] * mask[d][i][j];
                }
            }
        }

        return output;
    }

    /**
     * Performs the backward pass through the dropout layer, scaling the gradient by the dropout mask.
     *
     * @param gradient a 3D array representing the gradient of the loss with respect to the output
     * @return a 3D array representing the gradient of the loss with respect to the input
     * @throws IllegalStateException if the dropout mask dimensions do not match the gradient dimensions
     */
    @Override
    public double[][][] backward(double[][][] gradient) {
        if (!isTraining) {
            return gradient;
        }

        int depth = gradient.length;
        int height = gradient[0].length;
        int width = gradient[0][0].length;
        
        if (mask == null || mask.length != depth || mask[0].length != height || mask[0][0].length != width) {
            throw new IllegalStateException("Dropout mask dimensions do not match gradient dimensions");
        }

        double[][][] outputGradient = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    outputGradient[d][i][j] = gradient[d][i][j] * mask[d][i][j];
                }
            }
        }

        return outputGradient;
    }

    /**
     * Sets the training mode of the dropout layer.
     *
     * @param isTraining true if the layer is in training mode, false if in inference mode
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Computes the output shape of the layer given the input shape.
     *
     * @param inputShape an array of integers representing the dimensions of the input tensor
     * @return an array of integers representing the dimensions of the output tensor, which is the same as the input shape
     */
    @Override
    public int[] getOutputShape(int... inputShape) {
        return inputShape; // Dropout does not change the dimensions
    }
}