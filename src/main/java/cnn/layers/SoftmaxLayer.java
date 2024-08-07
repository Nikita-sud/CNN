package cnn.layers;

import java.io.Serializable;

import cnn.interfaces.Layer;

/**
 * A softmax layer in a neural network.
 * This layer applies the softmax function to the input tensor, which is typically used as the final layer in a classification network.
 */
public class SoftmaxLayer implements Layer, Serializable{

    @SuppressWarnings("unused")
    private double[][][] input;

    /**
     * Performs the forward pass through the softmax layer.
     * Applies the softmax function to the input tensor.
     *
     * @param input a 3D array representing the input tensor
     * @return a 3D array representing the output tensor after applying the softmax function
     */
    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        double[] flattenedInput = input[0][0];
        double[] softmaxOutput = softmax(flattenedInput);
        return new double[][][]{{softmaxOutput}};
    }

    /**
     * Applies the softmax function to the input array.
     *
     * @param input a 1D array representing the input values
     * @return a 1D array representing the softmax probabilities
     */
    private double[] softmax(double[] input) {
        double[] output = new double[input.length];
        double max = Double.NEGATIVE_INFINITY;

        for (double v : input) {
            if (v > max) {
                max = v;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    /**
     * Performs the backward pass through the softmax layer.
     * For the softmax layer, the backward pass typically returns the gradient unchanged.
     *
     * @param gradient a 3D array representing the gradient of the loss with respect to the output
     * @return a 3D array representing the gradient of the loss with respect to the input
     */
    @Override
    public double[][][] backward(double[][][] gradient) {
        return gradient;
    }

    /**
     * Computes the output shape of the layer given the input shape.
     *
     * @param inputShape an array of integers representing the dimensions of the input tensor
     * @return an array of integers representing the dimensions of the output tensor
     */
    @Override
    public int[] getOutputShape(int... inputShape) {
        return new int[]{inputShape[0]};
    }
}