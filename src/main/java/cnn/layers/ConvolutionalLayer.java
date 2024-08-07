package cnn.layers;

import cnn.utils.MatrixUtils;
import cnn.utils.ReLU;

import java.util.Random;

import cnn.interfaces.ActivationFunction;
import cnn.interfaces.AdaptiveLayer;
import cnn.interfaces.ParameterizedLayer;

/**
 * A convolutional layer in a neural network.
 * This layer applies a set of learnable filters to the input tensor, followed by an activation function.
 */
public class ConvolutionalLayer implements AdaptiveLayer, ParameterizedLayer {
    private int filterSize;
    private int numFilters;
    private int stride;
    private double[][][][] filters;
    private double[] biases;
    private double[][][] input;
    private double[][][] activatedOutput;
    private ActivationFunction activationFunction;
    private double[][][][] accumulatedFilterGradients;
    private double[] accumulatedBiasGradients;

    /**
     * Constructs a ConvolutionalLayer with the specified filter size, number of filters, stride, and activation function.
     *
     * @param filterSize the size of the filter
     * @param numFilters the number of filters
     * @param stride the stride of the convolution
     * @param activationFunction the activation function to apply
     */
    public ConvolutionalLayer(int filterSize, int numFilters, int stride, ActivationFunction activationFunction) {
        this.filterSize = filterSize;
        this.numFilters = numFilters;
        this.stride = stride;
        this.activationFunction = activationFunction;
    }

    /**
     * Constructs a ConvolutionalLayer with the specified filter size, number of filters, and activation function.
     * Uses a default stride of 1.
     *
     * @param filterSize the size of the filter
     * @param numFilters the number of filters
     * @param activationFunction the activation function to apply
     */
    public ConvolutionalLayer(int filterSize, int numFilters, ActivationFunction activationFunction) {
        this(filterSize, numFilters, 1, activationFunction);
    }

    /**
     * Constructs a ConvolutionalLayer with the specified filter size and number of filters.
     * Uses a default stride of 1 and ReLU activation function.
     *
     * @param filterSize the size of the filter
     * @param numFilters the number of filters
     */
    public ConvolutionalLayer(int filterSize, int numFilters) {
        this(filterSize, numFilters, 1, new ReLU());
    }

    /**
     * Initializes the filters with random values.
     *
     * @param inputDepth the depth of the input tensor
     */
    private void initializeFilters(int inputDepth) {
        filters = new double[numFilters][inputDepth][filterSize][filterSize];
        Random rand = new Random();
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filters[f][d][i][j] = rand.nextGaussian() * Math.sqrt(2.0 / (inputDepth * filterSize * filterSize));
                    }
                }
            }
        }
    }

    /**
     * Initializes the biases with random values.
     */
    private void initializeBiases() {
        biases = new double[numFilters];
        for (int i = 0; i < numFilters; i++) {
            biases[i] = Math.random();
        }
    }

    /**
     * Initializes the accumulated gradients to zero.
     */
    private void initializeAccumulatedGradients() {
        accumulatedFilterGradients = new double[numFilters][][][];
        for (int f = 0; f < numFilters; f++) {
            accumulatedFilterGradients[f] = new double[filters[f].length][filterSize][filterSize];
        }
        accumulatedBiasGradients = new double[numFilters];
    }

    /**
     * Initializes the layer with the given input shape.
     *
     * @param inputShape an array of integers representing the dimensions of the input tensor
     */
    @Override
    public void initialize(int... inputShape) {
        int inputDepth = inputShape[0];
        initializeFilters(inputDepth);
        initializeBiases();
        initializeAccumulatedGradients();
    }

    /**
     * Performs the forward pass through the convolutional layer.
     * Applies the convolution operation followed by the activation function.
     *
     * @param input a 3D array representing the input tensor
     * @return a 3D array representing the output tensor after convolution and activation
     */
    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        int inputDepth = input.length;
        int inputSize = input[0].length;
        int outputSize = (inputSize - filterSize) / stride + 1;

        double[][][] output = new double[numFilters][outputSize][outputSize];
        this.activatedOutput = new double[numFilters][outputSize][outputSize];

        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    int x = i * stride;
                    int y = j * stride;
                    double sum = 0;
                    for (int d = 0; d < inputDepth; d++) {
                        sum += MatrixUtils.applyFilter(input[d], filters[f][d], x, y);
                    }
                    output[f][i][j] = sum + biases[f];
                    activatedOutput[f][i][j] = activationFunction.activate(output[f][i][j]);
                }
            }
        }

        return activatedOutput;
    }

    /**
     * Performs the backward pass through the convolutional layer.
     * Computes the gradients of the loss with respect to the input tensor, filters, and biases.
     *
     * @param gradient a 3D array representing the gradient of the loss with respect to the output
     * @return a 3D array representing the gradient of the loss with respect to the input
     */
    @Override
    public double[][][] backward(double[][][] gradient) {
        int inputDepth = input.length;
        int inputSize = input[0].length;
        int outputSize = activatedOutput[0].length;
        double[][][] inputGradient = new double[inputDepth][inputSize][inputSize];

        // Backpropagation through activation function
        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    gradient[f][i][j] *= activationFunction.derivative(activatedOutput[f][i][j]);
                }
            }
        }

        // Calculate gradients for filters and inputs
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                // Calculate gradient for filters
                double[][] filterGrad = MatrixUtils.convolve(input[d], gradient[f], stride);
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        accumulatedFilterGradients[f][d][i][j] += filterGrad[i][j];
                    }
                }

                // Calculate gradient for input
                double[][] rotatedFilter = MatrixUtils.rotate180(filters[f][d]);
                double[][] inputGrad = MatrixUtils.fullConvolve(rotatedFilter, gradient[f]);
                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        inputGradient[d][i][j] += inputGrad[i][j];
                    }
                }
            }

            // Calculate gradient for biases
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    accumulatedBiasGradients[f] += gradient[f][i][j];
                }
            }
        }

        return inputGradient;
    }

    /**
     * Updates the parameters (filters and biases) of the layer using the accumulated gradients.
     *
     * @param learningRate the learning rate to use for updating the parameters
     * @param miniBatchSize the size of the mini-batch used for averaging the gradients
     */
    @Override
    public void updateParameters(double learningRate, int miniBatchSize) {
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < filters[f].length; d++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filters[f][d][i][j] -= learningRate * accumulatedFilterGradients[f][d][i][j] / miniBatchSize;
                        accumulatedFilterGradients[f][d][i][j] = 0;
                    }
                }
            }
            biases[f] -= learningRate * accumulatedBiasGradients[f] / miniBatchSize;
            accumulatedBiasGradients[f] = 0;
        }
    }

    /**
     * Resets the accumulated gradients for filters and biases to zero.
     */
    @Override
    public void resetGradients() {
        if (filters[0] == null) {
            return;
        }
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < filters[f].length; d++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        accumulatedFilterGradients[f][d][i][j] = 0;
                    }
                }
            }
            accumulatedBiasGradients[f] = 0;
        }
    }

    /**
     * Computes the output shape of the layer given the input shape.
     *
     * @param inputShape an array of integers representing the dimensions of the input tensor
     * @return an array of integers representing the dimensions of the output tensor
     */
    @Override
    public int[] getOutputShape(int... inputShape) {
        int inputSize = inputShape[1];
        int outputSize = (inputSize - filterSize) / stride + 1;
        return new int[]{numFilters, outputSize, outputSize};
    }
}
