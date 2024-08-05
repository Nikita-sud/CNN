package cnn.layers;

import cnn.utils.MatrixUtils;
import cnn.utils.ReLU;
import cnn.interfaces.ActivationFunction;
import cnn.interfaces.AdaptiveLayer;
import cnn.interfaces.ParameterizedLayer;

public class ConvolutionalLayer implements AdaptiveLayer, ParameterizedLayer{
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

    public ConvolutionalLayer(int filterSize, int numFilters, int stride, ActivationFunction activationFunction) {
        this.filterSize = filterSize;
        this.numFilters = numFilters;
        this.stride = stride;
        this.activationFunction = activationFunction;
    }

    public ConvolutionalLayer(int filterSize, int numFilters, ActivationFunction activationFunction) {
        this(filterSize, numFilters, 1, activationFunction);
    }

    public ConvolutionalLayer(int filterSize, int numFilters) {
        this(filterSize, numFilters, 1, new ReLU());
    }

    private void initializeFilters(int inputDepth) {
        filters = new double[numFilters][inputDepth][filterSize][filterSize];
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filters[f][d][i][j] = Math.random() * Math.sqrt(2.0 / (inputDepth * filterSize * filterSize));
                    }
                }
            }
        }
    }

    private void initializeBiases() {
        biases = new double[numFilters];
        for (int i = 0; i < numFilters; i++) {
            biases[i] = Math.random();
        }
    }

    private void initializeAccumulatedGradients() {
        accumulatedFilterGradients = new double[numFilters][][][];
        for (int f = 0; f < numFilters; f++) {
            accumulatedFilterGradients[f] = new double[filters[f].length][filterSize][filterSize];
        }
        accumulatedBiasGradients = new double[numFilters];
    }

    @Override
    public void initialize(int... inputShape) {
        int inputDepth = inputShape[0];
        initializeFilters(inputDepth);
        initializeBiases();
        initializeAccumulatedGradients();
    }

    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        int inputDepth = input.length;
        int inputSize = input[0].length;
        int outputSize = (inputSize - filterSize) / stride + 1;

        if (filters[0] == null || filters[0].length != inputDepth) {
            initializeFilters(inputDepth);
        }

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

    @Override
    public double[][][] backward(double[][][] gradient) {
        int inputDepth = input.length;
        int inputSize = input[0].length;
        int outputSize = activatedOutput[0].length;
        double[][][] inputGradient = new double[inputDepth][inputSize][inputSize];

        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    gradient[f][i][j] *= activationFunction.derivative(activatedOutput[f][i][j]);
                }
            }
        }

        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                double[][] filterGrad = MatrixUtils.convolve(input[d], gradient[f], stride);
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        accumulatedFilterGradients[f][d][i][j] += filterGrad[i][j];
                    }
                }
                double[][] rotatedFilter = MatrixUtils.rotate180(filters[f][d]);
                double[][] inputGrad = MatrixUtils.fullConvolve(rotatedFilter, gradient[f]);
                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        inputGradient[d][i][j] += inputGrad[i][j];
                    }
                }
            }
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    accumulatedBiasGradients[f] += gradient[f][i][j];
                }
            }
        }

        return inputGradient;
    }

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

    @Override
    public int[] getOutputShape(int... inputShape) {
        int inputSize = inputShape[1];
        int outputSize = (inputSize - filterSize) / stride + 1;
        return new int[]{numFilters, outputSize, outputSize};
    }
}
