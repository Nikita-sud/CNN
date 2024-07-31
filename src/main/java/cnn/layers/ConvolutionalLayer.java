package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;
import cnn.utils.ReLU;
import cnn.utils.TrainingConfig;
import cnn.interfaces.ActivationFunction;

public class ConvolutionalLayer implements Layer {
    private int filterSize;
    private int numFilters;
    private int stride;
    private double[][][][] filters;
    private double[] biases;
    private double[][][] input;
    private double[][][] activatedOutput;
    private ActivationFunction activationFunction;
    private TrainingConfig config;

    public ConvolutionalLayer(int filterSize, int numFilters, int stride, ActivationFunction activationFunction, TrainingConfig config) {
        this.filterSize = filterSize;
        this.numFilters = numFilters;
        this.stride = stride;
        this.filters = new double[numFilters][][][];
        this.biases = new double[numFilters];
        this.activationFunction = activationFunction;
        this.config = config;
        initializeBiases();
    }

    public ConvolutionalLayer(int filterSize, int numFilters, ActivationFunction activationFunction, TrainingConfig config) {
        this(filterSize, numFilters, 1, activationFunction, config);
    }

    public ConvolutionalLayer(int filterSize, int numFilters, TrainingConfig config) {
        this(filterSize, numFilters, 1, new ReLU(), config);
    }

    private void initializeFilters(int inputDepth) {
        for (int f = 0; f < numFilters; f++) {
            filters[f] = new double[inputDepth][filterSize][filterSize];
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
        for (int i = 0; i < numFilters; i++) {
            biases[i] = Math.random();
        }
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
        double[][][][] filterGradient = new double[numFilters][inputDepth][filterSize][filterSize];
        double[] biasGradient = new double[numFilters];

        // Обратное распространение через активационные функции
        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    gradient[f][i][j] *= activationFunction.derivative(activatedOutput[f][i][j]);
                }
            }
        }

        // Вычисление градиента для фильтров и входов
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                double[][] filterGrad = MatrixUtils.convolve(input[d], gradient[f], stride);
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filterGradient[f][d][i][j] += filterGrad[i][j];
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
                    biasGradient[f] += gradient[f][i][j];
                }
            }
        }

        // Обновление параметров
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int k = 0; k < filterSize; k++) {
                    for (int l = 0; l < filterSize; l++) {
                        filters[f][d][k][l] -= config.getLearningRate() * filterGradient[f][d][k][l];
                    }
                }
            }
            biases[f] -= config.getLearningRate() * biasGradient[f];
        }

        return inputGradient;
    }
}