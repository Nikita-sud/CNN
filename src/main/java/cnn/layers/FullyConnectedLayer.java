package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;
import cnn.interfaces.ActivationFunction;
import cnn.utils.TrainingConfig;

public class FullyConnectedLayer implements Layer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;
    private double[] input;
    private ActivationFunction activationFunction;
    private TrainingConfig config;

    public FullyConnectedLayer(int inputSize, int outputSize, ActivationFunction activationFunction, TrainingConfig config) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        this.activationFunction = activationFunction;
        this.config = config;
        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = Math.random();
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] = Math.random();
        }
    }

    @Override
    public double[][][] forward(double[][][] input) {
        this.input = MatrixUtils.flatten(input);
        double[] preActivation = MatrixUtils.multiply(this.input, weights, biases);
        double[] postActivation = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            postActivation[i] = activationFunction.activate(preActivation[i]);
        }
        return new double[][][] { { postActivation } };
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        double[] postActivationGradient = gradient[0][0];
        double[] preActivationGradient = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            preActivationGradient[i] = postActivationGradient[i] * activationFunction.derivative(input[i]);
        }

        double[] inputGradient = new double[inputSize];
        double[][] weightGradient = new double[inputSize][outputSize];
        double[] biasGradient = new double[outputSize];

        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                inputGradient[i] += preActivationGradient[j] * weights[i][j];
                weightGradient[i][j] += preActivationGradient[j] * input[i];
            }
            biasGradient[j] += preActivationGradient[j];
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] -= config.getLearningRate() * weightGradient[i][j];
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] -= config.getLearningRate() * biasGradient[j];
        }

        int depth = 1;
        int height = (int) Math.sqrt(inputSize);
        int width = height;
        double[][][] reshapedInputGradient = MatrixUtils.unflatten(inputGradient, depth, height, width);

        return reshapedInputGradient;
    }
}