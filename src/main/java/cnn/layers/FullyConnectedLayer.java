package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;
import cnn.interfaces.ActivationFunction;
import java.util.Random;

public class FullyConnectedLayer implements Layer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;
    private double[][][] input;
    private ActivationFunction activationFunction;
    private double[][] accumulatedWeightGradients;
    private double[] accumulatedBiasGradients;

    public FullyConnectedLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        this.activationFunction = activationFunction;
        initializeWeights();
        initializeAccumulatedGradients();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = rand.nextGaussian() * Math.sqrt(2.0 / inputSize);
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] = 0.0;
        }
    }

    private void initializeAccumulatedGradients() {
        accumulatedWeightGradients = new double[inputSize][outputSize];
        accumulatedBiasGradients = new double[outputSize];
    }

    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        double[] flattenedInput = input[0][0];

        double[] preActivation = MatrixUtils.multiply(flattenedInput, weights, biases);
        double[] postActivation = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            postActivation[i] = activationFunction.activate(preActivation[i]);
        }
        return new double[][][]{{postActivation}};
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        double[] postActivationGradient = gradient[0][0];
        double[] preActivationGradient = new double[outputSize];

        double[] flattenedInput = input[0][0];
        double[] preActivation = MatrixUtils.multiply(flattenedInput, weights, biases);

        for (int i = 0; i < outputSize; i++) {
            preActivationGradient[i] = postActivationGradient[i] * activationFunction.derivative(preActivation[i]);
        }

        double[] inputGradient = new double[flattenedInput.length];
        double[][] weightGradient = new double[inputSize][outputSize];
        double[] biasGradient = new double[outputSize];

        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                inputGradient[i] += preActivationGradient[j] * weights[i][j];
                weightGradient[i][j] += preActivationGradient[j] * flattenedInput[i];
            }
            biasGradient[j] += preActivationGradient[j];
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                accumulatedWeightGradients[i][j] += weightGradient[i][j];
            }
        }
        for (int j = 0; j < outputSize; j++) {
            accumulatedBiasGradients[j] += biasGradient[j];
        }

        return new double[][][]{{inputGradient}};
    }

    @Override
    public void updateParameters(double learningRate, int miniBatchSize) {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] -= learningRate * accumulatedWeightGradients[i][j] / miniBatchSize;
                accumulatedWeightGradients[i][j] = 0; // Reset accumulated gradient
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] -= learningRate * accumulatedBiasGradients[j] / miniBatchSize;
            accumulatedBiasGradients[j] = 0; // Reset accumulated gradient
        }
    }

    @Override
    public void resetGradients() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                accumulatedWeightGradients[i][j] = 0;
            }
        }
        for (int j = 0; j < outputSize; j++) {
            accumulatedBiasGradients[j] = 0;
        }
    }
}