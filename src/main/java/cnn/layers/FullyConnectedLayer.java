package cnn.layers;

import cnn.utils.MatrixUtils;
import cnn.interfaces.ActivationFunction;
import cnn.interfaces.AdaptiveLayer;
import cnn.interfaces.ParameterizedLayer;

import java.util.Random;

/**
 * A fully connected layer in a neural network, also known as a dense layer.
 * This layer connects every input neuron to every output neuron.
 */
public class FullyConnectedLayer implements AdaptiveLayer, ParameterizedLayer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;
    private double[][][] input;
    private ActivationFunction activationFunction;
    private double[][] accumulatedWeightGradients;
    private double[] accumulatedBiasGradients;

    /**
     * Constructs a FullyConnectedLayer with the specified output size and activation function.
     *
     * @param outputSize the number of neurons in the output layer
     * @param activationFunction the activation function to apply
     */
    public FullyConnectedLayer(int outputSize, ActivationFunction activationFunction) {
        this.outputSize = outputSize;
        this.activationFunction = activationFunction;
    }

    /**
     * Initializes the layer with the shape of the input tensor.
     * 
     * @param inputShape an array where the first element is the size of the input
     * @throws IllegalArgumentException if the input shape does not have exactly one dimension
     */
    @Override
    public void initialize(int... inputShape) {
        this.inputSize = inputShape[0];
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        initializeWeights();
        initializeAccumulatedGradients();
    }

    /**
     * Initializes the weights of the layer using He initialization.
     */
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

    /**
     * Initializes the accumulated gradients to zero.
     */
    private void initializeAccumulatedGradients() {
        accumulatedWeightGradients = new double[inputSize][outputSize];
        accumulatedBiasGradients = new double[outputSize];
    }

    /**
     * Performs the forward pass by computing the weighted sum of the inputs and applying the activation function.
     *
     * @param input a 3D array representing the input tensor, expected to be [1][1][inputSize]
     * @return a 3D array representing the output tensor, [1][1][outputSize]
     * @throws IllegalArgumentException if the input dimensions do not match the expected shape
     */
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

    /**
     * Performs the backward pass by computing the gradients of the loss with respect to the inputs and parameters.
     *
     * @param gradient a 3D array representing the gradient of the loss with respect to the output, [1][1][outputSize]
     * @return a 3D array representing the gradient of the loss with respect to the input, [1][1][inputSize]
     * @throws IllegalArgumentException if the gradient dimensions do not match the expected shape
     */
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

    /**
     * Updates the parameters of the layer using the accumulated gradients and a given learning rate.
     *
     * @param learningRate the learning rate for the update
     * @param miniBatchSize the size of the mini-batch for averaging the gradients
     */
    @Override
    public void updateParameters(double learningRate, int miniBatchSize) {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] -= learningRate * accumulatedWeightGradients[i][j] / miniBatchSize;
                accumulatedWeightGradients[i][j] = 0;
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] -= learningRate * accumulatedBiasGradients[j] / miniBatchSize;
            accumulatedBiasGradients[j] = 0;
        }
    }

    /**
     * Resets the accumulated gradients to zero.
     */
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

    /**
     * Computes the output shape of the layer given the input shape.
     *
     * @param inputShape an array where the first element is the size of the input
     * @return an array with one element representing the size of the output
     * @throws IllegalArgumentException if the input shape does not have exactly one dimension
     */
    @Override
    public int[] getOutputShape(int... inputShape) {
        return new int[]{outputSize};
    }

    /**
     * Gets the number of neurons in the output layer.
     *
     * @return the number of output neurons
     */
    public int getOutputSize() {
        return outputSize;
    }
}