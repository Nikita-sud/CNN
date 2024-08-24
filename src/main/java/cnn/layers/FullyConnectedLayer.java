package cnn.layers;

import cnn.utils.MatrixUtils;
import cnn.interfaces.ActivationFunction;
import cnn.interfaces.AdaptiveLayer;
import cnn.interfaces.ParameterizedLayer;

import java.io.Serializable;
import java.util.Random;

/**
 * A fully connected layer in a neural network, also known as a dense layer.
 * This layer connects every input neuron to every output neuron.
 * It supports L1 and L2 regularization.
 */
public class FullyConnectedLayer implements AdaptiveLayer, ParameterizedLayer, Serializable {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;
    private double lambdaL1;
    private double lambdaL2;
    private double[][][] input;
    private ActivationFunction activationFunction;
    private double[][] accumulatedWeightGradients;
    private double[] accumulatedBiasGradients;

    /**
     * Constructs a FullyConnectedLayer with the specified output size, activation function, 
     * and regularization coefficients.
     *
     * @param outputSize the number of neurons in the output layer
     * @param activationFunction the activation function to apply
     * @param lambdaL1 the L1 regularization coefficient
     * @param lambdaL2 the L2 regularization coefficient
     */
    public FullyConnectedLayer(int outputSize, ActivationFunction activationFunction, double lambdaL1, double lambdaL2) {
        this.outputSize = outputSize;
        this.activationFunction = activationFunction;
        this.lambdaL1 = lambdaL1;
        this.lambdaL2 = lambdaL2;
    }

    /**
     * Constructs a FullyConnectedLayer with the specified output size and activation function.
     * Regularization parameters are set to zero by default.
     *
     * @param outputSize the number of neurons in the output layer
     * @param activationFunction the activation function to apply
     */
    public FullyConnectedLayer(int outputSize, ActivationFunction activationFunction) {
        this(outputSize, activationFunction, 0, 0);
    }

    /**
     * Initializes the layer with the shape of the input tensor.
     * 
     * @param inputShape an array where the first element is the size of the input
     * @throws IllegalArgumentException if the input shape does not have exactly one dimension
     */
    @Override
    public void initialize(int... inputShape) {
        if (inputShape.length != 1) {
            throw new IllegalArgumentException("Input shape must have exactly one dimension");
        }
        this.inputSize = inputShape[0];
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        initializeWeights();
        initializeAccumulatedGradients();
    }

    /**
     * Initializes the weights of the layer using He initialization.
     * Weights are initialized with random values drawn from a Gaussian distribution.
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
     * This method is used to prepare for gradient accumulation during training.
     */
    private void initializeAccumulatedGradients() {
        accumulatedWeightGradients = new double[inputSize][outputSize];
        accumulatedBiasGradients = new double[outputSize];
    }

    /**
     * Performs the forward pass by computing the weighted sum of the inputs 
     * and applying the activation function.
     *
     * @param input a 3D array representing the input tensor, expected to be [1][1][inputSize]
     * @return a 3D array representing the output tensor, [1][1][outputSize]
     * @throws IllegalArgumentException if the input dimensions do not match the expected shape
     */
    @Override
    public double[][][] forward(double[][][] input) {
        if (input.length != 1 || input[0].length != 1 || input[0][0].length != inputSize) {
            throw new IllegalArgumentException("Input dimensions do not match the expected shape");
        }
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
     * Performs the backward pass by computing the gradients of the loss 
     * with respect to the inputs and parameters.
     *
     * @param gradient a 3D array representing the gradient of the loss with respect to the output, [1][1][outputSize]
     * @return a 3D array representing the gradient of the loss with respect to the input, [1][1][inputSize]
     * @throws IllegalArgumentException if the gradient dimensions do not match the expected shape
     */
    @Override
    public double[][][] backward(double[][][] gradient) {
        if (gradient.length != 1 || gradient[0].length != 1 || gradient[0][0].length != outputSize) {
            throw new IllegalArgumentException("Gradient dimensions do not match the expected shape");
        }
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
                
                // L1 regularization
                if (lambdaL1 != 0) {
                    accumulatedWeightGradients[i][j] += lambdaL1 * Math.signum(weights[i][j]);
                }
                
                // L2 regularization
                if (lambdaL2 != 0) {
                    accumulatedWeightGradients[i][j] += lambdaL2 * weights[i][j];
                }
            }
        }

        for (int j = 0; j < outputSize; j++) {
            accumulatedBiasGradients[j] += biasGradient[j];
        }

        return new double[][][]{{inputGradient}};
    }

    /**
     * Updates the parameters of the layer using the accumulated gradients 
     * and a given learning rate.
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
     * This is typically called after the parameters have been updated.
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