package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;
import cnn.interfaces.ActivationFunction;
import cnn.utils.TrainingConfig;
import java.util.Random;

public class FullyConnectedLayer implements Layer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;
    private double[][][] input;  // Изменено для хранения оригинальных входных данных
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
        Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = rand.nextGaussian() * Math.sqrt(2.0 / inputSize); // улучшенная инициализация весов
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] = 0.0; // начнем с нулевых смещений
        }
    }

    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input; // Сохраняем оригинальные входные данные
        double[] flattenedInput = MatrixUtils.flatten(input);
        double[] preActivation = MatrixUtils.multiply(flattenedInput, weights, biases);
        double[] postActivation = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            postActivation[i] = activationFunction.activate(preActivation[i]);
        }
        return new double[][][] { { postActivation } };
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        double[] postActivationGradient = MatrixUtils.flatten(gradient); // Используем flatten для преобразования градиентов в одномерный массив
        double[] preActivationGradient = new double[outputSize];

        // Получение предактивационных значений
        double[] flattenedInput = MatrixUtils.flatten(input);
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
                weights[i][j] -= config.getLearningRate() * weightGradient[i][j];
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] -= config.getLearningRate() * biasGradient[j];
        }

        return MatrixUtils.unflatten(inputGradient, input.length, input[0].length, input[0][0].length);
    }
}