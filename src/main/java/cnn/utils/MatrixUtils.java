package cnn.utils;

public class MatrixUtils {

    public static double[][] convolve(double[][] input, double[][] filter) {
        int inputSize = input.length;
        int filterSize = filter.length;
        int outputSize = inputSize - filterSize + 1;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double sum = 0;
                for (int k = 0; k < filterSize; k++) {
                    for (int l = 0; l < filterSize; l++) {
                        sum += input[i + k][j + l] * filter[k][l];
                    }
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    public static double[][] maxPooling(double[][] input, int poolSize) {
        int inputSize = input.length;
        int outputSize = inputSize / poolSize;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double max = input[i * poolSize][j * poolSize];
                for (int k = 0; k < poolSize; k++) {
                    for (int l = 0; l < poolSize; l++) {
                        if (input[i * poolSize + k][j * poolSize + l] > max) {
                            max = input[i * poolSize + k][j * poolSize + l];
                        }
                    }
                }
                output[i][j] = max;
            }
        }
        return output;
    }

    public static double[] multiply(double[] input, double[][] weights, double[] biases) {
        int inputSize = input.length;
        int outputSize = biases.length;
        double[] output = new double[outputSize];

        for (int j = 0; j < outputSize; j++) {
            double sum = biases[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * weights[i][j];
            }
            output[j] = sum;
        }
        return output;
    }

    public static double[][] convolveBackward(double[][] input, double[][] filter) {
        int inputSize = input.length;
        int filterSize = filter.length;
        int outputSize = inputSize + filterSize - 1;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                for (int k = 0; k < filterSize; k++) {
                    for (int l = 0; l < filterSize; l++) {
                        output[i + k][j + l] += input[i][j] * filter[k][l];
                    }
                }
            }
        }
        return output;
    }

    public static double[] flatten(double[][][] input) {
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        double[] flattened = new double[depth * height * width];
        
        int index = 0;
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    flattened[index++] = input[d][h][w];
                }
            }
        }
        return flattened;
    }

    public static double[][][] unflatten(double[] input, int depth, int height, int width) {
        double[][][] unflattened = new double[depth][height][width];
        
        int index = 0;
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    unflattened[d][h][w] = input[index++];
                }
            }
        }
        return unflattened;
    }
}