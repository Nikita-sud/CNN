package cnn.utils;

public class MatrixUtils {

    public static double applyFilter(double[][] input, double[][] filter, int startX, int startY) {
        int filterSize = filter.length;
        double sum = 0;

        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                sum += input[startX + i][startY + j] * filter[i][j];
            }
        }
        return sum;
    }

    public static double[][] rotate180(double[][] matrix) {
        int n = matrix.length;
        double[][] rotated = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                rotated[i][j] = matrix[n - 1 - i][n - 1 - j];
            }
        }
        return rotated;
    }

    public static double[][] maxPooling(double[][] input, int poolSize) {
        int inputSize = input.length;
        int outputSize = inputSize / poolSize;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double maxVal = input[i * poolSize][j * poolSize];
                for (int k = 0; k < poolSize; k++) {
                    for (int l = 0; l < poolSize; l++) {
                        if (input[i * poolSize + k][j * poolSize + l] > maxVal) {
                            maxVal = input[i * poolSize + k][j * poolSize + l];
                        }
                    }
                }
                output[i][j] = maxVal;
            }
        }
        return output;
    }

    public static double[][] averagePooling(double[][] input, int poolSize) {
        int inputSize = input.length;
        int outputSize = inputSize / poolSize;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double sum = 0.0;
                for (int k = 0; k < poolSize; k++) {
                    for (int l = 0; l < poolSize; l++) {
                        sum += input[i * poolSize + k][j * poolSize + l];
                    }
                }
                output[i][j] = sum / (poolSize * poolSize);
            }
        }
        return output;
    }

    // Перемножение матриц
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

    // Addition of two matrices
    public static double[][][] add(double[][][] a, double[][][] b) {
        int depth = a.length;
        int height = a[0].length;
        int width = a[0][0].length;
        double[][][] result = new double[depth][height][width];
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[d][h][w] = a[d][h][w] + b[d][h][w];
                }
            }
        }
        return result;
    }

    // Division of a matrix by a scalar
    public static double[][][] divide(double[][][] a, double scalar) {
        int depth = a.length;
        int height = a[0].length;
        int width = a[0][0].length;
        double[][][] result = new double[depth][height][width];
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[d][h][w] = a[d][h][w] / scalar;
                }
            }
        }
        return result;
    }

    // Flattening
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

    // Unflattening
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