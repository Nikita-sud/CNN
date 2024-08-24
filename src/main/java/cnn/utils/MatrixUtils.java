package cnn.utils;

import java.util.Arrays;

/**
 * A utility class for various matrix operations used in convolutional neural networks.
 */
public class MatrixUtils {

    /**
     * Pads the input tensor with zeros around the border.
     *
     * @param input the input 3D tensor (depth, height, width)
     * @param pad the amount of padding to add on each side
     * @return a new 3D tensor with padding added
     */
    public static double[][][] pad(double[][][] input, int pad) {
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;

        // Новые размеры с учетом паддинга
        int newHeight = height + 2 * pad;
        int newWidth = width + 2 * pad;

        // Создание нового тензора с добавлением паддинга
        double[][][] paddedInput = new double[depth][newHeight][newWidth];

        // Копирование данных из оригинального тензора в центр нового
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    paddedInput[d][i + pad][j + pad] = input[d][i][j];
                }
            }
        }

        return paddedInput;
    }

    /**
     * Removes padding from the input tensor.
     *
     * @param input the padded input 3D tensor (depth, height, width)
     * @param pad the amount of padding that was added on each side
     * @return a new 3D tensor with the padding removed
     */
    public static double[][][] unpad(double[][][] input, int pad) {
        int depth = input.length;
        int paddedHeight = input[0].length;
        int paddedWidth = input[0][0].length;

        // Новые размеры без паддинга
        int newHeight = paddedHeight - 2 * pad;
        int newWidth = paddedWidth - 2 * pad;

        // Создание нового тензора без паддинга
        double[][][] unpaddedInput = new double[depth][newHeight][newWidth];

        // Копирование данных из центрированной части в новый тензор
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < newHeight; i++) {
                for (int j = 0; j < newWidth; j++) {
                    unpaddedInput[d][i][j] = input[d][i + pad][j + pad];
                }
            }
        }

        return unpaddedInput;
    }

    /**
     * Applies a filter to a specific region of the input matrix starting at (startX, startY).
     *
     * @param input the input matrix
     * @param filter the filter matrix
     * @param startX the starting X-coordinate
     * @param startY the starting Y-coordinate
     * @return the sum of element-wise multiplication of the filter and the specified region of the input matrix
     */
    public static double applyFilter(double[][] input, double[][] filter, int startX, int startY) {
        int filterSize = filter.length;
        double sum = 0;

        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                int x = startX + i;
                int y = startY + j;
                if (x >= 0 && x < input.length && y >= 0 && y < input[0].length) {
                    sum += input[x][y] * filter[i][j];
                }
            }
        }
        return sum;
    }

    /**
     * Rotates a square matrix by 180 degrees.
     *
     * @param matrix the matrix to rotate
     * @return a new matrix that is the 180-degree rotation of the input matrix
     */
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

    /**
     * Performs a convolution operation on the input matrix using the given filter and stride.
     *
     * @param input the input matrix
     * @param filter the filter matrix
     * @param stride the stride for the convolution
     * @return the resulting matrix after the convolution
     */
    public static double[][] convolve(double[][] input, double[][] filter, int stride) {
        int inputSize = input.length;
        int filterSize = filter.length;
        int outputSize = (inputSize - filterSize) / stride + 1;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                output[i][j] = applyFilter(input, filter, i * stride, j * stride);
            }
        }
        return output;
    }

    /**
     * Performs a full convolution operation on the input matrix using the given filter.
     *
     * @param input the input matrix
     * @param filter the filter matrix
     * @return the resulting matrix after the full convolution
     */
    public static double[][] fullConvolve(double[][] input, double[][] filter) {
        int inputSize = input.length;
        int filterSize = filter.length;
        int outputSize = inputSize + filterSize - 1;
        double[][] output = new double[outputSize][outputSize];

        for (int i = -filterSize + 1; i < inputSize; i++) {
            for (int j = -filterSize + 1; j < inputSize; j++) {
                output[i + filterSize - 1][j + filterSize - 1] = applyFilter(input, filter, i, j);
            }
        }
        return output;
    }

    /**
     * Applies max pooling to the input matrix with the specified pool size.
     *
     * @param input the input matrix
     * @param poolSize the size of the pooling window
     * @return the resulting matrix after max pooling
     */
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

    /**
     * Applies average pooling to the input matrix with the specified pool size.
     *
     * @param input the input matrix
     * @param poolSize the size of the pooling window
     * @return the resulting matrix after average pooling
     */
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

    /**
     * Multiplies a vector by a matrix and adds a bias vector.
     *
     * @param input the input vector
     * @param weights the weight matrix
     * @param biases the bias vector
     * @return the resulting vector after the multiplication and bias addition
     */
    public static double[] multiply(double[] input, double[][] weights, double[] biases) {
        int outputSize = biases.length;
        double[] output = Arrays.copyOf(biases, outputSize); // Initialize with biases

        for (int j = 0; j < outputSize; j++) {
            double sum = 0.0;
            for (int i = 0; i < input.length; i++) {
                sum += input[i] * weights[i][j];
            }
            output[j] += sum;
        }
        return output;
    }

    /**
     * Unflattens a 1D array into a 3D matrix with the specified dimensions.
     *
     * @param input the input 1D array
     * @param depth the depth of the resulting 3D matrix
     * @param height the height of the resulting 3D matrix
     * @param width the width of the resulting 3D matrix
     * @return the resulting 3D matrix after unflattening
     */
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
