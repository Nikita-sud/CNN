package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;

/**
 * A pooling layer in a neural network, which reduces the spatial dimensions of the input tensor.
 * This layer can perform either max pooling or average pooling.
 */
public class PoolingLayer implements Layer {

    /**
     * Enumeration for the type of pooling operation.
     */
    public enum PoolingType {
        MAX,
        AVERAGE
    }

    private int poolSize;
    private double[][][] input;
    private PoolingType poolingType;

    /**
     * Constructs a PoolingLayer with the specified pool size and pooling type.
     *
     * @param poolSize the size of the pooling window
     * @param poolingType the type of pooling operation (MAX or AVERAGE)
     */
    public PoolingLayer(int poolSize, PoolingType poolingType) {
        this.poolSize = poolSize;
        this.poolingType = poolingType;
    }

    /**
     * Performs the forward pass by applying the pooling operation to the input tensor.
     *
     * @param input a 3D array representing the input tensor
     * @return a 3D array representing the output tensor after pooling
     */
    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        int inputDepth = input.length;
        int inputSize = input[0].length;
        int outputSize = inputSize / poolSize;

        double[][][] output = new double[inputDepth][outputSize][outputSize];

        for (int d = 0; d < inputDepth; d++) {
            if (poolingType == PoolingType.MAX) {
                output[d] = MatrixUtils.maxPooling(input[d], poolSize);
            } else if (poolingType == PoolingType.AVERAGE) {
                output[d] = MatrixUtils.averagePooling(input[d], poolSize);
            }
        }

        return output;
    }

    /**
     * Performs the backward pass by computing the gradient of the loss with respect to the input.
     *
     * @param gradient a 3D array representing the gradient of the loss with respect to the output
     * @return a 3D array representing the gradient of the loss with respect to the input
     */
    @Override
    public double[][][] backward(double[][][] gradient) {
        int inputDepth = input.length;
        int inputSize = input[0].length;
        int outputSize = gradient[0].length;
        double[][][] inputGradient = new double[inputDepth][inputSize][inputSize];

        for (int d = 0; d < inputDepth; d++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    int x = i * poolSize;
                    int y = j * poolSize;

                    if (poolingType == PoolingType.MAX) {
                        double maxVal = input[d][x][y];
                        int maxI = x, maxJ = y;
                        for (int k = 0; k < poolSize; k++) {
                            for (int l = 0; l < poolSize; l++) {
                                if (input[d][x + k][y + l] > maxVal) {
                                    maxVal = input[d][x + k][y + l];
                                    maxI = x + k;
                                    maxJ = y + l;
                                }
                            }
                        }
                        inputGradient[d][maxI][maxJ] = gradient[d][i][j];
                    } else if (poolingType == PoolingType.AVERAGE) {
                        double gradientValue = gradient[d][i][j] / (poolSize * poolSize);
                        for (int k = 0; k < poolSize; k++) {
                            for (int l = 0; l < poolSize; l++) {
                                inputGradient[d][x + k][y + l] = gradientValue;
                            }
                        }
                    }
                }
            }
        }
        return inputGradient;
    }

    /**
     * Computes the output shape of the layer given the input shape.
     *
     * @param inputShape an array where the first element is the depth and the second and third elements are the height and width of the input tensor
     * @return an array representing the output shape [depth, outputHeight, outputWidth]
     */
    @Override
    public int[] getOutputShape(int... inputShape) {
        int inputSize = inputShape[1];
        int outputSize = inputSize / poolSize;
        return new int[]{inputShape[0], outputSize, outputSize};
    }
}