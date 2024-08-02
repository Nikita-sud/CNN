package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;

public class PoolingLayer implements Layer {
    public enum PoolingType {
        MAX,
        AVERAGE
    }

    private int poolSize;
    private double[][][] input;
    private PoolingType poolingType;

    public PoolingLayer(int poolSize, PoolingType poolingType) {
        this.poolSize = poolSize;
        this.poolingType = poolingType;
    }

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

    @Override
    public void updateParameters(double learningRate, int miniBatchSize) {
        // Pooling layers do not have parameters to update.
    }

    @Override
    public void resetGradients() {
        // Pooling layers do not have gradients to reset.
    }
}
