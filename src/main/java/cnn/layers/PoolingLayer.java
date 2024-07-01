package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;
import cnn.utils.TrainingConfig;

public class PoolingLayer implements Layer {
    private int poolSize;
    private double[][][] input;
    @SuppressWarnings("unused")
    private TrainingConfig config;

    public PoolingLayer(int poolSize) {
        this.poolSize = poolSize;
    }

    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        int inputDepth = input.length;
        int inputSize = input[0].length;
        int outputSize = inputSize / poolSize;

        double[][][] output = new double[inputDepth][outputSize][outputSize];

        for (int d = 0; d < inputDepth; d++) {
            output[d] = MatrixUtils.maxPooling(input[d], poolSize);
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
                }
            }
        }
        return inputGradient;
    }
}
