package cnn.layers;

import cnn.interfaces.Layer;
import cnn.utils.MatrixUtils;

public class PoolingLayer implements Layer {
    private int poolSize;
    private double[][][] input;

    public PoolingLayer(int poolSize) {
        this.poolSize = poolSize;
    }

    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        int inputSize = input[0].length;
        int outputSize = inputSize / poolSize;
        double[][][] output = new double[input.length][outputSize][outputSize];

        for (int c = 0; c < input.length; c++) {
            output[c] = MatrixUtils.maxPooling(input[c], poolSize);
        }
        return output;
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        int inputSize = input[0].length;
        int outputSize = inputSize / poolSize;
        double[][][] inputGradient = new double[input.length][inputSize][inputSize];

        for (int c = 0; c < input.length; c++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double max = input[c][i * poolSize][j * poolSize];
                    int maxI = i * poolSize, maxJ = j * poolSize;
                    for (int k = 0; k < poolSize; k++) {
                        for (int l = 0; l < poolSize; l++) {
                            if (input[c][i * poolSize + k][j * poolSize + l] > max) {
                                max = input[c][i * poolSize + k][j * poolSize + l];
                                maxI = i * poolSize + k;
                                maxJ = j * poolSize + l;
                            }
                        }
                    }
                    inputGradient[c][maxI][maxJ] = gradient[c][i][j];
                }
            }
        }
        return input;
    }
}