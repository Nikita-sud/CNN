package cnn.layers;

import cnn.interfaces.Layer;

public class FlattenLayer implements Layer {

    private int depth;
    private int height;
    private int width;

    @Override
    public double[][][] forward(double[][][] input) {
        this.depth = input.length;
        this.height = input[0].length;
        this.width = input[0][0].length;
        double[][][] output = new double[1][1][depth * height * width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    output[0][0][d * height * width + i * width + j] = input[d][i][j];
                }
            }
        }

        return output;
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        double[][][] reshapedGradient = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    reshapedGradient[d][i][j] = gradient[0][0][d * height * width + i * width + j];
                }
            }
        }

        return reshapedGradient;
    }

    @Override
    public void updateParameters(double learningRate, int miniBatchSize) {
        // Flatten layer has no parameters to update
    }

    @Override
    public void resetGradients() {
        // Flatten layer has no gradients to reset
    }
}