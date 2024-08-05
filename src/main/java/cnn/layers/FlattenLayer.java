package cnn.layers;

import cnn.interfaces.AdaptiveLayer;

public class FlattenLayer implements AdaptiveLayer {

    private int depth;
    private int height;
    private int width;

    @Override
    public void initialize(int... inputShape) {
        this.depth = inputShape[0];
        this.height = inputShape[1];
        this.width = inputShape[2];
    }

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
    public int[] getOutputShape(int... inputShape) {
        int flatSize = inputShape[0] * inputShape[1] * inputShape[2];
        return new int[]{flatSize};
    }
}
