package cnn.layers;

import cnn.interfaces.Layer;
import java.util.Random;

public class DropoutLayer implements Layer {
    private double rate;
    private double[][][] mask;
    private boolean isTraining;

    public DropoutLayer(double rate) {
        this.rate = rate;
        this.isTraining = true;
    }

    @Override
    public double[][][] forward(double[][][] input) {
        if (!isTraining) {
            return input;
        }

        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        mask = new double[depth][height][width];
        Random random = new Random();

        double[][][] output = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    mask[d][i][j] = random.nextDouble() > rate ? 1.0 : 0.0;
                    output[d][i][j] = input[d][i][j] * mask[d][i][j];
                }
            }
        }

        return output;
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        if (!isTraining) {
            return gradient;
        }

        int depth = gradient.length;
        int height = gradient[0].length;
        int width = gradient[0][0].length;
        
        if (mask == null || mask.length != depth || mask[0].length != height || mask[0][0].length != width) {
            throw new IllegalStateException("Dropout mask dimensions do not match gradient dimensions");
        }

        double[][][] outputGradient = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    outputGradient[d][i][j] = gradient[d][i][j] * mask[d][i][j];
                }
            }
        }

        return outputGradient;
    }

    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    @Override
    public int[] getOutputShape(int... inputShape) {
        return inputShape; // Dropout не меняет размерность
    }
}
