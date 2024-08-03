package cnn.layers;

import cnn.interfaces.Layer;

public class BatchNormalizationLayer implements Layer {
    private double[] gamma;
    private double[] beta;
    private double[] mean;
    private double[] variance;
    private double[] x_hat;
    private double[] gammaGradient;
    private double[] betaGradient;
    private double epsilon = 1e-5;

    public BatchNormalizationLayer(int depth) {
        gamma = new double[depth];
        beta = new double[depth];
        mean = new double[depth];
        variance = new double[depth];
        x_hat = new double[depth];
        gammaGradient = new double[depth];
        betaGradient = new double[depth];
        for (int i = 0; i < depth; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    @Override
    public double[][][] forward(double[][][] input) {
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        double[][][] output = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            double sum = 0.0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    sum += input[d][i][j];
                }
            }
            mean[d] = sum / (height * width);

            double varSum = 0.0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    varSum += Math.pow(input[d][i][j] - mean[d], 2);
                }
            }
            variance[d] = varSum / (height * width);

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    x_hat[d] = (input[d][i][j] - mean[d]) / Math.sqrt(variance[d] + epsilon);
                    output[d][i][j] = gamma[d] * x_hat[d] + beta[d];
                }
            }
        }

        return output;
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        int depth = gradient.length;
        int height = gradient[0].length;
        int width = gradient[0][0].length;
        double[][][] inputGradient = new double[depth][height][width];

        for (int d = 0; d < depth; d++) {
            double dL_dgamma = 0.0;
            double dL_dbeta = 0.0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    dL_dgamma += gradient[d][i][j] * x_hat[d];
                    dL_dbeta += gradient[d][i][j];
                }
            }
            gammaGradient[d] += dL_dgamma;
            betaGradient[d] += dL_dbeta;

            double dL_dx_hat = 0.0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    dL_dx_hat = gradient[d][i][j] * gamma[d];
                    inputGradient[d][i][j] = (1.0 / (height * width)) * (1.0 / Math.sqrt(variance[d] + epsilon)) * (height * width * dL_dx_hat - dL_dgamma * x_hat[d] - dL_dbeta);
                }
            }
        }

        return inputGradient;
    }

    @Override
    public void updateParameters(double learningRate, int miniBatchSize) {
        for (int i = 0; i < gamma.length; i++) {
            gamma[i] -= learningRate * gammaGradient[i] / miniBatchSize;
            beta[i] -= learningRate * betaGradient[i] / miniBatchSize;
            gammaGradient[i] = 0.0;
            betaGradient[i] = 0.0;
        }
    }

    @Override
    public void resetGradients() {
        for (int i = 0; i < gamma.length; i++) {
            gammaGradient[i] = 0.0;
            betaGradient[i] = 0.0;
        }
    }
}