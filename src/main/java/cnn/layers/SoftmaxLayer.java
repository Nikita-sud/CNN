package cnn.layers;

import cnn.interfaces.Layer;

public class SoftmaxLayer implements Layer {
    @SuppressWarnings("unused")
    private double[][][] input;

    @Override
    public double[][][] forward(double[][][] input) {
        this.input = input;
        double[] flattenedInput = input[0][0];
        double[] softmaxOutput = softmax(flattenedInput);
        return new double[][][]{{softmaxOutput}};
    }

    private double[] softmax(double[] input) {
        double[] output = new double[input.length];
        double max = Double.NEGATIVE_INFINITY;

        for (double v : input) {
            if (v > max) {
                max = v;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        // Просто верните градиент, не учитывая производную softmax здесь
        return gradient;
    }

    @Override
    public int[] getOutputShape(int... inputShape) {
        // Softmax layer does not change the shape of the input
        return new int[]{inputShape[0]};
    }
}
