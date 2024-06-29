package cnn.layers;

import cnn.interfaces.Layer;

public class ConvolutionalLayer implements Layer {
    private int filterSize;
    private int numFilters;
    private double[][][] filters;

    public ConvolutionalLayer(int filterSize, int numFilters) {
        this.filterSize = filterSize;
        this.numFilters = numFilters;
        this.filters = new double[numFilters][filterSize][filterSize];
        initializeFilters();
    }

    private void initializeFilters() {
        for (int i = 0; i < numFilters; i++) {
            for (int j = 0; j < filterSize; j++) {
                for (int k = 0; k < filterSize; k++) {
                    filters[i][j][k] = Math.random();
                }
            }
        }
    }

    @Override
    public double[][][] forward(double[][][] input) {
        return new double[0][][];
    }

    @Override
    public double[][][] backward(double[][][] gradient) {
        return new double[0][][];
    }
}
