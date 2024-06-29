package cnn.layers;

import cnn.interfaces.Layer;

public class PoolingLayer implements Layer {
    private int poolSize;

    public PoolingLayer(int poolSize) {
        this.poolSize = poolSize;
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
