package cnn;

import cnn.interfaces.Layer;
import cnn.utils.TrainingConfig;

import java.util.ArrayList;
import java.util.List;

public class CNN {
    private List<Layer> layers;
    private TrainingConfig config;

    public CNN(TrainingConfig config) {
        this.layers = new ArrayList<>();
        this.config = config;
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public double[][][] forward(double[][][] input) {
        double[][][] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public double[][][] backward(double[][][] gradient) {
        double[][][] grad = gradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad);
        }
        return grad;
    }

    public TrainingConfig getConfig() {
        return config;
    }
}