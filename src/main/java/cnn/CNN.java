package cnn;

import cnn.interfaces.AdaptiveLayer;
import cnn.interfaces.Layer;
import cnn.interfaces.ParameterizedLayer;
import cnn.utils.ImageData;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CNN {
    private List<Layer> layers;
    private double learningRate;
    private int[] inputShape;

    public CNN(double learningRate, int... inputShape) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.inputShape = inputShape;
    }

    public void addLayer(Layer layer) {
        if (layers.isEmpty()) {
            if (layer instanceof AdaptiveLayer) {
                ((AdaptiveLayer) layer).initialize(inputShape);
                inputShape = layer.getOutputShape(inputShape);
            }
        } else {
            if (layer instanceof AdaptiveLayer) {
                ((AdaptiveLayer) layer).initialize(inputShape);
                inputShape = layer.getOutputShape(inputShape);
            }
        }
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

    public void updateParameters(int miniBatchSize) {
        for (Layer layer : layers) {
            if (layer instanceof ParameterizedLayer) {
                ((ParameterizedLayer) layer).updateParameters(learningRate, miniBatchSize);
            }
        }
    }

    public void resetGradients() {
        for (Layer layer : layers) {
            if (layer instanceof ParameterizedLayer) {
                ((ParameterizedLayer) layer).resetGradients();
            }
        }
    }

    public void SGD(List<ImageData> trainingData, int epochs, int miniBatchSize, List<ImageData> testData) {
        int nTest = testData.size();

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(trainingData);
            List<List<ImageData>> miniBatches = createMiniBatches(trainingData, miniBatchSize);

            miniBatches.parallelStream().forEach(miniBatch -> updateMiniBatch(miniBatch, miniBatchSize));

            if (nTest > 0) {
                System.out.println("Epoch " + (epoch + 1) + ": " + evaluate(testData) + " / " + nTest);
            }
        }
    }

    private List<List<ImageData>> createMiniBatches(List<ImageData> trainingData, int miniBatchSize) {
        List<List<ImageData>> miniBatches = new ArrayList<>();
        for (int i = 0; i < trainingData.size(); i += miniBatchSize) {
            miniBatches.add(trainingData.subList(i, Math.min(i + miniBatchSize, trainingData.size())));
        }
        return miniBatches;
    }

    private void updateMiniBatch(List<ImageData> miniBatch, int miniBatchSize) {
        resetGradients();
        for (ImageData data : miniBatch) {
            double[][][] output = forward(data.imageData);
            double[][][] lossGradient = computeLossGradient(output[0][0], data.label);
            backward(lossGradient);
        }
        updateParameters(miniBatchSize);
    }

    private double[][][] computeLossGradient(double[] output, double[] target) {
        double[][][] gradient = new double[1][1][output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[0][0][i] = output[i] - target[i];
        }
        return gradient;
    }

    public int evaluate(List<ImageData> testData) {
        int correct = 0;
        for (ImageData data : testData) {
            double[][][] output = forward(data.imageData);
            int predictedLabel = argMax(output[0][0]);
            int actualLabel = argMax(data.label);
            if (predictedLabel == actualLabel) {
                correct++;
            }
        }
        return correct;
    }

    private int argMax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}