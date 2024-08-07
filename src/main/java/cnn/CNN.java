package cnn;

import cnn.interfaces.AdaptiveLayer;
import cnn.interfaces.Layer;
import cnn.interfaces.ParameterizedLayer;
import cnn.utils.ImageData;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A Convolutional Neural Network (CNN) class that supports forward and backward propagation, parameter updates,
 * and training using Stochastic Gradient Descent (SGD).
 */
public class CNN implements Serializable {
    private static final long serialVersionUID = 1L;
    private List<Layer> layers;
    private double learningRate;
    private int[] inputShape;

    /**
     * Constructs a CNN with a specified learning rate and input shape.
     *
     * @param learningRate the learning rate for training
     * @param inputShape the shape of the input tensor
     */
    public CNN(double learningRate, int... inputShape) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.inputShape = inputShape;
    }

    /**
     * Adds a layer to the CNN. If the layer is adaptive, it initializes it with the current input shape.
     *
     * @param layer the layer to be added to the CNN
     */
    public void addLayer(Layer layer) {
        if (layer instanceof AdaptiveLayer) {
            ((AdaptiveLayer) layer).initialize(inputShape);
            inputShape = layer.getOutputShape(inputShape);
        } else {
            inputShape = layer.getOutputShape(inputShape);
        }
        layers.add(layer);
    }

    /**
     * Performs the forward pass through all layers of the CNN.
     *
     * @param input the input tensor
     * @return the output tensor after passing through all layers
     */
    public double[][][] forward(double[][][] input) {
        double[][][] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    /**
     * Performs the backward pass through all layers of the CNN.
     *
     * @param gradient the gradient of the loss with respect to the output
     * @return the gradient of the loss with respect to the input
     */
    public double[][][] backward(double[][][] gradient) {
        double[][][] grad = gradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad);
        }
        return grad;
    }

    /**
     * Updates the parameters of all parameterized layers in the CNN using accumulated gradients.
     *
     * @param miniBatchSize the size of the mini-batch used for averaging the gradients
     */
    public void updateParameters(int miniBatchSize) {
        for (Layer layer : layers) {
            if (layer instanceof ParameterizedLayer) {
                ((ParameterizedLayer) layer).updateParameters(learningRate, miniBatchSize);
            }
        }
    }

    /**
     * Resets the accumulated gradients for all parameterized layers in the CNN.
     */
    public void resetGradients() {
        for (Layer layer : layers) {
            if (layer instanceof ParameterizedLayer) {
                ((ParameterizedLayer) layer).resetGradients();
            }
        }
    }

    /**
     * Trains the CNN using Stochastic Gradient Descent (SGD) with mini-batches.
     *
     * @param trainingData the training data set
     * @param epochs the number of epochs to train for
     * @param miniBatchSize the size of each mini-batch
     * @param testData the test data set for evaluation
     * @param saveFilePath the file path to save the best model
     */
    public void SGD(List<ImageData> trainingData, int epochs, int miniBatchSize, List<ImageData> testData, String saveFilePath) {
        int nTest = testData.size();
        double bestAccuracy = 0.0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(trainingData);
            List<List<ImageData>> miniBatches = createMiniBatches(trainingData, miniBatchSize);

            miniBatches.parallelStream().forEach(miniBatch -> updateMiniBatch(miniBatch, miniBatchSize));

            if (nTest > 0) {
                int correct = evaluate(testData);
                double accuracy = (double) correct / nTest;
                System.out.println("Epoch " + (epoch + 1) + ": " + correct + " / " + nTest + " (" + accuracy * 100 + "%)");

                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    saveNetwork(saveFilePath);
                    System.out.println("New best model saved with accuracy: " + bestAccuracy * 100 + "%");
                }
            }
        }
    }

    /**
     * Trains the CNN using Stochastic Gradient Descent (SGD) with mini-batches.
     *
     * @param trainingData the training data set
     * @param epochs the number of epochs to train for
     * @param miniBatchSize the size of each mini-batch
     * @param testData the test data set for evaluation
     */
    public void SGD(List<ImageData> trainingData, int epochs, int miniBatchSize, List<ImageData> testData) {
        int nTest = testData.size();

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(trainingData);
            List<List<ImageData>> miniBatches = createMiniBatches(trainingData, miniBatchSize);

            miniBatches.parallelStream().forEach(miniBatch -> updateMiniBatch(miniBatch, miniBatchSize));

            if (nTest > 0) {
                int correct = evaluate(testData);
                double accuracy = (double) correct / nTest;
                System.out.println("Epoch " + (epoch + 1) + ": " + correct + " / " + nTest + " (" + accuracy * 100 + "%)");
            }
        }
    }

    /**
     * Creates mini-batches from the training data.
     *
     * @param trainingData the training data set
     * @param miniBatchSize the size of each mini-batch
     * @return a list of mini-batches
     */
    private List<List<ImageData>> createMiniBatches(List<ImageData> trainingData, int miniBatchSize) {
        List<List<ImageData>> miniBatches = new ArrayList<>();
        for (int i = 0; i < trainingData.size(); i += miniBatchSize) {
            miniBatches.add(trainingData.subList(i, Math.min(i + miniBatchSize, trainingData.size())));
        }
        return miniBatches;
    }

    /**
     * Updates the CNN parameters using a single mini-batch of training data.
     *
     * @param miniBatch the mini-batch of training data
     * @param miniBatchSize the size of the mini-batch
     */
    private void updateMiniBatch(List<ImageData> miniBatch, int miniBatchSize) {
        resetGradients();
        for (ImageData data : miniBatch) {
            double[][][] output = forward(data.getImageData());
            double[][][] lossGradient = computeLossGradient(output[0][0], data.getLabel());
            backward(lossGradient);
        }
        updateParameters(miniBatchSize);
    }

    /**
     * Computes the gradient of the loss function with respect to the output of the CNN.
     *
     * @param output the output of the CNN
     * @param target the target label
     * @return the gradient of the loss function
     */
    private double[][][] computeLossGradient(double[] output, double[] target) {
        double[][][] gradient = new double[1][1][output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[0][0][i] = output[i] - target[i];
        }
        return gradient;
    }

    /**
     * Evaluates the CNN on a test data set.
     *
     * @param testData the test data set
     * @return the number of correctly classified samples
     */
    public int evaluate(List<ImageData> testData) {
        int correct = 0;
        for (ImageData data : testData) {
            double[][][] output = forward(data.getImageData());
            int predictedLabel = argMax(output[0][0]);
            int actualLabel = argMax(data.getLabel());
            if (predictedLabel == actualLabel) {
                correct++;
            }
        }
        return correct;
    }

    /**
     * Returns the index of the maximum value in an array.
     *
     * @param array the array to search
     * @return the index of the maximum value
     */
    private int argMax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Saves the CNN to a file.
     *
     * @param filePath the path to the file where the CNN should be saved
     */
    public void saveNetwork(String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads a CNN from a file.
     *
     * @param filePath the path to the file from which the CNN should be loaded
     * @return the loaded CNN
     */
    public static CNN loadNetwork(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (CNN) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
}