package cnn;

import cnn.layers.FullyConnectedLayer;
import cnn.layers.SoftmaxLayer;
import cnn.utils.ImageData;
import cnn.utils.ReLU;

import java.io.IOException;
import java.util.List;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        double learningRate = 0.1;
        CNN cnn = new CNN(learningRate);

        cnn.addLayer(new FullyConnectedLayer(784, 60, new ReLU()));
        cnn.addLayer(new FullyConnectedLayer(60, 10, new ReLU()));
        cnn.addLayer(new SoftmaxLayer());    

        String trainImagesFile = "data/train-images.idx3-ubyte";
        String trainLabelsFile = "data/train-labels.idx1-ubyte";
        List<ImageData> trainDataset = MNISTReader.readMNISTData(trainImagesFile, trainLabelsFile);

        String testImagesFile = "data/t10k-images.idx3-ubyte";
        String testLabelsFile = "data/t10k-labels.idx1-ubyte";
        List<ImageData> testDataset = MNISTReader.readMNISTData(testImagesFile, testLabelsFile);

        cnn.SGD(trainDataset, 20, 32, testDataset);

        double[][][] input = testDataset.get(0).imageData;
        double[][][] output = cnn.forward(input);

        System.out.println("CNN output: " + Arrays.toString(output[0][0]));
        System.out.println("Actual output: " + Arrays.toString(testDataset.get(0).label));
    }
}