package cnn;

import cnn.layers.*;
import cnn.layers.PoolingLayer.PoolingType;
import cnn.utils.*;
import cnn.utils.activationFunctions.ELU;

import java.io.IOException;
import java.util.List;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        double learningRate = 0.01; 
        CNN cnn = new CNN(1, 28, 28);

        cnn.addLayer(new ConvolutionalLayer(3, 16, 1, new ELU(1)));
        cnn.addLayer(new BatchNormalizationLayer(16));
        cnn.addLayer(new ConvolutionalLayer(5, 8, 1, new ELU(1)));
        cnn.addLayer(new BatchNormalizationLayer(8));
        cnn.addLayer(new PoolingLayer(2, PoolingType.MAX));

        cnn.addLayer(new FlattenLayer());
        cnn.addLayer(new FullyConnectedLayer(128, new ELU(1)));
        cnn.addLayer(new BatchNormalizationLayer(128));
        cnn.addLayer(new FullyConnectedLayer(10, new ELU(1)));  
        cnn.addLayer(new SoftmaxLayer());

        cnn.printNetworkSummary();

        String trainImagesFile = "data/train-images.idx3-ubyte";
        String trainLabelsFile = "data/train-labels.idx1-ubyte";
        List<ImageData> trainDataset = MNISTReader.readMNISTData(trainImagesFile, trainLabelsFile);

        String testImagesFile = "data/t10k-images.idx3-ubyte";
        String testLabelsFile = "data/t10k-labels.idx1-ubyte";
        List<ImageData> testDataset = MNISTReader.readMNISTData(testImagesFile, testLabelsFile);

        cnn.SGD(trainDataset, 50, 32, testDataset,learningRate);

        double[][][] input = testDataset.get(2).getImageData();
        double[][][] output = cnn.forward(input);

        System.out.println("CNN output: " + Arrays.toString(output[0][0]));
        System.out.println("Actual output: " + Arrays.toString(testDataset.get(2).getLabel()));
    }
}