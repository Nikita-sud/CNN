package cnn;

import cnn.layers.ConvolutionalLayer;
import cnn.layers.FullyConnectedLayer;
import cnn.layers.PoolingLayer;
import cnn.layers.SoftmaxLayer;
import cnn.layers.PoolingLayer.PoolingType;
import cnn.utils.ImageData;
import cnn.utils.ReLU;
import cnn.utils.Sigmoid;
import cnn.utils.TrainingConfig;

import java.io.IOException;
import java.util.List;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        TrainingConfig config = new TrainingConfig(0.1); // Пример использования скорости обучения
        CNN cnn = new CNN(config);
        cnn.addLayer(new FullyConnectedLayer(784, 30, new Sigmoid(), config)); // Пример использования Tanh или другой активации
        cnn.addLayer(new FullyConnectedLayer(30, 10, new Sigmoid(), config));
        cnn.addLayer(new SoftmaxLayer()); // Добавляем слой Softmax отдельно

        // Чтение данных MNIST
        String trainImagesFile = "data/train-images.idx3-ubyte";
        String trainLabelsFile = "data/train-labels.idx1-ubyte";
        List<ImageData> trainDataset = MNISTReader.readMNISTData(trainImagesFile, trainLabelsFile);

        String testImagesFile = "data/t10k-images.idx3-ubyte";
        String testLabelsFile = "data/t10k-labels.idx1-ubyte";
        List<ImageData> testDataset = MNISTReader.readMNISTData(testImagesFile, testLabelsFile);

        // Создание и запуск тренера
        CNNTrainer trainer = new CNNTrainer(cnn);
        trainer.train(trainDataset, testDataset, 10, 100); // 10 эпох

        // Пример использования с тестовым изображением после обучения
        double[][][] input = testDataset.get(0).imageData;
        double[][][] output = cnn.forward(input);

        // Вывод результата
        System.out.println("Output: " + Arrays.deepToString(output));
    }
}