package cnn;

import cnn.layers.ConvolutionalLayer;
import cnn.layers.FullyConnectedLayer;
import cnn.layers.PoolingLayer;
import cnn.utils.ImageData;
import cnn.utils.ReLU;
import cnn.utils.Sigmoid;
import cnn.utils.TrainingConfig;

import java.io.IOException;
import java.util.List;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        TrainingConfig config = new TrainingConfig(0.01); // Пример использования скорости обучения
        CNN cnn = new CNN(config);
        cnn.addLayer(new ConvolutionalLayer(3, 8, new ReLU(), config));
        cnn.addLayer(new PoolingLayer(2));
        cnn.addLayer(new FullyConnectedLayer(1352, 10, new ReLU(), config)); // Пример использования ReLU
        cnn.addLayer(new FullyConnectedLayer(10, 1, new Sigmoid(), config)); // Пример использования Sigmoid

        // Чтение данных MNIST
        String imagesFile = "data/train-images.idx3-ubyte";
        String labelsFile = "data/train-labels.idx1-ubyte";
        List<ImageData> dataset = MNISTReader.readMNISTData(imagesFile, labelsFile);

        // Создание и запуск тренера
        CNNTrainer trainer = new CNNTrainer(cnn);
        trainer.train(dataset, 10); // 10 эпох

        // Пример использования с тестовым изображением после обучения
        double[][][] input = dataset.get(0).imageData;
        double[][][] output = cnn.forward(input);

        // Вывод результата
        System.out.println("Output: " + Arrays.deepToString(output));
    }
}