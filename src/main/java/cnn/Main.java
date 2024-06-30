package cnn;

import cnn.layers.ConvolutionalLayer;
import cnn.layers.FullyConnectedLayer;
import cnn.layers.PoolingLayer;
import cnn.utils.ReLU;
import cnn.utils.Sigmoid;
import cnn.utils.TrainingConfig;

public class Main {
    public static void main(String[] args) {
        TrainingConfig config = new TrainingConfig(0.01); // Пример использования скорости обучения
        CNN cnn = new CNN(config);
        cnn.addLayer(new ConvolutionalLayer(3, 8,new ReLU(), config));
        cnn.addLayer(new PoolingLayer(2));
        cnn.addLayer(new FullyConnectedLayer(1352, 10, new ReLU(), config)); // Пример использования ReLU
        cnn.addLayer(new FullyConnectedLayer(10, 1, new Sigmoid(), config)); // Пример использования Sigmoid

        // Пример использования с фиктивными данными
        double[][][] input = new double[1][28][28]; // Пример входных данных (1 канал, 28x28 пикселей)
        double[][][] output = cnn.forward(input);

        // Вывод результата
        System.out.println("Output: " + output);

        // Пример обратного прохода
        double[][][] gradient = new double[1][1][10]; // Пример градиента
        cnn.backward(gradient);
    }
}
