package cnn;

import java.util.List;
import cnn.utils.ImageData;

public class CNNTrainer {
    private CNN cnn;

    public CNNTrainer(CNN cnn) {
        this.cnn = cnn;
    }

    // Метод для обучения
    public void train(List<ImageData> dataset, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            for (ImageData data : dataset) {
                // Прямой проход
                double[][][] output = cnn.forward(data.imageData);

                // Вычисление потерь
                double loss = computeLoss(output[0][0], data.label);
                totalLoss += loss;

                // Вычисление градиента функции потерь
                double[][][] lossGradient = computeLossGradient(output[0][0], data.label);

                // Обратное распространение
                cnn.backward(lossGradient);

                // Обновление весов будет происходить внутри слоев во время обратного прохода
            }
            System.out.println("Epoch " + epoch + ", Loss: " + (totalLoss / dataset.size()));
        }
    }

    // Метод для вычисления функции потерь (кросс-энтропийная ошибка)
    private double computeLoss(double[] output, double[] target) {
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            loss -= target[i] * Math.log(output[i]);
        }
        return loss / output.length;
    }

    // Метод для вычисления градиента функции потерь
    private double[][][] computeLossGradient(double[] output, double[] target) {
        double[][][] gradient = new double[1][1][output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[0][0][i] = output[i] - target[i];
        }
        return gradient;
    }
}