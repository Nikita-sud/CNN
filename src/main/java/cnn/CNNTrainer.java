package cnn;

import java.util.List;
import java.util.Collections;

import cnn.utils.ImageData;
import cnn.utils.MatrixUtils;

public class CNNTrainer {
    private CNN cnn;

    public CNNTrainer(CNN cnn) {
        this.cnn = cnn;
    }

    // Метод для обучения
    public void train(List<ImageData> trainDataset, List<ImageData> testDataset, int epochs, int batchSize) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(trainDataset); // Перемешивание данных для каждой эпохи
            double totalLoss = 0;
            @SuppressWarnings("unused")
            int batchCount = 0;

            for (int batchStart = 0; batchStart < trainDataset.size(); batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, trainDataset.size());
                List<ImageData> miniBatch = trainDataset.subList(batchStart, batchEnd);

                double[][][] accumulatedGradient = null;
                for (ImageData data : miniBatch) {
                    // Прямой проход
                    double[][][] output = cnn.forward(data.imageData);

                    // Вычисление потерь
                    double loss = computeLoss(output[0][0], data.label);
                    totalLoss += loss;

                    // Вычисление градиента функции потерь
                    double[][][] lossGradient = computeLossGradient(output[0][0], data.label);

                    // Накопление градиентов
                    if (accumulatedGradient == null) {
                        accumulatedGradient = lossGradient;
                    } else {
                        accumulatedGradient = MatrixUtils.add(accumulatedGradient, lossGradient);
                    }
                }

                // Обновление весов по накопленному градиенту среднего значения
                cnn.backward(MatrixUtils.divide(accumulatedGradient, miniBatch.size()));
                batchCount++;
            }

            double averageLoss = totalLoss / trainDataset.size();
            double accuracy = evaluateAccuracy(cnn, testDataset);
            System.out.println("Epoch " + (epoch + 1) + ", Loss: " + averageLoss + ", Accuracy: " + accuracy);
        }
    }

    // Метод для вычисления функции потерь (например, среднеквадратичная ошибка)
    private double computeLoss(double[] output, double[] target) {
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            loss += Math.pow(output[i] - target[i], 2);
        }
        return loss / output.length;
    }

    // Метод для вычисления градиента функции потерь
    private double[][][] computeLossGradient(double[] output, double[] target) {
        double[][][] gradient = new double[1][1][output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[0][0][i] = 2 * (output[i] - target[i]) / output.length;
        }
        return gradient;
    }

    // Метод для вычисления точности
    private double evaluateAccuracy(CNN cnn, List<ImageData> dataset) {
        int correct = 0;
        for (ImageData data : dataset) {
            double[][][] output = cnn.forward(data.imageData);
            int predictedLabel = argMax(output[0][0]);
            int actualLabel = argMax(data.label);
            if (predictedLabel == actualLabel) {
                correct++;
            }
        }
        return (double) correct / dataset.size();
    }

    // Метод для нахождения индекса максимального элемента в массиве
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
