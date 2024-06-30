package cnn.utils;

public class TrainingConfig {
    private double learningRate;

    public TrainingConfig(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
