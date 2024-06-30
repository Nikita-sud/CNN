package cnn.utils;

public class ImageData {
    public double[][][] imageData;
    public double[] label;

    public ImageData(double[][][] imageData, double[] label) {
        this.imageData = imageData;
        this.label = label;
    }

    public double[][][] getImageData() {
        return imageData;
    }

    public double[] getLabel() {
        return label;
    }
}