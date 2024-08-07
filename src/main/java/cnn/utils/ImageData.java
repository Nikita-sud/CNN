package cnn.utils;

import java.io.Serializable;

/**
 * A class representing image data and its corresponding label.
 * This class is used to store and retrieve the input data and labels for training and evaluating a neural network.
 */
public class ImageData implements Serializable{
    private double[][][] imageData;
    private double[] label;

    /**
     * Constructs an ImageData object with the specified image data and label.
     *
     * @param imageData a 3D array representing the image data
     * @param label a 1D array representing the label
     */
    public ImageData(double[][][] imageData, double[] label) {
        this.imageData = imageData;
        this.label = label;
    }

    /**
     * Returns the image data.
     *
     * @return a 3D array representing the image data
     */
    public double[][][] getImageData() {
        return imageData;
    }

    /**
     * Returns the label.
     *
     * @return a 1D array representing the label
     */
    public double[] getLabel() {
        return label;
    }
}