package cnn;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import cnn.utils.ImageData;
import cnn.utils.ImageAugmentation; // Assuming you have a utility class for augmentations

/**
 * A utility class for reading MNIST data from IDX file format.
 * The MNIST dataset consists of images of handwritten digits and their corresponding labels.
 */
public class MNISTReader {

    /**
     * Main method for reading and processing the MNIST data.
     *
     * @param args command-line arguments
     * @throws IOException if there is an error reading the files
     */
    public static void main(String[] args) throws IOException {
        String imagesFile = "data/train-images.idx3-ubyte";
        String labelsFile = "data/train-labels.idx1-ubyte";
        List<ImageData> dataset = readMNISTData(imagesFile, labelsFile);
        Collections.shuffle(dataset);
        System.out.println(Arrays.deepToString((Object[])dataset.get(0).getImageData()));
        System.out.println(Arrays.toString(dataset.get(0).getLabel()));
        // Use the dataset as needed
    }

    /**
     * Reads MNIST image and label data from IDX files and returns a list of ImageData objects.
     *
     * @param imagesFile the path to the images IDX file
     * @param labelsFile the path to the labels IDX file
     * @return a list of ImageData objects containing the image data and corresponding labels
     * @throws IOException if there is an error reading the files
     */
    @SuppressWarnings("unused")
    public static List<ImageData> readMNISTData(String imagesFile, String labelsFile) throws IOException {
        try (DataInputStream images = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesFile)));
             DataInputStream labels = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsFile)))) {

            int magicNumberImages = images.readInt();
            int numberOfImages = images.readInt();
            int rows = images.readInt();
            int cols = images.readInt();

            int magicNumberLabels = labels.readInt();
            int numberOfLabels = labels.readInt();

            List<ImageData> dataset = new ArrayList<>();
            for (int i = 0; i < numberOfImages; i++) {
                // Read and normalize image data
                double[][] imageData = new double[rows][cols];
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        imageData[r][c] = (images.readUnsignedByte() & 0xff) / 255.0;
                    }
                }

                // Read label
                int label = labels.readUnsignedByte();
                double[] arrayLabel = new double[10];
                arrayLabel[label] = 1.0;

                // Original image
                dataset.add(new ImageData(new double[][][]{imageData}, arrayLabel));

                // Augmented images
                for (int j = 0; j < 4; j++) {
                    double[][] augmentedImage = ImageAugmentation.augment(imageData, 2, 5);
                    dataset.add(new ImageData(new double[][][]{augmentedImage}, arrayLabel));
                }
            }

            return dataset;
        }
    }
}