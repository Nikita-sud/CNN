package cnn;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import cnn.utils.ImageData;

public class MNISTReader {

    public static void main(String[] args) throws IOException {
        String imagesFile = "data/train-images.idx3-ubyte";
        String labelsFile = "data/train-labels.idx1-ubyte";
        List<ImageData> dataset = readMNISTData(imagesFile, labelsFile);
        Collections.shuffle(dataset);
        System.out.println(Arrays.deepToString((Object[])dataset.get(0).getImageData()));
        System.out.println(Arrays.toString(dataset.get(0).getLabel()));
        // Use the dataset as needed
    }

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

                // Store the image data and label in the dataset
                dataset.add(new ImageData(new double[][][]{imageData}, arrayLabel));
            }

            return dataset;
        }
    }
}