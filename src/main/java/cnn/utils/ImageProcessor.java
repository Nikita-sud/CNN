package cnn.utils;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

/**
 * A utility class for processing images for use in a convolutional neural network.
 * This class includes methods for resizing and normalizing images.
 */
public class ImageProcessor {

    /**
     * Processes a given BufferedImage by resizing it to 28x28 pixels and normalizing the pixel values.
     *
     * @param image the input BufferedImage to be processed
     * @return a 3D array representing the processed image, normalized to values between 0 and 1
     */
    public static double[][][] processImage(BufferedImage image) {
        BufferedImage resizedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(image, 0, 0, 28, 28, null);
        g2d.dispose();

        double[][][] input = new double[1][28][28];
        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                int color = resizedImage.getRGB(x, y) & 0xFF;
                input[0][y][x] = color / 255.0; 
            }
        }

        return input;
    }
}