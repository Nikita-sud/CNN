package cnn.utils;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;

/**
 * Utility class for performing image augmentation, including random shifts and rotations.
 * This class is primarily used for data augmentation in machine learning models.
 */
public class ImageAugmentation {

    /**
     * Augments an image by applying random shifts and rotations. The image data is assumed 
     * to be normalized between 0 and 1, where 0 represents black and 1 represents white.
     *
     * @param imageData a 2D array of doubles representing the pixel values of the image, 
     *                  normalized between 0 and 1. The array dimensions should match the 
     *                  dimensions of the image.
     * @param maxShift the maximum number of pixels by which the image can be shifted 
     *                 in both the x and y directions. The actual shift will be randomly 
     *                 chosen within this range.
     * @param maxRotation the maximum rotation angle (in degrees) by which the image can 
     *                    be rotated. The actual rotation will be randomly chosen within 
     *                    this range, in both clockwise and counterclockwise directions.
     * @return a 2D array of doubles representing the augmented image, with the same 
     *         dimensions as the input array. The pixel values are normalized between 
     *         0 and 1.
     */
    public static double[][] augment(double[][] imageData, int maxShift, int maxRotation) {
        int rows = imageData.length;
        int cols = imageData[0].length;

        // Convert to BufferedImage for manipulation
        BufferedImage image = new BufferedImage(cols, rows, BufferedImage.TYPE_BYTE_GRAY);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int pixel = (int) (imageData[r][c] * 255);
                image.setRGB(c, r, (pixel << 16) | (pixel << 8) | pixel);
            }
        }

        // Apply transformations
        BufferedImage augmentedImage = new BufferedImage(cols, rows, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = augmentedImage.createGraphics();
        AffineTransform transform = new AffineTransform();

        // Calculate random shift within bounds
        int maxShiftX = Math.min(maxShift, cols / 2);
        int maxShiftY = Math.min(maxShift, rows / 2);
        double shiftX = (Math.random() * 2 * maxShiftX) - maxShiftX;
        double shiftY = (Math.random() * 2 * maxShiftY) - maxShiftY;
        transform.translate(shiftX, shiftY);

        // Random rotation
        double rotation = (Math.random() - 0.5) * 2 * Math.toRadians(maxRotation);
        transform.rotate(rotation, cols / 2.0, rows / 2.0);

        g2d.drawImage(image, transform, null);
        g2d.dispose();

        // Convert back to double array
        double[][] augmentedData = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int pixel = (augmentedImage.getRGB(c, r) & 0xff);
                augmentedData[r][c] = pixel / 255.0;
            }
        }

        return augmentedData;
    }
}