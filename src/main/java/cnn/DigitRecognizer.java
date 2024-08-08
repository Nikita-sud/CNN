package cnn;

import javax.swing.*;

import cnn.digitsDrawing.DrawingPanel;
import cnn.utils.ImageProcessor;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * A class for recognizing hand-drawn digits using a convolutional neural network (CNN).
 * The class provides functionality to process an image and predict the digit.
 */
public class DigitRecognizer {
    private CNN cnn;

    /**
     * Constructs a DigitRecognizer with the specified CNN.
     *
     * @param cnn the convolutional neural network to use for digit recognition
     */
    public DigitRecognizer(CNN cnn) {
        this.cnn = cnn;
    }

    /**
     * Returns the index of the maximum value in an array.
     *
     * @param array the array to search
     * @return the index of the maximum value
     */
    private int argMax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Recognizes the digit in the given image by processing the image and using the CNN to predict the digit.
     *
     * @param image the input BufferedImage containing the hand-drawn digit
     * @return the recognized digit (0-9)
     */
    public int recognize(BufferedImage image) {
        double[][][] input = ImageProcessor.processImage(image);
        double[][][] output = cnn.forward(input);
        return argMax(output[0][0]);
    }

    /**
     * Main method for running the DigitRecognizer application.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        CNN cnn = CNN.loadNetwork("savedNetwork/my_cnn.dat");
        if (cnn == null) {
            System.out.println("Failed to load CNN.");
            return;
        } else {
            cnn.printNetworkSummary();
        }

        DigitRecognizer recognizer = new DigitRecognizer(cnn);
        JFrame frame = new JFrame("Draw a digit");
        DrawingPanel panel = new DrawingPanel(28, 28);
        frame.add(panel);

        JLabel resultLabel = new JLabel("Recognized digit: ");
        resultLabel.setFont(new Font("Serif", Font.BOLD, 24)); 
        frame.add(resultLabel, BorderLayout.SOUTH);

        // Timer to update the result label every 500 milliseconds (0.5 seconds)
        Timer timer = new Timer(500, e -> {
            BufferedImage image = panel.getImage();
            int digit = recognizer.recognize(image);
            resultLabel.setText("Recognized digit: " + digit);
        });
        timer.start();

        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}