package cnn.digitsDrawing;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;

/**
 * A panel for drawing digits, designed for creating and visualizing hand-drawn digits.
 * The panel allows drawing with the mouse and provides functionality to clear the drawing.
 */
public class DrawingPanel extends JPanel {
    private BufferedImage image;
    private Graphics2D g2;
    private int prevX, prevY;
    private static final int SCALE = 20;
    private static final int LINE_THICKNESS = 2; 

    /**
     * Constructs a DrawingPanel with the specified width and height.
     *
     * @param width the width of the panel in pixels
     * @param height the height of the panel in pixels
     */
    public DrawingPanel(int width, int height) {
        setPreferredSize(new Dimension(width * SCALE, height * SCALE));
        image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        g2 = image.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setStroke(new BasicStroke(LINE_THICKNESS));
        clear();

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                prevX = e.getX() / SCALE;
                prevY = e.getY() / SCALE;
            }
        });

        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                int x = e.getX() / SCALE;
                int y = e.getY() / SCALE;
                g2.drawLine(prevX, prevY, x, y);
                prevX = x;
                prevY = y;
                repaint();
            }
        });

        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> clear());

        add(clearButton);
    }

    /**
     * Clears the drawing panel by filling it with black and setting the drawing color to white.
     */
    public void clear() {
        g2.setColor(Color.BLACK);
        g2.fillRect(0, 0, image.getWidth(), image.getHeight());
        g2.setColor(Color.WHITE);
        repaint();
    }

    /**
     * Returns the current drawing as a BufferedImage.
     *
     * @return the current drawing as a BufferedImage
     */
    public BufferedImage getImage() {
        return image;
    }

    /**
     * Paints the component by scaling up the image and drawing it on the panel.
     *
     * @param g the Graphics object to protect
     */
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int color = image.getRGB(x, y) & 0xFF;
                g.setColor(new Color(color, color, color));
                g.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
            }
        }
    }
}