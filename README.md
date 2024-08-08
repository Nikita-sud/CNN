
# Convolutional Neural Network (CNN) Project

## Overview
This project implements a Convolutional Neural Network (CNN) from scratch in Java. The CNN is designed to process image data, perform forward and backward propagation, and train using Stochastic Gradient Descent (SGD). The project includes various types of layers, activation functions, and utility classes to facilitate the construction and training of the network.

## Features
- Convolutional Layers
- Pooling Layers (Max and Average)
- Batch Normalization Layers
- Fully Connected Layers
- Dropout Layers
- Flatten Layers
- Softmax Layer
- Various Activation Functions (ReLU, LeakyReLU, ELU, Sigmoid, Tanh)
- MNIST data reader
- SGD training with mini-batches
- Model saving and loading
- Drawing panel for digit input
- Digit recognizer for hand-drawn digits

## Directory Structure
```
|-- .vscode
|   |-- settings.json
|-- data
|   |-- data/t10k-images.idx3-ubyte
|   |-- data/t10k-labels.idx1-ubyte
|   |-- data/train-images.idx3-ubyte
|   |-- data/train-labels.idx1-ubyte
|-- savedNetwork
|   |-- my_cnn.dat
|-- src
|   |-- main/java/cnn
|   |   |-- digitsDrawing
|   |       |-- DrawingPanel.java
|   |   |-- interfaces
|   |       |-- ActivationFunction.java
|   |       |-- AdaptiveLayer.java
|   |       |-- Layer.java
|   |       |-- ParameterizedLayer.java
|   |   |-- layers
|   |       |-- BatchNormalizationLayer.java
|   |       |-- ConvolutionalLayer.java
|   |       |-- DropoutLayer.java
|   |       |-- FlattenLayer.java
|   |       |-- FullyConnectedLayer.java
|   |       |-- PoolingLayer.java
|   |       |-- SoftmaxLayer.java
|   |   |-- utils
|   |       |-- ELU.java
|   |       |-- ImageData.java
|   |       |-- ImageProcessor.java
|   |       |-- LeakyReLU.java
|   |       |-- MatrixUtils.java
|   |       |-- ReLU.java
|   |       |-- Sigmoid.java
|   |       |-- Tanh.java
|   |   |-- CNN.java
|   |   |-- DigitRecognizer.java
|   |   |-- Main.java
|   |   |-- MNISTReader.java
|-- test/java/cnn
|-- target
|-- pom.xml
|-- README.md
```

## Getting Started
### Prerequisites
- Java Development Kit (JDK) 8 or higher
- Git

### Installing
1. Clone the repository:
    ```sh
    git clone https://github.com/Nikita-sud/CNN
    ```

## Usage
### Training the Network
The network can be trained using the MNIST dataset, which is included in the `data` directory. The `Main` class reads the dataset, constructs the CNN, and trains it using SGD. The trained model is saved to `savedNetwork/my_cnn.dat`.

### Loading a Saved Network
To load a previously saved network, use the `CNN.loadNetwork` method:
```java
CNN cnn = CNN.loadNetwork("savedNetwork/my_cnn.dat");
```

### Evaluating the Network
To evaluate the network on a test dataset, use the `evaluate` method:
```java
List<ImageData> testDataset = MNISTReader.readMNISTData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
int correct = cnn.evaluate(testDataset);
System.out.println("Test accuracy: " + (double) correct / testDataset.size());
```

### Drawing and Recognizing Digits
You can use the `DigitRecognizer` class to draw and recognize hand-drawn digits. The `DigitRecognizer` class uses the trained CNN model to predict the digit drawn on a `DrawingPanel`.

To run the `DigitRecognizer` application:
```java
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
```

## Project Structure
### Interfaces
- `ActivationFunction`: Defines methods for applying an activation function and its derivative.
- `Layer`: Represents a layer in the neural network with methods for forward and backward propagation.
- `AdaptiveLayer`: Extends `Layer` to include methods for initialization.
- `ParameterizedLayer`: Extends `Layer` to include methods for parameter updates.

### Layers
- `BatchNormalizationLayer`: Normalizes the input to have zero mean and unit variance.
- `ConvolutionalLayer`: Applies learnable filters to the input tensor.
- `DropoutLayer`: Randomly sets a fraction of input units to zero during training.
- `FlattenLayer`: Flattens a 3D input tensor into a 1D output tensor.
- `FullyConnectedLayer`: Connects every input neuron to every output neuron.
- `PoolingLayer`: Reduces the spatial dimensions of the input tensor.
- `SoftmaxLayer`: Applies the softmax function to the input tensor.

### Utilities
- `MatrixUtils`: Contains various matrix operations used in convolutional neural networks.
- `ImageData`: Represents image data and its corresponding label.
- `ImageProcessor`: Utility class for processing images for use in a CNN.
- Activation Functions: Implementations of various activation functions (`ReLU`, `LeakyReLU`, `ELU`, `Sigmoid`, `Tanh`).

### Main Class
- `Main`: Demonstrates how to construct, train, and evaluate the CNN using the MNIST dataset.

### Drawing and Recognizing Digits
- `DrawingPanel`: Panel for drawing digits and visualizing hand-drawn digits.
- `DigitRecognizer`: Class for recognizing hand-drawn digits using a CNN.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License
This project is for educational purposes and is not licensed under any specific terms.

For any questions or suggestions, feel free to open an issue or contact the project maintainer through your profile contact information.