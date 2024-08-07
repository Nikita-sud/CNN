
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

## Directory Structure
```
|-- .vscode
|-- data
|-- savedNetwork
|-- src
|   |-- main/java/cnn
|   |   |-- interfaces
|   |   |-- layers
|   |   |-- utils
|   |   |-- CNN.java
|   |   |-- Main.java
|   |   |-- MNISTReader.java
|   |-- test/java/cnn
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
- Activation Functions: Implementations of various activation functions (`ReLU`, `LeakyReLU`, `ELU`, `Sigmoid`, `Tanh`).

### Main Class
- `Main`: Demonstrates how to construct, train, and evaluate the CNN using the MNIST dataset.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License
This project is for educational purposes and is not licensed under any specific terms.

For any questions or suggestions, feel free to open an issue or contact the project maintainer through your profile contact information.
