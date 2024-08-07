package cnn.interfaces;

/**
 * Interface representing an activation function used in neural networks.
 * An activation function determines the output of a node given an input or set of inputs.
 */
public interface ActivationFunction {
    
    /**
     * Applies the activation function to a single input value.
     *
     * @param x the input value to be activated
     * @return the activated output value
     */
    double activate(double x);
    
    /**
     * Computes the derivative of the activation function for a given input value.
     * This is often used in backpropagation during the training of neural networks.
     *
     * @param x the input value for which the derivative is calculated
     * @return the derivative of the activation function at the given input value
     */
    double derivative(double x);
    
    /**
     * Applies the activation function to an array of input values.
     * This method uses the single input {@link #activate(double)} method to process each element in the input array.
     *
     * @param input the array of input values to be activated
     * @return an array containing the activated output values
     */
    default double[] activate(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = activate(input[i]);
        }
        return output;
    }
}