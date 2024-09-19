# python-neural-networks

### Neural Network Code Explanation

This program demonstrates a simple feedforward neural network model using Python and NumPy. It features the following structure:

- Input Layer: 4 input neurons
- Hidden Layer: 3 neurons
- Output Layer: 3 neurons

The program initializes random weights for two weight matrices and processes the input through the network using the sigmoid activation function. Below is a detailed explanation of each part of the code.

### Components of the Code

##### Sigmoid Activation Function: 
The sigmoid function is used as the activation function. It introduces non-linearity to the network and squashes the output between 0 and 1.

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

##### Input Vector: 
The input vector x is initialized with 4 elements randomly sampled from a uniform distribution between 0.1 and 0.9. The shape of the input vector is (1, 4).

`x = np.random.uniform(low=0.1, high=0.9, size=(1, 4))`

##### Weight Matrices:

W1: The weight matrix between the input layer and the hidden layer is initialized with random values sampled from a normal distribution. The shape of W1 is (3, 4), meaning there are 3 hidden layer neurons and 4 input neurons.

W2: The weight matrix between the hidden layer and the output layer is initialized similarly, with a shape of (3, 3) for 3 output neurons and 3 hidden layer neurons.

    W1 = np.random.normal(loc=0, scale=1.5, size=(3, 4))
    W2 = np.random.normal(loc=0, scale=1.3, size=(3, 3))

##### Forward Propagation:

Hidden Layer Activations: The dot product between the input vector x and the transpose of the weight matrix W1 is calculated. This produces a (1, 3) matrix, which is passed through the sigmoid function to produce the hidden layer activations z2.

    z = np.dot(x, W1.transpose())
    z2 = sigmoid(z)

Output Layer Activations: The dot product of the hidden layer activations z2 and the weight matrix W2 is computed, followed by applying the sigmoid function to produce the final output o.

    z3 = np.dot(z2, W2)
    o = sigmoid(z3)

### Print Statements:

The program prints the following:

- Input vector
- Weight matrices (W1 and W2)
- The final output layer values

This helps you understand how the input is propagated through the network and transformed by the weights at each layer.

    print("\ninput layer: \nvector:\n", x, "\n")
    print("\nhidden layer: \nweight matrix number 1:\n", W1, "\n")
    print("weight matrix number 2:\n", W2, "\n")
    print("\nexit layer: \noutput:\n", o)

### How the Network Works

- Input Layer: The input vector x consists of 4 elements, representing 4 input features.
- Hidden Layer: This layer has 3 neurons. The dot product between the input vector and the weight matrix W1 is computed, and the result is passed through the sigmoid function to introduce non-linearity.
- Output Layer: The output from the hidden layer is multiplied by the second weight matrix W2. After applying the sigmoid activation function again, the final output values o are generated.

### How to Run the Code

Install NumPy if you haven't already:

    pip install numpy

Copy and paste the code into a Python script (e.g., neural_network.py).
Run the script using Python:

    python neural_network.py

### Example Output


input layer: 
vector:
 [[0.32319874 0.53840696 0.57651231 0.61748694]] 

hidden layer: 
weight matrix number 1:
 [[ 1.4891489  -1.6909135   0.45768887 -0.57104836]
  [-0.30725307  0.67958519  1.02938035  2.42103324]
  [-2.68710222 -0.72703881  2.42426982  0.8566211 ]] 

weight matrix number 2:
 [[-0.03373565  2.45273279 -0.43008313]
  [-2.09310807 -0.97527661  0.62315233]
  [ 1.2578198   2.26008715 -0.20737622]] 

exit layer: 
output:
 [[0.35903069 0.12897535 0.79842741]]

The printed output will show the randomly initialized input vector, weight matrices, and the final output vector.

### Modifying the Network

To change the number of neurons in the hidden or output layers, modify the shape of the weight matrices W1 and W2.
You can use different activation functions or add more layers for experimentation.

### Conclusion

This is a simple demonstration of how a feedforward neural network with one hidden layer operates using basic matrix operations. This structure can be extended to build more complex models.

### Contributions

Krzysztof Piotrowski

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Contact

For any questions or issues, please contact krzysztof.piotrowski.in@gmail.com
