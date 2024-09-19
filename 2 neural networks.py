import numpy as np
import random

#Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Initialise Input Vector with 4 elements sampled from a uniform distribution
x = np.random.uniform(low=0.1, high=0.9, size=(1, 4))

#Initialise Weight Matrices
W1 = np.random.normal(loc=0,
                          scale=1.5,
                          size=(3, 4))
W2 = np.random.normal(loc=0,
                          scale=1.3,
                          size=(3, 3))

#Compute Hidden Layer Activations
z = np.dot(x, W1.transpose())
z2 = sigmoid(z)

#Compute Output Layer Activations
z3 = np.dot(z2, W2)
o = sigmoid(z3)


print("\ninput layer: \nvector:\n", x, "\n")

print("\nhidden layer: \nweight matrix number 1:\n", W1, "\n")

print("weight matrix number 2:\n", W2, "\n")

print("\nexit layer: \noutput:\n", o)


