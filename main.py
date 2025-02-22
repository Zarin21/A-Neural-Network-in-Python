import numpy as np

# Sigmoid function

"""
The sigmoid function is a mathematical function that produces an "S" shaped curve called a sigmoid curve.
It is used in machine learning, particularly in logistic regression and neural networks, as an activation function. It maps any real-valued number into the range (0, 1), making it useful for models that need to output probabilities.

The sigmoid function is defined as: f(x) = 1/(1+e^(-x)) 
Where: 
    e is the base of the natural logarithm, 
    x is the input to the function, 
    and f(x) is the output.

The nonlin function implements the sigmoid function when deriv is False. 
When deriv is True, it returns the derivative of the sigmoid function, which is useful for backpropagation in neural networks. 
The derivative of the sigmoid function is f'(x) = f(x)(1-f(x)).

Parameters:
    x: input to the sigmoid function
    deriv: boolean flag to indicate whether to return the derivative of the sigmoid function

"""
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x) # Derivative of the sigmoid function
    return 1/(1+np.exp(-x)) # Sigmoid function

# Input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

