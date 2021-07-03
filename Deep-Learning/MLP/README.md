# Multi Layer Perceptron
&nbsp;&nbsp; Before delveing into MLP, let go through Universl Approximation Theorem.
## Universal Approximation Theorem

<p align="center"><img src = "images/UAT.jpg"><br/>
  
 In summary, theorem says it is possible to approximate any function with a single hidden layer perceptron if it is wide enough. 
A neural network is merely a computation graph representing the compositions of a set of functions.
  <p align="center"><img src = "images/ANN-Graph.gif"><br/>
    
  The above image shows a MLP with two hidden layers. the depth of the network refers to the number of hidden layers and the width, refers to the number of neurons in each hidden layer. We must design the architecture of the network, including how many layers the network should contain (depth), how these layers should be connected to each other, and how many units should be in each layer (width). We cshould carefully design our network;  Some times a deep network results in paralel useless linear computings and a very wide network results in overfitting and both of them lead to poor generalization. The number of network parameters will be more sensitive to width of the network. For example, a MLP with six hidden neurons in one layer will have more parameters than a MLP with six neurons in two hidden layers. In the code, we explore how changing the width and depth of the network will affect the result.<br/>
    
The Activation Function is another important factor which computes the hidden layers values. We explored one of them, the Logistic Function and its charactristics in [Logistic Regression.](https://github.com/HosseinDvz/Python/tree/main/Deep-Learning/Logistic%20%26%20Softmax%20Regression). Another important activation function is Rectified Linear Unit activation function (ReLU): g(z) = max{0,z}.
<p align="center"><img src = "images/ReLu.jpg"><br/>
  
  This activation function is the default activation function recommended for use with most feedforward neural networks. Applying this function to the output of a linear transformation yields a nonlinear transformation. However, the function remains very close to linear, in the sense that is a piecewise linear function with two linear pieces. Because rectified linear units are nearly linear, they preserve many of the properties that make linear models easy to optimize with gradientbased methods. They also preserve many of the properties that make linear models generalize well.(Deep Learning, Goodfellow). <br/>
  
 g(z) = max{0,z} + a * min{0,z} is the generalization of ReLU and depend on the values of a, we have different functions:
  - Leaky ReLU (another commonly used activation function) :very small a
 <p align="center"><img src = "images/LeakyReLU.jpg"><br/>
   
   
 - Absolute value function : a = -1 <br/>
   
 - Prametric RelU (PReLU) : learnable a
