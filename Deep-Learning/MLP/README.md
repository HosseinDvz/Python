# Multi Layer Perceptron
&nbsp;&nbsp; Before delveing into MLP, let go through Universl Approximation Theorem.
## Universal Approximation Theorem

<p align="center"><img src = "images/UAT.jpg"><br/>
  
 In summary, theorem says it is possible to approximate any function with a single hidden layer perceptron if it is wide enough. 
A neural network is merely a computation graph representing the compositions of a set of functions.
  <p align="center"><img src = "images/ANN-Graph.gif"><br/>
    
  The above image shows a MLP with two hidden layers. the depth of the network refers to the number of hidden layers and the width, refers to the number of neurons in each hidden layer. these two parameters are two important factors in designing a MLP and depends on the nature of the problem. We should be very careful about them; some times a deep network results in paralel useless linear computings and a very wide network results in overfitting and both of them lead to poor generalization. The number of network parameters will be more sensitive to width of the network. For example, a MLP with six hidden neurons in one layer will have more parameters than a MLP with six neurons in two hidden layers. In the code, we explore how changing the width and depth of the network will affect the result.<br/>
    
  
 
  
  
