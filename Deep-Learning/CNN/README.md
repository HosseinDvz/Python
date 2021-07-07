# Convolutional Neural Networks
 First we try to understand what is convolution:
 ## Definition:
 &nbsp;&nbsp;In mathematics (in particular, functional analysis), convolution is a mathematical operation on two functions (f and g) that produces a third function (f*g) that expresses how the shape of one is modified by the other. The term convolution refers to both the result function and to the process of computing it.<br/>
 
 <p align="center"><img src = "images/ConvFormula.jpg"><br/>

Convolution is commutative.<br/> 
  I became familirar with convolution in Signal Processing. To completely undesrtand what is convolution and how to calculate it watch this [video](https://www.youtube.com/watch?v=LIs0h34iFN8&list=PLJ-OcUCIty7evBmHvYRv66RcuziszpSFB&index=11). <br/>
In the context of Deep Learning, when we talk about the *Convolution* of f and g, we actually talk about the **correlation** of f and g which is sliding one signal (f) through the other one(g) without flipping it but in Convolotion we flip one signal over y axis. Correlation is measurement of the similarity between two signals/sequences. Convolution is measurement of effect of one signal on the other signal.Correlation is NOT commutative. The following image from Wikipedia shows the difference.
 <p align="center"><img src = "images/ConVsCor.png"><br/>
 
 ## Main Idea behind the CNNs
 The idea is similar to filtering the signals; for example, when we pass a signal through a lowpass system(filter), it rejects the high frequencies and keeps low frequencies in the signal. <br/>
   In the context of CNN, the function f would be our input, the function g will be our filter. We are going to have a couple of filters (Kernel). Each kernel extract a feature in the picture. For example one kernel may find horizonal lines and one kernel the vertical lines or a specific color. Those feature will be combined and passed to a MLP and then classified. Actually, what CNNs do in recognizing the picture is very what we do; first we identify the important parts of the picture and then decide what it is. <br/>
   
   
 There are two benefits in filtering the image before passing it to a MLP. Firstly, it dramatically decreases the dimentionality of input. Each colored image in CIFAR10 has 32*32*3 dimentions and after flattening it becomes an input of  size 3072 for a single image which is too many for MLP. Secondly, CNNs do not ignore the vicinity of two pixles.  In a normal network, we would have connected every pixel in the input image to a neuron in the next layer. In doing so, we would not have taken advantage of the fact that pixels in an image are close together for a reason and have special meaning.
