 # Regression
 
 In this project we are going to explore the fitting and generalization of regression models via simulation whitout using Python libraries. Here, we can see how changing the paramters of our regression model (namely, degree of the polynomial)  , changing the the number of data and changing the quality of data (by adding noise) can affect the result.
 
 Suppose that X and Y are both real valued random variables, where X takes value in (0, 1) and Y depends on X according to: <br/>
                                           <p align="center">**Y = cos(2*pi*X) + Z**<br/>
  
 where Z is a zero mean Gaussian random variable with variance *sigma^2*, and independent of X. But we assume
that we do not know this dependency of Y on X and that you only observe a sample of N (X, Y ) pairs.
Based on the observed sample, we will learn a polynomial regression model and examine the fitting and
generalization capability of our model in relation to the model complexity and sample size.<br/>
  
  
  ## Manual Implementation of Gradient Descent (GD)
  
  Gradient Descent is probably the most famous optimization algorithm and the back bone of many other optimization algorithms. Here, I manually implemented this algorithm.
To understand how Gradient Descent works, go this link (min 53).<br/> (https://www.youtube.com/watch?v=qSTHZvN8hzs). <br/>
  
The main problem in implementing the GD is taking the drivative of loss function. The loss function here is Mean Square Error (MSE). I have used a very simple and creative idea to take the derivative. 
The function we are going to fit to our data is a polynomial with degree d (do not confuse degree with dimention):<br/>
                                       <p align="center">**Y=a0 + a1X + a2X^2 + ...+ adX^d** <br/>
                                         
We are going to find (a0,a1,...,ad) so we need to take the derivative with respect to (a0,a1,...,ad)  which all of them are degree 1. therefore, their derivative would be their coefficients X,X^2, ..., X^d. The following fuction does this job in the code and provide the deraivative matrix: <br/>
                                         
                                         
```
def grad_matrix(x,d):
    
     if d==0:
         return np.ones((len(x),1))
     elif d==1:
         return np.c_[np.ones((len(x),1)),x]
     
     X_grad = x    
     for i in range(2,d):
        x_add=[]
        x_add = x**i
        X_grad = np.concatenate((X_grad,x_add), axis=1)
        
     X_grad_b = np.c_[np.ones((len(x),1)),X_grad]

```
<br/> d is degree of the model which is one of the user's input and x is our data. I have implemented Gradient Descent, Stochastic Gradient Descent(SGD) and Mini Batch Gradient Descent. the x will be different in any of this methods. GD uses all data in each iteration, mini batch GD uses a batch of data (the size of the batch is one of the user's inputs) and SGD uses just one data point in each iteration. So, the most accurate models is GD , then mini batch GD and the fastest and least accurate one is SGD for the fixed number of iterations for all optimization algorithms. You will see how MSE changes at any of these algorithms. Choosing between optimization algorithms depends on the situations.                                
                           
  *To be contiued ....*
                        
 
 
