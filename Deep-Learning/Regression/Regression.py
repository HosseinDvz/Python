# -*- coding: utf-8 -*-
"""
@author: Hossein DVZ
"""

import numpy as np
import matplotlib.pyplot as plt



def getData(N,var):
    x=np.random.uniform(0,1,(N,1))
    y=np.cos(2*x*np.pi) +np.random.normal(0.0, var,(N,1))
    return x,y


def  getMSE(theta,X,y):
    
    m = len(y)
    
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost


#While the model degree is n, we need to estimate n+1 coeficients, bios + X coeficients
def getdegree(d):
    return d+1

#calculating the gradient matrix
def grad_matrix(x,d):
    
     if d==0:
         return np.ones((len(x),1))
     elif d==1:
         return np.c_[np.ones((len(x),1)),x]
     
     X_grad = x    
     for i in range(1,d):
        x_add=[]
        x_add = x**i
        X_grad = np.concatenate((X_grad,x_add), axis=1)
        
     X_grad_b = np.c_[np.ones((len(x),1)),X_grad]
    
     return X_grad_b




#taking user input to start the job
number_of_tdata = int(input("Enter the number of training data point(integer): "))
noise_var = float(input("Enter the variance of the noise: "))
d = int(input("Enter the model degree (integer): "))
iterations = int(input("Enter the number of iteration(integer): "))
learning_rate = float(input("Enter the learning rate: "))
batch_size=int(input("Enter the batch size for mini-batch GD(integer): "))
wieght_decay=input("do you want to apply weight decay?(y/n) ")

if wieght_decay.casefold()=='y' :
    wd=float(input("Enter the weight decay constant: "))
else:
    wd=0
    
print("\n")
X,y = getData(number_of_tdata,noise_var)
init_theta = np.random.randn(getdegree(d),1)

def gradient_descent(X,y,d,learning_rate,iterations,init_theta):
   
    m = len(y)
    theta_G=init_theta    
    X_b = grad_matrix(X, d)
   
 
    for it in range(iterations):
        
        prediction = np.dot(X_b,theta_G)
        theta_G = theta_G -(1/m)*learning_rate*( X_b.T.dot(prediction - y)+2*wd*theta_G)
        
       
        
    return theta_G,X_b



def stocashtic_gradient_descent(X,y,d,learning_rate,iterations,init_theta):
    
    m = len(y)
    theta_SGD=init_theta
    #cost_history = np.zeros(iterations)
    data = np.concatenate((X,y), axis=1)

    for i in range(iterations):
        if iterations<=m:
            #rand_ind=np.random.randint(0,len(y))
            X_i = data[i][0].reshape(1,X.shape[1])
            X_i = grad_matrix(X_i, d)
            y_i=data[i][1].reshape(1,1)
            prediction = np.dot(X_i,theta_SGD)
            theta_SGD = theta_SGD -(1/m)*learning_rate*( X_i.T.dot(prediction - y_i)+2*wd*theta_SGD) 

        else:
            for i in range(0,int(iterations/m)):
                for j in range(0,len(y)):
                    #rand_ind=np.random.randint(0,len(y))
                    X_i = data[j][0].reshape(1,X.shape[1])
                    X_i = grad_matrix(X_i, d)
                    y_i=data[j][1].reshape(1,1)
                    prediction = np.dot(X_i,theta_SGD)
                    theta_SGD = theta_SGD -(1/m)*learning_rate*( X_i.T.dot(prediction - y_i)+2*wd*theta_SGD) 
 
        return theta_SGD#, cost_history

def minibatch_gradient_descent(X,y,learning_rate,iterations,batch_size,init_theta):
    
    theta_mini=init_theta
    if batch_size >= len(y):
        theta_mini,X_b=gradient_descent(X, y, d, learning_rate, iterations, init_theta)
        return theta_mini   
    
    count =0
    while(count<=iterations):
        X_i=[]
        y_i=[]
        if count<len(y):
            X_i=X[count:count+batch_size+1]
            y_i=y[count:count+batch_size+1]
            theta_mini,X_b=gradient_descent(X_i, y_i, d, learning_rate,
                                            min(iterations,int(iterations/batch_size)),theta_mini)
        
        count+=batch_size
          
    return theta_mini
    
        
    

def fitData(X,y,d):
        theta1,X_b=gradient_descent(X,y,d,learning_rate,iterations,init_theta)
        print("Gradient descent coeffients: ", theta1.T)
        print("MSE for grdient descent: ", getMSE(theta1,X_b,y))
        print("================================================\n")
        
        
        theta2=stocashtic_gradient_descent(X,y,d,learning_rate,iterations,init_theta)
        print("Stochastic Gradient descent coeffients: ", theta2.T)
        print("MSE for Stochastic Gradient descent: ", getMSE(theta2,X_b,y))
        print("================================================\n")
        
        
        theta3=minibatch_gradient_descent(X,y,learning_rate,iterations,batch_size,init_theta)
        print("Mini batch Gradient Descent coeffients: ", theta3.T)
        print("MSE for Mini batch Gradient Descent: ", getMSE(theta3, X_b, y))
        print("================================================\n")
        
        X_test,y_test=getData(1000, noise_var)
        X_test_b = grad_matrix(X_test,d)
        print("E_out for Gradient Descent(GD) for 1000 test data: {:.3f} ".format(getMSE(theta1,X_test_b,y_test)))
        print("E_out for Stochastic GD for 1000 test data: {:.3f}".format(getMSE(theta2,X_test_b,y_test)))
        print("E_out for mini GD for 1000 test data: {:.3f}".format(getMSE(theta3,X_test_b,y_test)))
        
        
        
        #plotting all models
        y_pred1 = X_b.dot(theta1)
        y_pred2 = X_b.dot(theta2)
        y_pred3 = X_b.dot(theta3)
        plt.plot(X,y_pred1,'r.',label='GD')
        plt.plot(X,y_pred2,'b.',label='SGD')
        plt.plot(X,y_pred3,'g.',label='mini_GD')
        plt.plot(X,y,'m.',label='Original data')
        plt.xlabel("$x$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        _ =plt.axis([0,1,-2,3])
        plt.legend(loc='upper right', frameon=True);
        plt.show()
        
        return theta1,getMSE(theta1,X_b,y),getMSE(theta1,X_test_b,y_test)
        
        
    
        

theta,Ein,Eout = fitData(X, y, d)


#creating all possible combination of number of training data set, degree and variance
def combination():

    comb=[]
    for i in [2, 5, 10, 20, 50, 100, 200]:
        for j in list(range(0,21)):
            for k in [.01,.1,1]:
                comb.append([i,j,k])
    return comb


    
def experiment():
    
    
      
    #array to store results
     final_results=[]
    #taking all 441 possible combination of N,d,var
     all_comb=combination()
     
     for i in range(0,len(all_comb)):
         N=all_comb[i][0]
         d=all_comb[i][1]
         var=all_comb[i][2]
         lr=0.5
         
         if d==0:
             theta_all=np.empty((1,1))
         elif d==1:
             theta_all=np.empty((2,1))
         else:
             theta_all=np.empty((d+1,0))
        

         Ein_all=np.empty((1,1))
         Eout_all=np.empty((1,1))
         M=0
         while(M<50):
            M+=1
            x_train,y_train=getData(N, var)
            theta,X_b=gradient_descent(x_train,y_train, d, lr, 100,np.random.randn(getdegree(d),1))
            x_test,y_test=getData(1000, var)
            E_in=np.array(getMSE(theta, X_b, y_train)).reshape(1,1)
            X_test_b = grad_matrix(x_test,d)
            E_out=np.array(getMSE(theta,X_test_b,y_test)).reshape(1,1)
        
            Ein_all=np.concatenate((Ein_all,E_in))
            Eout_all=np.concatenate((Eout_all,E_out))
            theta_all=np.concatenate((theta_all,theta),axis=1)
       
        
         theta_final=theta_all.mean(1).reshape(getdegree(d),1)
         
         Ein_avg=Ein_all.mean(0).item()
         print("for N= ",N,"      degree= ",d, "     variance= ",var)
         print("Average in sample error (Ein-bar): {:0.3f}".format(Ein_avg))
         Eout_avg=Eout_all.mean(0).item()
         print("Average out of sample error(Eout-bar): {:0.3f} ".format(Eout_avg))
         x,y=getData(1000, var)
         x_b = grad_matrix(x,d)
         E_bias=getMSE(theta_final, x_b, y)
         print("Out of sample error of averaged hypothesis(E-bias): {:.3f} ".format(E_bias))
         print("===================================================")
         final_results.append([N,d,var,Eout_avg,E_bias])
     return final_results
    
    #return Ein_avg,Eout_avg,E_bias,theta_all,Ein_all,Eout_all,theta_final


#Ein_avg,Eout_avg,E_bias,theta_all,Ein_all,Eout_all,theta_final=

wieght_decay=input("do you want to apply weight decay for the experiment?(y/n) ")
if wieght_decay.casefold()=='y' :
    wd=float(input("Enter the weight decay constant: "))
else:
    wd=0
execute=input("do you want to run the experiment?(y/n) ")
if execute.casefold()=='y':
    final_results=experiment()
else:
    print("program terminated by the user")
    raise KeyboardInterrupt
    

    
    
#*************************plotting result of experiment **********************#   
plot_list1=[]
plot_list2=[]
plot_list3=[]    
for i in range(0,len(final_results)):
    if final_results[i][0] in [2] and float(final_results[i][2])==.1:
        plot_list1.append(final_results[i][4])
        
    elif final_results[i][0] in [10] and float(final_results[i][2])==.1:
        plot_list2.append(final_results[i][4])
        
    elif final_results[i][0] in [200] and float(final_results[i][2])==.1:
        plot_list3.append(final_results[i][4]) 

#setting the very larg errors to zero
for i in range(0,len(plot_list1)):
    if plot_list1[i]>350000:
        plot_list1[i]=0
        
for i in range(0,len(plot_list2)):
    if plot_list2[i]>350000:
        plot_list2[i]=0

for i in range(0,len(plot_list3)):
    if plot_list3[i]>350000:
        plot_list3[i]=0
#end of setting======================        
          
dct={'list_2':plot_list1, 'list_10': plot_list2, 'list_200':plot_list3}
dim=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

for i in [2,10,200]:
    plt.scatter(dim,dct['list_%s' %i], label='N= %s' % i)
    plt.xlabel("$degree$", fontsize=15)
    plt.ylabel("$Ebias$", fontsize=15)
    _ =plt.axis([-1,22,-10000,360000])
    plt.title("Change in bias vs degree for fixed variance(var=0.1)",fontsize=12)
plt.legend()
plt.show()


########################################################
plot_varlist1=[]
plot_varlist2=[]
plot_varlist3=[]
#plot_varlist4=[]

for i in range(0,len(final_results)):
    if final_results[i][0] in [10] and float(final_results[i][1])==8:
        plot_varlist1.append(final_results[i][4])
        
    elif final_results[i][0] in [50] and float(final_results[i][1])==8:
        plot_varlist2.append(final_results[i][4])
        
    elif final_results[i][0] in [100] and float(final_results[i][1])==8:
        plot_varlist3.append(final_results[i][4])
        
    #elif final_results[i][0] in [200] and float(final_results[i][1])==8:
        #plot_varlist4.append(final_results[i][4]) 
        
dct2={'list_10':plot_varlist1,'list_50':plot_varlist2,'list_100':plot_varlist3}#,'list_200':plot_varlist4}
variances=[.01,.1,1]

for i in [10,50,100]:#,200]:
    plt.scatter(variances,dct2['list_%s' %i], label='N= %s' % i)
    plt.xlabel("$Variance$", fontsize=15)
    plt.ylabel("$Ebias$", fontsize=15)
    _ =plt.axis([-.01,1.1,-10000,650000])
    plt.title("change in Ebias vs variance for fixed degree(dim=8)",fontsize=12)
plt.legend()
plt.show()
###################################################

plot_E_N=[]

for i in range(0,len(final_results)):
    if final_results[i][0] in [2,5,10,20,50,100,200] and final_results[i][1] in [7] and float(final_results[i][2])==.1:
        plot_E_N.append([final_results[i][0],final_results[i][3]])

for i in range(0,len(plot_E_N)):
    if plot_E_N[i][1] > 500000:
        plot_E_N[i][1]=0
        
        
for i in range(0,len(plot_E_N)):
    plt.scatter(plot_E_N[i][0],plot_E_N[i][1])
    plt.xlabel("Number of training data",fontsize=15)
    plt.ylabel("$EOut$", fontsize=15)
    plt.title("change in EOut by increasing the number of training data")
plt.show()    
    












































