# Logistic Regression
   ## Introduction 
&nbsp;&nbsp; Here we have a classification problem; developing a classifier which assigns a random example X (possibly unseen) a class label
Y . In summary, in discriminative modeling,we have a data set D = {(Xi, Yi), i: 1,2,...,N} (e.g (Xi = image of a dog, Yi = dog)). we are choosing a family of hypothesis H for conditional distribution ![](images/cond.jpg) (Modelling), then, we try to find best distribution, ![](images/cond2.png) , in H based on our dataset by using Maximum Likelihood Estimator or Maximum a *Posteriori* Estimator (Learning). In the case where Y = {0,1} for an unseen Xi, we declare Y = 1 if ![](images/pred1.png) and Y=0 otherwise (Classification/Prediction).
## Logistic Regression Model
