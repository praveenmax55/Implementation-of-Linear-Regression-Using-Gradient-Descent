# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot
2. Trace the best fit line and calculate the cost function
3. Calculate the gradient descent and plot the graph for it
4. Predict the profit for two population sizes.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Praveen D
RegisterNumber: 212222240076

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X, y , theta and generate the cost fuction in a linear regression model
  """
  m=len(y)
  h=X.dot(theta)  # length of training data
  square_err=(h-y)**2 

  return 1/(2*m) * np.sum(square_err)  
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)   # function call

def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
 predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0))) 
*/
```

## Output:
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/746f5932-314f-46b2-85b0-0f9cbe7a125d)
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/8abcd571-8162-4cde-8afa-026a45ca72c8)
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/bdaccf74-d538-4582-9dd6-c56e38ad20d6)
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/b149fdce-5bea-4046-a91f-3f01981f8510)
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/4bf77e21-4b9f-4664-8e8b-a73aa733983d)
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/3297c9d8-219d-4100-84ef-76fe4e7556d2)
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/de77c6a7-0774-4a0f-82f7-91d0ebbc209d)
![image](https://github.com/praveenmax55/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497509/d17650de-4231-49dd-b539-1c451056b30f)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
