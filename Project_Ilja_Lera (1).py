#!/usr/bin/env python
# coding: utf-8

# In[219]:


import math
import random
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt
import time


# In[39]:


A = np.array([[2,11],[10,-11]]) #решение (2,1)
b = np.array([15,9])
l = 0


# In[41]:


#оценим константу липшица
B=2*A.T@A
L=np.trace(B)
print(L)
print(B)


# In[234]:


iris = sklearn.datasets.load_iris()
n = 2
m = 150
X = iris.data[:, :n]
y = (iris.target != 0) * 1
e = np.full(X.shape[0], 1)
ex = np.column_stack((e, X))
exy = np.column_stack((e, X, y))
#print(exy)
np.random.shuffle(exy)
#print(exy)


# In[ ]:


#print("Введите размерность пространства: ")
#n = int(input()) 
print("Введите матрицу ")
A = np.array([list(map(int, input().split())), list(map(int, input().split()))])
    
print("Введите столбец значений ") 
b=np.array([int(j) for j in input().split()])


# In[42]:


def Function (x):
    return np.linalg.norm(A.dot(x)-b)**2 + l*np.linalg.norm(x)


# In[43]:


def Grad (x):
    if np.linalg.norm(x) == 0:
        return 2*(A.T).dot(A.dot(x)-b)
    return 2*(A.T).dot(A.dot(x)-b) + l*x/np.linalg.norm(x)


# In[44]:


def MirrorDescent (x0):
    k=1
    #h = 0.0001/k
    g = Grad(x0)
    sum =np.sum(x0*np.exp(-g/L))
    if sum != 0:
        x1=x0*np.exp(-g/L)/sum
    else:
        return "SUM = 0"
    #d = np.append(xarr,x1)
    while np.linalg.norm(x1-x0) > 0.000001:
        #h = 0.0001/k
        x0=x1
        g = Grad(x0)
        sum = np.sum(x0*np.exp(-g/L))
        if sum != 0:
            x1=x0*np.exp(-g/L)/sum
        else:
            return "SUM = 0"
        k += 1
        #d = np.append(d,x1)
    return x1


# In[45]:


def makeData ():
    x = np.arange (-10, 10, 0.1)
    y = np.arange (-10, 10, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.ndarray(shape=(200,200))
    for i in range(0,199):
        for j in range(0,199):
            zgrid[i][j] = Function(np.array([x[i],y[j]]))
    return xgrid, ygrid, zgrid


# In[46]:


def Draw(d):
    m=d.size
    k=int(m/2)
    coord1=[0 for i in range(k)]
    coord2=[0 for i in range(k)]
    for i in range (0,m,2):
        coord1[int(i/2)]=d[i]
        coord2[int(i/2)]=d[i+1]
    coords = np.array([coord1,coord2])
    coord3=[0 for i in range(k)]
    for i in range (k):
        coord3[i]=Function(np.array([coords[0][i],coords[1][i]]))
    x, y, z = makeData()

    fig = pylab.figure()
    axes = Axes3D(fig)
    
    axes.scatter(coords[0], coords[1], coord3, color = 'black')
    axes.plot_surface(x, y, z, alpha = 0.3, color = 'yellow')
    

    pylab.show()


# In[145]:


def Grad_descent(x0):
    start_time = time.time()
    xarr=np.array([])
    #h=0.0001
    x1=x0-Grad(x0)/L
    d = np.append(xarr,x1)
    while (np.linalg.norm(x1-x0) > 0.000001):
        x0=x1
        x1= x0-Grad(x0)/L
        d = np.append(d,x1)
    #Draw(d)
    print("Time - ", time.time() - start_time)
    return x1


# In[146]:


def sigmoid(theta, x):
    x = np.delete(x,n+1)
    return 1 / (1 + np.exp(-np.dot(theta.T,x)))


# In[235]:


def Grad_descent_log():
    theta = np.zeros(n+1)
    lr = 0.1
    for _ in range(2000):
        _sum = 0
        for j in range(0,n+1):
            #print("hey")
            for i in range(0,math.ceil(m/2)):
                _sum += (sigmoid(theta, exy[i]) - exy[i][n+1])*exy[i][j]
            #print(_sum)
            theta[j] -= lr*_sum
        #print(theta)
    return theta


# In[238]:


def check():
    theta = Grad_descent_log()
    #print(theta)
    counter = 0
    for i in range(math.ceil(m/2), m):
        ans = sigmoid(theta.T, exy[i])
        #print(ans, y[i])
        if math.ceil(ans-0.5) != exy[i][n+1]:
            counter += 1
    print("Ошибок в предсказании: ", counter)


# In[239]:


check()


# In[11]:


def Delayed_Grad_descent(x0):
    start_time = time.time()
    #h=0.0001
    arrx = []
    list.append(arrx, x0)
    g0 = Grad(x0)
    x0 = x0 - g0/L
    g1 = Grad(x0)
    list.append(arrx, x0)
    x0 = x0 - g0/L
    list.append(arrx, x0)
    #d = np.append(xarr,x1)
    while (np.linalg.norm(arrx[-1]-arrx[-2]) > 0.000001):
        list.append(arrx, (arrx[-1]-g1/L))
        g1 = g0
        g0 = Grad(arrx[-1])
        #d = np.append(d,x1)
    #Draw(d)
    print("Time - ", time.time() - start_time)
    return arrx[-1]


# In[12]:


def Dual(x0, z):
    start_time = time.time()
    x1 = x0 + 1
    xarr=np.array([])
    k = 1
    d = np.append(xarr,x0)
    while (np.linalg.norm(x1-x0) > 0.00001):
        #q = 1/k
        g = Grad(x0)
        x1 = x0
        z = z + g
        x0 = -z/L
        if np.linalg.norm(x0) < 100000: d = np.append(d,x0)
        #print("X: ", x0)
        k += 1
    #Draw(d)
    #print(d)
    print("Time - ", time.time() - start_time)
    return x0
        


# In[13]:


print(Grad_descent(np.array([3,2])))


# In[14]:


print(MirrorDescent(np.array([3,2])))


# In[15]:


print(Delayed_Grad_descent(np.array([3,2])))


# In[16]:


print(Dual(np.array([3,2]), np.array([1,2])))


# In[ ]:




