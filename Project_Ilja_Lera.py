#!/usr/bin/env python
# coding: utf-8

# In[177]:


from math import *
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
import time


# In[178]:


A = np.array([[2,11],[10,-11]]) #решение (2,1)
b = np.array([15,9])
l = 0


# In[179]:


def Function (x):
    return np.linalg.norm(A.dot(x)-b)**2 + l*np.linalg.norm(x)


# In[180]:


def Grad (x):
    if np.linalg.norm(x) == 0:
        return 2*(A.T).dot(A.dot(x)-b)
    return 2*(A.T).dot(A.dot(x)-b) + l*x/np.linalg.norm(x)


# In[181]:


def MirrorDescent (x0):
    k=1
    h = 0.0001/k
    g = Grad(x0)
    sum =np.sum(x0*np.exp(-h*g))
    if sum != 0:
        x1=x0*np.exp(-h*g)/sum
    else:
        return "SUM = 0"
    #d = np.append(xarr,x1)
    while np.linalg.norm(x1-x0) > 0.000001:
        h = 0.0001/k
        x0=x1
        g = Grad(x0)
        sum = np.sum(x0*np.exp(-h*g))
        if sum != 0:
            x1=x0*np.exp(-h*g)/sum
        else:
            return "SUM = 0"
        k += 1
        #d = np.append(d,x1)
    return x1


# In[182]:


def makeData ():
    x = np.arange (-10, 10, 0.1)
    y = np.arange (-10, 10, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.ndarray(shape=(200,200))
    for i in range(0,199):
        for j in range(0,199):
            zgrid[i][j] = Function(np.array([x[i],y[j]]))
    return xgrid, ygrid, zgrid


# In[183]:


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


# In[184]:


def Grad_descent(x0):
    start_time = time.time()
    xarr=np.array([])
    h=0.0001
    x1=x0-Grad(x0)*h
    d = np.append(xarr,x1)
    while (np.linalg.norm(x1-x0) > 0.000001):
        x0=x1
        x1= x0-Grad(x0)*h
        d = np.append(d,x1)
    #Draw(d)
    print("Time - ", time.time() - start_time)
    return x1


# In[185]:


def Delayed_Grad_descent(x0):
    start_time = time.time()
    h=0.0001
    arrx = []
    list.append(arrx, x0)
    g0 = Grad(x0)
    x0 = x0 - h*g0
    g1 = Grad(x0)
    list.append(arrx, x0)
    x0 = x0 - h*g0
    list.append(arrx, x0)
    #d = np.append(xarr,x1)
    while (np.linalg.norm(arrx[-1]-arrx[-2]) > 0.000001):
        list.append(arrx, (arrx[-1]-h*g1))
        g1 = g0
        g0 = Grad(arrx[-1])
        #d = np.append(d,x1)
    #Draw(d)
    print("Time - ", time.time() - start_time)
    return arrx[-1]


# In[201]:


def Dual(x0, z):
    start_time = time.time()
    x1 = x0 + 1
    xarr=np.array([])
    k = 1
    d = np.append(xarr,x0)
    while (np.linalg.norm(x1-x0) > 0.00001):
        q = 1/k
        g = Grad(x0)
        x1 = x0
        z = z + g
        x0 = -q*z
        if np.linalg.norm(x0) < 100000: d = np.append(d,x0)
        #print("X: ", x0)
        k += 1
    #Draw(d)
    #print(d)
    print("Time - ", time.time() - start_time)
    return x0
        


# In[187]:


print(Grad_descent(np.array([3,2])))


# In[188]:


print(MirrorDescent(np.array([3,2])))


# In[192]:


print(Delayed_Grad_descent(np.array([3,2])))


# In[202]:


print(Dual(np.array([3,2]), np.array([1,2])))


# In[ ]:




