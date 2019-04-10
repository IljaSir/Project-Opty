#!/usr/bin/env python
# coding: utf-8

# In[103]:


from math import *
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt 


# In[172]:


A = np.array([[2,11],[10,-11]]) #решение (2,1)
b = np.array([15,9])
l = 0
xarr=np.array([])


# In[105]:


def Function (x):
    return np.linalg.norm(A.dot(x)-b)**2 + l*np.linalg.norm(x)


# In[106]:


def Grad (x):
    if np.linalg.norm(x) == 0:
        return 2*(A.T).dot(A.dot(x)-b)
    return 2*(A.T).dot(A.dot(x)-b) + l*x/np.linalg.norm(x)


# In[169]:


def MirrorDescent (x0):
    h = 0.0001
    g = Grad(x0)
    sum =np.sum(x0*np.exp(-h*g))
    if sum != 0:
        x1=x0*np.exp(-h*g)/sum
    else:
        return "SUM = 0"
    #d = np.append(xarr,x1)
    while np.linalg.norm(x1-x0) > 0.000001:
        x0=x1
        g = Grad(x0)
        sum = np.sum(x0*np.exp(-h*g))
        if sum != 0:
            x1=x0*np.exp(-h*g)/sum
        else:
            return "SUM = 0"
        #d = np.append(d,x1)
    return x1


# In[125]:


def makeData ():
    x = np.arange (-10, 10, 0.1)
    y = np.arange (-10, 10, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.ndarray(shape=(200,200))
    for i in range(0,199):
        for j in range(0,199):
            zgrid[i][j] = Function(np.array([x[i],y[j]]))
    return xgrid, ygrid, zgrid


# In[126]:


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


# In[127]:


def Grad_descent(x0):
    h=0.0001
    x1=x0-Grad(x0)*h
    d = np.append(xarr,x1)
    while (np.linalg.norm(x1-x0) > 0.000001):
        x0=x1
        x1= x0-Grad(x0)*h
        d = np.append(d,x1)
    #Draw(d)
    return x1


# In[168]:


print(Grad_descent(np.array([3,2])))


# In[173]:


print(MirrorDescent(np.array([2,1])))


# In[143]:


res = np.array([1,2])*np.exp(np.array([1,1]))
print(res)


# In[ ]:




