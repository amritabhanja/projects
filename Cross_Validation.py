
# coding: utf-8

# In[4]:

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')

def load_data():
    # load data
    data = matrix(genfromtxt('Desktop/wavy_data.csv', delimiter=','))
    x = asarray(data[:,0])
    y = asarray(data[:,1])
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=10)
    #print(X_train)
    return (x,y,X_train,X_test,y_train,y_test)

def plot_model(w,D,ax):
    s = asarray(linspace(0,1,100))
    s.shape = (size(s),1)
    f = []
    for m in range(1,D+1):
        f.append(w[2*m-1]*cos(2*pi*m*s))
        f.append(w[2*m]*sin(2*pi*m*s))
    f = asarray(f)
    f = sum(f,axis = 0) + w[0]
    ax.plot(s,f,'-r', linewidth = 2)
    
def fourier_features(x,D):
    X=[]
    for i in range(1,D+1):
        X.extend((cos(2*pi*x*i),sin(2*pi*x*i)))
    F=asarray(X)
    F.shape=(len(X),20)
    F=F.T
    temp = shape(F)
    temp = ones((temp[0],1))
    F = concatenate((temp,F),1)
    F=F.T
    return(F)

def fourier_features1(x,D):
    X=[]
    for i in range(1,D+1):
        X.extend((cos(2*pi*x*i),sin(2*pi*x*i)))
    F=asarray(X)
    F.shape=(len(X),10)
    F=F.T
    temp = shape(F)
    temp = ones((temp[0],1))
    F = concatenate((temp,F),1)
    F=F.T
    return(F)
def fourier_features_data(x,D):
    X=[]
    for i in range(1,D+1):
        X.extend((cos(2*pi*x*i),sin(2*pi*x*i)))
    F=asarray(X)
    F.shape=(len(X),30)
    F=F.T
    temp = shape(F)
    temp = ones((temp[0],1))
    F = concatenate((temp,F),1)
    F=F.T
    return(F)

def plot_mses(mses1,mses2,deg):
    f,ax = plt.subplots(1,facecolor = 'white')
    ax.plot(deg,mses1)
    ax.xaxis.set_ticks(deg)
    ax.plot(deg,mses2)
    ax.xaxis.set_ticks(deg)
    ax.set_ylim(0,1)
    ax.set_xlabel('$D$',fontsize=20,labelpad = 10)
    ax.set_ylabel('$MSE $',fontsize=20,rotation = 0,labelpad = 20)



def main():
    x,y,X_train,X_test,y_train,y_test = load_data()
    deg = array([1,2,3,4,5,6,7,8])
    f,axs = plt.subplots(2,4,facecolor = 'white')
    for i in range(0,4):
        axs[0,i].scatter(X_train,y_train,color = 'b')
        axs[0,i].scatter(X_test,y_test,color = 'y')
        s = 'D = ' + str(deg[i])
        axs[0,i].set_title(s,fontsize=15)
        
        axs[1,i].scatter(X_train,y_train,color = 'b')
        axs[1,i].scatter(X_test,y_test,color = 'y')
        s = 'D = ' + str(deg[i + 3])
        axs[1,i].set_title(s,fontsize=15)
        
        #axs[2,i].scatter(X_train,y_train,color = 'b')
        #axs[2,i].scatter(X_test,y_test,color = 'y')
        #s = 'D = ' + str(deg[i + 5])
        #axs[2,i].set_title(s,fontsize=15)
    mses_train = []
    mses_test = []
    for D in range(0,8):
        F_train = fourier_features(X_train,deg[D])
        w_train = dot(linalg.pinv(dot(F_train,F_train.T)),dot(F_train,y_train))
        new_mse_train = linalg.norm(dot(F_train.T,w_train) - y_train)/size(y_train)
        mses_train.append(new_mse_train)

        F_test = fourier_features1(X_test,deg[D])
        w_test = dot(linalg.pinv(dot(F_test,F_test.T)),dot(F_test,y_test))
        new_mse_test = linalg.norm(dot(F_test.T,w_train) - y_test)/size(y_test)
        mses_test.append(new_mse_test)

        n = 0
        m = D
        if D > 3:
            n = 1
            m = D - 4
        #if D >= 6:
         #   n = 2
        #   m = D - 6
        plot_model(w_train,deg[D],axs[n,m])

    plot_mses(mses_train,mses_test,deg)
    a=mses_test.index(min(mses_test))
    show()
    print(a)
    plt.scatter(x,y, s = 30)
    s = 'D = ' + str(deg[a])
    plt.suptitle(s,fontsize=15)
    #plt.scatter(X_test,y_test, s = 30,color = 'y')
    F = fourier_features_data(x,deg[a])
    w = dot(linalg.pinv(dot(F,F.T)),dot(F,y))
    if a<3:
        n=0
        m=a
    else:
        n=1
        m=a-4
    s = asarray(linspace(0,1,100))
    s.shape = (size(s),1)
    f = []
    for m in range(1,deg[a]+1):
        f.append(w[2*m-1]*cos(2*pi*m*s))
        f.append(w[2*m]*sin(2*pi*m*s))
    f = asarray(f)
    f = sum(f,axis = 0) + w[0]
    plt.plot(s,f,'-r', linewidth = 2)
    
   
    

    show()
main()


# In[ ]:




# In[ ]:



