
# coding: utf-8

# In[10]:

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')

def load_data():
    # load data
    data = matrix(genfromtxt('Desktop/galileo_ramp_data.csv', delimiter=','))
    x = asarray(data[:,0])
    y = asarray(data[:,1])
    return (x,y)

def plot_model(w,D,ax):
    s = asarray(linspace(0,8,100))
    s.shape = (size(s),1)
    f = []
    for k in range(1,D+1):
        f.append(w[k]*s**k)
    f = asarray(f)
    f = sum(f,axis = 0) + w[0]
    ax.plot(s,f,'-r', linewidth = 2)
    
def poly_features(x,D):
    X=[]
    for i in range(1,D+1):
        X.append(x**i)
    F=asarray(X)
    F.shape=(D,5)
    F=F.T
    temp = shape(F)
    temp = ones((temp[0],1))
    F = concatenate((temp,F),1)
    F=F.T
    return(F)

def poly_features1(x,D):
    X=[]
    for i in range(1,D+1):
        X.append(x**i)
    F=asarray(X)
    F.shape=(D,1)
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
    ax.set_ylim(0,0.05)
    ax.set_xlabel('$D$',fontsize=20,labelpad = 10)
    ax.set_ylabel('$MSE $',fontsize=20,rotation = 0,labelpad = 20)

def main():
    x,y = load_data()
    mses_train_final = [0,0,0,0,0,0]
    mses_test_final = [0,0,0,0,0,0]
    for k in range(6):
        X_test,y_test = x[k],y[k]
        c = np.delete(x,k)
        d = np.delete(y,k)
        X_train,y_train = c,d
        deg = array([1,2,3,4,5,6])
        f,axs = plt.subplots(2,3,facecolor = 'white')
        for i in range(0,3):
            axs[0,i].scatter(X_train,y_train,color = 'b')
            axs[0,i].scatter(X_test,y_test,color = 'y')
            axs[0,i].set_ylim(-.2,1.5)
            s = 'D = ' + str(deg[i])
            axs[0,i].set_title(s,fontsize=15)
        
            axs[1,i].scatter(X_train,y_train,color = 'b')
            axs[1,i].scatter(X_test,y_test,color = 'y')
            axs[1,i].set_ylim(-.2,1.5)
            s = 'D = ' + str(deg[i + 3])
            axs[1,i].set_title(s,fontsize=15)
        
        mses_train = []
        mses_test = []
        for D in range(0,6):
            F_train = poly_features(X_train,deg[D])
            w_train = dot(linalg.pinv(dot(F_train,F_train.T)),dot(F_train,y_train))
            new_mse_train = linalg.norm(dot(F_train.T,w_train) - y_train)/size(y_train)
            mses_train.append(new_mse_train)

            F_test = poly_features1(X_test,deg[D])
            w_test = dot(linalg.pinv(dot(F_test,F_test.T)),dot(F_test,y_test))
            new_mse_test = linalg.norm(dot(F_test.T,w_train) - y_test)/size(y_test)
            mses_test.append(new_mse_test)
            n = 0
            m = D
            if D > 2:
                n = 1
                m = D - 3
            plot_model(w_train,deg[D],axs[n,m])
        mses_test_final = (list(np.array(mses_test_final) + np.array(mses_test)))
        myInt = 6
        mses_test_final[:] = [x / myInt for x in mses_test_final]
        mses_train_final = (list(np.array(mses_train_final) + np.array(mses_train)))
        mses_train_final[:] = [x / myInt for x in mses_train_final]

    plot_mses(mses_train_final,mses_test_final,deg)
    

    a=mses_test.index(min(mses_test))
    show()
    print(a)
    plt.scatter(X_train,y_train, s = 30,color = 'b')
    s = 'D = ' + str(deg[a])
    plt.suptitle(s,fontsize=15)
    plt.scatter(X_test,y_test, s = 30,color = 'y')
    F_train = poly_features(X_train,deg[a])
    w_train = dot(linalg.pinv(dot(F_train,F_train.T)),dot(F_train,y_train))
    if a<2:
        n=0
        m=a
    else:
        n=1
        m=a-3
    s = asarray(linspace(0,8,100))
    s.shape = (size(s),1)
    f = []
    for k in range(1,deg[a]+1):
        f.append(w_train[k]*s**k)
    f = asarray(f)
    f = sum(f,axis = 0) + w_train[0]
    plt.plot(s,f,'-r', linewidth = 2)
    show()
main()


# In[ ]:



