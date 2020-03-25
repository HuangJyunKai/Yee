# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:57:52 2019

@author: 07067
"""

import numpy as np



def polyfit(x,y,n):
    if n>1:
        in_x=x.copy()
        for s_n in range(n,1,-1):
            x=np.c_[in_x**s_n,x]
        
    Ndata=len(y)
    x=np.c_[np.ones([Ndata,1]),x]
    # Beta_hat=pinv(x'*x)*x'*y; % Least square approach
    xt=x.transpose()
    C=np.dot(xt,x)
    p2=np.dot(xt,y)
    Beta_hat=np.dot(np.linalg.pinv(C),p2)
    return Beta_hat

def polyval(Beta_hat,x,n):
    if n>1:
        in_x=x.copy()
        for s_n in range(n,1,-1):
            x=np.c_[in_x**s_n,x]
    if np.size(x)==1:
        Ndata=1
    else:
        Ndata=len(x)
    x=np.c_[np.ones([Ndata,1]),x]
    y_hat=np.dot(x,Beta_hat)
    return y_hat

def ransac(data,y,order=1,Iter=200,sigma=1, Nums=2):
    # （Random sample consensus，RANSAC）
    #Ransac
    res = np.zeros((Iter,order+2))
    for i in range(Iter):
        idx= np.random.permutation(data.shape[0])[0:Nums]
        if data.ndim==1:
            sample = data[idx]
        else:
            sample = data[idx,:]
        sample_y=y[idx]
        Beta_hat=polyfit(sample,sample_y,order)
        res[i,:order+1]=Beta_hat
        res[i,order+1]=np.where(abs((polyval(Beta_hat,data,order)-y))<sigma)[0].size
    
    pos=np.argmax(res[:,order+1])
    Beta_hat = res[pos,0:order+1];
    y_hat_ransac = polyval(Beta_hat,data,order)
    return Beta_hat, y_hat_ransac,res




#        
#order=1
#Beta_hat=polyfit(data,y,order)
#y_hat=polyval(Beta_hat,data,order)
#Beta_hat2, y_hat_ransac,res=ransac(data,y,order=order,Iter=100, sigma=1, Nums=5)
#       
#plt.plot(data,y,'*')
#plt.plot(data,y_hat,'r-')
#plt.plot(data,y_hat_ransac,'g-')
