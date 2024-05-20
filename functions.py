import numpy as np
import scipy as sp
import ot

from numpy import linalg as LA

from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy.sparse import csr_matrix
from scipy.special import softmax
import networkx as nx

from IPython import display
import time

from numpy import matlib
import matplotlib.pyplot as plt

np.random.seed(1)

def mirror(data,   C,  numItermax=100000, stopThr=1e-9, verbose = False):
    
    d= C.flatten()# vectorized cost matrix
    
    m = data.shape[0] # number of measures
    n = data.shape[1] # img_size**
    #define algorithm constants
    dnorm = LA.norm(d, np.inf)
    eta = 1/(4*dnorm*np.sqrt(3*n*np.log(n)))
    alpha = 2*dnorm*eta*n
    beta =  6*dnorm*eta*np.log(n)
    gamma= 3*eta*np.log(n)
    losses = []
    

    #define inputs
    p = np.ones(n) / n
    s = np.copy(p)
    x = np.ones((m, n**2)) / (m*n**2)
    u = np.copy(x)
    y = np.zeros((m, 2*n))
    v = np.copy(y)
    shat = np.zeros_like(p)

    
    #edge-incidence matrix

    A = np.zeros((2*n, n**2))
    i=0
    j=0
    for i in range(n):
        A[i, j:j+n]=1
        j+=n
    t=0
    for i in range(n,2*n):
        for j in range(t, n**2, n):
            A[i,j]=1
        t+=1

    A = csr_matrix(A)
    res = []

    #Algorithm
    err = 1
    cpt=0
    while (err > stopThr and cpt < numItermax):
        cpt += 1
        if cpt % 100 == 0:
            print(cpt)
        
    # for k in range(0, numItermax):

        for i in range (0,m):
            b = np.concatenate((p,data[i]),axis=0)
            v[i] = y[i] + alpha * (A.dot(x[i]) - b) # variable for y
           
            flag = np.abs(v[i]) >1
            v[i, flag] = np.sign(v[i, flag])

            u[i] =  x[i]*np.exp(-gamma * (d + 2*dnorm * A.transpose().dot(y[i]))) / np.sum( x[i]*np.exp(-gamma * (d + 2*dnorm * A.transpose().dot(y[i]))) )  # variable for x
            
        s = p*np.exp(beta*np.sum(y[:,:n],0)) / np.sum( p*np.exp(beta*np.sum(y[:,:n],0)) )  # variable for p
        pold = np.copy(shat)
        shat +=s       
        for i in range (0,m):
            b = np.concatenate((s,data[i]),axis=0)
            y[i] = y[i] + alpha * (A.dot(u[i]) - b) 

            flag = np.abs(y[i]) >1
            y[i, flag] = np.sign(y[i, flag])

            x[i] = x[i]*np.exp(-gamma * (d + 2*dnorm * A.transpose().dot(v[i]))) / np.sum( x[i]*np.exp(-gamma * (d + 2*dnorm * A.transpose().dot(v[i]))) ) 
       
        p = p*np.exp(beta*np.sum(v[:,:n],0)) / np.sum( p*np.exp(beta*np.sum(v[:,:n], 0)) )  
        
        if i % 10 == 1:
            err = np.sum(np.abs(pold - shat/cpt))
            
        phat = shat / cpt
        if verbose and cpt % 1000 == 0:
            print('Pass {} iterations'.format(cpt), flush=True)
            with open('WBimages/mnist/iter{}.npy'.format(cpt), 'wb') as f:
                np.save(f, phat)
        if (cpt)%20000 == 0:
            res.append(phat)

        gaus = data
        Wc = 0
        for i in range(gaus.shape[0]): 
            Wc += ot.emd2(phat ,gaus[0],C)
        Wc = Wc/gaus.shape[0]

        losses.append(Wc)
    #Output   

    return phat, res, losses



def incidence_matrix(n):
    A = np.zeros((2*n, n**2))
    for i in range(n):
        for j in range(n):
            A[i][i*n+j] = 1
            A[n+j][i*n+j] = 1
    return A


def sample_from_cat(p):
    u = np.random.uniform(size=p.shape[0])
    cum_prob = np.cumsum(p, axis=1)
    samples = np.argmax(cum_prob > u[:, np.newaxis], axis=1)
    return samples

def proj(y):
    mask = np.abs(y) > 1
    y[mask] = np.sign(y[mask])
    return y


def my_mirror(data, C,  numItermax=100000, stopThr=1e-9, K=2):

    d= C.flatten()# vectorized cost matrix
    gaus = data
    m = data.shape[0] # number of measures
    n = data.shape[1] # img_size**

    #define algorithm constants
    dnorm = np.longdouble(LA.norm(d, np.inf))
    tau = np.longdouble(1 / (4*dnorm*np.sqrt(3*n*np.log(n))))
    Rx = np.longdouble(np.sqrt(3*m*np.log(n)))
    Ry =  np.longdouble(np.sqrt(m*n))
    # Ry = np.longdouble(np.sqrt(2*m*n*(2*np.log(n)+np.exp(-1))))
    prob = 1.0 / K
    alpha = 1 - prob
    res = []

    #define inputs
    x, y, p = np.ones((m, n**2), dtype = np.longdouble) / (m*n**2), np.zeros((m, 2*n), dtype = np.longdouble), np.ones(n, dtype = np.longdouble) / n # z
    x_hat, p_hat = np.log(x), np.log(p)

    u, v, s = np.copy(x), np.copy(y), np.copy(p) # w
    u_next, v_next, s_next = np.copy(x), np.copy(y), np.copy(p) # w_next
    u_, v_, s_ = np.copy(x), np.copy(y), np.copy(p) #term_w

    u_hat, s_hat =  np.copy(x_hat), np.copy(p_hat) # w_hat
    u_hat_next, s_hat_next =  np.copy(x_hat), np.copy(p_hat) # w_hat_next
    u_hat_, s_hat_ =  np.copy(x_hat), np.copy(p_hat) #term_ w_hat

    res_u = np.zeros(x.shape)
    res_v = np.zeros(y.shape)
    res_s = np.zeros(p.shape)
    losses = []
    #edge-incidence matrix
    A = incidence_matrix(n)
    A = csr_matrix(A, dtype = np.longdouble)
    #Algorithm
    cpt=0
    for iter in range(1, numItermax + 1):
        if iter % 100 == 0: print(iter)
        for k in range(1,K+1):

            # first step 
            for i in range (m):
                b = np.concatenate((s, data[i]),axis=0)
                Fu = (d + 2*dnorm * A.transpose().dot(v[i])) / m
                Fv = 2*dnorm * (A.dot(u[i]) - b) / m

                u_hat_[i] =  alpha * x_hat[i] + (1-alpha) * u_hat[i] - tau * Rx**2 * Fu
                u_[i] = softmax(u_hat_[i])

                v_[i] = alpha * y[i] + (1-alpha) * v[i] + tau * Ry**2 * Fv
                # v_[i] = y[i]**alpha * v[i]**(1-alpha)*np.exp(tau * Ry**2 * Fv) - 1
                v_[i] = proj(v_[i])

            Fs = 2*dnorm * np.sum(v[:,:n],0) / m    
            s_hat_ = alpha * p_hat + (1-alpha) * s_hat + tau * Rx**2 * Fs
            s_ = softmax(s_hat_)
            res_v, res_u, res_s = res_v + v_, res_u + u_, res_s + s_
            
            # # sampling for u naive
            # oracle_u = np.zeros(u.shape)
            # for i in range(m):
            #     ind = np.random.choice(np.arange(n**2))
            #     oracle_u[i][ind] = (d + 2*dnorm * A.transpose().dot(v_[i] - v[i]))[ind] / m

            # sampling for u umom
            oracle_u = csr_matrix(u.shape, dtype=np.longdouble)
            for i in range(m):
                diff = v_[i] - v[i]
                l1_diff = np.abs(diff)
                norm_l1_diff = np.sum(l1_diff)
                ind = np.random.choice(np.arange(2*n), p=l1_diff / norm_l1_diff)
                oracle_u[i] = A[ind] * np.sign(l1_diff[ind]) * norm_l1_diff * 2 * dnorm / m

            # sampling for v naive
            oracle_v = np.zeros(v.shape)
            for i in range(m):
                ind = np.random.choice(np.arange(2*n))
                oracle_v[i][ind] =  2*dnorm * A.dot(u_[i] -u[i])[ind] / m

            # # sampling for v
            # oracle_v = csr_matrix(v.shape, dtype=np.longdouble)
            # for i in range(m):
            #     diff = u_[i] - u[i] 
            #     ind = np.random.choice(np.arange(n**2))
            #     oracle_v[i] = A.transpose()[ind] * diff[ind] * (2 * dnorm / m) 

            # oracle_u = 0
            # oracle_v = 0
            oracle_s = 0

            # second step 
            x_hat = np.array(u_hat_ - tau * Rx**2 * oracle_u)
            x = softmax(u_hat)
            y = proj(np.array(v_ + tau * Ry**2 * oracle_v))
            p_hat = np.array(s_hat_ + tau * Rx**2 * oracle_s)
            p = softmax(s_hat)

            #recalculation
            u_next = (x + (k-1) * u_next) / k
            v_next = (y + (k-1) * v_next) / k
            s_next = (p + (k-1) * s_next) / k
            u_hat_next = (x_hat + (k-1) * u_hat_next) / k
            s_hat_next = (p_hat + (k-1) * s_hat_next) / k


        Wc = 0
        for i in range(gaus.shape[0]): 
            Wc += ot.emd2(res_s / (iter*K) ,gaus[0],C)
        Wc = Wc/gaus.shape[0]
        losses.append(Wc)
        u, v, s = u_next, v_next, s_next
        u_hat, s_hat = u_hat_next, s_hat_next

        if iter%20000 == 0:
            res.append([res_v / (numItermax*K), res_u / (numItermax*K), res_s / (numItermax*K)])


    return res_v / (numItermax*K), res_u / (numItermax*K), res_s / (numItermax*K), res, losses
