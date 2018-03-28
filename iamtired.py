# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 02:55:05 2018

@author: Ana
"""

import numpy as np
from matplotlib import pyplot as py
from scipy.stats import invgamma

T = 4000
N = 4000 # Made smaller because my pc is slow.
mean = np.zeros(N)
stdv = np.zeros(N)
mu_mock = -1
sigma_eta_mock = np.sqrt(0.05) # Reduced by factor of 10
#--------------------
#         ARTIFICAL TIME SERIES
#--------------------
#exponent_ht = mu_mock + np.random.normal(0, sigma_eta_mock, T)
#sigma_t = np.exp(exponent_ht/2)
#epsilon_t = np.random.normal(0, 1, T)
#y_t = sigma_t * epsilon_t

def time_series(T, mu, sigma_eta):

    """

    Generates times series of length T when function is called.

    

    Notes: Seems to work correctly.

    """

    y = []

    for t in range(T):

        epsilon = np.random.normal(0, 1)

        eta = np.random.normal(0, sigma_eta)

        h_t = mu + eta

        y.append(np.exp(h_t/2)*epsilon)

    return y

y_t = np.array(time_series(T, mu_mock, sigma_eta_mock))

#--------------------
#         SAMPLING OF THETA - POSTERIOR DISTRIBUTION
#--------------------
#print(sigma_eta_guess**2, 'b4')
def var_eta_func(ht,mu):
    A = 0.5*(np.sum((ht - mu)**2))
    y = invgamma.rvs(T/2, scale = A)
    assert((np.isnan(y)==False))
    return y
    
def mu_func(ht,sigma_eta):   
    C = np.sum(ht)
    B = T
    y = np.random.normal(C/B,sigma_eta*np.sqrt(1/B))
    assert((np.isnan(y)==False))
    return y
    
#print(sum(ht_guess), 'ht b4')

def ht_func(ht, mu, sigma_eta): 
    ht_new = ht
    proposal = np.random.normal(mu, sigma_eta , T)
    ht_cur = (-0.5)*(proposal + (y_t**2)/np.exp(proposal))
    ht_prev = (-0.5)*(ht + (y_t**2)/np.exp(ht))
    alpha = np.exp(ht_cur-ht_prev)
    y = np.random.uniform(0,1,T)
    ind = np.where((y<=alpha))
    ht_new[ind] = proposal[ind]
    #toms was ht_new[ind[0]] = proposal[ind[0]]
    assert((np.size(ht_cur)==np.size(ht_prev)==np.size(alpha)== T))
    assert((np.isnan(alpha.any())==False))
    return ht_new
#--------------------
#         EXPONENT OF LIKELIHOOD FUNCTION AND PRIOR
#--------------------
def likelihood_prior_exp(ht,mu,sig,y_t):  
    y= np.sum((-ht/2) - ((y_t**2/2)*np.exp(-ht)) \
         - ((ht - mu)**2/(2*(sig**2)))) - (T/2+1)*np.log((sig**2))
#    y = np.sum((-ht/2) - (y_t**2)/(2*np.exp(ht)) \
#         - ((ht - mu)**2)/(2*sig**2)) + T*np.log((sig**2)**(-1/2))
         
         
         
         
    #print(sum(- ht/2), sum(np.log((2*(np.exp(ht/2))**2)**(-1/2))), 'list1') #np.log((2*(np.exp(ht/2))**2)**(-1/2)))), 'list1')
    #print(sum(- ((ht - mu)**2)/(2*sig**2) + np.log((2*sig**2)**(-1/2))), 'list2')
         #- (T/2+1)*np.log((sig**2))
#    sigma_t = np.exp(ht/2)
#    print(sum(sigma_t), 'sig_t')
#    list1 = -(y_t**2)/(2*sigma_t**2) + np.log((2*np.pi*sigma_t**2)**(-1/2))
#    print(sum(list1), ' list1')
#    list2 = -((ht-mu)**2)/(2*sig**2) + np.log((2*np.pi*sig**2)**(-1/2))
#    print(sum(list2), 'list2')
#    likelihood = (np.sum(list1 + list2))
#    y = likelihood #* (1/sig**2)

#    y = (-ht/2) - (y_t**2)/(2*np.exp(ht)) \
#         - ((ht - mu)**2)/(2*(sig**2))
#    y = np.sum(y)
#    y = y - (T/2+1)*np.log((sig**2))

#    print('inside likelyhood',np.sum(-(ht/2) - ((y_t**2/2)*np.exp(-ht)) 
#         - ((ht - mu)**2/(2*(sig**2)))), -(T/2+1)*np.log((sig**2)),y)
    return y
#--------------------
#         INITIAL CONDITIONS
#--------------------
mu_guess =  1
sigma_eta_guess =  np.sqrt(0.5)
ht_guess = np.full(T,0.6)
#mu_guess =  1
#sigma_eta_guess = np.sqrt(0.5) #can't be zero or else f_ht_theta dont work
#ht_guess = np.full(T,0)#np.array([np.random.normal() for i in range(T)])#np.array([-.6]*T)#[np.random.normal() for i in range(T)]) 

#--------------------
#         GlOBAL ACCEPT REJECT STEP
#--------------------  
previous = likelihood_prior_exp(ht_guess, mu_guess,sigma_eta_guess,y_t)
count = 0
#print('p1')
for i in range(N):
    #print(mu_guess,sigma_eta_guess,ht_guess[100], 'old')
    new_sig = np.sqrt(var_eta_func(ht_guess, mu_guess))#np.sqrt(var_eta_func(ht_guess, mu_guess))
# order matters so this order is good    
#technically doesnt matter if ht_guess changes because each step a new ht is 
    #calculated whether previous ht accpeted or not from the previous ht
    new_mu = mu_func(ht_guess, sigma_eta_guess)
    new_ht = ht_func(ht_guess, mu_guess, sigma_eta_guess)
   # print(new_sig, 'new_sig')
    current = likelihood_prior_exp(new_ht, new_mu, new_sig,y_t) 
    previous = likelihood_prior_exp(ht_guess, mu_guess,sigma_eta_guess,y_t)    
#    print(current,'current')
#    print(previous,'previous')
    #break
    exponent = current - previous
    correction = np.exp(exponent) 
    #print(correction)
    #break
    
    alpha = min(1, correction)
    if ((np.random.uniform(0,1) < alpha)):
#        previous = current 
        sigma_eta_guess = new_sig
        mu_guess = new_mu
        ht_guess = np.array(new_ht)
        count+= 1
        mean[i] = mu_guess
        stdv[i] = sigma_eta_guess
        #break
    else:
        ht_guess = ht_guess
        mean[i] = mu_guess
        stdv[i] = sigma_eta_guess

    #print(i)
    #break
##--------------------
##         PLOT GRAPHS
##--------------------  
t = np.linspace(0,N,N)  
py.figure(figsize=(20,10))
#py.xlim(t[1*N/5],N)
py.plot(t,stdv, label = ' stdv')
#py.figure()
py.plot(t,mean, label= 'mean')
py.axhline(y=np.mean(mean[N/100:]), color='r', linestyle='-', label = 'avg mean = %s'%np.mean(mean[N/10:]))
py.axhline(y=np.mean(stdv[N/100:]), color='k', linestyle='-', label = 'avg stdv = %s'%np.mean(stdv[N/10:]))
py.legend()
py.show()
print(count)