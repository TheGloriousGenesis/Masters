# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:41:38 2018

@author: Ana
Extra stuff:

    #ht_test = np.full(T,0.1)
    #mu_test = -1
    #sigma_test = np.sqrt(0.2)
    #yt_test = np.full(T,0.3)
    #
    #test = likelihood_func(ht_test,mu_test,sigma_test**2,yt_test)
    #print(test)

    #def likelihood_func(ht,mu,var):
    ##    f_et_ot = -0.5*((ht) + ((y_t**2)/np.exp(ht)))
    ##    f_ht_theta = -0.5*(np.log(var) + ((ht - mu)**2/var))
    ##    y = (f_et_ot + f_ht_theta)   
    #    i = np.sum(-0.5*ht - (y_t**2/(2*np.exp(ht))) - \
    #        0.5*np.log(var) - (ht-mu)**2/(2*var))
    #    y = i - np.log(var)   
    #    return y

    #    y = np.sum(-0.5*ht - (y_t**2/(2*np.exp(ht))) \
    #         - (ht-mu)**2/(2*var)) - ((T/2)-1)*np.log(var)
    #assert((np.isnan(y)==False))

"""

import numpy as np
from matplotlib import pyplot as py
from scipy.stats import invgamma

T = 5000
N = 30000
mean = np.zeros(N)
stdv = np.zeros(N)
mu_mock = -1
sigma_eta_mock = np.sqrt(0.5)
#--------------------
#         ARTIFICAL TIME SERIES
#--------------------
exponent_ht = mu_mock + np.random.normal(0, sigma_eta_mock, T)
sigma_t = np.exp(exponent_ht/2)
epsilon_t = np.random.normal(0, 1, T)
y_t = sigma_t * epsilon_t

#--------------------
#         INITIAL CONDITIONS
#--------------------
mu_guess =  0
sigma_eta_guess = 1 #can't be zero or else f_ht_theta dont work
ht_guess = np.full(T,0.6)
#--------------------
#         SAMPLING OF THETA - POSTERIOR DISTRIBUTION
#--------------------
def var_eta_func(ht,mu):
    A = 0.5*np.sum(((ht-mu)**2))
    y = invgamma.rvs((T/2), scale = A)
    assert((np.isnan(y)==False))
    return y
    
def mu_func(ht,sigma_eta):   
    C = np.sum(ht)
    B = T
    y = np.random.normal(C/B,sigma_eta/np.sqrt(B))
    assert((np.isnan(y)==False))
    return y

def ht_func(ht, mu, sigma_eta):    
    ht_new = np.copy(ht)
    proposal = np.random.normal(mu,sigma_eta,T)  
    ht_cur = (-0.5)*(proposal + (y_t**2)/np.exp(proposal))
    ht_prev = (-0.5)*(ht + (y_t**2)/np.exp(ht))
    alpha = np.exp((ht_cur - ht_prev))
    y = np.random.uniform(0,1,T)
    ind = np.where(((alpha>=1) | ((alpha<=1) & (y<=alpha))))
    ht_new[ind] = proposal[ind]
    assert((np.size(ht_cur)==np.size(ht_prev)==np.size(alpha)== T))
    assert((np.isnan(alpha.any())==False))
    return ht_new
#--------------------
#         EXPONENT OF LIKELIHOOD FUNCTION AND PRIOR
#--------------------
def likelihood_prior_exp(ht,mu,var,y_t):  
    y= np.sum(-(ht/2) - ((y_t**2/2)*np.exp(-ht)) +\
         - ((ht - mu)**2/(2*var))) +(-(T/2)-1)*np.log(var)
    return y
#--------------------
#         GlOBAL ACCEPT REJECT STEP
#--------------------  
previous = likelihood_prior_exp(ht_guess, mu_guess,(sigma_eta_guess**2),y_t)

count = 0
for i in range(N):
    new_var = var_eta_func(ht_guess, mu_guess)
    new_mu = mu_func(ht_guess, sigma_eta_guess)
    new_ht = ht_func(ht_guess, mu_guess, sigma_eta_guess)
    current = likelihood_prior_exp(new_ht, new_mu, new_var,y_t)

    exponent = current - previous
    correction = np.exp(exponent) 
    alpha = min(1, correction)
    
    assert((np.isnan(correction)==False))
    
    if ((np.random.uniform(0,1)<= alpha)):
        previous = current
        ht_guess = new_ht 
        mu_guess = new_mu 
        sigma_eta_guess = np.sqrt(new_var)
        count+= 1
        mean[i] = new_mu
        stdv[i] = new_var
    else:
        previous = previous
        ht_guess = ht_guess
        mu_guess = mu_guess
        sigma_eta_guess = sigma_eta_guess
        mean[i] = mu_guess
        stdv[i] = sigma_eta_guess
##--------------------
##         PLOT GRAPHS
##--------------------  
t = np.linspace(0,N,N)  
py.figure(figsize=(20,10))
#py.xlim(t[1*N/5],N)
py.plot(t,stdv)
py.plot(t,mean)
py.show()


 