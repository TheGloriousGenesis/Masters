# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:41:38 2018

@author: Ana
"""

import numpy as np
from matplotlib import pyplot as py
from scipy.stats import invgamma

T = 5000
N = 10000
mean = np.zeros(N)
stdv = np.zeros(N)
mu_mock = -1
sigma_eta_mock = np.sqrt(0.05)
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
    
    A = 0.5*sum(((ht-mu)**2))
    y = invgamma.rvs(T/2, scale = A)
    #print(y,'var')
    return y
    
def mu_func(ht,sigma_eta):   
    C = sum(ht)
    B = T
    y = np.random.normal(C/B,sigma_eta/np.sqrt(B))
    return y

def ht_func(ht, mu, sigma_eta):
    
    ht_new = np.copy(ht)
    proposal = np.random.normal(mu,sigma_eta,T)
    
    ht_cur = (-0.5)*(proposal + (y_t**2)/np.exp(proposal))
    ht_prev = (-0.5)*(ht + (y_t**2)/np.exp(ht))
    alpha = np.exp((ht_cur - ht_prev))
    y = np.random.uniform(0,1)
    ind = np.where(((alpha>=1) | ((alpha<=1) & (y<=alpha))))
    ht_new[ind] = proposal[ind]
    return ht_new
#--------------------
#         LIKELIHOOD FUNCTION
#--------------------
#def likelihood_func(ht,mu,var):
#    f_et_ot = -0.5*((ht) + ((y_t**2)/np.exp(ht)))
#    f_ht_theta = -0.5*(np.log(var) + ((ht - mu)**2/var))
#    y = (f_et_ot + f_ht_theta)
#    return y, f_et_ot, f_ht_theta
def likelihood_func(ht,mu,var):  
    y= -1*(ht/2) + ((y_t**2/2)*np.exp(-1*ht)) +\
        (-(T/2)-1)*np.log(var) - ((ht - mu)**2/(2*var))
    return y
#--------------------
#         GlOBAL ACCEPT REJECT STEP
#--------------------  
previous = likelihood_func(ht_guess, mu_guess,(sigma_eta_guess**2))

count = 0
for i in range(N):
    new_var = var_eta_func(ht_guess, mu_guess)
    new_mu = mu_func(ht_guess, sigma_eta_guess)
    new_ht = ht_func(ht_guess, mu_guess, sigma_eta_guess)
    current = likelihood_func(new_ht, new_mu, new_var)
    mean[i] = new_mu
    stdv[i] = new_var
    heo = np.sum((current - previous))
   # print(heo)
#    u  = sigma_eta_guess**2
    correction = np.exp(heo) 
    #G1 = (current - previous)  
    #G2 = np.log(new_var**-1) - np.log((sigma_eta_guess**2)**-1)
    #correction = np.exp(G1)*((sigma_eta_guess**2)/(new_var**(-1)))
    #*np.exp(G2)
    #print(G2)
    alpha = min(1, correction)
    assert((np.isnan(correction)==False))
    assert((correction<=1) and (correction>=0))
    
    if ((np.random.uniform(0,1)<= alpha)):
        previous = current
        ht_guess = new_ht 
        mu_guess = new_mu 
        sigma_eta_guess = np.sqrt(new_var)
        count+= 1
    else:
        previous = previous
        ht_guess = ht_guess
        mu_guess = mu_guess
        sigma_eta_guess = sigma_eta_guess
##--------------------
##         PLOT GRAPHS
##--------------------  
#t = np.linspace(0,N,N)  
#py.figure(figsize=(20,10))
##py.xlim(t[1*N/5],N)
#py.plot(t,mean)

#print((plot_ht(t)))
py.show()


 