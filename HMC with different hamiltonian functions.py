# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:54:27 2018

@author: Ana
"""
import numpy as np
from matplotlib import pyplot as py

def time_series_2(mu,sigma_eta,T): 
    ln_sd2 = np.zeros(T)
    e = np.random.normal(0,1,T)
    nu = np.random.normal(0,sigma_eta,T)
    ln_sd2[0] = 0    
    for n in range(1,T):
        ln_sd2[n] = mu + nu[n]
    y=np.exp((ln_sd2)/2)*e
    return y, np.mean(ln_sd2),np.var(ln_sd2) 

y_t = time_series_2(-1,np.sqrt(0.05),300)[0]

#def HybridMonteCarlo(T, mu, sigma):
#    n = int(T*0.1)
#    delta = 1/n
#    def Hamilitionian(xj, p): 
#        kinetic_E =  0.5*(p**2)
#        potential_E = 0.5*(xj**2)
#        return kinetic_E + potential_E
#    def dH_dx(xi, mu):
#        y = xi
#        return y
#    x = np.zeros(T)
#    p0 = np.random.normal(mu,sigma)
#    i = 0
#    while i+1 < T:
#        i = i+1
#        p0 = np.random.normal(mu,sigma)
#        pStar = p0
#        acc = 0
#        #-----------for the first half step        
#        pStar = p0 - (delta/2)*dH_dx(x[i-1], mu)
#        xStar = x[i-1] + (delta)*pStar
#        for u in range(1,T):
#            #-----------for the full steps 
#            pStar = pStar - delta*dH_dx(xStar, mu)
#            xStar = xStar + (delta)*pStar
#            py.plot(xStar,pStar,'ko', linewidth = 20)
#            py.xlim(-6, 6) 
#            py.ylim(-6, 6)
#            py.xlabel('x') 
#            py.ylabel('p')
#            py.title('Phase Space')
#            #-----------for the last half step
#            
#        pStar = pStar - (delta/2)*dH_dx(xStar, mu)
#        current = -Hamilitionian(xStar, pStar)
#        previous = -Hamilitionian(x[i], p0)
#       # print(type(previous))
#        alpha2 = min(1, np.exp(current - previous))
#        if (np.random.uniform(0,1) < alpha2):
#            x[i] = xStar
#        else:
#            x[i] = x[i] # ht[i]
#        py.show()
#    return x
#    
#y = HybridMonteCarlo(300, 1, 0.5)

#def HybridMonteCarlo(T, mu, sigma, yt, var_eta):
#    n = int(T*0.1)
#    delta = 1/n
#    
#    #lamda = int(n*delta) #L
#    def Hamilitionian(xj, p, yt, mu, var_eta, jj): 
#        kinetic_E =  0.5*(p**2)
#        one = 0.5*(xj + ((yt[jj])**2)*np.exp(-xj))
#        two = ((xj - mu)**2)/(2*var_eta)
#        potential_E = one + two
#        return kinetic_E + potential_E
#    def dH_dx(yt, xi, mu, var_eta, ii):
#        y = 0.5* (1 - ((yt[ii])**2)*xi*np.exp(-xi)) + \
#            (xi - mu)/var_eta
#        return y
#
#    x = np.zeros(T)
#    p0 = np.random.normal(mu,sigma)
#    i = 0
#    #while i+1 < T:
#    i = i+1
#    p0 = np.random.normal(mu,sigma)
#    pStar = p0
#    acc = 0
#    #-----------for the first half step        
#    pStar = p0 - (delta/2)*dH_dx(yt, x[i-1], mu, var_eta, i)
#    xStar = x[i-1] + (delta)*pStar
#    for u in range(1,T):
#        #-----------for the full steps 
#        pStar = pStar - delta*dH_dx(yt, xStar, mu, var_eta, i)
#        xStar = xStar + (delta)*pStar
#        py.plot(xStar,pStar,'ko', linewidth = 20)
#        py.xlim(-6, 6) 
#        py.ylim(-6, 6)
#        py.xlabel('x') 
#        py.ylabel('p')
#        py.title('Phase Space')
#        #-----------for the last half step
#    pStar = pStar - (delta/2)*dH_dx(yt, xStar, mu, var_eta, i)
#    current = -Hamilitionian(xStar, pStar,yt, mu, var_eta, i)
#    previous = -Hamilitionian(x[i], p0,yt, mu, var_eta, i)
#    alpha2 = min(1, np.exp(current - previous))
#    if (np.random.uniform(0,1) < alpha2):
#        x[i] = xStar
#    else:
#        x[i] = x[i]
#    py.show()
#    return x
#    
#y = HybridMonteCarlo(300, -1, np.sqrt(0.05),y_t, 0.05)


#def HybridMonteCarlo(T, mu, sigma, yt, var_eta):
#    n = 100#int(T*0.1)
#    delta = 1/n
#    
#    #lamda = int(n*delta) #L
#    def Hamilitionian(xj, p, yt, mu, var_eta, jj): 
#        kinetic_E =  0.5*(p**2)
#        one = 0.5*(xj + ((yt[jj])**2)*np.exp(-xj))
#        two = ((xj - mu)**2)/(2*var_eta)
#        potential_E = one + two
#        return kinetic_E + potential_E
#    def dH_dx(yt, xi, mu, var_eta, ii):
#        y = 0.5* (1 - ((yt[ii])**2)*np.exp(-xi)) +\
#            (xi - mu)/var_eta
#        return y
#
#    x = np.zeros(T)
#    p0 = np.random.normal(mu,sigma)
#    i = 0
#    while i+1 < T:
#        i = i+1
#        p0 = np.random.normal(mu,sigma)
#        pStar = p0
#        acc = 0
#        #-----------for the first half step        
#        pStar = p0 - delta*dH_dx(yt, x[i-1], mu, var_eta, i)
#        xStar = x[i-1] + (delta)*pStar
#        for u in range(1,n-1):
#            #-----------for the full steps 
#            pStar = pStar - (delta)*dH_dx(yt, xStar, mu, var_eta, i)
#            xStar = xStar + (delta/2)*pStar
#            if i ==270 :
#                py.plot(xStar,pStar,'x', linewidth = 20)
#                #py.xlim(-6, 6) 
#                #py.ylim(-6, 6)
#                py.xlabel('x') 
#                py.ylabel('p')
#                py.title('Phase Space')
#            #-----------for the last half step
#        pStar = pStar - (delta/2)*dH_dx(yt, xStar, mu, var_eta, i)
#        current = -Hamilitionian(xStar, pStar,yt, mu, var_eta, i)
#        previous = -Hamilitionian(x[i], p0,yt, mu, var_eta, i)
#        alpha2 = min(1, np.exp(current - previous))
#        if (np.random.uniform(0,1) < alpha2):
#            x[i] = xStar
#        else:
#            x[i] = x[i]
#        py.show()
#    return x
#    
#y = HybridMonteCarlo(300, -1, np.sqrt(0.05),y_t, 0.05)

def HybridMonteCarlo(T, mu, sigma, yt, ht, phi, var_eta):
    n = int(T*0.1)
    delta = 1/n
    def Hamilitionian(xj, xminus_1, p, yt, mu, var_eta, phi, j): 
        kinetic_E =  0.5*(p**2)
        one = 0.5*(xj + ((yt[j])**2)*np.exp(-xj))
        two = ((xj - mu - phi*(xminus_1 - mu))**2)/(2*var_eta)
        potential_E = one + two
        return kinetic_E + potential_E
    def dH_dx(yt, xi, xminus_1, mu, phi, var_eta, i):
        y = 0.5* (1 - ((yt[i])**2)*np.exp(-xi)) + \
            (xi - mu - phi*(xminus_1 - mu))/var_eta
        return y
    x = np.zeros(T)
    p0 = np.random.normal(mu,sigma)
    #--------------i = 0, First step (technically 2nd) (1 in T) is calculated differently
    dH = (0.5*(1 - ((yt[0])**2)*np.exp(-x[0]))) + \
            (2*(x[0] - mu)**2)/((2*var_eta**2)/(1-phi**2))
    #-----half step
    pStar = p0 - (delta/2)*dH
    xStar = x[0] + (delta)*pStar
    #-----full step
    for u in range(n-1):
        pStar = pStar - delta*dH
        xStar = xStar + (delta)*pStar
    pStar = pStar - (delta/2)*dH
    
    current = -(0.5*pStar**2 + 0.5*(xStar + ((yt[0])**2)*np.exp(-xStar)) \
                + ((xStar - mu)**2)/((2*var_eta)/(1-phi**2)))
    previous = -(0.5*p0**2 + 0.5*(x[0] + ((yt[0])**2)*np.exp(-x[0])) \
                + ((x[0] - mu)**2)/((2*var_eta)/(1-phi**2)))
    correction = current - previous
    alpha2 = min(1, np.exp(correction))
    u = np.random.uniform(0,1)
    if (u <= alpha2):
        x[0] = xStar
    else:
        x[0] = x[0]
    #--------------Other steps (i= 1-->T) calculated
    i = 0
    while i+1 < T:
        i = i+1
        p0 = np.random.normal(mu,sigma)
        acc = 0
        #-----------for the first half step  
        xminus_1 = x[i-1]
        pStar = p0 - (delta/2)*dH_dx(yt, x[i-1],xminus_1, mu, phi, var_eta, i)
        xStar = x[i-1] + (delta)*pStar
        for u in range(1,n-1):
            #-----------for the full steps 
            pStar = pStar - delta*dH_dx(yt, xStar,xminus_1, mu,phi, var_eta, i)
            xStar = xStar + (delta)*pStar
            py.plot(xStar,pStar, linewidth = 20)
            if i ==3:
                py.plot(xStar,pStar,'x', linewidth = 20)
                py.xlabel('x') 
                py.ylabel('p')
                py.title('Phase Space')
            #-----------for the last half step
        pStar = pStar - (delta/2)*dH_dx(yt, xStar, xminus_1, mu, phi, var_eta, i)
        
        current = -Hamilitionian(xStar, xminus_1, pStar, yt, mu ,var_eta, phi,i)
        previous = -Hamilitionian(x[i], xminus_1, p0, yt, mu, var_eta, phi,i)  
        alpha2 = min(1, np.exp((current - previous)))
        if (np.random.uniform(0,1) < alpha2):
            x[i] = xStar
            acc+=1
        else:
            x[i] = x[i] # ht[i]
        py.show()
    return x

y = HybridMonteCarlo(300, -1, np.sqrt(0.05), y_t, np.full(300,0.6), 0.97, 0.05)