#    y = (var**-2)*sig_sqr**-0.5 * np.exp(-(h_old/2 + \
#        (yt**2/2) * np.exp(-h_old) + (h_old-u_t)**2/2*sig_sqr))

    #y = -(ht/2 + (yt**2*np.exp(-ht))/2) -\
#                        (ht-mu)**2/(2*var) -\
#                        (T/2+1)*np.log(var)

import numpy as np
from matplotlib import pyplot as py
from scipy.stats import invgamma
#------------------------------------------------------------------
T = 5000
N = 1000
mean = np.zeros(N)
stdv = np.zeros(N)
#------------------------------------------------------------------
"""
Initial values for variables needed to calculate likelihood function

    ht_t    = array (T)
                equal to ln((sigma_t)^2), related to volatility
    mu      = float
                the mean value of time series
    var_eta = float
                variance of the time series
"""
ht_t = np.full(T,-1.)
u = 1
var_eta = np.sqrt(.15)
#------------------------------------------------------------------
"""
Artifical Time series

    error terms = array (T)
    
"""
u_mock = -1
var_eta_mock = 0.05
sigma_eta_mock = np.sqrt(var_eta_mock)
ot = np.exp((u_mock + np.random.normal(0,sigma_eta_mock,T))/2)
#ot = np.exp(u_mock/2)
et = np.random.normal(0,1,T)
yt = ot*et
#n = T/10
#p, x = np.histogram(yt, bins=n) # bin it into n = N/10 bins
#x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
#q = np.linspace(0,T, T)
#py.plot(q, yt)
#py.show()
#--------------------

"""
Functions determining current step in values needed for
likelihood
"""
def var_eta_func(ht,mu):
    A = 0.5 * sum(((ht - mu)**2)) 
    sigma_squared = invgamma.rvs(T/2, scale = A)
    assert((np.size(sigma_squared)==1))
    return sigma_squared
        
def mu_func(ht,var):
    x = np.random.normal(sum(ht)/T,np.sqrt(var/T))
    assert((np.size(x)==1))  
    return x
  
def h_t(ht,mu,var):
    """
    Spends more time in dense areas that in less dense areas
    Gives a acceptance probability which the h_new is accepted with
    changing the correction facto changings the accepetance rate of the
    global accept reject step    
    """
    h_new = np.copy(ht)
    proposal = np.random.normal(mu,np.sqrt(var),T)
#    a = -0.5*((proposal-ht)+(yt**2*np.exp(-2*proposal)-\
#        yt**2*np.exp(-2*ht)))
#    a = -0.5*((proposal-ht)+(yt**2/np.exp(proposal)-\
#    yt**2/np.exp(ht)))
#    a =-(proposal/2 + yt**2/2*np.exp(-proposal)) -\
#        (ht/2 - yt**2/2*np.exp(-ht))
    a = 0.5*(-proposal + ht - yt**2*(np.exp(-proposal) - np.exp(-ht)))

    correction = np.exp(a)
    ind = (np.where((correction> 1)) or (np.where((correction<= 1) & \
           (np.random.uniform(0,1)<=correction))))
    h_new[ind] = proposal[ind]
    assert((np.size(ht)==np.size(correction)==np.size(proposal)==T))
    return h_new    

def likelihood_func(ht,var,mu):
    et_ot_2 = np.log((2.*np.pi*np.exp(ht))**(-0.5)) - (yt**2/2.*np.exp(-ht))
    ht_theta = np.log((2.*np.pi*var)**(-0.5)) - ((ht-mu)**2/(2.*var))
    y =  et_ot_2 + ht_theta
        
    #et = 
    #y = -0.5*np.log(2*np.pi*var)-0.5*np.log(2*np.pi)-\
     #   0.5*ht - (yt**2/(2*np.exp(ht))) - ((ht-mu)**2/(2*var))

    assert((np.size(y)==T))
    return y
    

                        
previous = likelihood_func(ht_t,var_eta,u)

#--------------------
"""
Global accept reject step to check validity of likelihood function values
newly created from h_t, mu_func, var_nu

must save mean ans stdv of every step to estimate stochastic parameters back
"""

count = 0
for i in range(N):
    
    new_ht = h_t(ht_t,u,var_eta)
    new_var_eta = var_eta_func(ht_t,u)
    new_mut = mu_func(ht_t,var_eta)
    previous_m = np.mean(previous)
    current = likelihood_func(new_ht,new_var_eta,new_mut)
    mean[i] = new_mut
    stdv[i] = np.sqrt(new_var_eta)
    
    heo = np.sum((current - previous))
    he = np.exp(heo)*(var_eta/new_var_eta)
#    cuu = np.sum(current) - np.log(new_var_eta)
#    puu = np.sum(previous) - np.log(var_eta)
#    he = np.exp((cuu - puu))
    #print(var_eta/new_var_eta)
    alpha = min(1,he)
    
    assert((np.isnan(he)==False))
    assert((he<=1) and (he>0))
    
    if ((np.random.uniform(0,1) <= alpha)):
        previous = current
        ht_t = new_ht 
        u = new_mut 
        var_eta = new_var_eta
        count+= 1
        #print(var_eta, 2)
    else:
        previous = previous
        ht_t = ht_t
        u = u
        var_eta = var_eta

#--------------------  
t = np.linspace(0,N,N)  
py.figure(figsize=(20,10))
#py.xlim(t[1*N/5],N)
py.plot(t,mean)
py.show()
 
