import numpy as np
from matplotlib import pyplot as py
from scipy.stats import invgamma

T = 6000
N = 35000

mu_mock = -1
sigma_eta_mock = np.sqrt(0.05)
varrrr = 0.05
#--------------------
#         ARTIFICAL TIME SERIES
#--------------------
def time_series_2(mu,sigma_eta,T): 
    ln_sd2 = np.zeros(T)
    e = np.random.normal(0,1,T)
    nu = np.random.normal(0,sigma_eta,T)
    ln_sd2[0] = 0    
    for n in range(1,T):
        ln_sd2[n] = mu + nu[n]
    y=np.exp((ln_sd2)/2)*e
    return y, np.mean(ln_sd2),np.var(ln_sd2) 

y_t,ytmean,ytvar = time_series_2(mu_mock,sigma_eta_mock,T)
#--------------------
#         SAMPLING OF THETA PARAMETERS - PRIOR DISTRIBUTIONS
#--------------------
def var_eta_func(ht,mu):
    A = 0.5*(np.sum((ht - mu)**2))
    y = invgamma.rvs(T/2, scale = A)
    assert((np.isnan(y)==False))
    return y
    
def mu_func(ht,sigma_eta):  
    B = T
    C = np.sum(ht)
    y = np.random.normal(C/B,sigma_eta*np.sqrt(1/B))
    assert((np.isnan(y)==False))
    return y
    
def ht_func(ht, mu, sigma_eta): 
    ht_new = ht
    proposal = np.random.normal(mu, sigma_eta , T)
    ht_cur = (-0.5)*(proposal + (y_t**2)/np.exp(proposal))
    ht_prev = (-0.5)*(ht + (y_t**2)/np.exp(ht))
    alpha = np.exp(ht_cur - ht_prev)
    y = np.random.uniform(0,1,T)
    ind = np.where((y<alpha))    
    ht_new[ind] = proposal[ind]
    return ht_new
    assert((np.size(ht_cur)==np.size(ht_prev)==np.size(alpha)== T))
    assert((np.isnan(alpha.any())==False))
    return ht_new

#--------------------
#         EXPONENT OF LIKELIHOOD FUNCTION * PRIOR
#--------------------
def likelihood_prior_exp(ht,mu,var_eta,y_t):  
    y= np.sum((-ht/2) - (y_t**2)/(2*np.exp(ht)) \
         - ((ht - mu)**2/(2*var_eta))) - (T/2+1)*np.log(var_eta)
    return y
#--------------------
#         INITIAL CONDITIONS
#--------------------
var_eta_guess = 0.5
sigma_eta_guess =  np.sqrt(var_eta_guess) #can't be zero or else f_ht_theta dont work
mu_guess =  1
ht_guess = np.random.normal(0,1,T)
mean = np.zeros(N)
stdv = np.zeros(N)
ht_100 = np.zeros(N)
#--------------------
#         GlOBAL ACCEPT REJECT STEP
#--------------------  
count = 0
for i in range(N):
    new_var = var_eta_func(ht_guess, mu_guess)
    new_mu = mu_func(ht_guess, sigma_eta_guess)
    new_sig = np.sqrt(new_var)
    new_ht = ht_func(ht_guess, mu_guess, sigma_eta_guess)
    current = likelihood_prior_exp(new_ht, new_mu, new_var,y_t)
    previous = likelihood_prior_exp(ht_guess, mu_guess,var_eta_guess,y_t)
    exponent = current - previous
    correction = np.exp(exponent) 
    alpha1 = min(1, correction)
    if ((np.random.uniform(0,1) < alpha1)):
        var_eta_guess = new_var
        sigma_eta_guess = new_sig
        mu_guess = new_mu
        ht_guess = new_ht
        count+= 1
        mean[i] = mu_guess
        stdv[i] = var_eta_guess
    else:
        mean[i] = mu_guess
        stdv[i] = var_eta_guess
        
    ht_100[i] = ht_guess[200]

