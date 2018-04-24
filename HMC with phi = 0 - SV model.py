import numpy as np
from matplotlib import pyplot as py
from scipy.stats import invgamma

T = 1000
N = 1000

mu_mock = -1.00
sigma_eta_mock = np.sqrt(0.05)
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

y_t = time_series_2(mu_mock,sigma_eta_mock,T)[0]
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

def HybridMonteCarlo(T, mu, sigma, yt, var_eta):
    n = 5#int(T*0.1)
    delta = 1/n
    def Hamilitionian(xj, p, yt, mu, var_eta, jj): 
        kinetic_E =  0.5*(p**2)
        one = 0.5*(xj + ((yt[jj])**2)*np.exp(-xj))
        two = ((xj - mu)**2)/(2*var_eta)
        potential_E = one + two
        return kinetic_E + potential_E
    def dH_dx(yt, xi, mu, var_eta, ii):
        y = 0.5* (1 - ((yt[ii])**2)*np.exp(-xi)) +\
            (xi - mu)/var_eta
        return y

    x = np.zeros(T)
    p0 = np.random.normal(mu,sigma)
    i = 0
    acc = 0
    while i+1 < T:
        i = i+1
        p0 = np.random.normal(mu,sigma)
        pStar = p0
        #-----------for the first half step        
        pStar = p0 - (delta/2)*dH_dx(yt, x[i-1], mu, var_eta, i)
        xStar = x[i-1] + (delta/2)*pStar
        for u in range(1,n-1):
            #-----------for the full steps 
            pStar = pStar - (delta)*dH_dx(yt, xStar, mu, var_eta, i)
            xStar = xStar + (delta/2)*pStar
            if i ==270 :
                py.plot(xStar,pStar,'x', linewidth = 20)
                #py.xlim(-6, 6) 
                #py.ylim(-6, 6)
                py.xlabel('x') 
                py.ylabel('p')
                py.title('Phase Space')
            #-----------for the last half step
        pStar = pStar - (delta/2)*dH_dx(yt, xStar, mu, var_eta, i)
        current = -Hamilitionian(xStar, pStar,yt, mu, var_eta, i)
        previous = -Hamilitionian(x[i], p0,yt, mu, var_eta, i)
        alpha2 = min(1, np.exp(current - previous))
        if (np.random.uniform(0,1) < alpha2):
            x[i] = xStar
            acc+=1
        else:
            x[i] = x[i]
        py.show()
    return x

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
ht_guess = np.random.normal(0,1,T)#np.full(T,0.6)
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
    previous = likelihood_prior_exp(ht_guess, mu_guess, var_eta_guess, y_t) 
    new_ht = HybridMonteCarlo(T, mu_guess, sigma_eta_guess, y_t, var_eta_guess)   
    current = likelihood_prior_exp(new_ht, new_mu, new_var,y_t)
    print(sum(ht_guess), sum(new_ht))
#    previous = likelihood_prior_exp(ht_guess, mu_guess,var_eta_guess,y_t)
    exponent = current - previous
    correction = np.exp(exponent) 
    alpha1 = min(1, correction)
    #print(current, previous)
    #break
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
    ht_100[i] = ht_guess[100]
##--------------------
##         ERROR ANALYSIS
##-------------------- 
#def estimated_autocorrelation(x):
#    n = len(x)
#    variance = x.var()
#    x = x-x.mean()
#    r = np.correlate(x, x, mode = 'full')[-n:]
#    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
#    result = r/(variance*(np.arange(n, 0, -1)))
#    return result
#    
#def acf(series):
#    timeseries = (series)
#    timeseries -= np.mean(timeseries)
#    autocorr_f = np.correlate(timeseries, timeseries, mode='full')
#    temp = autocorr_f[autocorr_f.size/2:]/autocorr_f[autocorr_f.size/2]
#    return temp
#    
#cutoff = N/10#10000
#    
#mu_n_mean = np.mean(mean[cutoff:])
#mu_n_std = np.std(mean[cutoff:])
#mu_n_diff = abs(mu_mock - mu_n_mean)
#
#sigma_eta_sq_n_mean = np.mean(stdv[cutoff:])
#sigma_eta_sq_n_std = np.std(stdv[cutoff:])
#sigma_eta_sq_n_diff = abs(sigma_eta_mock - sigma_eta_sq_n_mean)
#
#std_err = sigma_eta_mock/np.sqrt(N) 
#confi_95_mean = 1.96*np.std(mean[cutoff:])
#confi_95_stdv = 1.96*np.std(stdv[cutoff:])
#
###--------------------
###         PLOT GRAPHS
###--------------------
#t = np.linspace(0,N,N)  
#py.figure(figsize=(20,15))#.add_axes([0.1, 0.2, 0.8, 0.75])
#py.title(r'$Estimate\ for\ Variance,\ \sigma_{\eta}^{2}$', y = 1.02, size="40")
#py.plot(t[cutoff:], stdv[cutoff:], label = r'$\sigma_{\eta}^2$')
#py.axhline(y=np.mean((stdv[cutoff:])), color='r', linestyle='-', label = r'$Average\ \sigma_{\eta}^2 = %.3f +/- %.3f$' % (np.mean(stdv[cutoff:]), confi_95_stdv))
#py.axhline(y=sigma_eta_mock**2, color='k', linestyle='-', label = r'$Actual\ \sigma_{\eta}^2 = %.3f$' % sigma_eta_mock**2)
#py.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
#          fancybox=True, shadow=True, ncol=3, prop={'size':20})          
#py.xlim(cutoff,N)
#py.tick_params(labelsize=20)
#py.xlabel(r'$N$', size="30")
#py.ylabel(r'$\mu$', size="30")
#
#
#py.figure(figsize=(20,15))#.add_axes([0.1, 0.2, 0.8, 0.75])#
#py.title(r'$Estimate\ for\ Mean,\ \mu$', y = 1.02, size="40")
#py.plot(t[cutoff:], mean[cutoff:], label = r'$\mu$', color='g')
#py.axhline(y=np.mean(mean[cutoff:]), color='r', linestyle='-', label = r'$Average\ \mu = %.3f+/- %.3f$' %(np.mean(mean[cutoff:]),confi_95_mean))
#py.axhline(y=mu_mock, color='k', linestyle='-', label = r'$Actual \mu = %.3f$' % mu_mock)
#py.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
#          fancybox=True, shadow=True, ncol=3, prop={'size':20})          
#py.xlim(cutoff,N)
#py.tick_params(labelsize=20)
#py.xlabel(r'$N$', size="30")
#py.ylabel(r'$\sigma$', size="30")
##0.5, -0.05
##py.figure()
##py.title(r'$h_{100} lag$')
##py.scatter(ht_100[1:201],ht_100[:200], marker = 'x')
##py.xlabel(r'$ht_{100}_{i-1}$')
##py.ylabel(r'$ht_{100}_{i}$')
##
##py.figure()
##py.title(r'$h_{100} acf$')
##py.plot(t[:200],acf(ht_100[1:201]), label= 'ht[100]')
#
###--------------------no error bars needed
#py.figure(figsize=(20,15))
#py.title(r'$h_{100}\ time\ series$', y = 1.02, size="40")
#py.plot(t[cutoff:], ht_100[cutoff:], label= r'$ht_{100}$')
#py.tick_params(labelsize=20)
#py.xlabel(r'$N$',size="30")
#py.ylabel(r'$ht_{100}$', size="30")
#py.xlim(cutoff,N)
##
#t = np.linspace(0,T,T)
#py.figure(figsize=(20,15))
#py.title(r'$Time\ series\ for\ SV\ model$', y = 1.02, size="40")
#py.plot(t, y_t)
#py.tick_params(labelsize=20)
#py.xlabel(r'$T$', size="30")
#py.ylabel(r'$y_{t}$', size="30")
#py.show()
#print(count)