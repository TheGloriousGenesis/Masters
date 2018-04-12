import numpy as np
from matplotlib import pyplot as py
from scipy.stats import invgamma
import timeit

T = 100
N = 500

mean = np.zeros(N)
stdv = np.zeros(N)
ht_10 = np.zeros(N)
mu_mock = -1
sigma_eta_mock = np.sqrt(0.05) 
phi_mock = 0.97
#--------------------
#         ARTIFICAL TIME SERIES
#--------------------
#exponent_ht = mu_mock + phi_mock*(np.random.normal(0, sigma_eta_mock, T)
#sigma_t = np.exp(exponent_ht/2)
#epsilon_t = np.random.normal(0, 1, T)
#y_t = sigma_t * epsilon_t
def time_series_2(u,phi,sd_nu2,T): 
    """
    Outputs time series relating to the stochastic model
    
    Parameters
    -----------
    u = float
        Analogous to 
        unobservable factor to be estimated
    phi =  float
        Analogous to "existence" of volatility in financial model
        If <1 produces a white noise looking process
        unobservable factor to be estimated
    ln_sd = array
        the natural logarithm of the standard deviation
        Analogous to the dependancy of asset return value on flucations in
        the financial market due to new information. circumstances e.t.c
        equal to log of volatility
        follows AR(1) process
        unobservable factor to be estimated
    e = array 
        independant normally distributed random number.
        Analogous to 
    mu = array
        independant and identically distributed to e (normally distributed 
        random number).
    y = array
        Analogous to value (or return) of an asset at given time t
    """    
    ln_sd = np.zeros(T)
    e = np.random.normal(0,1,T)
    nu = np.random.normal(0,sd_nu2,T)
    ln_sd[0] = 0    
    
    for n in range(1,T):
        ln_sd[n] = u + phi * (ln_sd[(n-1)]-u) + nu[n]
    
    y=np.exp(ln_sd)*e
    
    assert((not any(np.isnan(ln_sd)) and all(np.isfinite(ln_sd)) and \
    all(np.isreal(ln_sd))))
    assert((not any(np.isnan(e)) and all(np.isfinite(e)) and \
    all(np.isreal(e))))
    assert((not any(np.isnan(nu)) and all(np.isfinite(nu)) and \
    all(np.isreal(nu))))
    assert((not any(np.isnan(y)) and all(np.isfinite(y)) and \
    all(np.isreal(y))))
    assert((np.shape(e)==np.shape(ln_sd)==np.shape(nu)==np.shape(y)))
    assert((type(e)==type(ln_sd)==type(nu)==type(y)))
    
    return y, np.mean(ln_sd),np.var(ln_sd) 

y_t = time_series_2(mu_mock,phi_mock,sigma_eta_mock**2,T)[0]
#--------------------
#         SAMPLING OF THETA - POSTERIOR DISTRIBUTION
#--------------------
def var_eta_func(ht,mu, phi):
    x = np.sum((ht[1:] - mu - phi*(ht[:len(ht)-1] - mu))**2)
    A = 0.5*((1-phi**2)*(ht[0] - mu)**2 + x)
    y = invgamma.rvs(T/2, scale = A)
    assert((np.isnan(y)==False) and (A>0))
    return y
    
def mu_func(ht,sigma_eta,phi):   
    B = (1-phi**2) + (T-1)*(1-phi)**2
    C = (1-phi**2)*ht[0] + (1-phi)*np.sum(ht[1:] - phi*ht[:len(ht)-1])
    y = np.random.normal(C/B,sigma_eta*np.sqrt(1/B))
    assert((np.isnan(y)==False))
    return y

def phi_func(ht,mu,sigma_eta, phi):
    prev = phi
    D = -(ht[0] - mu)**2 + np.sum((ht[1:len(ht)-1] - mu)**2)
    E = np.sum((ht[1:] - mu)*(ht[:len(ht)-1] - mu))
    while 2==2:
        proposal = np.random.normal(E/D,sigma_eta*np.sqrt(1/D))
        if proposal>-1 and proposal<1:
            break
    Prob_Ta = min(1,np.sqrt((1-proposal**2)/(1-prev**2)))      
    if np.random.uniform()<=Prob_Ta:
        return proposal   
    else:
        return prev

def Hamilitionian(ht, p, yt, mu, var_eta, phi): 
    kinetic_E   =  0.5*sum(p**2)
    one = 0.5*sum(ht + (yt**2)*np.exp(-ht)) + \
                    ((ht[0] - mu)**2)/((2*var_eta)/(1-phi**2))
    two = sum(((ht[1:] - mu - phi*(ht[:len(ht)-1] - mu))**2)/(2*var_eta))
    potential_E = one + two
    return kinetic_E + potential_E

start_time = timeit.default_timer()
def HybridMonteCarlo(del_t, T, mu, sigma, yt, ht, phi, var_eta):
    """
    cannot simulate Hamiltonian dynamics exactly because of time discretization
    must take into consideration time discretization. we go forward through
    finite differencing. del_t and rounding errors will make tranformations not
    perfectly reversible. BUT HMC cancels this effect by adding Metropolis accept
    reject after lAMDA leapfrog steps. The new state made is then accepted if
    P = MIN(1, EXP(-H')/EXP(-H))
    
    guess   = [x,p]
            where x = position, p = velocity
    del_t   = time step. This will not perserve Hamiltonian precisely
    dH_dx   = momemtum evolution (p dot)
    N       = I think it is the time series length?
    x_half  = stores half step update of position to calculate full step
    x       = stores full step update of position
    n       = trajectory length
    new     = keeps values of x, x_half,p  as p[0,1,2] respectively
    g_cur   = new proposals for momemtum variable
    
    #might have the same issue with ht changing
    dH_dp = as it isnt a sum
    """
    n = int(T*del_t)
    x = np.full(n,0)#np.zeros((n,))
    x_half = np.full(n,0)#np.zeros((n,))
    p = np.full(n,0)#np.zeros((n,))
    p_prev = np.random.normal(mu,sigma,T)
    p_cur  = np.full(T,0) #chsnge
    #print(n)
    ht_prev = np.array(ht)
    ht_cur = np.full(T,0)#change
    #--------------First step (1 in T) is calculated differently
    x[0] = ht[0] 
    p[0] = p_prev[0]
        
    x_half[0] = x[0] + (del_t/2)*p[0]
    p[0] = p[0] - del_t*(0.5*(1 - (yt[0]**2)*ht[0]*np.exp(-ht[0])))
    x[0] = x_half[0] + (del_t/2)*p[0]
        
    p_cur[0] = p[0]
    ht_cur[0] = x[0]
    #--------------Other steps (2-->T) calculated
    for i in range(1,T):
        """
        for each element in ht
        """
        x[0] = ht[i] 
        p[0] = p_prev[i]

        dH_dx = 0.5* (1 - (yt[i]**2)*ht[i]*np.exp(-ht[i])) + \
                (ht[i] - mu - phi*(ht[i-1] - mu))/var_eta 
        for t in range(n-1):
            """
            evole said element in ht
            """
            x_half[t+1] = x[t] + (del_t/2)*p[t]            
            p[t+1] = p[t] - del_t*dH_dx
            x[t+1] = x_half[t+1] + (del_t/2)*p[t+1]
        #print(x[-1])
        p_cur[i] = p[-1]
        ht_cur[i] = x[-1]
       # print('i', i, 'p',p_cur[i], 'ht',x[-1] )
    current = -Hamilitionian(ht_cur, p_cur, yt, mu,sigma**2, phi)
    previous = -Hamilitionian(ht_prev, p_prev, yt, mu, sigma**2, phi)
    correction = np.exp((current - previous))
    alpha = min(1, correction)
    #print(np.round(ht_cur-ht_prev, decimals =3),)

    u = np.random.uniform(0,1)
    #print('ht_cur', sum(ht_cur), alpha)
    if (u <= alpha):
        return ht_cur
    else:
        return ht_prev
elapsed = timeit.default_timer() - start_time
#--------------------
#         EXPONENT OF LIKELIHOOD FUNCTION AND PRIOR
#--------------------
def likelihood_prior_exp(ht,mu,var_eta,yt,phi): 
    part_one = np.sum(-ht/2 - (yt**2)/(2*np.exp(ht)))
    part_two = np.sum(- (ht[1:] - mu - phi*(ht[:len(ht)-1])**2)/(2*var_eta)) - (T/2+1)*np.log(var_eta)
    part_three = np.sum(- 0.5*np.log(var_eta/(1-phi**2)) - ((ht[0] - mu)**2)/((2*var_eta)/(1-phi**2)))
    y = part_one + part_two + part_three
#    y = np.sum((-ht/2) - (yt**2)/(2*np.exp(ht)) \
#         - (ht[1:] - mu - phi*(ht[:len(ht)-1])**2)/(2*var_eta)) - (T/2+1)*np.log(var_eta)\
#         - 0.5*np.log(var_eta/(1-phi**2)) - ((ht[0] - mu)**2)/((2*var_eta)/(1-phi**2))
    return y
#--------------------
#         INITIAL CONDITIONS
#--------------------
var_eta_guess = 0.5
sigma_eta_guess =  np.sqrt(var_eta_guess) #can't be zero or else f_ht_theta dont work
mu_guess =  1
ht_guess = np.full(T,0.6)
phi_guess = 0.1
del_t_step = 0.1 #change back to 0.01
#--------------------
#         GlOBAL ACCEPT REJECT STEP
#--------------------  
count = 0
for i in range(N):
    new_var = var_eta_func(ht_guess, mu_guess, phi_guess)
    new_mu = mu_func(ht_guess, sigma_eta_guess, phi_guess)
    new_sig = np.sqrt(new_var)
    #previous = likelihood_prior_exp(ht_guess, mu_guess, var_eta_guess, y_t) 
    new_phi = phi_func(ht_guess, mu_guess,sigma_eta_guess,phi_guess)    
    new_ht = HybridMonteCarlo(del_t_step,T, mu_guess, sigma_eta_guess,y_t,ht_guess, phi_guess, var_eta_guess)
    #break
    current = likelihood_prior_exp(new_ht, new_mu, new_var,y_t, new_phi)
    previous = likelihood_prior_exp(ht_guess, mu_guess,var_eta_guess,y_t, phi_guess)
#possible something to do with var_eta    
    exponent = current - previous
    print(current, previous)
    correction1 = np.exp(exponent) 
    alpha1 = min(1, correction1)
    #print('Acceptance', correction1)
    #break

    if ((np.random.uniform(0,1) <= alpha1)):
        var_eta_guess = new_var
        sigma_eta_guess = new_sig
        mu_guess = new_mu
        ht_guess = new_ht
        phi_guess = new_phi
        count+= 1
        mean[i] = mu_guess
        stdv[i] = sigma_eta_guess
    else:
        mean[i] = mu_guess
        stdv[i] = sigma_eta_guess

    #ht_10[i] = ht_guess[100]
    
##--------------------
##         PLOT GRAPHS
##--------------------  
time = np.linspace(0,N,N)  
py.figure()
py.title("mean/ stdv time series")
py.plot(time, stdv, label = 'stdv')
py.plot(time, mean, label = 'mean')
#py.plot(time[10000:], stdv[10000:], label = ' stdv')
#py.plot(time[10000:], mean[10000:], label= 'mean')
#py.axhline(y=np.mean(mean[10000:]), color='r', linestyle='-', label = 'avg mean = %.3g'%np.mean(mean[10000:]))
#py.axhline(y=mu_mock, color='k', linestyle='-', label = 'Actual result = %.3g'%mu_mock)
#py.axhline(y=np.mean(stdv[10000:]), color='r', linestyle='-', label = 'avg stdv = %.3g'%np.mean(stdv[10000:]))
#py.axhline(y=sigma_eta_mock, color='k', linestyle='-', label = 'Actual result = %.3g' %sigma_eta_mock)
py.legend()

#py.figure()
#py.title("h_100 time series")
#py.plot(time[30000:], ht_10[30000:], label= 'ht[100]')
#

py.show()
print(count)