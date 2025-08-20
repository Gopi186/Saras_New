#!/usr/bin/env python
# coding: utf-8

import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from math import e
import statistics as st
import cmath
import scipy
import scipy.stats as ss
import random
#import seaborn as sns
import pandas as pd
import emcee
import corner
import h5py
import numpy.ma as ma
from scipy.interpolate import interp1d
import pickle
import scipy.constants as const
from scipy import interpolate
from multiprocessing import Pool
import arviz as az


# ## RLC S11 Data

RLC_S11_df = pd.read_csv(
    "/home/saurabhs/Documents/gopi/Saras_New/Calibrated_ADHOC_S11_TEST4_SARAS4.csv"
)

RLC_S11 = list(RLC_S11_df.columns)
v = RLC_S11_df.iloc[0]
v = v * 1e6
RLC_S11 = [float(i) for i in RLC_S11]

fit = np.polyfit(v, RLC_S11, 6)  # estimating fit (coeffs)
p0, p1, p2, p3, p4, p5, p6 = fit

model = np.polyval(fit, v)
res1 = model - RLC_S11  # residuals


# ## RLC Avg Data

avg = h5py.File(
    "/home/saurabhs/Documents/amarnath/hdatasets/2025-04-13_053004.pyavg", "r"
)


# ### Setting freq range
# Frequencies
avg["pyavg"].keys()
sfreq = avg.attrs["sfreq"]
print(avg.attrs.keys())
freq = np.arange(0, 8193, 1) * sfreq * 1e3
ind_freq = np.logical_and((freq >= 40), (freq <= 110))
req_freq = freq[ind_freq] * 1e6
len_freq = len(req_freq)


# ### Interpolating the RLC for the required frequencies

# Interpolating the RLC for the required frequencies
interpolator = interp1d(v, RLC_S11, kind="linear")
freq_new = np.linspace(v.min(), v.max(), len_freq)
# print(len(freq_new))
RLC_new = interpolator(freq_new)  ##Interpolated RLC

# plt.plot(req_freq,RLC_new)
# plt.show()



# New Fit Coeffs
fit = np.polyfit(req_freq, RLC_new, 6)
p0, p1, p2, p3, p4, p5, p6 = fit
model = np.polyval(fit, req_freq)
# res=model-RLC_new

# ### Reading the avg data

SPCD_avg = avg["SPCD"][0, ind_freq].real
SPCF_avg = avg["SPCF"][0, ind_freq].real
mask_avg = avg["mask"][0, ind_freq].real

# ### Masking

# Masking the unwanted frequencies
TMeas_exp_avg = ma.masked_array(SPCD_avg, np.logical_not(mask_avg))  ##Masking
# plt.plot(req_freq,TMeas_exp_avg,'r.', label='Masked data', alpha=0.8)
# plt.plot(req_freq,SPCD_avg, alpha=0.8, label='Data')
# plt.xlabel('freq (MHz)')
# plt.legend()
# plt.show()

##Masking Uncertainities
uncert = ma.masked_array(SPCF_avg, np.logical_not(mask_avg))

"""
plt.plot(req_freq,uncert,'r.', label='Masked data', alpha=0.8)
#plt.plot(req_freq,SPCF_avg, alpha=0.8, label='Data')
plt.xlabel('freq (MHz)')
plt.legend()
plt.show()
print(uncert.shape)
"""


# ## Model

# ### For RLC

def Compute_Tmeas_RLC(
    PA, Pref, p0, p1, p2, p3, p4, p5, p6, PN, gamma_N, f, l, req_freq
):
    A = np.zeros_like(req_freq, dtype=float)
    B = np.zeros_like(req_freq, dtype=float)
    C = np.zeros_like(req_freq, dtype=float)
    l = l
    itr = 3
    Tmeas = []
    phi_A = 0
    phi_N = 0
    phi_f = 0
    P_ref = Pref
    c = const.c
    req_freq = req_freq
    # global req_freq
    gamma_A = np.polyval(
        [
            p0 * 1e-48,
            p1 * 1e-40,
            p2 * 1e-31,
            p3 * 1e-23,
            p4 * 1e-16,
            p5 * 1e-8,
            p6 * 1e-1,
        ],
        req_freq,
    )

    phi = (4 * math.pi * (req_freq) * l) / (0.7 * c)
    # print(phi.size)
    for n in range(itr):
        # For A term: l index sum
        l_values = np.arange(n + 1)
        sum_l = np.sum(
            np.cos((2 * l_values[:, None] - n) * (phi_N + phi_A + phi)), axis=0
        )
        # print(sum_l.size)
        # For C term: c index sum (same structure as A)
        c_values = np.arange(n + 1)
        sum_c = np.sum(
            np.cos((2 * c_values[:, None] - n) * (phi_N + phi_A + phi)), axis=0
        )

        # Powers
        gammaA_pow_n = np.abs(gamma_A) ** n
        gammaN_pow_n = np.abs(gamma_N) ** n

        # A term update
        A += gammaA_pow_n * gammaN_pow_n * sum_l

        # B term update
        B += (
            2
            * np.abs(f)
            * (np.abs(gamma_A) ** (n + 1))
            * gammaN_pow_n
            * np.cos(phi_f + (n + 1) * (phi_A + phi) + n * phi_N)
        )

        # C term update
        C += gammaA_pow_n * gammaN_pow_n * sum_c

    # Tmeas = (PA * A - Pref) + PN * (B + (np.abs(f)**2) * (np.abs(gamma_A)**2) * A)

    Tmeas = (PA * (1 - gamma_A**2) * A - Pref) + PN * (
        B + (np.abs(f) ** 2) * (np.abs(gamma_A) ** 2) * C
    )

    return Tmeas

"""
#Simulated signal vs Actual averaged signal
T=Compute_Tmeas_RLC_trial(300,300, -2.52097554,  9.26316121, -1.39878668,  1.25855337,\
     -7.14339265,  1.24118382,  8.85812772, 70, 0.3, 0.1, 0.15, req_freq)
T1=Compute_Tmeas_RLC(313.44495298597064, 307.2088350626911, -1.662375689423965, 7.231646700605323, -1.1829738388449063, 0.9588036878869409, -4.277921997353682, \
          0.31665161309467715, 9.891971880968821,  74.23509620659401, \
             0.19840997630322196, 0.4260990319365141, 0.21653155250707493, req_freq)
,
plt.plot(req_freq,T, label="Actual S11")
plt.plot(req_freq,T1,'|r',label="Est S11")
plt.plot(req_freq,TMeas_exp_avg,',k',label="Data")
plt.legend()
plt.xlabel("Freq")
plt.ylabel("T(K)")
plt.title("TMeas vs Freq")
plt.show()
"""


# ### For Open

def Compute_Tmeas_Open(PA, Pref, gamma_A, PN, gamma_N, f, l, req_freq):
    A = np.zeros_like(req_freq, dtype=float)
    B = np.zeros_like(req_freq, dtype=float)
    C = np.zeros_like(req_freq, dtype=float)
    l = l
    itr = 3
    Tmeas = []
    phi_A = 0
    phi_N = 0
    phi_f = 0
    P_ref = Pref
    c = const.c
    req_freq = req_freq
    # global req_freq
    gamma_A = gamma_A

    phi = (4 * math.pi * (req_freq) * l) / (0.7 * c)
    # print(phi.size)
    for n in range(itr):
        # For A term: l index sum
        l_values = np.arange(n + 1)
        sum_l = np.sum(
            np.cos((2 * l_values[:, None] - n) * (phi_N + phi_A + phi)), axis=0
        )
        # print(sum_l.size)
        # For C term: c index sum (same structure as A)
        c_values = np.arange(n + 1)
        sum_c = np.sum(
            np.cos((2 * c_values[:, None] - n) * (phi_N + phi_A + phi)), axis=0
        )

        # Powers
        gammaA_pow_n = np.abs(gamma_A) ** n
        gammaN_pow_n = np.abs(gamma_N) ** n

        # A term update
        A += gammaA_pow_n * gammaN_pow_n * sum_l

        # B term update
        B += (
            2
            * np.abs(f)
            * (np.abs(gamma_A) ** (n + 1))
            * gammaN_pow_n
            * np.cos(phi_f + (n + 1) * (phi_A + phi) + n * phi_N)
        )

        # C term update
        C += gammaA_pow_n * gammaN_pow_n * sum_c

    # Tmeas = (PA * A - Pref) + PN * (B + (np.abs(f)**2) * (np.abs(gamma_A)**2) * A)

    Tmeas = (PA * A - Pref) + PN * (B + (np.abs(f) ** 2) * (np.abs(gamma_A) ** 2) * C)

    return Tmeas


# ## 21cm signal

path_to_file = "/home/saurabhs/Documents/gopi/Saras_New/Data_18March_wMFP.mat"
path_to_freq = "/home/saurabhs/Documents/gopi/Saras_New/freq_saras.txt"

data = scipy.io.loadmat(path_to_file)
data = data["Data2"] / 1e3  # Kelvin units

fr = np.loadtxt(path_to_freq)  # Frequency in MHz
signal_function = interpolate.interp1d(fr, data, fill_value="extrapolate")

F_MIN = 40  # define your own

F_MAX = 110  # define your own

NO_OF_CHANNELS = len_freq  # define your own

freq_array = np.linspace(F_MIN, F_MAX, NO_OF_CHANNELS)

signals = signal_function(freq_array)

sig_idx = 63
sig21cm = signals[sig_idx]

# plt.plot(req_freq, sig21cm)
# plt.title('21 cm profile')
# plt.xlabel('frequency (MHz)')
# plt.ylabel('T (K)')
# plt.grid(True)
# plt.show()


# ### Adding the 21cm signal (Avg signal + 21cm signal)

a = 1
T21 = a * sig21cm + TMeas_exp_avg

# plt.plot(req_freq,T21, 'k', zorder=1)
# plt.plot(req_freq,TMeas_exp_avg, '.r', zorder=2)


# ###  Open Run Data

data = map(
    lambda x: [x[0], x[1], x[2]],
    np.loadtxt(
        "/home/saurabhs/Documents/gopi/Saras_New/saras3_S11_200mm_above_water.s1p",
        skiprows=5,
    ),
)

d = list(data)  # List of the format [Frequency, Magnitude, Phase (in degrees)]
c = const.c  # speed of light (m/s)
freq = [d[i][0] for i in range(len(d) - 1)]  # frequency
# df=pd.read_csv('/home/saurabhs/Documents/gopi/Saras_New/S11_for_freq.csv')
# rlc = df[df.columns[1]].values.tolist()
"""
RLC_S11_df = pd.read_csv('/home/saurabhs/Documents/gopi/Saras_New/Calibrated_ADHOC_S11_TEST4_SARAS4.csv')

RLC_S11 = list(RLC_S11_df.columns)
v=RLC_S11_df.iloc[0]
RLC_S11 = [float(i) for i in RLC_S11]
rlc= RLC_S11[::-1]
print(v[0],RLC_S11[0])
plt.plot(v, RLC_S11)
"""


# ## MCMC for Open

########## Define the log-likelihood function
def log_likelihood_o(params, seed, v, s, TA_exp):
    PA, Pref, gamma_A, PN, gamma_N, f, l = params

    np.random.seed(seed)

    # Compute model TA
    TA_model = Compute_Tmeas_Open(PA, Pref, gamma_A, PN, gamma_N, f, l, v)
    TA_model = np.array(TA_model)

    # Compute chi-squared with weights

    chi2 = np.sum([((TA_exp - TA_model) / s) ** 2])

    return -0.5 * chi2  # - np.log(math.sqrt(2*np.pi)*s)  # Likelihood function


########## Define the log-prior function
def log_prior_o(params):
    PA, Pref, gamma_A, PN, gamma_N, f, l = params

    if (
        (0 <= gamma_A <= 1)
        and (0 <= gamma_N <= 1)
        and PA > 0
        and Pref > 0
        and PN > 0
        and f > 0
        and l > 0
    ):
        return 0.0  # Reject sample
    return -np.inf  # Uniform prior


######### Define the full log-probability function


def log_probability_o(params, seed, v, s, TA_exp):
    lp = log_prior_o(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_o(params, seed, v, s, TA_exp)


########### MCMC for Open

def evolve_MCMC_o(**kwargs):  # Define parameter names
    ndim = 7  # Number of parameters
    nwalkers = 150  # Number of MCMC walkers
    nsteps = kwargs["nsteps"]  # Number of MCMC steps per walker

    ## Initialize walkers around a random point in parameter space
    initial_value = kwargs["initial_value"]

    seed = kwargs["seed"]
    v = kwargs["v"]
    s = kwargs["s"]
    TA_exp = kwargs["TA_exp"]

    np.random.seed(seed + 1)
    ## Starting positions for the walkers (making sure that the initial positions satisfy the priors)

    pos_valid = []
    while len(pos_valid) < nwalkers:
        trial = np.array(initial_value) + 1e-1 * np.random.normal(size=ndim)

        if np.isfinite(log_prior_o(trial)):
            pos_valid.append(trial)
    pos = np.array(pos_valid)

    # pos = np.array(initial_value)[None,:] + 1e-1*np.random.normal(0,1,size=(nwalkers,ndim))

    ## for making sure only positive positional values
    """
    for idim in range(ndim):
        ind_pos = pos[:,idim]<0
        pos[ind_pos,idim] = np.abs(pos[ind_pos,idim])
    """

    ## Set up the MCMC sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_o, args=(seed, v, s, TA_exp), pool=pool
        )

        ## Run MCMC

        print("Running MCMC for Open...")
        sampler.run_mcmc(pos, nsteps, progress=True)

    ## Extract the samples
    burn = 250
    full_samples = sampler.get_chain()
    samples = sampler.get_chain(discard=burn, flat=True)
    tau = sampler.get_autocorr_time(discard=burn, tol=0, quiet=True)
    print("tau (best-effort):", tau)
    ## Compute log-likelihoods for all samples

    log_likes = np.array([log_likelihood_o(p, seed, v, s, TA_exp) for p in samples])

    ## Find the index of the maximum likelihood

    max_likelihood_index = np.argmax(log_likes)

    ## Get the best-fit parameters

    best_fit_params = samples[max_likelihood_index].tolist()

    return full_samples, samples, best_fit_params, pos

# evolve MCMC
ip_params = [0, 300, 1, 70, 0.3, 0.1, 0.15]  # input parameters
nsteps = 1000
initials = [0.007, 303, 0.97, 70.7, 0.29, 0.1, 0.13]
s = 0.001
seed = 63
v = req_freq
TA_exp = Compute_Tmeas_Open(*ip_params, req_freq)
TA_exp = np.array(TA_exp) + np.random.normal(0, 0.001, len(v))
kwargs = {
    "s": s,
    "seed": seed,
    "TA_exp": TA_exp,
    "nsteps": nsteps,
    "ip_params": ip_params,
    "v": v,
    "initial_value": initials,
}
full_samples_o, samples_o, best_fit_params_o, pos = evolve_MCMC_o(**kwargs)


# parameter_names = ["PA","Pref", "gamma_A", "PN", "gamma_N", "f","l"]
# #corner plot
# fig = corner.corner(samples_o, labels=parameter_names, truths=ip_params, color='C1',  # Change color of the contour and histograms
# hist_kwargs={"color": "royalblue"},  # Set histogram color
# contour_kwargs={"colors": ["royalblue"]},  # Set contour color
# truth_color="green" ) # Color for the true parameter values)
# plt.suptitle("Open Param Est with gA, gN, PA, PN, f & l Constrained", fontsize=25)
# #plt.savefig('/home/saurabhs/Documents/gopi/Saras_Model/Untitled Folder/Open_All_15cm_1.png', dpi=300)
# plt.show()

# #print("Expected Paramerters:",ip_params)
# print("Best-fit parameters (Maximum Likelihood Estimate):", best_fit_params_o)


# ## MCMC for RLC real data with S11 fixed
########## Define the log-likelihood function
def log_likelihood_full(params, seed, v, s, TA_exp, p0, p1, p2, p3, p4, p5, p6, signal):
    # Compute TA expected
    # seed,v,s,TA_exp,p0,p1,p2,p3,p4,p5,p6,signal
    # ip_params=kwargs["ip_params"]
    #     v=kwargs["v"]
    #     s=kwargs["s"]
    #     TA_exp=kwargs["TMeas_exp"]
    #     p0, p1, p2, p3, p4, p5, p6 = kwargs["S11_params"]
    #     signal=kwargs["signal"]

    np.random.seed(seed)

    # Compute model TA

    a, PA, Pref, PN, gamma_N, f, l = params
    TA_model = Compute_Tmeas_RLC(
        PA, Pref, p0, p1, p2, p3, p4, p5, p6, PN, gamma_N, f, l, v
    )

    TA_model = np.array(TA_model) + a * signal

    # Compute chi-squared
    chi2 = np.sum([((TA_exp - TA_model) / s) ** 2])

    return -0.5 * chi2  # - np.log(math.sqrt(2*np.pi)*s)


########## Define the log-prior function
def log_prior_full(params):
    a, PA, Pref, PN, gamma_N, f, l = params
    # v=kwargs["v"]
    # PA, p0, p1, p2, p3, p4, p5, p6, PN, gamma_N, f, l = params
    # gamma_A=np.polyval([p0*1e-48, p1*1e-39, p2*1e-31, p3*1e-23, p4*1e-15, p5*1e-08, p6*1e-01],v)

    if (
        a > 0
        and 0 <= gamma_N <= 1
        and PA > 0
        and Pref > 0
        and PN > 0
        and f > 0
        and l > 0
    ):  # np.amin(gamma_A)>=0 and np.amax(gamma_A)<=1 and ):
        return 0.0  # Accept sample

    return -np.inf  # Reject prior


######### Define the full log-probability function


def log_probability_full(
    params, seed, v, s, TA_exp, p0, p1, p2, p3, p4, p5, p6, signal
):
    lp = log_prior_full(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_full(
        params, seed, v, s, TA_exp, p0, p1, p2, p3, p4, p5, p6, signal
    )


########## MCMC for RLC Sys Params
def evolve_MCMC_Full(**kwargs):
    # Define parameter names
    # parameter_names = ["PA", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "PN", "gamma_N", "f", "l"]

    # args=args

    ## Number of walkers and steps

    ndim = 7  # Number of parameters
    nwalkers = 100  # Number of MCMC walkers
    nsteps = kwargs["nsteps"]  # Number of MCMC steps per walker

    ## Initialize walkers around a random point in parameter space

    initial_value = kwargs["initial_value"]

    seed = kwargs["seed"]
    v = kwargs["v"]
    s = kwargs["s"]
    TA_exp = kwargs["TA_exp"]
    p0, p1, p2, p3, p4, p5, p6 = kwargs["S11_params"]
    signal = kwargs["signal"]

    ## Starting positions for the walkers (making sure that the initial positions satisfy the priors)
    np.random.seed(seed + 1)

    pos_valid = []
    while len(pos_valid) < nwalkers:
        trial = np.array(initial_value) + 1e-1 * np.random.normal(
            size=ndim
        )  # initial_value + 1e-1 * np.random.randn(ndim)
        # print(len(trial))
        if np.isfinite(log_prior_full(trial)):
            pos_valid.append(trial)
    pos = np.array(pos_valid)

    # pos = np.array(initial_value)[None,:] + 1e-1*np.random.normal(0,1,size=(nwalkers,ndim))

    ## for making sure only positive positional values
    """
    for idim in range(ndim):
        ind_pos = pos[:,idim]<0
        pos[ind_pos,idim] = np.abs(pos[ind_pos,idim])
    """

    ## Set up the MCMC sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability_full,
            args=(seed, v, s, TA_exp, p0, p1, p2, p3, p4, p5, p6, signal),
            pool=pool,
        )

        ## Run MCMC

        print("Running MCMC for RLC...")
        sampler.run_mcmc(pos, nsteps, progress=True)

    ## Extract the samples
    burn = 1400
    full_samples = sampler.get_chain()
    samples = sampler.get_chain(discard=burn, flat=True)
    tau = sampler.get_autocorr_time(discard=burn, tol=0, quiet=True)
    # Acceptance Fraction
    af_per_walker = sampler.acceptance_fraction  # array shaped (nwalkers,)
    af_mean = af_per_walker.mean()
    print("tau (best-effort):", tau)
    print("tau_max:", np.max(tau))

    ## Compute log-likelihoods for all samples

    log_likes = np.array(
        [
            log_likelihood_full(
                p, seed, v, s, TA_exp, p0, p1, p2, p3, p4, p5, p6, signal
            )
            for p in samples
        ]
    )

    ## Find the index of the maximum likelihood

    max_likelihood_index = np.argmax(log_likes)

    ## Get the best-fit parameters

    best_fit_params = samples[max_likelihood_index].tolist()

    return full_samples, samples, best_fit_params, pos, af_per_walker, af_mean, tau


# evolve MCMC full with initials for system parameters from Open
## Initialising kwargs
S11_params = [
    -1.662375689423965,
    7.231646700605323,
    -1.1829738388449063,
    0.9588036878869409,
    -4.277921997353682,
    0.31665161309467715,
    9.891971880968821,
]
seed = 29
v = req_freq
a = 1  ## multiplication factor for the signal
PAo, Prefo, gAo, PNo, gNo, fo, lo= best_fit_params_o ## In case initials are taken from best fit of the open run
PA, Pref, PN, gN, f, l = [310, 305, 73, 0.18, 0.4, 0.15]
initials = [a, PA, Prefo, PNo, gNo, fo, lo]  ##initials for mcmc walkers
nsteps = 5000
signal = sig21cm  ## 21cm signal
# p0, p1, p2, p3, p4, p5, p6=S11_params
# TMeas_=Compute_Tmeas_RLC_trial(300, 300,p0, p1, p2, p3, p4, p5, p6, 70,0.3, 0.1, 0.15, req_freq)#T21
TMeas_exp = T21  ## Expected Signal
s = uncert  # 0.001 ## uncertainity

kwargs = {
    "s": s,
    "v": v,
    "seed": seed,
    "nsteps": nsteps,
    "initial_value": initials,
    "TA_exp": TMeas_exp,
    "S11_params": S11_params,
    "signal": signal,
}  # "ip_params":ip_params}

(
    full_samples_full,
    samples_full,
    best_fit_params_full,
    initial_pos,
    af_per_walker,
    af_mean,
    tau,
) = evolve_MCMC_Full(**kwargs)


# ## Generating and Saving Pickel File)

# Ask user for confirmation
parameter_names = ["a", "PA", "Pref", "PN", "gamma_N", "f", "l"]
true_params = [
    1,
    313.44495298597064,
    307.2088350626911,
    74.23509620659401,
    0.19840997630322196,
    0.4260990319365141,
    0.21653155250707493,
]
confirm = input("Do you want to save the pickle file? (y/n): ").strip().lower()
if confirm == "y":
    result = {
        "seed": seed,
        "req_freq": req_freq,
        "full_samples": full_samples_full,
        "samples": samples_full,
        "best_fit": best_fit_params_full,
        "param_names": parameter_names,
        "true_params": true_params,
        "Real Data": TMeas_exp,
        "sig21cm": sig21cm,
        "Mean Acceptance Factor": af_mean,
        "Auto corr time": tau,
    }

    with open(f"results_{seed}_a1s1_realdata.pkl", "wb") as f:  # open a text file
        pickle.dump(result, f)  # serialize the list

    f.close()
    print("Pickle File saved!")
else:
    print("Save cancelled.")

