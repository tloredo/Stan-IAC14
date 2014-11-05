"""
PyStan gamma-Poisson demo for IAC Winter School on Bayesian Astrophysics

For PyStan info:

https://pystan.readthedocs.org/en/latest/getting_started.html

Created 2014-11-04 by Tom Loredo
"""

from collections import namedtuple

import numpy as np
import scipy
from scipy import stats
import matplotlib as mpl

# Pollute the namespace!
from matplotlib.pyplot import *
from scipy import *

from stanfit import StanFit

# Interactive plotting customizations:
import myplot
from myplot import close_all, csavefig
ion()
# myplot.tex_on()
csavefig.save = False


# Stan code defining a gamma-Poisson MLM for number counts
# (log N - log S) fitting.
code = """
data {
    int<lower=0> N; 
    int<lower=0> counts[N];
    real  exposures[N]; 
} 

parameters {
    real<lower=0> alpha; 
    real<lower=0> beta;
    real<lower=0> fluxes[N];
}

model {
    alpha ~ exponential(1.0);
    beta ~ gamma(0.1, 1.0);
    for (i in 1:N){
        fluxes[i] ~ gamma(alpha, beta);
        counts[i] ~ poisson(fluxes[i] * exposures[i]);
  }
}
"""


# Setup for stellar observations:
if False:
    # Define gamma dist'n parameters alpha & phi_cut:
    Jy_V0 = 3640.  # V=0 energy flux in Jy
    phi_V0 = 1.51e7 * 0.16 * Jy_V0  # V=0 photon number flux (s m^2)^{-1}
    V_cut = 24.  # V magnitude at rollover
    phi_cut = phi_V0 * 10.**(-0.4*V_cut)  # flux at rollover
    alpha = .4  # power law part has exponent alpha-1; requires alpha > 0

    # Variables describing the data sample:
    n_s = 10
    area = pi*(8.4**2 - 5**2)  # LSST primary area (m^2)
    exposures = 10.*area*ones(n_s)  # LSST single-image integration time * area
    mid = n_s//2
    exposures[mid:] *= 10  # last half use 10x default exposure

# Setup for GRB observations:
if True:
    # Define gamma dist'n parameters alpha & phi_cut:
    phi_cut = 10.  # peak flux of bright BATSE GRB, photons/s/cm^2
    alpha = .4  # power law part has exponent alpha-1; requires alpha > 0

    # Variables describing the data sample:
    n_s = 100
    area = 335.  # Single BATSE LAD effective area, cm^2
    # Fake projected areas for a triggered detector:
    areas = area*stats.uniform(loc=.5, scale=.5).rvs(n_s)
    exposures = .064*areas  # use 64 ms peak flux time scale


# Define the true flux dist'n as a gamma dist'n.
beta = 1./phi_cut  # Stan uses the inverse scale
ncdistn = stats.gamma(a=alpha, scale=phi_cut)

# Sample some source fluxes from the flux population dist'n.
fluxes = ncdistn.rvs(n_s)


# Generate observations of the flux sample.
def gen_data():
    """
    Simulate photon count data from the Poisson distribution, gathering
    the data and descriptive information in a dict as needed by Stan.
    """
    n_exp = fluxes*exposures  # expected counts for each source
    counts = stats.poisson.rvs(n_exp)
    return dict(N=n_s, exposures=exposures, counts=counts)

data = gen_data()



# Invoke Stan.
if False:
    fit = pystan.stan(model_code=code, data=data,
                    iter=1000, chains=4)

    # More iterations:
    #fit2 = pystan.stan(fit=fit1, data=schools_dat, iter=10000, chains=4)

    chains = fit.extract(permuted=True)  # return a dictionary of arrays
    alphas = chains['alpha']
    betas = chains['beta']
    flux_chains = chains['fluxes']


    ## return an array of three dimensions: iterations, chains, parameters
    a = fit.extract(permuted=False)

fit = StanFit(code, data, 4, 1000)
