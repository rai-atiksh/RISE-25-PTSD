# ----------------------------------------------------------------------------
# Contributors: Tawan T. A. Carvalho
#               Luana B. Domingos
#               Renan O. Shimoura
#               Nilton L. Kamiji
#               Vinicius Lima
#               Mauro Copelli
#               Antonio C. Roque
# ----------------------------------------------------------------------------
# References:
#
# *Context-dependent encoding of fear and extinctionmemories in a large-scale
# network model of the basal amygdala*,
# I. Vlachos, C. Herry, A. Lüthi, A. Aertsen and A. Kumar,
# PLoS Comput. Biol. 7 (2011), e1001104.
# ----------------------------------------------------------------------------
# File description:
#
# Main parameter values.
# ----------------------------------------------------------------------------

from brian2 import *


#############################################################################
# Network parameters
#############################################################################

# Source: doi:10.1523/JNEUROSCI.2700-20.2021 (Roughly 4:1)
NE = 400   #excitatory neurons
NI = 100    #inhibitory neurons 

NA = NB = int(NE*0.2) #number of excitatory neurons divided in subpopulation

# Abstraction, speeds up simulation 
# connection probabilities
pcon =  [[0.01,     # excitatory to excitatory
          0.15],    # excitatory to inhibitory
         [0.15,     # inhibitory to excitatory
          0.10]]    # inhibitory to inhibitory

#############################################################################
# Neuron parameters
#############################################################################

Vt      = -50.0*mV         # threshold
tref    = 2.0*ms           # refractory time
Ek      = -70.0*mV         # reset potential
E0      = -70.0*mV         # resting potential
Eexc    = 0.0*mV           # reversal potential for excitatory synapses
Einh    = -80.0*mV         # reversal potential for inhibitory synapses
taum    = 15.0*ms          # membrane time constant
Cm      = 250.0*pF         # membrane capacitance
Gl      = 16.7*nS          # leakage conductance

#############################################################################
# Synapse parameters
#############################################################################]
# TODO: Very quick for some reason
# https://doi.org/10.1152/jn.1996.76.3.1958
tauexc_rise  = 1*ms       #+- 0.04 
tauexc_decay = 3.6*ms     #+- 0.18
tauinh_rise  = 1*ms       #+- 0.03
tauinh_decay = 5.16*ms    #+- 0.14
# tauexc_rise  = 0.326*ms # excitatory rise time constant
# tauexc_decay = 0.326*ms # excitatory decay time constant
# tauinh_rise  = 0.326*ms # inhibitory rise time constant
# tauinh_decay = 0.326*ms # inhibitory decay time constant

# synaptic weights (in nS)
# Typically 1/2, this is fine source: (https://doi.org/10.7554/elife.89519)
wsyn = [[1.25*nS,  #wee: excitatory to excitatory
         1.25*nS], #wei: excitatory to inhibitory
        [2.5*nS,   #wie: inhibitory to excitatory
         2.5*nS]]  #wii: inhibitory to inhibitory

# TODO: Test decreased synapse weights (chemical changes)
wsyn_ratio = 0.9
wsyn_impaired = [[1.25*nS,  #wee: excitatory to excitatory
                  1.25*nS * wsyn_ratio], #wei: excitatory to inhibitory
                 [2.5*nS,   #wie: inhibitory to excitatory
                  2.5*nS * wsyn_ratio]]  #wii: inhibitory to inhibitory

wcs  = 'randn()*0.1*nS + 0.9*nS'    #from CS to all neurons
wctx = 'randn()*0.05*nS + 0.4*nS'   #from CTX to all neurons

# synaptic delay
# 1-5 ms is fine
sdelay = [['(randn()*0.1 + 2.0)*ms',
           '(randn()*0.1 + 2.0)*ms'],
          ['(randn()*0.1 + 2.0)*ms',
           '(randn()*0.1 + 2.0)*ms']]

# synaptic plasticity parameters
tauh = 10.0*ms
tauc = 10.0*ms
w_min = 0.4*nS
w_max = 4.0*nS
alpha = 2e-3
# TODO: Test Learning Rate (10%-70% of original alpha)
alpha_impaired = 0.3 * alpha
c_u = 0.35
h_u = 0.35
w_e     = 1.25*nS   # synaptic weight

#############################################################################
# Poisson background input parameters
#############################################################################
# 5/6 is higher end, Abstraction
rate_E  = 1.5*Hz    # Poisson spiking firing rate to excitatory neurons
rate_I  = 1.7*Hz    # Poisson spiking firing rate to inhibitory neurons

# TODO: Test increased background activity (between 10%-50%)
rate_impaired_ratio = 1.5
rate_E_impaired = rate_E * rate_impaired_ratio
rate_I_impaired = rate_I * rate_impaired_ratio

#############################################################################
# Defining input parameters
#############################################################################
# Abstraction 
fCS			= 500
fCTX		= 300

# Ratio of impaired CTX firing rate to normal in PTSD
# TODO: Test boost of CTX-A (Ratio)
fCTX_boosted_r = 1.1        # Small boost in acquisition
fCTX_impaired_r = 0.1       # Small boost in extinction

nCSA 		= 4				# Number of CS presentations to population A
nCSB 		= 5				# Number of CS presentations to population B
tCS_dur  	= 100.0			# CS duration in ms
tCS_off 	= 300.0			# Time in ms between two consecutive CS presentation

tCTXA_dur = nCSA*(tCS_dur+tCS_off)	# CTX_A duration in ms
tCTXB_dur = nCSB*(tCS_dur+tCS_off)	# CTX_B duration in ms
tCTX_off  = 200.0					# Time with both CTX turned off

tinit	 = 200.0									# Initial time for transient
tsim 	 = tinit + tCTXA_dur + tCTX_off + tCTXB_dur	# Total time of simulation
delta_tr = 0.1                						# Temporal resolution (ms)
nbins    = int(tsim/delta_tr)        				# Length of simulation (bins)
tstim	 = np.arange(0.0, tsim, delta_tr)			# Times discretized

input_vars={
            'cs_rate'  : 0.0*Hz,
            'ctxA_rate': 0.0*Hz,
            'ctxB_rate': 0.0*Hz,
            }

#############################################################################
# Defining simulation parameters
#############################################################################
defaultclock.dt =  delta_tr*ms                      # time step for numerical integration

#default settings for plotting
rcParams["figure.figsize"] = [10,6]
rcParams.update({'font.size': 18})
