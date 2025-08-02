# ----------------------------------------------------------------------------
# NOTE: Portions of the comments in this file were generated with ChatGPT.
#       See https://chat.openai.com for details.
# ----------------------------------------------------------------------------

import numpy as np
from scipy import signal

from params     import *
from models_eq  import *

#############################################################################
# Beta function normalization factor
#############################################################################
# Compute normalization factor for a double-exponential (beta) synaptic kernel
# - If tau_rise == tau_decay, defaults to the alpha-function normalization
# - Ensures the peak amplitude of the kernel is 1 before scaling by synaptic weight
def beta_normalization_factor(tau_rise, tau_decay):
    numeric_limit = 1e-16*ms
    # difference between rise and decay time constants; used to determine whether use alpha or beta functions
    tau_diff = tau_decay - tau_rise

    # auxiliary parameters for beta function
    peak_value = 0.0

    if abs(tau_diff) > numeric_limit:  # tau_rise != tau_decay; use beta function
        # time to peak
        t_peak = tau_decay * tau_rise * np.log( tau_decay / tau_rise ) / tau_diff
        # peak_value of the beta function (difference of exponentials)
        peak_value = np.exp( -t_peak / tau_decay ) - np.exp( -t_peak / tau_rise )

    if abs(peak_value) < numeric_limit/ms: # tau_rise == tau_decay; use alpha function
        normalization_factor = np.exp(1) / tau_decay
    else: # tau_rise != tau_decay; use beta function
        normalization_factor = (1. / tau_rise - 1. / tau_decay ) / peak_value

    return normalization_factor

# Precompute baseline normalization factors for excitatory and inhibitory synapses
Gexc_0 = beta_normalization_factor(tauexc_rise, tauexc_decay)
Ginh_0 = beta_normalization_factor(tauinh_rise, tauinh_decay)

#############################################################################
# Network structure with all inputs connected
#############################################################################
def amygdala_net(input=False, input_vars=input_vars, pcon=pcon, wsyn=wsyn, sdel=sdelay, PTSD=False, record_weights=True):
    """
    Build the amygdala network:
      - Leaky integrate-and-fire neurons (exc + inh)
      - Recurrent synapses with plasticity or static rules
      - Background Poisson noise
      - Optional CS and context (CTX) Poisson inputs with plastic synapses
    Returns all network objects and monitors for use in simulations.
    """
    #############################################################################
    # Neuron group definitions
    #############################################################################
    # Combined excitatory (NE) + inhibitory (NI) neurons
    neurons = NeuronGroup(
        NE+NI,
        eq_LIF,                 # LIF Equations
        threshold='v>Vt',       # Spike Condition
        reset=reset_LIF,        # Reset Dynamics
        refractory=tref,        # Absolute Refractory Period 
        method='rk4'            # Calculation Method
    )

    # Initialize membrane potential around resting plus noise
    neurons.v    = 'E0 + randn()*3.0*mV'
    # Initialize synaptic weight state variables
    neurons.wcs  = wcs
    neurons.wctx = wctx

    # Split into excitatory and inhibitory populations
    pop = []
    pop.append(neurons[0:NE])   # excitatory subset
    pop.append(neurons[NE:])    #inhibitory neurons

    # Further split excitatory group into subpopulations A and B
    pop_A = pop[0][:NA]     #subneuronsulation A - 20% of excitatory neurons
    pop_B = pop[0][-NB:]    #subneuronsulation B - 20% of excitatory neurons

    #############################################################################
    # Recurrent synapse creation
    #############################################################################
    # Use precomputed normalization factors for synaptic kernels
    G_0    = [Gexc_0, Ginh_0]

    conn = []  # list to hold all Synapses objects
    for pre in range(0,2):          # loop over pre populations: 0=exc, 1=inh
        for post in range(0,2):     # loop over post populations
            # baseline weights
            if PTSD == True and post == 0:
                ws = wsyn_impaired[pre][post]
            else:
                ws  = wsyn[pre][post]
            g_0 = G_0[pre]          # normalization factor

            conn.append(Synapses(
                        pop[pre], pop[post],
                        model = syn_model,      # conductance-based model
                        on_pre=pre_eq[pre])     # pre-spike update rule
                    )
            conn[-1].connect(condition='i!=j', p=pcon[pre][post])   # random connectivity
            conn[-1].w     = '(randn()*0.1*nS + ws)'                # initial weights
            conn[-1].delay = sdel[pre][post]                        # synaptic delay

    ###########################################################################
	# Creating poissonian background inputs
	###########################################################################
    # Excitatory background onto excitatory and INH pop
    if PTSD == True:
        Pe = PoissonInput(pop[0], 'Gexc_aux', 1000, rate_E_impaired, weight=w_e*Gexc_0)
        Pi = PoissonInput(pop[1], 'Gexc_aux', 1000, rate_I_impaired, weight=w_e*Gexc_0)
    else:
        Pe = PoissonInput(pop[0], 'Gexc_aux', 1000, rate_E, weight=w_e*Gexc_0)
        Pi = PoissonInput(pop[1], 'Gexc_aux', 1000, rate_I, weight=w_e*Gexc_0)

    # Optional CS and CTX inputs
    if input==True:
        #############################################################################
        # Poisson stimulus groups for CS and context
        #############################################################################
        #initially the inputs are not active.
        PG_cs    = PoissonGroup(len(neurons), rates = input_vars['cs_rate'])
        PG_ctx_A = PoissonGroup(len(pop_A), rates = input_vars['ctxA_rate'])
        PG_ctx_B = PoissonGroup(len(pop_B), rates = input_vars['ctxB_rate'])

        ###########################################################################
    	# Connecting CS and CTX to neuron populations
    	###########################################################################
        # Plastic CS->E synapses (one-to-one mapping)
        CS_e = Synapses(PG_cs[:NE], pop[0], model = syn_plast, on_pre=pre_cs)
        CS_e.connect(j='i')
        # CS_e.m = input_vars['mt_array']

        #connecting CS to all inhibitory neurons using static synapses
        CS_i = Synapses(PG_cs[NE:], pop[1], model = syn_model, on_pre = pre_exc)
        CS_i.connect(j='i')
        CS_i.w = 'randn()*0.1*nS + 0.9*nS'

        #############################################################################
        # Connect context A & B groups with plastic synapses
        #############################################################################
        CTX_A = Synapses(PG_ctx_A, pop_A, model = syn_plast, on_pre=pre_ctx)
        CTX_A.connect(j='i')
        # CTX_A.m = input_vars['mt_array']

        #Context B connected with subpopulation B using synaptic plasticity
        CTX_B = Synapses(PG_ctx_B, pop_B, model = syn_plast, on_pre=pre_ctx)
        if (PTSD == True): 
            CTX_B.pre.code = pre_ctx_impaired
        CTX_B.connect(j='i')
        # CTX_B.m = input_vars['mt_array']

    else:
        # If no external inputs, set placeholders
        PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = [],[],[],[],[],[],[]

	###########################################################################
	# Monitors for spikes and synaptic weights
	###########################################################################
    spikemon_ne = SpikeMonitor(pop[0], record=True)  # all excitatory
    spikemon_ni = SpikeMonitor(pop[1], record=True)  # all inhibitory
    spikemon_A  = SpikeMonitor(pop_A, record=True)   # subpop A
    spikemon_B  = SpikeMonitor(pop_B, record=True)   # subpop B

    spikemon = [spikemon_ne, spikemon_ni, spikemon_A, spikemon_B]

    statemon_CS    = StateMonitor(pop[0], 'wcs', record=record_weights)
    statemon_CTX_A = StateMonitor(pop_A, 'wctx', record=record_weights)
    statemon_CTX_B = StateMonitor(pop_B, 'wctx', record=record_weights)

    statemon = [statemon_CS, statemon_CTX_A, statemon_CTX_B]

	###########################################################################
	# Assemble network and return handles
	###########################################################################
    net = Network(collect())
    net.add(neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B)
    return(net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B)

#############################################################################
# Function to smooth curves in graphs
#############################################################################
def matlab_smooth(data, window_size):
    # assumes the data is one dimensional
    n = data.shape[0]
    c = signal.lfilter(np.ones(window_size)/window_size, 1, data)
    idx_begin = range(0, window_size - 2)
    cbegin = data[idx_begin].cumsum()
    # select every second elemeent and divide by their index
    cbegin = cbegin[0::2] / range(1, window_size - 1, 2)
    # select the list backwards
    idx_end = range(n-1, n-window_size + 1, -1)
    cend = data[idx_end].cumsum()
    # select every other element until the end backwards
    cend = cend[-1::-2] / (range(window_size - 2, 0, -2))
    c = np.concatenate([cbegin, c[window_size-1:], cend])
    return c
