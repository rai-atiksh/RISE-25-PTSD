from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

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

    #############################################################################
    return normalization_factor

def sim0():
    start_scope()
    
    ###     LIF Parameters     ###

    N = 4000        # Number of total neurons
    N_exc = 3400    # Number of excitatory neurons
    N_inh = 600     # Number of inhibatory neurons
    N_ain = int(N_exc * 0.2)  # Number of EXC neurons getting input from ctxA
    N_bin = int(N_exc * 0.2)  # Number of EXC neurons getting input from ctxB

    E_rest = -70.0 * mV         # Membrane resting potential
    E_reset = -70.0 * mV        # MEmbrane reset potential
    E_exc = 0.0 * mV
    E_inh = -80.0 * mV
    theta = -50.0 * mV          # spike threshold

    C_m = 250.0 * pF            # Membrane capitance

    tau_ref = 2.0 * ms          # refractory period
    tau_rise = 0.326 * ms       # Same for EXC and INH
    tau_decay = 0.326 * ms      # Same for EXC and INH

    wcs  = 'randn()*0.1*nS + 0.9*nS'    #from CS to all neurons
    wctx = 'randn()*0.05*nS + 0.4*nS'   #from CTX to all neurons

    ###     Differential Equations     ###
    
    # LIF function for voltage
    eqs = '''
        dv/dt = ((E_rest - v) + g_exc * (E_exc - v) + g_inh * (E_inh - v)) / C_m: volt
        
        dg_exc/dt = g_exc_aux - g_exc/tau_rise : siemens
        dg_exc_aux/dt = -g_exc_aux/tau_decay : siemens/second

        dg_inh/dt = Ginh_aux - Ginh/tau_rise : siemens
        dg_inh_aux/dt = -g_inh_aux/tau_decay : siemens/second
    '''
    
    # Synaptic conductance equations
    syn_model = ''' w   : siemens'''
    pre_exc   = '''
                Gexc_aux_post += w*Gexc_0
                '''

    pre_inh   = '''
                Ginh_aux_post += w*Ginh_0
                '''
    pre_eq    = [pre_exc, pre_inh]

    ###     Neuron Nodes    ###

    neurons = NeuronGroup(N, eqs, threshold='v>theta', reset='v=theta', refractory=tau_ref, method='exact')
    neurons.v = E_rest
    neurons.add_attribute('wcs')
    neurons.add_attribute('wctx')
    neurons.wcs  = wcs
    neurons.wctx = wctx

    pop = []
    pop.append(neurons[:N_exc])       # 3400 Excitatory Neurons
    pop.append(neurons[N_exc:])       # 600 Inhibitory Neurons

    A = pop[0][:N_ain]                 # 20% of neurons getting ctxA input
    B = pop[0][N_ain:N_ain+N_bin]      # 20% of neurons getting ctxB input

    ###     Creating synapse connections     ###
    
    g_exc_0 = beta_normalization_factor(tau_rise, tau_decay)
    g_inh_0 = beta_normalization_factor(tau_rise, tau_decay)

    g_0 = [g_exc_0, g_inh_0]

    # Probability rates of each synapse connection
    p_ee = 0.01     # EXC -> EXC
    p_ei = 0.15     # INH -> EXC
    p_ie = 0.15     # EXC -> INH
    p_ii = 0.1      # INH -> INH
    
    # Connection weights
    wsyn = [[1.25*nS,   #wee: excitatory to excitatory
         1.25*nS],      #wei: excitatory to inhibitory
        [2.5*nS,        #wie: inhibitory to excitatory
         2.5*nS]]       #wii: inhibitory to inhibitory
    
    # Connection probabilities
    pcon =  [[0.01,     # excitatory to excitatory
          0.15],        # excitatory to inhibitory
         [0.15,         # inhibitory to excitatory
          0.10]]        # inhibitory to inhibitory
    
    # 
    sdelay = [['(randn()*0.1 + 2.0)*ms',
           '(randn()*0.1 + 2.0)*ms'],
          ['(randn()*0.1 + 2.0)*ms',
           '(randn()*0.1 + 2.0)*ms']]

    conn = [] # Stores connections
    # Makes each connection
    for pre in range(0,2):
        for post in range(0,2):
            ws  = wsyn[pre][post]
            # G_0 = g_0[pre]

            conn.append(Synapses(pop[pre], pop[post], model = syn_model, on_pre=pre_eq[pre]))
            conn[-1].connect(condition='i!=j', p=pcon[pre][post])
            conn[-1].w     = '(randn()*0.1*nS + ws)'
            conn[-1].delay = sdelay[pre][post]
    
    ###     Poisson background input parameters     ###
    w_e     = 1.25*nS   # synaptic weight
    rate_E  = 5.0*Hz    # Poisson spiking firing rate to excitatory neurons
    rate_I  = 6.0*Hz    # Poisson spiking firing rate to inhibitory neurons

    # Creating poissonian background inputs
    Pe = PoissonInput(pop[0], 'g_exc_aux', 1000, rate_E, weight=w_e*g_exc_0)    # EXC pop
    Pi = PoissonInput(pop[1], 'g_exc_aux', 1000, rate_I, weight=w_e*g_exc_0)    # Is this correct? INH pop

    # Defining input parameters
    #############################################################################
    fCS			= 500.0			# CS firing rate
    fCTX		= 300.0			# CTX firing rate

    nCSA 		= 5				# Number of CS presentations to population A
    nCSB 		= 6				# Number of CS presentations to population B
    tCS_dur  	= 50.0			# CS duration in ms
    tCS_off 	= 150.0			# Time in ms between two consecutive CS presentation

    tCTXA_dur = nCSA*(tCS_dur+tCS_off)	# CTX_A duration in ms
    tCTXB_dur = nCSB*(tCS_dur+tCS_off)	# CTX_B duration in ms
    tCTX_off  = 100.0					# Time with both CTX turned off

    tinit	 = 100.0									# Initial time for transient
    tsim 	 = tinit + tCTXA_dur + tCTX_off + tCTXB_dur	# Total time of simulation
    delta_tr = 0.1                						# Temporal resolution (ms)
    nbins    = int(tsim/delta_tr)        				# Length of simulation (bins)
    tstim	 = np.arange(0.0, tsim, delta_tr)			# Times discretized
    
    t1 = tinit+tCTXA_dur
    t2 = t1+tCTX_off
    t3 = t2+tCTXB_dur+tCTX_off
    input_vars = {
        'cs_rate'  : 'stimulus(t)',
        'ctxA_rate': '((t>='+str(tinit)+'*ms)*(t<='+str(t1)+'*ms)+(t>='+str(t3)+'*ms))*'+str(fCTX)+'*Hz',
        'ctxB_rate': '((t>='+str(t2)+'*ms)*(t<='+str(t2+tCTXB_dur)+'*ms))*'+str(fCTX)+'*Hz'
    }
    # cs_rate = 0.0*Hz
    # ctxA_rate = 0.0*Hz
    # ctxB_rate = 0.0*Hz

    ###     Creating CS and CTX inputs     ###

    #initially the inputs are not active.
    PG_cs = PoissonGroup(len(neurons), rates = input_vars['cs_rate'])
    PG_ctx_A = PoissonGroup(len(A), rates = input_vars['ctxA_rate'])
    PG_ctx_B = PoissonGroup(len(B), rates = input_vars['ctxB_rate'])

    # Synaptic plasticity model equations
    
    ### TODO: understand this ###
    syn_plast  =''' delta_t : second
                '''
    pre_cs     ='''
                tcs_post = t
                c_post += c_u
                delta_t = abs(tcs_post - tctx_post)
                wcs_post += mt(t)*alpha*h_post*abs(w_max-wcs_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wcs_post)*c_post*(delta_t>100.0*ms)
                wctx_post += mt(t)*alpha*h_post*abs(w_max-wctx_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wctx_post)*c_post*(delta_t>100.0*ms)
                Gexc_aux_post += wcs_post * Gexc_0
                '''
    pre_ctx    ='''
                tctx_post = t
                h_post+= h_u
                delta_t = abs(tcs_post - tctx_post)
                wcs_post+= mt(t)*alpha*h_post*abs(w_max-wcs_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wcs_post)*c_post*(delta_t>100.0*ms)
                wctx_post+= mt(t)*alpha*h_post*abs(w_max-wctx_post)*c_post*(delta_t<100.0*ms) - mt(t)*alpha*abs(w_min-wctx_post)*c_post*(delta_t>100.0*ms)
                Gexc_aux_post += wctx_post * Gexc_0
                '''

    ###     Connecting CS and CTX to neuron populations     ###

    #connecting CS to all excitatory neurons using the plasticity rule
    CS_e = Synapses(PG_cs[:N_exc], pop[0], model = syn_plast, on_pre=pre_cs)
    CS_e.connect(j='i')
    # CS_e.m = input_vars['mt_array']

    #connecting CS to all inhibitory neurons using static synapses
    CS_i = Synapses(PG_cs[N_exc:], pop[1], model = syn_model, on_pre = pre_exc)
    CS_i.connect(j='i')
    CS_i.w = 'randn()*0.1*nS + 0.9*nS'

    #Context A connected with subpopulation A using synaptic plasticity
    CTX_A = Synapses(PG_ctx_A, A, model = syn_plast, on_pre=pre_ctx)
    CTX_A.connect(j='i')
    # CTX_A.m = input_vars['mt_array']

    #Context B connected with subpopulation B using synaptic plasticity
    CTX_B = Synapses(PG_ctx_B, B, model = syn_plast, on_pre=pre_ctx)
    CTX_B.connect(j='i')
    # CTX_B.m = input_vars['mt_array']

    ###     Creating monitors     ###
    record_weights = True
    spikemon_ne = SpikeMonitor(pop[0], record=True)
    spikemon_ni = SpikeMonitor(pop[1], record=True)
    spikemon_A  = SpikeMonitor(A, record=True)
    spikemon_B  = SpikeMonitor(B, record=True)

    spikemon = [spikemon_ne, spikemon_ni, spikemon_A, spikemon_B]

    statemon_CS    = StateMonitor(pop[0], 'wcs', record=record_weights)
    statemon_CTX_A = StateMonitor(A, 'wctx', record=record_weights)
    statemon_CTX_B = StateMonitor(B, 'wctx', record=record_weights)

    statemon = [statemon_CS, statemon_CTX_A, statemon_CTX_B]

    ### Visualization ###

        ### Run simulation ###
    run(tsim*ms)

    ### ---- Visualization ---- ###
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Raster plot
    # Combine spikes for excitatory and inhibitory neurons
    axes[0].set_title("Raster Plot: Neuron Spikes")
    axes[0].plot(spikemon_ne.t/ms, spikemon_ne.i, 'b.', markersize=2, label="Excitatory")
    axes[0].plot(spikemon_ni.t/ms, spikemon_ni.i + N_exc, 'r.', markersize=2, label="Inhibitory")

    # Highlight A and B subpopulations (subset of excitatory)
    axes[0].axhspan(0, N_ain, color='green', alpha=0.1, label="CTX-A subpop")
    axes[0].axhspan(N_ain, N_ain + N_bin, color='orange', alpha=0.1, label="CTX-B subpop")

    axes[0].set_ylabel("Neuron Index")
    axes[0].legend(loc='upper right')

    # 2. Average population firing rate (in bins)
    bin_size = 10*ms
    exc_counts, exc_edges = np.histogram(spikemon_ne.t, bins=np.arange(0, tsim*ms+bin_size, bin_size))
    inh_counts, _         = np.histogram(spikemon_ni.t, bins=np.arange(0, tsim*ms+bin_size, bin_size))

    exc_rate = exc_counts / N_exc / (bin_size)
    inh_rate = inh_counts / N_inh / (bin_size)
    t_bins = exc_edges[:-1]/ms

    axes[1].plot(t_bins, exc_rate, 'b-', label="Excitatory Rate")
    axes[1].plot(t_bins, inh_rate, 'r-', label="Inhibitory Rate")
    axes[1].set_ylabel("Firing Rate (Hz)")
    axes[1].legend()

    # 3. Show CS and CTX input timing
    axes[2].set_title("Stimulus timing")
    # CS pulses occur at known intervals
    for n in range(nCSA):
        start = tinit + n*(tCS_dur+tCS_off)
        axes[2].axvspan(start, start+tCS_dur, color='purple', alpha=0.4)
    for n in range(nCSB):
        start = tinit + tCTXA_dur + tCTX_off + n*(tCS_dur+tCS_off)
        axes[2].axvspan(start, start+tCS_dur, color='purple', alpha=0.2)
    # Context A/B periods
    axes[2].axvspan(tinit, tinit + tCTXA_dur, color='green', alpha=0.1, label="CTX-A")
    axes[2].axvspan(tinit + tCTXA_dur + tCTX_off, tsim, color='orange', alpha=0.1, label="CTX-B")

    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Stimulus")
    axes[2].legend()

    plt.tight_layout()
    plt.show()




def sim1():
    start_scope()
    
    tau_leak = 0.1
    E_rest = 0.1 * mV
    g_exc = 0.1 * mV
    g_inh = 0.1
    E_exc = 0.1
    E_inh = 0.1
    eq = '''
        dv/dt = (E_rest - v) + g_exc * (E_exc - v) + g_inh * (E_inh - v) : volt
    '''
    

if __name__ == '__main__':
    sim0()