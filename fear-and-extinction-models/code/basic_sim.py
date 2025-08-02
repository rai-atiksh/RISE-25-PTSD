# ----------------------------------------------------------------------------
# Contributors: Tawan T. A. Carvalho
#               Luana B. Domingos
#               Renan O. Shimoura
#               Nilton L. Kamiji
#               Vinicius Lima
#               Mauro COpelli
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
# Definition of the protocols used to reproduce the results for the spiking
# model.
# ----------------------------------------------------------------------------
# NOTE: Portions of the comments in this file were generated with ChatGPT.
#       See https://chat.openai.com for details.
# ----------------------------------------------------------------------------

from params     import *
from amygdala   import *

import sys
import os
import multiprocessing as mp
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

protocol = int(sys.argv[1])

seed(100) # seed of the random number generator


# Number of parallel repeat simulations
n_simulations = 2


def make_figure(fear_stages_simulations, n_simulations, t1, t2, filename="normal"):
    """
    Runs parallel simulations, loads the last sim's spike data, computes
    time-binned firing rates for the final run, and then generates & saves:
      - A raster plot of spikes
      - Time-series of firing rates for A, B, and inhibitory populations
      - Summary panels (mean ± SEM) of firing rates, CS weights, and CTX weights
    Args:
      fear_stages_simulations : function to run one protocol repeat
      n_simulations            : total repeats to average over
      t1, t2                   : protocol phase boundary times (ms)
    """
    # Only execute when script is run directly (not when imported)
    if __name__ == '__main__':
        processing_simulations = mp.Pool(2)
        # Map each simulation ID (0…n_simulations-1) to a worker
        results = processing_simulations.map(fear_stages_simulations, range(n_simulations))


        # --------------------------------------------------------------------
        # 2) Load the **last** simulation’s raw spike data
        # --------------------------------------------------------------------
        data = np.load('fear_stages/last_simulation_data_' + filename + '.npy', allow_pickle=True)
        # Extract arrays by key
        spk_ni = data.item().get('spk_ni_t')    # inhibitory spike times (ms)
        spk_ne = data.item().get('spk_ne_t')    # excitatory spike times (ms)
        spk_A = data.item().get('spk_A')        # sub-pop A spike times
        spk_B = data.item().get('spk_B')        # sub-pop B spike times
        ID_ni = data.item().get('ID_ni')        # inhibitory neuron indices
        ID_ne = data.item().get('ID_ne')        # excitatory neuron indices
        ID_A = data.item().get('ID_A')          # sub-pop A neuron indices
        ID_B = data.item().get('ID_B')          # sub-pop B neuron indices
        cs_intervals = data.item().get('cs_intervals')  # list of (start,end) CS windows

         # --------------------------------------------------------------------
        # 3) Compute time‐binned firing rates for the **last** simulation
        # --------------------------------------------------------------------
        bin_f = 15 # time bin to calculate the time series of the trigger rate of the last simulation.
        # Histogram counts in each bin for each population
        fr_IH_last = np.histogram(spk_ni, bins=np.arange(0, tsim, bin_f))
        fr_A_last = np.histogram(spk_A, bins=np.arange(0, tsim, bin_f))
        fr_B_last = np.histogram(spk_B, bins=np.arange(0, tsim, bin_f))


        ###########################################################################
        # Final results
        ###########################################################################

        # --------------------------------------------------------------------
        # 4) Raster & rate‐time‐series plot for the **last** run
        # --------------------------------------------------------------------
        fig = plt.figure(constrained_layout=False, figsize=(15,10))
        gs = fig.add_gridspec(13,1)

        ax = fig.add_subplot(gs[0:7, 0])
        ax.plot(spk_ne, ID_ne, 'k.', ms=2) #all excitatory exneurons
        ax.plot(spk_ni, ID_ni + 3400, 'r.', ms=2) #inhibitory neurons
        ax.plot(spk_A, ID_A, '.', ms=4, color="tab:orange") #sub-pop A - fear neurons
        ax.plot(spk_B, ID_B + 2720 , '.', color="tab:blue", ms=4) #sub-pop B - extinction neurons

        for i,j in cs_intervals:
            ax.plot([i,j],[-100,-100],'k-', lw = 3)
        # plt.plot(tstim[nonzero_id],-100.0*np.ones(size(nonzero_id)),'.k')

        # annotate protocol phases
        ax.text(350, -350, 'CONDITIONING')
        ax.text(1650, -350, 'EXTINCTION')
        ax.text(2430, -350, 'RENEWAL')
        # phase‐boundary lines
        ax.axvline(t1, ls='--', lw=2, color='black')
        ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
        ax.set_ylim(-400,NE+NI)
        ax.set_xlim(50,tsim)
        ax.set_ylabel('# Neuron')
        ax.set_xlabel('Time (ms)')
        ax.text(-200, 3800,"A", weight="bold", fontsize=30)

        ax.get_xaxis().set_visible(False)
        
        # 4b) Firing‐rate time‐series for sub-pops A & B (rows 7–9)
        ax = fig.add_subplot(gs[7:10, 0])

        ax.plot(fr_B_last[1][:-1], matlab_smooth(fr_B_last[0]*1000/(bin_f*NI), 5), lw=2)
        ax.plot(fr_A_last[1][:-1], matlab_smooth(fr_A_last[0]*1000/(bin_f*NI), 5), lw=2)

        #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
        #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

        # overlay CS bars & phase lines
        for i,j in cs_intervals:
            ax.plot([i,j],[-0.5,-0.5],'k-', lw = 3)
        ax.axvline(t1, ls='--', lw=2, color='black')
        ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
        ax.set_xlim(50,tsim)
        ax.set_ylim(-0.7,4.5)
        ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')

        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (ms)")
        ax.text(-200, 3.8,"B", weight="bold", fontsize=30)
        ax.get_xaxis().set_visible(False)

        # 4c) Firing‐rate time‐series for inhibitory pop (rows 10–12)
        ax = fig.add_subplot(gs[10:, 0])
        #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
        ax.plot(fr_IH_last[1][:-1], matlab_smooth(fr_IH_last[0]*1000/(bin_f*NI), 5), 'r-')
        #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

        # overlay CS bars & phase lines
        for i,j in cs_intervals:
            ax.plot([i,j],[-2,-2],'k-', lw = 3)
        ax.axvline(t1, ls='--', lw=2, color='black')
        ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
        ax.set_ylim(-3,30)
        ax.set_yticks(range(0, 31, 10))
        ax.set_xlim(50,tsim)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (ms)")
        #ax.get_xaxis().set_visible(False)
        ax.text(-200, 17,"C", weight="bold", fontsize=30)

        # save raster + time‐series figure
        plt.tight_layout()
        plt.savefig('fear_stages/raster_' + filename + '.png', dpi = 200)


        # --------------------------------------------------------------------
        # 5) Summary figure: mean ± SEM across all repeats
        # --------------------------------------------------------------------
        n_CS = nCSA + nCSB + 1 #total CS presentation

        fig = plt.figure(constrained_layout=True, figsize=(7,10))
        gs = fig.add_gridspec(6,1)

        ax = fig.add_subplot(gs[0:2, 0])
        ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[0], yerr=np.std(results, axis=0)[0], fmt='s', ms=10, capsize=3, label = r"$pop_A$", color=plt.cm.tab10(1))
        ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[1], yerr=np.std(results, axis=0)[1], fmt='o', ms=10, capsize=3, label = r"$pop_B$", color=plt.cm.tab10(0))
        ax.axvline(5.5, ls='--', lw=2, color='black')
        ax.axvline(11.5, ls='--', lw=2, color='black')

        ax.set_xticks(range(n_CS+1))
        ax.set_ylim(-0.2,5)
        ax.text(0.8, 4.5, 'CONDITIONING')
        ax.text(6.9, 4.5, 'EXTINCTION')
        ax.text(11.9, 3.5, 'RENEWAL', horizontalalignment='center', verticalalignment='center', rotation=270)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_xlabel('CS presentations')
        legend = ax.legend(bbox_to_anchor=(0.25,0.87), fontsize=15)
        legend.get_frame().set_alpha(0)
        ax.get_xaxis().set_visible(False)
        ax.text(-2, 4.4,"A", weight="bold", fontsize=30)


        ax = fig.add_subplot(gs[2:4, 0])
        ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[3], yerr=np.std(results, axis=0)[3], fmt='o', ms=10, capsize=3)
        ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[2], yerr=np.std(results, axis=0)[2], fmt='s', ms=10, capsize=3)
        ax.axvline(5.5, ls='--', lw=2, color='black')
        ax.axvline(11.5, ls='--', lw=2, color='black')

        ax.set_xticks(range(n_CS+1))
        ax.set_ylabel('CS weights (nS)')
        ax.set_xlabel('CS presentations')
        ax.set_ylim(0,3)
        ax.get_xaxis().set_visible(False)
        ax.text(-2, 2.7,"B", weight="bold", fontsize=30)


        ax = fig.add_subplot(gs[4:6, 0])
        ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[5], yerr=np.std(results, axis=0)[5], fmt='o', ms=10, capsize=3)
        ax.errorbar(range(1,n_CS+1),np.mean(results, axis=0)[4], yerr=np.std(results, axis=0)[4], fmt='s', ms=10, capsize=3)
        ax.axvline(5.5, ls='--', lw=2, color='black')
        ax.axvline(11.5, ls='--', lw=2, color='black')

        ax.set_xticks(range(n_CS+1))
        ax.set_ylabel('HPC weights (nS)')
        ax.set_xlabel('CS presentations')
        ax.text(-2, 2.7,"C", weight="bold", fontsize=30)
        ax.set_ylim(0,3)

        # save summary figure
        plt.savefig('fear_stages/average_' + filename + '.png', dpi = 200)
        # plt.show()


# ----------------------------------------------------------------------------
# Normal Fear Extinction (Protocol 1)
# ----------------------------------------------------------------------------

if protocol == 1:
    # Create output directory for this protocol
    os.system('mkdir fear_stages')
    filename = 'normal'

    # Build a single array 'aux' that represents one CS on/off cycle:
    #   1 for duration tCS_dur, then 0 for duration tCS_off
    aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
    aux[:int(tCS_dur/delta_tr)] = 1

    # Concatenate the stimulus pattern:
    #   1) initial no-stim period of length tinit
    #   2) nCSA repetitions of aux (CS + CTX-A on/off)
    #   3) no-stim gap (CTX off)
    #   4) nCSB repetitions of aux (CS + CTX-B on/off)
    #   5) another no-stim gap
    #   6) one final aux cycle
    m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSA),\
                            np.zeros(int(tCTX_off/delta_tr)),np.tile(aux, nCSB),\
                            np.zeros(int(tCTX_off/delta_tr)),np.tile(aux, 1)])
    
    # Wrap the raw on/off array into Brian2 TimedArray objects:
    #   - mt: unitless 0/1; stimulus: firing rate (scaled by fCS)
    mt       = TimedArray(m_array, dt=delta_tr*ms)
    stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

    # Compute key time‐points for the protocol
    t1 = tinit+tCTXA_dur            # end of first CTX A period
    t2 = t1+tCTX_off                # start of CTX B period
    t3 = t2+tCTXB_dur+tCTX_off      # Start of renewal


    # Prepare a dict of input‐rate expressions for the amygdala network
    new_input_vars={
                # CS neurons fire at 'stimulus' rate throughout
                'cs_rate'  : 'stimulus(t)',
                # CTX A active during initial CS A & final renewal window
                'ctxA_rate': '((t>='+str(tinit)+'*ms)*(t<='+str(t1)+'*ms)+(t>='+str(t3)+'*ms))*'+str(fCTX)+'*Hz',
                # CTX B active only in middle extinction window
                'ctxB_rate': '((t>='+str(t2)+'*ms)*(t<='+str(t2+tCTXB_dur)+'*ms))*'+str(fCTX)+'*Hz'
                }

    # Total simulation time is end of last CS on/off cycle
    tsim = t3 + tCS_dur + tCS_off
    tstim= np.arange(0.0, tsim, delta_tr)

    def fear_stages_simulations(l):
        """
        Repeats the conditioning-extinction-renewal protocol.
        l : integer seed offset (so each sim has distinct RNG)
        Returns arrays of firing rates and synaptic weights.
        """

        seed(99+l)
        print("Running simulations: please wait")
        print("Simulation ID: {}".format(l+1))

        # (Re)initialize Brian2 network & objects
        start_scope()
        net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = \
            amygdala_net(input=True, input_vars=new_input_vars, pcon=pcon, wsyn=wsyn, sdel=sdelay, PTSD=False, record_weights=True)
        # Run the full protocol
        net.run(tsim*ms, report='stdout')

        # Unpack spike monitors for each population
        spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
        statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

        ###########################################################################
        # Results Analysis
        ###########################################################################
        
        # 1) Identify the time windows of each CS presentation
        nonzero_id = np.nonzero(m_array)
        winsize  = int(tCS_dur/delta_tr)
        ind  = 0
        cs_intervals = []
        for i in range(nCSA+nCSB+1):
            cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
            ind+=winsize

        # 2) Compute average firing rates of A & B populations during each CS
        
        print("Calculating the average firing rate for each subpopulation")
        
        timesA = spikemon_A.t/ms
        timesB = spikemon_B.t/ms

        fr_A = []
        fr_B = []

        for i, j in cs_intervals:
            fr_A.append(np.sum((timesA>=i) & (timesA<=j))/(NA*tCS_dur/1000.0))
            fr_B.append(np.sum((timesB>=i) & (timesB<=j))/(NB*tCS_dur/1000.0))

        
        # 3) Compute mean CS & CTX synaptic weights over time
        print("Calculating the average CS and CTX weights")
        
        # Extract and average weights from state monitors
        wCS_A = np.array(statemon_CS.wcs[:NA])
        wCS_B = np.array(statemon_CS.wcs[-NB:])

        wCS_A = wCS_A.mean(axis=0)
        wCS_B = wCS_B.mean(axis=0)

        wCS_A = wCS_A[nonzero_id]
        wCS_B = wCS_B[nonzero_id]

        #CTX
        wCTX_A = np.array(statemon_CTX_A.wctx)
        wCTX_B = np.array(statemon_CTX_B.wctx)

        wCTX_A = wCTX_A.mean(axis=0)
        wCTX_B = wCTX_B.mean(axis=0)

        wCTX_A = wCTX_A[nonzero_id]
        wCTX_B = wCTX_B[nonzero_id]

        # 4) Average weights in each CS window
        CS_A = []
        CS_B = []
        CTX_A = []
        CTX_B = []
        aux = 0

        for i in range(nCSA+nCSB+1):
            CS_A.append(wCS_A[aux:aux+winsize].mean())
            CS_B.append(wCS_B[aux:aux+winsize].mean())

            CTX_A.append(wCTX_A[aux:aux+winsize].mean())
            CTX_B.append(wCTX_B[aux:aux+winsize].mean())

            aux+=winsize

        # 5) Optionally save the last simulation's raw data
        if((l+1)==n_simulations):
            np.save('fear_stages/last_simulation_data_' + filename + '.npy', \
                    {'spk_ni_t': spikemon_ni.t/ms,
                    'spk_ne_t': spikemon_ne.t/ms,
                    'spk_A': spikemon_A.t/ms,
                    'spk_B': spikemon_B.t/ms,
                    'ID_ni': np.array(spikemon_ni.i),
                    'ID_ne': np.array(spikemon_ne.i),
                    'ID_A': np.array(spikemon_A.i),
                    'ID_B': np.array(spikemon_B.i),
                    'cs_intervals': np.array(cs_intervals)})

        return(fr_A, fr_B, CS_A/nS, CS_B/nS, CTX_A/nS, CTX_B/nS)

    make_figure(fear_stages_simulations, n_simulations, t1, t2, filename)

# ----------------------------------------------------------------------------
# PTSD Fear Extinction (Protocol 2)
# ----------------------------------------------------------------------------
# Changes:
# 1. Uses alpha_impaired for fear extinction neuron plasticity TODO: Source
#       - Decreases population b neuron plasticity and activity in E
# 2. Increased background activity on both (consistent with https://doi.org/10.1002/hbm.23886)
#       - Increases activity of both populations, but a larger relative increase in Fear pathway
# 3. Increase CTX-A activity throughout fear extinction stage to represent impaired LI? activity
#       - Inhibits decrease of fear pathway during neutral contexts
#       - Makes Renewal worsen the situation more. TODO: source that it gets worse?
#       - Also increases during acquisition stages
# 4. TODO: Excitation–Inhibition (E/I) imbalance
#       - In practice, one can lower the strength of active interneurons in the amygdala network.
#       

elif protocol == 2: 
    # Create output directory for this protocol
    os.system('mkdir fear_stages')
    filename = 'PTSD'

    # Build a single array 'aux' that represents one CS on/off cycle:
    #   1 for duration tCS_dur, then 0 for duration tCS_off
    aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
    aux[:int(tCS_dur/delta_tr)] = 1

    # Concatenate the stimulus pattern:
    #   1) initial no-stim period of length tinit
    #   2) nCSA repetitions of aux (CS + CTX-A on/off)
    #   3) no-stim gap (CTX off)
    #   4) nCSB repetitions of aux (CS + CTX-B on/off)
    #   5) another no-stim gap
    #   6) one final aux cycle
    m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),\
                              np.tile(aux, nCSA),\
                              np.zeros(int(tCTX_off/delta_tr)),\
                              np.tile(aux, nCSB),\
                              np.zeros(int(tCTX_off/delta_tr)),\
                              np.tile(aux, 1)])
    
    # Wrap the raw on/off array into Brian2 TimedArray objects:
    #   - mt: unitless 0/1; stimulus: firing rate (scaled by fCS)
    mt       = TimedArray(m_array, dt=delta_tr*ms)
    stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

    # Compute key time‐points for the protocol
    t1 = tinit+tCTXA_dur            # end of first CTX A period
    t2 = t1+tCTX_off                # start of CTX B period
    t3 = t2+tCTXB_dur+tCTX_off      # end of second CTX B period


    # Prepare a dict of input‐rate expressions for the amygdala network
    new_input_vars={
                # CS neurons fire at 'stimulus' rate throughout
                'cs_rate'  : 'stimulus(t)',
                # CTX A active during initial CS A & final renewal window
                # ((init<=t<=t1) + 
                # (t3<=init) +
                # (t2<=t<=t3) * fCTX * Hz
                'ctxA_rate': '((t>='+str(tinit)+'*ms)*(t<='+str(t1)+'*ms)*fCTX_boosted_r+ \
                             (t>='+str(t3)+'*ms)+ \
                             (t>='+str(t2)+'*ms)*(t<='+str(t2+tCTXB_dur)+'*ms)*fCTX_impaired_r)' + '*'+str(fCTX)+'*Hz', 
                # CTX B active only in middle extinction window
                'ctxB_rate': '((t>='+str(t2)+'*ms)*(t<='+str(t2+tCTXB_dur)+'*ms))*'+str(fCTX)+'*Hz'
                }

    # Total simulation time is end of last CS on/off cycle
    tsim = t3 + tCS_dur + tCS_off
    tstim= np.arange(0.0, tsim, delta_tr)

    def fear_stages_simulations(l):
        """
        Repeats the conditioning-extinction-renewal protocol.
        l : integer seed offset (so each sim has distinct RNG)
        Returns arrays of firing rates and synaptic weights.
        """

        seed(99+l)
        print("Running simulations: please wait")
        print("Simulation ID: {}".format(l+1))

        # (Re)initialize Brian2 network & objects
        start_scope()
        net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = \
            amygdala_net(input=True, input_vars=new_input_vars,pcon=pcon, wsyn=wsyn, sdel=sdelay,PTSD=True,record_weights=True)
        # Run the full protocol
        net.run(tsim*ms, report='stdout')

        # Unpack spike monitors for each population
        spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
        statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

        ###########################################################################
        # Results Analysis
        ###########################################################################
        
        # 1) Identify the time windows of each CS presentation
        nonzero_id = np.nonzero(m_array)
        winsize  = int(tCS_dur/delta_tr)
        ind  = 0
        cs_intervals = []
        for i in range(nCSA+nCSB+1):
            cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
            ind+=winsize

        # 2) Compute average firing rates of A & B populations during each CS
        
        print("Calculating the average firing rate for each subpopulation")
        
        timesA = spikemon_A.t/ms
        timesB = spikemon_B.t/ms

        fr_A = []
        fr_B = []

        for i, j in cs_intervals:
            fr_A.append(np.sum((timesA>=i) & (timesA<=j))/(NA*tCS_dur/1000.0))
            fr_B.append(np.sum((timesB>=i) & (timesB<=j))/(NB*tCS_dur/1000.0))

        
        # 3) Compute mean CS & CTX synaptic weights over time
        print("Calculating the average CS and CTX weights")
        
        # Extract and average weights from state monitors
        wCS_A = np.array(statemon_CS.wcs[:NA])
        wCS_B = np.array(statemon_CS.wcs[-NB:])

        wCS_A = wCS_A.mean(axis=0)
        wCS_B = wCS_B.mean(axis=0)

        wCS_A = wCS_A[nonzero_id]
        wCS_B = wCS_B[nonzero_id]

        #CTX
        wCTX_A = np.array(statemon_CTX_A.wctx)
        wCTX_B = np.array(statemon_CTX_B.wctx)

        wCTX_A = wCTX_A.mean(axis=0)
        wCTX_B = wCTX_B.mean(axis=0)

        wCTX_A = wCTX_A[nonzero_id]
        wCTX_B = wCTX_B[nonzero_id]

        # 4) Average weights in each CS window
        CS_A = []
        CS_B = []
        CTX_A = []
        CTX_B = []
        aux = 0

        for i in range(nCSA+nCSB+1):
            CS_A.append(wCS_A[aux:aux+winsize].mean())
            CS_B.append(wCS_B[aux:aux+winsize].mean())

            CTX_A.append(wCTX_A[aux:aux+winsize].mean())
            CTX_B.append(wCTX_B[aux:aux+winsize].mean())

            aux+=winsize

        # 5) Optionally save the last simulation's raw data
        if((l+1)==n_simulations):
            np.save('fear_stages/last_simulation_data_' + filename + '.npy', \
                    {'spk_ni_t': spikemon_ni.t/ms,
                    'spk_ne_t': spikemon_ne.t/ms,
                    'spk_A': spikemon_A.t/ms,
                    'spk_B': spikemon_B.t/ms,
                    'ID_ni': np.array(spikemon_ni.i),
                    'ID_ne': np.array(spikemon_ne.i),
                    'ID_A': np.array(spikemon_A.i),
                    'ID_B': np.array(spikemon_B.i),
                    'cs_intervals': np.array(cs_intervals)})

        return(fr_A, fr_B, CS_A/nS, CS_B/nS, CTX_A/nS, CTX_B/nS)

    make_figure(fear_stages_simulations, n_simulations, t1, t2, 'PTSD')

