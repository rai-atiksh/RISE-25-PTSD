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


###############################################################################
# Simulation protocols
###############################################################################



'''

protocol = 0:   simulation of spontaneous network activity (Figure 4)

protocol = 1:   dynamics of conditioning and extinction processes (Figures 5 and 6)

protocol = 3:   gamma oscillations for high network connectivity (Figure 9)

protocol = 4:   effects of connectivity, synaptic weights and delays of the
                inhibitory population on synchronization (Figures 10 and 11)
                
protocol = 5:   blockage of inhibition (Figure 12 and 13)


'''
def make_figure(renewal_fear_simulations, n_simulations, t1, t2):
    if __name__ == '__main__':
        processing_simulations = mp.Pool(3)
        results = processing_simulations.map(renewal_fear_simulations, range(n_simulations))


        #loading the data from the last simulation
        data = np.load('renewal_fear/last_simulation_data.npy', allow_pickle=True)
        spk_ni = data.item().get('spk_ni_t')
        spk_ne = data.item().get('spk_ne_t')
        spk_A = data.item().get('spk_A')
        spk_B = data.item().get('spk_B')
        ID_ni = data.item().get('ID_ni')
        ID_ne = data.item().get('ID_ne')
        ID_A = data.item().get('ID_A')
        ID_B = data.item().get('ID_B')
        cs_intervals = data.item().get('cs_intervals')

        bin_f = 15 #time bin to calculate the time series of the trigger rate of the last simulation.
        fr_IH_last = np.histogram(spk_ni, bins=np.arange(0, tsim, bin_f))
        fr_A_last = np.histogram(spk_A, bins=np.arange(0, tsim, bin_f))
        fr_B_last = np.histogram(spk_B, bins=np.arange(0, tsim, bin_f))


        ###########################################################################
        # Final results
        ###########################################################################

        # Raster plot for the last simulation
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

        ax.text(350, -350, 'CONDITIONING')
        ax.text(1650, -350, 'EXTINCTION')
        ax.text(2430, -350, 'RENEWAL')
        ax.axvline(t1, ls='--', lw=2, color='black')
        ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
        ax.set_ylim(-400,NE+NI)
        ax.set_xlim(50,tsim)
        ax.set_ylabel('# Neuron')
        ax.set_xlabel('Time (ms)')
        ax.text(-200, 3800,"A", weight="bold", fontsize=30)

        ax.get_xaxis().set_visible(False)

        ax = fig.add_subplot(gs[7:10, 0])

        ax.plot(fr_B_last[1][:-1], matlab_smooth(fr_B_last[0]*1000/(bin_f*NI), 5), lw=2)
        ax.plot(fr_A_last[1][:-1], matlab_smooth(fr_A_last[0]*1000/(bin_f*NI), 5), lw=2)

        #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
        #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

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

        ax = fig.add_subplot(gs[10:, 0])
        #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
        #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
        ax.plot(fr_IH_last[1][:-1], matlab_smooth(fr_IH_last[0]*1000/(bin_f*NI), 5), 'r-')
        #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

        for i,j in cs_intervals:
            ax.plot([i,j],[-2,-2],'k-', lw = 3)
        ax.axvline(t1, ls='--', lw=2, color='black')
        ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
        ax.set_ylim(-3,20)
        ax.set_xlim(50,tsim)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (ms)")
        #ax.get_xaxis().set_visible(False)
        ax.text(-200, 17,"C", weight="bold", fontsize=30)
        plt.tight_layout()
        plt.savefig('renewal_fear/raster.png', dpi = 200)


        ###########################################################################
        # Average firing rate,CS and CTX for each subpopulation
        ###########################################################################
        n_CS = 12 #total CS presentation

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
        plt.savefig('renewal_fear/avarages.png', dpi = 200)
        # plt.show()


# protocol = 2:   fear renewal (Figures 7 and 8)

# Normal Fear Extinction
if protocol == 1:
    os.system('mkdir renewal_fear')

    # Makes array of on/off CS input
    aux	= np.zeros(int((tCS_dur+tCS_off)/delta_tr))
    aux[:int(tCS_dur/delta_tr)] = 1
    m_array = np.concatenate([np.zeros(int(tinit/delta_tr)),np.tile(aux, nCSA),\
                            np.zeros(int(tCTX_off/delta_tr)),np.tile(aux, nCSB),\
                            np.zeros(int(tCTX_off/delta_tr)),np.tile(aux, 1)])

    mt       = TimedArray(m_array, dt=delta_tr*ms)
    stimulus = TimedArray(m_array*fCS*Hz, dt=delta_tr*ms)

    t1 = tinit+tCTXA_dur
    t2 = t1+tCTX_off
    t3 = t2+tCTXB_dur+tCTX_off
    new_input_vars={
                'cs_rate'  : 'stimulus(t)',
                'ctxA_rate': '((t>='+str(tinit)+'*ms)*(t<='+str(t1)+'*ms)+(t>='+str(t3)+'*ms))*'+str(fCTX)+'*Hz',
                'ctxB_rate': '((t>='+str(t2)+'*ms)*(t<='+str(t2+tCTXB_dur)+'*ms))*'+str(fCTX)+'*Hz'
                }
    input_vars.update(new_input_vars)
    tsim = t3 + tCS_dur + tCS_off
    tstim= np.arange(0.0, tsim, delta_tr)			# Times discretized

    n_simulations = 2 #total of simulations

    def renewal_fear_simulations(l):
        seed(100+l)
        print("Running simulations: please wait")
        print("Simulation ID: {}".format(l+1))
        start_scope()
        net, neurons, conn, Pe, Pi, spikemon, statemon, PG_cs, PG_ctx_A, PG_ctx_B, CS_e, CS_i, CTX_A, CTX_B = amygdala_net(input=True, input_vars=input_vars)
        net.run(tsim*ms, report='stdout')

        spikemon_ne, spikemon_ni, spikemon_A, spikemon_B = [spikemon[0],spikemon[1],spikemon[2],spikemon[3]]
        statemon_CS, statemon_CTX_A, statemon_CTX_B = [statemon[0], statemon[1], statemon[2]]

        ###########################################################################
        # Results
        ###########################################################################
        nonzero_id = np.nonzero(m_array)
        winsize  = int(tCS_dur/delta_tr)

        ind  = 0
        cs_intervals = []
        for i in range(nCSA+nCSB+1):
            cs_intervals.append([tstim[nonzero_id][ind],tstim[nonzero_id][ind+winsize-1]])
            ind+=winsize


        ###########################################################################
        # Average firing rate for each subpopulation
        ###########################################################################
        print("Calculating the average firing rate for each subpopulation")

        timesA = spikemon_A.t/ms
        timesB = spikemon_B.t/ms

        fr_A = []
        fr_B = []

        for i, j in cs_intervals:
            fr_A.append(np.sum((timesA>=i) & (timesA<=j))/(NA*tCS_dur/1000.0))
            fr_B.append(np.sum((timesB>=i) & (timesB<=j))/(NB*tCS_dur/1000.0))

        ###########################################################################
        # Average of CS and CTX weights
        ###########################################################################
        print("Calculating the average CS and CTX weights")

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

        #saving the data from the last simulation
        if((l+1)==30):
            np.save('renewal_fear/last_simulation_data.npy', \
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

    make_figure(renewal_fear_simulations, n_simulations, t1, t2)
