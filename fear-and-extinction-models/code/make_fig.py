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

import multiprocessing as mp
import matplotlib.pyplot as plt

def make_old_figure(fear_stages_simulations, n_simulations, t1, t2, filename="normal"):
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
        # ax.text(-200, 3800,"A", weight="bold", fontsize=30)

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
        # ax.text(-200, 3.8,"B", weight="bold", fontsize=30)
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
        ax.set_ylim(-3,100)
        ax.set_xlim(50,tsim)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (ms)")
        #ax.get_xaxis().set_visible(False)
        ax.text(-200, 17,"C", weight="bold", fontsize=30)

        # save raster + time‐series figure
        plt.tight_layout()
        plt.savefig('fear_stages/raster_' + filename + '.png', dpi = 200)

        # plt.show()


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
    
    # TODO: remove commented lines
    processing_simulations = mp.Pool(2)
    # Map each simulation ID (0…n_simulations-1) to a worker
    processing_simulations.map(fear_stages_simulations, range(n_simulations))

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
    fig = plt.figure(constrained_layout=False, figsize=(10,10))
    gs = fig.add_gridspec(21,1)

    ax = fig.add_subplot(gs[0:10, 0])
    ax.plot(spk_ne, ID_ne, 'k.', ms=2) #all excitatory exneurons
    ax.plot(spk_ni, ID_ni + 3400, 'r.', ms=2) #inhibitory neurons
    ax.plot(spk_A, ID_A, '.', ms=4, color="tab:orange") #sub-pop A - fear neurons
    ax.plot(spk_B, ID_B + 2720 , '.', color="tab:blue", ms=4) #sub-pop B - extinction neurons

    for i,j in cs_intervals:
        ax.plot([i,j],[-100,-100],'k-', lw = 3)
    # plt.plot(tstim[nonzero_id],-100.0*np.ones(size(nonzero_id)),'.k')

    # annotate protocol phases
    # TODO change values here
    ax.text(250, -350, 'CONDITIONING', fontsize=15)
    ax.text(1250, -350, 'EXTINCTION', fontsize=15)
    ax.text(2100, -350, 'REN.', fontsize=15)
    # phase‐boundary lines
    ax.axvline(t1, ls='--', lw=2, color='black')
    ax.axvline(t2+tCTXB_dur, ls='--', lw=2, color='black')
    ax.set_ylim(-400,NE+NI)
    ax.set_xlim(50,tsim)
    ax.set_ylabel('# Neuron')
    ax.set_xlabel('Time (ms)')
    # ax.text(-250, 3800, "A", weight="bold", fontsize=30)

    ax.get_xaxis().set_visible(False)
    
    # 4b) Firing‐rate time‐series for sub-pops A & B (rows 7–9)
    ax = fig.add_subplot(gs[11:21, 0])

    ax.plot(fr_B_last[1][:-1], matlab_smooth(fr_B_last[0]*1000/(bin_f*NI), 5), lw=2)
    ax.plot(fr_A_last[1][:-1], matlab_smooth(fr_A_last[0]*1000/(bin_f*NI), 5), lw=2)

    ax.set_xlim(50, tsim)
    ax.set_ylim(-0.25, 2.3)       # match your desired frequency range
    ax.set_yticks(range(0, 3)) # adjust ticks if needed

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (ms)")
    # ax.text(-250, 2.15, "B", weight="bold", fontsize=30)

    #ax.plot(fr_BB[1][:-1], fr_BB[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_AA[1][:-1], fr_AA[0]*1000/(bin_f*NI),lw=2)
    #ax.plot(fr_IH[1][:-1], fr_IH[0]*1000/(bin_f*NI), 'r-')
    #plt.plot(fr_EX[1][:-1], fr_EX[0]*1000/(bin_f*NI), 'b-')

    # overlay CS bars & phase lines
    for i, j in cs_intervals:
        ax.plot([i, j], [0, 0], 'k-', lw=3)  # move to y=0 (or any fixed visible baseline)

    ax.axvline(t1, ls='--', lw=2, color='black')
    ax.axvline(t2 + tCTXB_dur, ls='--', lw=2, color='black')

    # save raster + time‐series figure
    plt.savefig('fear_stages/raster_' + filename + '.png', dpi = 200)

    # plt.show()

