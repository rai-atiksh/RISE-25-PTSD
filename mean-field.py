import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

class MeanValueModel:
    def __init__(self):
        pass
    
    # Initializes the decay value dictionary, the time vector and then returns them
    def initialize(self, T, dt):
        time = np.arange(0, T * dt, dt)
        alpha = {
            'BLA': 0.5,
            'CeA': 0.5,
            'ITC': 0.5,
            'IL': 0.5,
            'PL': 0.5,
            'vHPC': 0.5,
            'dHPC': 0.5
        }
        return time, alpha

    #Defines and initializes the activity arrays for the sub-regions and then returns it
    def init_activity(self, alpha, time):
        activity = {region: np.zeros(len(time)) for region in alpha}
        activity['BLA'][0] = 0.1
        activity['CeA'][0] = 0.1
        activity['ITC'][0] = 0.1
        activity['IL'][0] = 0.1
        activity['PL'][0] = 0.1
        activity['vHPC'][0] = 0.1
        activity['dHPC'][0] = 0.1
        return activity

    # Returns the dictionaries containing the synaptic weighting
    # The values were tweeked and experimented with to showcase the most biologically plausible results
    def init_weight(self):
        return {
            'vHPC_to_IL': 0.7,
            'dHPC_to_PL': 0.6,
            'vHPC_to_BLA': 0.5,
            'PL_to_BLA': 0.6,
            'IL_to_ITC': 0.7,
            'ITC_to_CeA': 0.8,
            'BLA_to_CeA': 0.6
        }

    # Runs the simulation
    def run_sim(self, T, dt, time, alpha, activity, w, ptsd_factor, I_fear, I_context_safe, I_context_threat, I_des_amp, des_active_acq, des_active_ex, des_active_re):
        # Arrays to keep track of context, fear input, and des amplitutde over time
        context_trace = np.ones(len(time))
        I_fear_trace = np.zeros(len(time))
        I_des = np.zeros(len(time))
        
        # Initialize the DES array based on when you want it active
        if des_active_acq:
            I_des[: int(len(time) / 3)] = I_des_amp
        if des_active_ex:
            I_des[int(len(time) / 3) : int(2 * len(time) / 3)] = I_des_amp
        if des_active_re:
            I_des[int(2 * len(time) / 3) : ] = I_des_amp

        # Initializing pulse parameters for fear input
        pulse_duration = int(T / 40)
        pulse_period = int(T / 12)

        # Initializing synaptic weighting(and learning rate) of ITC to CeA to reflect Hebbian plasticity
        w_ITC_to_CeA = np.zeros(len(time))
        w_ITC_to_CeA[0] = w['ITC_to_CeA']
        learning_rate = 0.01
        
        #Making the middle third of context represent the 'safe' enviornment
        context_trace[int(len(time)/3) : int(2*len(time)/3)] = 0.0

        # Initializing the fear input pulses
        for t in range(int(T / 50), len(time), pulse_period):
            I_fear_trace[t : t + pulse_duration] = I_fear

        # The actual simulation
        for t in range(1, len(time)):
            I_vHPC = I_context_safe * (1 - context_trace[t])
            I_dHPC = I_context_threat * context_trace[t]
            # Changing activitis of hippocampal areas based on context
            activity['vHPC'][t] = max(0, activity['vHPC'][t-1] + dt * (-alpha['vHPC'] * activity['vHPC'][t-1] + I_vHPC))
            activity['dHPC'][t] = max(0, activity['dHPC'][t-1] + dt * (-alpha['dHPC'] * activity['dHPC'][t-1] + I_dHPC))

            # Changing cortical areas based on des and hippocampal activity
            # IL activity is affected by ptsd here in our model
            # Note: des input can be moved to different brain regions to simulate effect of moving des stimulation to different region
            activity['IL'][t] = max(0, activity['IL'][t-1] + dt * (-alpha['IL'] * activity['IL'][t-1] + w['vHPC_to_IL'] * activity['vHPC'][t] * ptsd_factor + I_des[t]))
            activity['PL'][t] = max(0, activity['PL'][t-1] + dt * (-alpha['PL'] * activity['PL'][t-1] + w['dHPC_to_PL'] * activity['dHPC'][t]))

            # Changing BLA activity based on vHPC and also PL activities
            activity['BLA'][t] = max(0, activity['BLA'][t-1] + dt * (-alpha['BLA'] * activity['BLA'][t-1] - 
                                        w['vHPC_to_BLA'] * activity['vHPC'][t] * ptsd_factor + 
                                        w['PL_to_BLA'] * activity['PL'][t] * I_fear_trace[t] + I_fear_trace[t]))
            # Changing ITC activity based on IL and ptsd severity
            activity['ITC'][t] = max(0, activity['ITC'][t-1] + dt * (-alpha['ITC'] * activity['ITC'][t-1] + 
                                        w['IL_to_ITC'] * activity['IL'][t] * ptsd_factor + I_des[t]))

            # Amount of inhibition received by CeA from the ITC 
            inhibition = w_ITC_to_CeA[t-1] * activity['ITC'][t]
            # Changing CeA activity based on inhibition value and BLA activity
            activity['CeA'][t] = max(0, activity['CeA'][t-1] + dt * (-alpha['CeA'] * activity['CeA'][t-1] + 
                                    w['BLA_to_CeA'] * activity['BLA'][t] - inhibition))

            # Changing synaptic weight between ITC and CeA based on hebbian plasticity
            delta_w = learning_rate * activity['ITC'][t] * activity['CeA'][t] * ptsd_factor
            w_ITC_to_CeA[t] = w_ITC_to_CeA[t-1] + delta_w

        return activity, context_trace, I_fear_trace, w_ITC_to_CeA

    # Plots the graphs of the subregion activities, the context, and the fear input over time
    # Each 3rd of the graph represents fear aquisition, extinction, and renewal/association respectively
    def plot(self, time, activity, context_trace, I_fear_trace, w_ITC_to_CeA, ptsd_factor):
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        # Note: graphs with '#' after aren't necessary to plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(time, activity['CeA'], label='CeA (Fear Output)', linewidth=2)
        ax1.plot(time, activity['BLA'], label='BLA (Associative Input)', linestyle='--')
        ax1.plot(time, activity['ITC'], label='ITC (Inhibitory Gate)')
        ax1.plot(time, activity['IL'], label='IL (Extinction Controller)')
        #ax1.plot(time, activity['PL'], label='PL (Fear Promoter)')      #
        #ax1.plot(time, activity['vHPC'], label='vHPC (Safe Context)')   #
        #ax1.plot(time, activity['dHPC'], label='dHPC (Threat Context)') #
        #ax1.plot(time, w_ITC_to_CeA, label='w_ITC→CeA (Plastic Inhibition)', linestyle='--')    #
        ax1.set_title("Fear Extinction Circuit Activity (ptsd_factor = " + str(ptsd_factor) + ")", fontsize=14)
        ax1.set_ylabel("Activity", fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(time, context_trace, label='Context (0 = Safe, 1 = Threat)', color='purple')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Safe', 'Threat'])
        ax2.set_ylabel("Context", fontsize=12)
        ax2.grid(True)
        ax2.set_title("Context Over Time", fontsize=13)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(time, I_fear_trace, label='Fear Stimulus Input', color='red')
        ax3.set_xlabel("Time (seconds)", fontsize=12)
        ax3.set_ylabel("I_fear", fontsize=12)
        ax3.set_title("Fear Stimulus Over Time", fontsize=13)
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig('mean-field-out.png')
        plt.show()

    # Creating bar graphs of the difference in values of fear peak in acquisition vs renewal
    # for a normal person, person with ptsd, and a person with ptsd and applied DES at different times
    def plot_bars(self, T, dt, time, alpha, ptsd_factor, I_fear, I_context_safe, I_context_threat, I_des_amp):
        # Running 5 different models to get 5 different values of activities
        length = len(time)
        normal = MeanValueModel()
        ptsd_no_des = MeanValueModel()
        ptsd_des_acquisition = MeanValueModel()
        ptsd_des_extinction = MeanValueModel()
        ptsd_des_ex_re = MeanValueModel()

        # Initializing their activites
        activity_normal = normal.init_activity(alpha, time)
        activity_ptsd_no_des = ptsd_no_des.init_activity(alpha, time)
        activity_ptsd_des_acquisition = ptsd_des_acquisition.init_activity(alpha, time)
        activity_ptsd_des_extinction = ptsd_des_extinction.init_activity(alpha, time)
        activity_ptsd_des_ex_re = ptsd_des_ex_re.init_activity(alpha, time)
        # Initializing their weights
        w_normal = normal.init_weight()
        w_ptsd_no_des = ptsd_no_des.init_weight()
        w_ptsd_des_acquisition = ptsd_des_acquisition.init_weight()
        w_ptsd_des_extinction = ptsd_des_extinction.init_weight()
        w_ptsd_des_ex_re = ptsd_des_ex_re.init_weight()

        # Running their simulation
        activity_normal, _, _, _ = normal.run_sim(T, dt, time, alpha, activity_normal, w_normal, 1.0, I_fear,
                                                                        I_context_safe, I_context_threat, I_des_amp, 
                                                                        False, False, False)
        activity_ptsd_no_des, _, _, _ = ptsd_no_des.run_sim(T, dt, time, alpha, activity_ptsd_no_des, w_ptsd_no_des, ptsd_factor, I_fear,
                                                                        I_context_safe, I_context_threat, I_des_amp, 
                                                                        False, False, False)
        activity_ptsd_des_acquisition, _, _, _ = ptsd_des_acquisition.run_sim(T, dt, time, alpha, activity_ptsd_des_acquisition, w_ptsd_des_acquisition, ptsd_factor, I_fear,
                                                                        I_context_safe, I_context_threat, I_des_amp, 
                                                                        True, False, False)
        activity_ptsd_des_extinction, _, _, _ = ptsd_des_extinction.run_sim(T, dt, time, alpha, activity_ptsd_des_extinction, w_ptsd_des_extinction, ptsd_factor, I_fear,
                                                                        I_context_safe, I_context_threat, I_des_amp, 
                                                                        False, True, False)
        activity_ptsd_des_ex_re, _, _, _ = ptsd_des_ex_re.run_sim(T, dt, time, alpha, activity_ptsd_des_ex_re, w_ptsd_des_ex_re, ptsd_factor, I_fear,
                                                                        I_context_safe, I_context_threat, I_des_amp, 
                                                                        False, True, True)
        # Extracting CeA activity
        cea_normal = activity_normal['CeA']
        cea_ptsd_no_des = activity_ptsd_no_des['CeA']
        cea_ptsd_des_acquisition = activity_ptsd_des_acquisition['CeA']
        cea_ptsd_des_extinction = activity_ptsd_des_extinction['CeA']
        cea_ptsd_des_ex_re = activity_ptsd_des_ex_re['CeA']

        # Finding the % change in their peaks during acquisition vs renewal
        diff_normal = (max(cea_normal[int(2 * length / 3) : ]) - max(cea_normal[ : int(length / 3)])) / max(cea_normal[ : int(length / 3)]) * 100
        diff_ptsd_no_des = (max(cea_ptsd_no_des[int(2 * length / 3) : ]) - max(cea_ptsd_no_des[ : int(length / 3)])) /  max(cea_ptsd_no_des[ : int(length / 3)]) * 100
        diff_ptsd_des_acquisition = (max(cea_ptsd_des_acquisition[int(2 * length / 3) : ]) - max(cea_ptsd_des_acquisition[ : int(length / 3)])) / max(cea_ptsd_des_acquisition[ : int(length / 3)]) * 100
        diff_ptsd_des_extinction = (max(cea_ptsd_des_extinction[int(2 * length / 3) : ]) - max(cea_ptsd_des_extinction[ : int(length / 3)])) / max(cea_ptsd_des_extinction[ : int(length / 3)]) * 100
        diff_ptsd_des_ex_re = (max(cea_ptsd_des_ex_re[int(2 * length / 3) : ]) - max(cea_ptsd_des_ex_re[ : int(length / 3)])) / max(cea_ptsd_des_ex_re[ : int(length / 3)]) * 100

        # Plotting it
        catgories = ["Normal", "PTSD(No DES)", "PTSD(DES acquisition)", "PTSD(DES extinction)", "PTSD(DES extinction & renewal)"]
        values = [diff_normal, diff_ptsd_no_des, diff_ptsd_des_acquisition ,diff_ptsd_des_extinction, diff_ptsd_des_ex_re]
        plt.figure(figsize=(13, 6))
        plt.bar(catgories, values)
        plt.ylabel("% Change in fear peaks(Acquisition - Renewal)", fontsize=14)
        plt.tick_params(axis="both", labelsize=10)
        plt.title("Fear Response Drop off Across Conditions (ptsd_factor = " + str(ptsd_factor) + ")", fontsize=16)

        plt.show()



def main():
    T = 300
    dt = 0.1
    ptsd_factor = 0.5         # 0 = more severe, 1 = normal 
    I_fear = 1.0              # Fear input
    I_context_safe = 0.3      # Stimulating current used to represent perception of 'safe' envirornment
    I_context_threat = 0.5    # Stimulating current used to represent perception of 'threatening' envirornment
    I_des_amp = 0.05           # The amplitutde of the stimulating des current
    des_active_acq = False    # whether des is active during acquisition
    des_active_ex = False     # whether des is active during extinction
    des_active_re = False     # whether des is active during renewal

    model = MeanValueModel()
    time, alpha = model.initialize(T, dt)
    activity = model.init_activity(alpha, time)
    w = model.init_weight()

    activity, context_trace, I_fear_trace, w_ITC_to_CeA = model.run_sim(T, dt, time, alpha, activity, w, ptsd_factor, I_fear,
                                                                        I_context_safe, I_context_threat, I_des_amp, 
                                                                        des_active_acq, des_active_ex, des_active_re)

    model.plot(time, activity, context_trace, I_fear_trace, w_ITC_to_CeA, ptsd_factor)
    model.plot_bars(T, dt, time, alpha, ptsd_factor, I_fear, I_context_safe, I_context_threat, I_des_amp)

if __name__ == "__main__":
    main()
