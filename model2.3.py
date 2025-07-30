import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

class MeanValueModel:
    def __init__(self):
        pass

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

    def run_sim(self, T, dt, time, alpha, activity, w, ptsd_factor, I_fear, I_context_safe, I_context_threat, I_dbs_amp, dbs_active):
        context_trace = np.ones(len(time))
        I_fear_trace = np.zeros(len(time))
        I_dbs = np.zeros(len(time))
        if dbs_active:
            I_dbs[int(len(time) / 3) : ] = I_dbs_amp

        pulse_duration = int(T / 60)
        pulse_period = int(T / 10)

        w_ITC_to_CeA = np.zeros(len(time))
        w_ITC_to_CeA[0] = w['ITC_to_CeA']
        learning_rate = 0.01

        context_trace[int(len(time)/3) : int(2*len(time)/3)] = 0.0
        for t in range(0, len(time), pulse_period):
            I_fear_trace[t : t + pulse_duration] = I_fear

        for t in range(1, len(time)):
            I_vHPC = I_context_safe if context_trace[t] == 0.0 else 0
            I_dHPC = I_context_threat if context_trace[t] == 1.0 else 0

            activity['vHPC'][t] = max(0, activity['vHPC'][t-1] + dt * (-alpha['vHPC'] * activity['vHPC'][t-1] + I_vHPC))
            activity['dHPC'][t] = max(0, activity['dHPC'][t-1] + dt * (-alpha['dHPC'] * activity['dHPC'][t-1] + I_dHPC))

            activity['IL'][t] = max(0, activity['IL'][t-1] + dt * (-alpha['IL'] * activity['IL'][t-1] + w['vHPC_to_IL'] * activity['vHPC'][t] * ptsd_factor + I_dbs[t]))
            activity['PL'][t] = max(0, activity['PL'][t-1] + dt * (-alpha['PL'] * activity['PL'][t-1] + w['dHPC_to_PL'] * activity['dHPC'][t]))

            activity['BLA'][t] = max(0, activity['BLA'][t-1] + dt * (-alpha['BLA'] * activity['BLA'][t-1] + 
                                        w['vHPC_to_BLA'] * activity['vHPC'][t] + 
                                        w['PL_to_BLA'] * activity['PL'][t] + I_fear_trace[t]))

            activity['ITC'][t] = max(0, activity['ITC'][t-1] + dt * (-alpha['ITC'] * activity['ITC'][t-1] + 
                                        w['IL_to_ITC'] * activity['IL'][t] * ptsd_factor + I_dbs[t]))

            inhibition = w_ITC_to_CeA[t-1] * activity['ITC'][t]
            activity['CeA'][t] = max(0, activity['CeA'][t-1] + dt * (-alpha['CeA'] * activity['CeA'][t-1] + 
                                    w['BLA_to_CeA'] * activity['BLA'][t] - inhibition))

            delta_w = learning_rate * activity['ITC'][t] * activity['CeA'][t]
            w_ITC_to_CeA[t] = w_ITC_to_CeA[t-1] + delta_w

        return activity, context_trace, I_fear_trace, w_ITC_to_CeA

    def plot(self, time, activity, context_trace, I_fear_trace, w_ITC_to_CeA, ptsd_factor):
        # Note: graphs with '#' after aren't necessary to plot
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(time, activity['CeA'], label='CeA (Fear Output)', linewidth=2)
        ax1.plot(time, activity['BLA'], label='BLA (Associative Input)', linestyle='--')
        ax1.plot(time, activity['ITC'], label='ITC (Inhibitory Gate)')
        ax1.plot(time, activity['IL'], label='IL (Extinction Controller)')
        ax1.plot(time, activity['PL'], label='PL (Fear Promoter)')      #
        ax1.plot(time, activity['vHPC'], label='vHPC (Safe Context)')   #
        ax1.plot(time, activity['dHPC'], label='dHPC (Threat Context)') #
        ax1.plot(time, w_ITC_to_CeA, label='w_ITC→CeA (Plastic Inhibition)', linestyle='--')    #
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
        ax3.set_xlabel("Time (arbitrary units)", fontsize=12)
        ax3.set_ylabel("I_fear", fontsize=12)
        ax3.set_title("Fear Stimulus Over Time", fontsize=13)
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

def main():
    T = 300
    dt = 0.1
    ptsd_factor = 0
    I_fear = 1.0
    I_context_safe = 0.6
    I_context_threat = 0.6
    I_dbs_amp = 0.2
    dbs_active = True

    model = MeanValueModel()
    time, alpha = model.initialize(T, dt)
    activity = model.init_activity(alpha, time)
    w = model.init_weight()

    activity, context_trace, I_fear_trace, w_ITC_to_CeA = model.run_sim(T, dt, time, alpha, activity, w, ptsd_factor, I_fear,
                                                                        I_context_safe, I_context_threat, I_dbs_amp, 
                                                                        dbs_active)

    model.plot(time, activity, context_trace, I_fear_trace, w_ITC_to_CeA, ptsd_factor)

if __name__ == "__main__":
    main()
