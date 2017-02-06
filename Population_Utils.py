from IPython.core.debugger import Tracer
import numpy as np
import matplotlib.pyplot as plt
import nest

def get_population_spikes(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return:    list with spikearrays containing spiketimes per population
    """
    spikes_arraylist = []
    for idx, pop in enumerate(pop_list):
        spiketimes = nest.GetStatus(pop.recorder._spike_detector.device, 'events')[0]['times']
        spikes_arraylist.append(spiketimes)
    return spikes_arraylist

def get_population_spikes_spiNNaker(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return:    list with spikearrays containing spiketimes per population
    """
    spikes_arraylist = []
    for idx, pop in enumerate(pop_list):
        spiketimes = pop.getSpikes()
        spikes_arraylist.append(spiketimes)
    return spikes_arraylist

def get_spiketrains(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return:    list with spiketrains
    """
    spiketrains_list = []
    for idx, pop in enumerate(pop_list):
        spiketrains = pop.get_data('spikes').segments[0].spiketrains
        spiketrains_list.append(spiketrains)
    return spiketrains_list


def retrieve_voltage_data(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return: v_list: list with voltage values
    """
    v_list =[]
    for pop in pop_list:
        v= pop[0].get_data('v').segments[0].analogsignalarrays[0]
        v_list.append(v)
    return v_list

def retrieve_voltage_data_spiNNaker(pop_list,N=None):
    """
    :param pop_list: list with pyNN populations
    :param N:        max number of populations to get voltages from
    :return: v_list: list with voltage values
    """

    if N==None or N>len(pop_list):
        N=len(pop_list)


    v_list =[]
    for pop in pop_list[:N]:
        v= pop.get_v()
        v_list.append(v)
    return v_list

def get_population_activities(spikes, timebin, start, stop, pop_size):
    """
    calculate Population Activity (average number of spikes per timebin)
    :param spikes: [ np.array(spiketimes), ... ]
    :param timebin: ms, bin width to calculate population activity
    :param start:  ms, start time from where to calculate population activity
    :param stop: ms, stop time until where to calculate population activity
    :param pop_size: population size
    :return: population_activities, [[fl, fl, ...], [], ...] list of lists of floats
    """
    population_activities = []

    """
    for idx in range(len(spikes_arraylist)):
        population_activity = []
        for t in np.arange(0, sim_duration, PA_timebin):
            spikes_hiddenP = spikes_arraylist[idx]
            Nspikes_pop = len(np.where(np.logical_and(spikes_hiddenP >= t, spikes_hiddenP < t + PA_timebin))[0])
            PA = (Nspikes_pop / float(hiddenPexc_size)) / (0.001 * PA_timebin)  # population activity in Hz
            population_activity.append(PA)
        population_activities.append(population_activity)
    return population_activities
    """
    for spiketimes in spikes:
        population_activity = []
        for t in np.arange(start, stop, timebin):
            if t+timebin>stop: # dont calculate PA if bin not in [start,stop]
                continue
            Nspikes_pop = len(np.where(np.logical_and(spiketimes >= t, spiketimes < t + timebin))[0])
            # if Nspikes_pop>0:
                # Tracer()()
            PA = (Nspikes_pop / float(pop_size)) / (0.001 * timebin)  # population activity in Hz
            population_activity.append(PA)
        population_activities.append(population_activity)
    return population_activities
"""
def get_mean_PA(pop_activities):
    for pop_idx in range(len(pop_activities)):
        mean_PA = []
        for idx in range(inputs.shape[1]):
            PA_count = 0
            N_count = 0
            t_start = idx * (FR_sim_dur + pause_dur) + warmup_dur + 0.01
            t_stop = t_start + FR_sim_dur

            for idx, t in enumerate(np.arange(0, sim_duration, PA_timebin)):
                if np.logical_and(t > t_start, t <= t_stop):
                    PA_count += population_activities[pop_idx][idx]
                    N_count += 1
            mean_PA.append(PA_count / N_count)
        mean_PAs.append(mean_PA)
"""

def plot_simulation(spiketrains_in,spiketrains_hidden_exc,spiketrains_hidden_inh,population_activities,input_population_activities,PA_timebin,res_size,input_size,sim_duration,hiddenP_size,hiddenPexc_size,plot_voltage_traces=False,v_hidden_populations=None):

    N_hidden_populations = res_size
    # change figure size
    plt.rcParams["figure.figsize"][0] = 11.0
    plt.rcParams["figure.figsize"][1] = 16.0

    fig0 = plt.figure()

    # subplot 1 - plot input spikes
    ax1 = fig0.add_subplot(411)
    for idx in range(input_size):
        spiketrains_input = spiketrains_in[idx]
        for st in spiketrains_input:
            y = np.ones((st.size)) * st.annotations['source_index'] + idx * (hiddenPexc_size + 10)
            ax1.plot(st, y, '|', color='blue', mew=2, markersize=1.5)
    ax1.set_xlim(0, sim_duration)
    ax1.set_ylabel('Neuron ID')
    ax1.set_title('Input Spikes')

    # subplot 2 - plot hiddenP spikes
    if plot_voltage_traces:  # if voltage traces to be plotted use only one row, else two
        ax2 = fig0.add_subplot(412, sharex=ax1)
    elif not (plot_voltage_traces):
        ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=2, sharex=ax1)
    for idx in range(N_hidden_populations):
        spiketrains_hiddenPexc = spiketrains_hidden_exc[idx]
        spiketrains_hiddenPinh = spiketrains_hidden_inh[idx]
        for st in spiketrains_hiddenPexc:
            y = np.ones((st.size)) * st.annotations['source_index'] + idx * (hiddenP_size + 10)
            ax2.plot(st, y, '|', color='black', mew=2, markersize=1.5)
        for st in spiketrains_hiddenPinh:
            y = np.ones((st.size)) * st.annotations['source_index'] + idx * (hiddenP_size + 10) + hiddenPexc_size
            ax2.plot(st, y, '|', color='red', mew=2, markersize=1.5)
    ax2.set_xlim(0, sim_duration)
    ax2.set_ylabel('Neuron ID')
    ax2.set_title('Hidden Population Spikes')
    ax2.legend()

    # subplot 3 (optional) - plot voltage of first X excitatory reservoir neurons
    if plot_voltage_traces:
        Nvoltage_traces = 3
        ax3 = fig0.add_subplot(413, sharex=ax1)
        for idx in range(N_hidden_populations):
            v_hiddenPexc = v_hidden_populations[0]
            for x in range(Nvoltage_traces):
                signal = v_hiddenPexc[:, x]
                signal = np.array([float(s) for s in signal])
                ax3.plot(v_hiddenPexc.times, signal + 30 * x + idx * 30 * Nvoltage_traces, color='black')
            v_hiddenPinh = v_hidden_populations[1]
            for x in range(Nvoltage_traces):
                signal = v_hiddenPinh[:, x]
                signal = np.array([float(s) for s in signal])
                ax3.plot(v_hiddenPinh.times, signal + 30 * x + 2 * idx * 30 * Nvoltage_traces, color='red')
        # for x in range(Nvoltage_traces):
        #     signal = v_input_populations[0][:, x]
        #     signal = np.array([float(s) for s in signal])
        #     ax3.plot(v_input_populations[0].times, signal + 30 * x + 2 * idx * 30 * Nvoltage_traces, color='blue')

        ax3.set_ylabel('membrane Voltage (mV)')
        ax3.set_title('membrane potential of x Hidden Population neurons')
        ax3.legend()

    # subplot 4 - plot population activity
    ax4 = fig0.add_subplot(414, sharex=ax1)
    for idx in range(N_hidden_populations):
        population_activity = np.array(population_activities[idx])
        if idx == 0:  # add label only once
            ax4.plot(np.arange(0, sim_duration, PA_timebin), population_activity + idx * 1000, color='black',
                     label='Population Activity')
        else:
            ax4.plot(np.arange(0, sim_duration, PA_timebin), population_activity + idx * 1000, color='black')
    summed_input_activities = [sum(x) for x in zip(*input_population_activities)]
    for idx in range(input_size):
        if idx ==0: # add label only once
            ax4.plot(np.arange(0, sim_duration, PA_timebin), np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue',
             label='Input Population Activity')
        else:
            ax4.plot(np.arange(0, sim_duration, PA_timebin),np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue')
    ax4.set_xlabel("time (ms)")
    ax4.set_ylabel("Activity (Hz)")
    ax4.set_ylim(0,45000)
    ax4.legend()

    # plot
    plt.tight_layout()
    plt.show()

    # reset default plot window size
    plt.rcParams["figure.figsize"][0] = 8.0
    plt.rcParams["figure.figsize"][1] = 6.0
    return

def plot_simulation_spiNNaker(spikes_in,spikes_hidden_exc,spikes_hidden_inh,population_activities,input_population_activities,PA_timebin,res_size,input_size,sim_duration,hiddenP_size,hiddenPexc_size,plot_voltage_traces=False,v_hidden_populations=None):

    N_hidden_populations = res_size
    # change figure size
    plt.rcParams["figure.figsize"][0] = 11.0
    plt.rcParams["figure.figsize"][1] = 16.0

    fig0 = plt.figure()

    # subplot 1 - plot input spikes
    ax1 = fig0.add_subplot(411)
    for idx in range(input_size):
        spikes = spikes_in[idx]
        y = spikes[:,0] + idx * (hiddenPexc_size + 10)
        ax1.plot(spikes[:,1], y, '|', color='blue', mew=2, markersize=1.5)
    ax1.set_xlim(0, sim_duration)
    ax1.set_ylabel('Neuron ID')
    ax1.set_title('Input Spikes')

    # subplot 2 - plot hiddenP spikes
    if plot_voltage_traces:  # if voltage traces to be plotted use only one row, else two
        ax2 = fig0.add_subplot(412, sharex=ax1)
    elif not (plot_voltage_traces):
        ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=2, sharex=ax1)
    for idx in range(N_hidden_populations):
        spikes_exc = spikes_hidden_exc[idx]
        spikes_inh = spikes_hidden_inh[idx]

        y_exc = spikes_exc[:, 0] + idx * (hiddenP_size + 10)
        ax2.plot(spikes_exc[:,1], y_exc, '|', color='black', mew=2, markersize=1.5)

        y_inh = spikes_inh[:, 0] + idx * (hiddenP_size + 10) + hiddenPexc_size
        ax2.plot(spikes_inh[:,1], y_inh, '|', color='red', mew=2, markersize=1.5)

    ax2.set_xlim(0, sim_duration)
    ax2.set_ylabel('Neuron ID')
    ax2.set_title('Hidden Population Spikes')
    ax2.legend()

    # subplot 3 (optional) - plot voltage of first X excitatory neurons per population
    if plot_voltage_traces:
        Nvoltage_traces = 3
        ax3 = fig0.add_subplot(413, sharex=ax1)
        for idx in range(len(v_hidden_populations[0])):
            v_hiddenPexc = v_hidden_populations[0]
            for neuronID in range(Nvoltage_traces):
                voltages = v_hiddenPexc[idx][np.where(v_hiddenPexc[idx][:,0]==neuronID)]
                x = voltages[:,1]
                y = voltages[:,2]
                ax3.plot(x, y + 30 * neuronID + idx * 30 * 2 * Nvoltage_traces, color='black')

            v_hiddenPinh = v_hidden_populations[1]
            for neuronID in range(Nvoltage_traces):
                voltages = v_hiddenPinh[idx][np.where(v_hiddenPinh[idx][:,0]==neuronID)]
                x = voltages[:,1]
                y = voltages[:,2]
                ax3.plot(x, y + 30 * neuronID + idx * 30 * 2 * Nvoltage_traces + 30*Nvoltage_traces, color='red')

            """
            for x in range(Nvoltage_traces):
                signal = v_hiddenPexc[:, x]
                signal = np.array([float(s) for s in signal])
                ax3.plot(v_hiddenPexc.times, signal + 30 * x + idx * 30 * Nvoltage_traces, color='black')
            v_hiddenPinh = v_hidden_populations[1]
            for x in range(Nvoltage_traces):
                signal = v_hiddenPinh[:, x]
                signal = np.array([float(s) for s in signal])
                ax3.plot(v_hiddenPinh.times, signal + 30 * x + 2 * idx * 30 * Nvoltage_traces, color='red')
            """
        # for x in range(Nvoltage_traces):
        #     signal = v_input_populations[0][:, x]
        #     signal = np.array([float(s) for s in signal])
        #     ax3.plot(v_input_populations[0].times, signal + 30 * x + 2 * idx * 30 * Nvoltage_traces, color='blue')

        ax3.set_ylabel('membrane Voltage (mV)')
        ax3.set_title('membrane potential of x Hidden Population neurons')
        ax3.legend()

    # subplot 4 - plot population activity
    ax4 = fig0.add_subplot(414, sharex=ax1)
    for idx in range(N_hidden_populations):
        population_activity = np.array(population_activities[idx])
        if idx == 0:  # add label only once
            ax4.plot(np.arange(0, sim_duration, PA_timebin), population_activity + idx * 1000, color='black',
                     label='Population Activity')
        else:
            ax4.plot(np.arange(0, sim_duration, PA_timebin), population_activity + idx * 1000, color='black')
    summed_input_activities = [sum(x) for x in zip(*input_population_activities)]
    for idx in range(input_size):
        if idx ==0: # add label only once
            ax4.plot(np.arange(0, sim_duration, PA_timebin), np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue',
             label='Input Population Activity')
        else:
            ax4.plot(np.arange(0, sim_duration, PA_timebin),np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue')
    ax4.set_xlabel("time (ms)")
    ax4.set_ylabel("Activity (Hz)")
    # ax4.set_ylim(0,45000)
    ax4.legend()

    # plot
    plt.tight_layout()
    plt.show()

    # reset default plot window size
    plt.rcParams["figure.figsize"][0] = 8.0
    plt.rcParams["figure.figsize"][1] = 6.0
    return


def compare_states(ANN_states,SNN_states,res_size,n_it):
    """
    :param ANN_states: states of rate-coded ANN
    :param SNN_states: states of SNN
    :param res_size: number of nodes in reservoir
    :param n_it: number of iterations
    :return:
    """
    """
    # Calculate mrse
    rse = 0
    for i in range(res_size):
        rse += np.sum(np.sqrt(np.square(ANN_states[:, i][20:] - np.array(SNN_states[i][21:]) / 500.0)))
    mrse = rse / (res_size * (n_it - 20))
    """
    # set plot window size
    plt.rcParams["figure.figsize"][0] = 3.0
    plt.rcParams["figure.figsize"][1] = 12.0
    fig1 = plt.figure()
    # plot SNN_states
    [plt.plot([y / 500 + i * 0.5 for y in SNN_states[i]], color='black') for i in range(res_size)]
    plt.plot(0, 0, color='black', label="Population Activity")  # for label
    # plot ANN_states
    [plt.plot(range(1, n_it), ANN_states[:, x] + x * 0.5, color='green') for x in range(res_size)]
    plt.plot(0, 0, color='green', label="Rate-based neuron state")  # for label
    plt.legend()
    plt.xlabel("timesteps")
    plt.ylabel("Neuron state/population activity")
    # plt.title("MRSE = " + str(mrse) + " (discarded first 20 timesteps)")
    plt.show()
    # reset default plot window size
    plt.rcParams["figure.figsize"][0] = 8.0
    plt.rcParams["figure.figsize"][1] = 6.0
