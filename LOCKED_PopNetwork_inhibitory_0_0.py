"""

can create reasonable CPG

simulate recurrent network of populations & compare network behaviour with similar rate-based network

populations mimic max(0, 1.05*x/(1.6+x)) IO

LIF population with 20% inhibitory neurons
connections between inhibitory and excitatory population
intra- and interconnection probability = 0.1

"""



import pyNN.nest as sim
import matplotlib.pyplot as plt
import numpy as np
import time
import nest
import RC_Utils
import sklearn.linear_model
import Population_Utils as PU

from IPython.core.debugger import Tracer
# import rateBasedSim0 as rbs
from IPython.core.debugger import Tracer

import rateBasedReservoir_invertedInput
reload(rateBasedReservoir_invertedInput)
input_size = 0
res_size = 50
output_size = 1
FR_sim_dur = 10.0  # ms, duration of 1 (artificial net) timestep

# Create input signal
input_duration = 1500  # ms
input_freq =8       # Hz
sample_freq = 100   # Hz
n_it = int(input_duration/1000.*sample_freq)
y_signal, y_time = RC_Utils.createInputSine(input_freq=input_freq, input_duration=input_duration/1000., sample_freq=sample_freq)
# shift and scale input signal
y = y_signal+abs(min(y_signal))
y = y/max(y)
y = y*0.4+0.1
inputs = y.reshape(1,-1)


# Create the network
spec_rad = 0.85 #should be coupled to average amount of inputs neurons receive
fraction_inverted = 0.5
ratebased_network = rateBasedReservoir_invertedInput.ReservoirNet(n_in=input_size, n_out=1, n_res=res_size, spec_rad=spec_rad, leakage=0.7,
                                                     scale_bias=0.0,scale_fb=2.0, scale_noise=0.005, scale_fb_noise=0.005, negative_weights=True,fraction_inverted=fraction_inverted)# set input weights (default random)
N_uninverted = int(res_size - res_size*fraction_inverted)

# ratebased_network.w_fb[-5:] = 0.0
"""
ratebased_network.w_in[:]=0.0
for x in range(7):
    ratebased_network.w_in[x]=x*0.06
# ratebased_network.w_in[1]=0.5
# ratebased_network.w_in[2]=0.9# run the network
# ratebased_network.w_res = np.zeros((ratebased_network.w_res.shape))
"""
# ratebased_network.run(n_it=n_it, U=np.transpose(inputs))
ratebased_network.train(Ytrain=inputs.reshape(-1,1))
# ratebased_network.run(n_it=500)
# retrieve network data
res_weights = ratebased_network.w_res
input_weights = ratebased_network.w_in
output_weights = ratebased_network.w_out
output_weights_ridge = ratebased_network.w_out_ridge
alphas = ratebased_network.alphas
print "output_weights OLS"
print output_weights
# print "output_weights_ridge"
# print output_weights_ridge
# for x in output_weights:
#     x[0] = 1.0
feedback_weights = ratebased_network.w_fb
states = ratebased_network.X
outputs = []
outputs_ridge = []
for st in states:
    outputs.append(np.dot(st,output_weights)[0])
    outputs_ridge.append(np.dot(st, output_weights_ridge)[0])

# print network info
print "========Network info==========="
print "spectral radius: "+str(spec_rad)
print "average weight: " + str(np.mean(res_weights)) + " " + u"\u00B1" + str(np.std(res_weights))
print "max(weigths): "+str(np.max(res_weights))
print "min(weigths): "+str(np.min(res_weights))
print "=============================="
# plot
"""
fig = plt.figure()
# [plt.plot(range(1,n_it),states[:,x]+x) for x in range(res_size)]
plt.plot(outputs,color='red',linewidth=2)
# plt.plot(outputs_ridge,color='orange',linewidth=2,label="ridge reg")
plt.title('rate-based reservoir output neuron states')
plt.ylabel('Neuron states')
plt.xlabel('timestep')
plt.legend()
plt.show()
"""

########################### Parameters #########################
record_voltage_traces = False
plot_voltage_traces = False

num_hidden_populations = res_size
hiddenP_size = 100
fraction_inhibitory = 0.2
hiddenPinh_size = int(np.round(hiddenP_size * fraction_inhibitory))
hiddenPexc_size = hiddenP_size - hiddenPinh_size
PA_timebin = 1.0  # ms, duration in which population activity is calculated
Pconnect = 0.1 # connnection probability between any two neurons of two populations

# define input spiketrains
# input_size = 1
pause_dur = 0.0  # ms, time between input spiketrains of different frequency
warmup_dur = 50.0  # ms, warmup period
cooldown_dur = 0.0  # ms, cooldown period
input_spike_train = [ [] for input in range(input_size)]
input_FRs = inputs*500.0  # convert input for rate based neuron ([0.0,1.0]) to Firing Rate ([0,1000.0/msrefractoryperiod]),
# Tracer()()
"""
for N_input in range(input_size):
    for idx, FR in enumerate(input_FRs[N_input,:]):
        temp_spike_train = []
        # create spike train for this FR in time window [0,FR_sim_dur)
        for x in np.arange(0, FR_sim_dur, 1000./FR):
            # input_spike_train[N_input].append(np.around(x,2)+1)
            temp_spike_train.append(x)
        if len(temp_spike_train)>0:
            # get residual time
            residual_time = FR_sim_dur-temp_spike_train[-1]
            # add some time < residual time ( firing doesnt have to start at timepoint 0)
            temp_spike_train = temp_spike_train + np.random.rand()*residual_time
            # shift spike times to correct timewindow
            temp_spike_train = temp_spike_train + idx*(FR_sim_dur + pause_dur) + warmup_dur
            # add spikes to spike train
            for x in temp_spike_train:
                input_spike_train[N_input].append(x)
"""
sim_duration = warmup_dur + inputs.shape[1]*(FR_sim_dur+pause_dur)+ cooldown_dur  # ms

neuron_parameters = {
                'cm': 0.2,
                'v_reset': -75,
                'v_rest': -65,
                'v_thresh': -50,
                'tau_m': 30.0,  # sim.RandomDistribution('uniform', (10.0, 15.0)),
                'tau_refrac': 2.0,
                'tau_syn_E': 0.5,  # np.linspace(0.1, 20, hiddenP_size),
                'tau_syn_I': 0.5,
                'i_offset': 0.0
            }

num_threads = 4
####################### End Parameters ###########################

# setup simulation
sim.setup(timestep=0.1, min_delay=0.1, max_delay=1001.0, threads=num_threads)

# =========================================================================
#                              create network
# =========================================================================

# create connector
connector_inp_hiddenP = sim.FixedProbabilityConnector(p_connect=Pconnect)
connector_hiddenPexc_hiddenPinh = sim.FixedProbabilityConnector(p_connect=Pconnect)
connector_hiddenPinh_hiddenPexc = sim.FixedProbabilityConnector(p_connect=Pconnect)
connector_hiddenP_hiddenP = sim.FixedProbabilityConnector(p_connect=Pconnect)

input_size = 2
# create populations
input_populations = []
for idx in range(input_size):
    input_populations.append(sim.Population(hiddenPexc_size, sim.IF_curr_exp, neuron_parameters))

hidden_populations = [ [] for x in range(num_hidden_populations)]
for idx in range(num_hidden_populations):
    hiddenPexc = sim.Population(hiddenPexc_size, sim.IF_curr_exp, neuron_parameters)
    hiddenPinh = sim.Population(hiddenPinh_size, sim.IF_curr_exp, neuron_parameters)
    hidden_populations[idx].append(hiddenPexc)
    hidden_populations[idx].append(hiddenPinh)

# create interpopulation projections
projections_inp_hiddenP = []
for idx, pop in enumerate(hidden_populations[:N_uninverted]):
    projection_inp_hiddenP = sim.Projection(input_populations[0], pop[0], connector_inp_hiddenP,
                                        sim.StaticSynapse(weight=feedback_weights[idx,0])) #input_weights[idx,0] [neuron.weights for x in range(hiddenP_size)]
    projections_inp_hiddenP.append(projection_inp_hiddenP)  # ? is it necessary to store them in a list ?
for idx, pop in enumerate(hidden_populations[N_uninverted:]):
    projection_inp_hiddenP = sim.Projection(input_populations[1], pop[0], connector_inp_hiddenP,
                                        sim.StaticSynapse(weight=feedback_weights[idx+N_uninverted,0])) #input_weights[idx,0] [neuron.weights for x in range(hiddenP_size)]
    projections_inp_hiddenP.append(projection_inp_hiddenP)  # ? is it necessary to store them in a list ?


projections_hiddenP_hiddenP = []
for idx0 in range(num_hidden_populations):
    pop0exc = hidden_populations[idx0][0]
    for idx1 in range(num_hidden_populations):
        pop1exc = hidden_populations[idx1][0]
        projection = sim.Projection(pop0exc, pop1exc, connector_hiddenP_hiddenP,
                                        sim.StaticSynapse(weight=res_weights[idx1,idx0])) # 0.5
        projections_hiddenP_hiddenP.append(projection)  # ? is it necessary to store them in a list ?

# create intrapopulation projections
for idx in range(num_hidden_populations):
    hiddenPexc = hidden_populations[idx][0]
    hiddenPinh = hidden_populations[idx][1]
    projection_hiddenPexc_hiddenPinh = sim.Projection(hiddenPexc, hiddenPinh, connector_hiddenPexc_hiddenPinh,
                                                      sim.StaticSynapse(weight=1.0))
    # projection_hiddenPexc_hiddenPexc = sim.Projection(hiddenPexc, hiddenPexc, connector_hiddenPexc_hiddenPexc,
    #                                                   sim.StaticSynapse(weight=0.1))
    projection_hiddenPinh_hiddenPexc = sim.Projection(hiddenPinh, hiddenPexc, connector_hiddenPinh_hiddenPexc,
                                                      sim.StaticSynapse(weight=-1.0))
    # projection_hiddenPinh_hiddenPinh = sim.Projection(hiddenPinh, hiddenPinh, connector_hiddenPinh_hiddenPinh,
    #                                                   sim.StaticSynapse(weight=-0.1))

# create and inject white noise current in populations
white_noise = sim.NoisyCurrentSource(mean=0.0, stdev=1.5, start=0.0)
hidden_populations[0][0].inject(white_noise)
for pop in hidden_populations:
    pop[0].inject(white_noise)

white_noise_input = sim.NoisyCurrentSource(mean=0.0, stdev=4.0, start=0.0)
input_populations[0].inject(white_noise_input)
input_populations[1].inject(white_noise_input)

# pulse = sim.DCSource(amplitude=1.0,start=200.0, stop=300.0) # ,
# pulse.inject_into(hidden_populations[0][0])
# pulse.inject_into(hidden_populations[1][0])
# pulse.inject_into(hidden_populations[2][0])
# pulse = sim.DCSource(amplitude=5.0,start=400.0, stop=450.0) # ,
# pulse.inject_into(hidden_populations[0][0])

def getPulseAmplitude(input):
    """ calculates pulse amplitude to crate spiking population with FR = input"""
    if input == 0:
        amplitude = 0
    elif input>0 and input<138:
        amplitude = (input-26)/112
    elif input>138:
        amplitude = np.power(np.e,(input-125)/125 )
    elif input <0:
        amplitude = 0
        print "WARNING negative input is turned into 0 input !!!"
    else:
        raise ValueError('prohibited input FR')
    return amplitude

pulses = []
pulses_inverted = []
for input in input_FRs[0]:
    pulse = getPulseAmplitude(input)
    pulses.append(pulse)

    pulse_inverted = getPulseAmplitude(250-input)
    pulses_inverted.append(pulse_inverted)

for idx, amplitude in enumerate(pulses):
    start = idx*(FR_sim_dur+pause_dur)+warmup_dur
    pulse = sim.DCSource(amplitude=amplitude, start=start, stop=start+FR_sim_dur)
    pulse.inject_into(input_populations[0])

for idx, amplitude in enumerate(pulses_inverted):
    start = idx*(FR_sim_dur+pause_dur)+warmup_dur
    pulse = sim.DCSource(amplitude=amplitude, start=start, stop=start+FR_sim_dur)
    pulse.inject_into(input_populations[1])


# record spikes and membrane voltage
for input_pop in input_populations:
    input_pop.record(['spikes'])
    input_pop.record(['v'])

for pop in hidden_populations:
    pop[0].record(['spikes'], sampling_interval=0.1)  # exc populations
    pop[1].record(['spikes'], sampling_interval=0.1)  # inh populations # 'v',
    if record_voltage_traces:
        pop[0].record(['v'], sampling_interval=0.1)  # populations
        pop[1].record(['v'], sampling_interval=0.1)


def getSpikes(hidden_populations):
    spiketrains_hidden_populations_exc = []
    spiketrains_hidden_populations_inh = []
    spikes_hidden_populations = [[] for x in range(num_hidden_populations)]
    for idx, pop in enumerate(hidden_populations):
        spiketrains_hiddenPexc = pop[0].get_data('spikes').segments[0].spiketrains
        spiketrains_hidden_populations_exc.append(spiketrains_hiddenPexc)
        spikes_hiddenPexc = []
        for spiketrain in spiketrains_hiddenPexc:
            for spike in spiketrain:
                spikes_hiddenPexc.append(float(spike))
        spikes_hidden_populations[idx].append(np.array(spikes_hiddenPexc))

        spiketrains_hiddenPinh = pop[1].get_data('spikes').segments[0].spiketrains
        spiketrains_hidden_populations_inh.append(spiketrains_hiddenPinh)
        spikes_hiddenPinh = []
        for spiketrain in spiketrains_hiddenPinh:
            for spike in spiketrain:
                spikes_hiddenPinh.append(float(spike))
        spikes_hidden_populations[idx].append(np.array(spikes_hiddenPinh))
    return spikes_hidden_populations

"""
def getPopAct(spikes, timebin, start, stop):

    calculate Population Activity (average number of spikes per timebin)
    :param spikes: [ [spiketimes], ... ]
    :param timebin: ms, bin width to calculate population activity
    :param start:  ms, start time from where to calculate population activity
    :param stop: ms, stop time until where to calculate population activity
    :return: population_activities, [[fl, fl, ...], [], ...] list of lists of floats

    pop_size = 100.0
    population_activities = []
    # Tracer()()
    for spiketimes in spikes:
        population_activity = []
        for t in np.arange(start, stop, timebin):
            # spikes_hiddenP = spikes_hidden_populations[idx][0]
            if t+timebin>stop: # dont calculate PA if bin not in [start,stop]
                continue
            Nspikes_pop = len(np.where(np.logical_and(spiketimes >= t, spiketimes < t + timebin))[0])
            PA = (Nspikes_pop / pop_size) / (0.001 * timebin)  # population activity in Hz
            population_activity.append(PA)
        population_activities.append(population_activity)
    return population_activities
"""
# Create and inject input current into input neuron
# pulse = sim.DCSource(amplitude=amplitude, start=start, stop=stop)
# sine = sim.ACSource(start=warmup_dur, stop=warmup_dur+input_duration, amplitude=2.7, offset=2.8,frequency=input_freq, phase=0.0)
# sine.inject_into(input_populations[0])
#create inverted sine
# sine2 = sim.ACSource(start=warmup_dur, stop=warmup_dur+input_duration, amplitude=2.7, offset=2.8,frequency=input_freq, phase=180.0)
# sine2.inject_into(input_populations[1])

# =========================================================================
#                              run simulation
# =========================================================================
print "simulating "+str(sim_duration)+"ms network time"
start_time = time.time()
sim.run(sim_duration)
sim_time = time.time() - start_time
print "%f s for simulation" % (sim_time)

# =========================================================================
#                         retrieve + process data
# =========================================================================

time0 = time.time()
spikes_input_populations = PU.get_population_spikes(input_populations)
spikes_hidden_populations_exc = PU.get_population_spikes(list(np.array(hidden_populations)[:,0]))
spikes_hidden_populations_inh = PU.get_population_spikes(list(np.array(hidden_populations)[:,1]))
spikes_hidden_populations = [ [x,y] for x,y in zip(spikes_hidden_populations_exc,spikes_hidden_populations_inh) ]
time1 = time.time()
print '%f s for retrieving spiketrains'%(time1-time0)

time0 = time.time()
spiketrains_in = PU.get_spiketrains(input_populations)
spiketrains_hidden_exc = PU.get_spiketrains(list(np.array(hidden_populations)[:,0]))
spiketrains_hidden_inh = PU.get_spiketrains(list(np.array(hidden_populations)[:,1]))
time1 = time.time()
print '%f s for retrieving spiketrains'%(time1-time0)

"""
spiketrains_input_populations = []
spikes_input_populations = [ [] for x in range(input_size)]
for idx, pop in enumerate(input_populations):
    spiketrains_input = pop.get_data('spikes').segments[0].spiketrains
    spiketrains_input_populations.append(spiketrains_input)
    spikes_input = []
    for spiketrain in spiketrains_input:
        for spike in spiketrain:
            spikes_input.append(float(spike))
    spikes_input_populations[idx].append(np.array(spikes_input))

spiketrains_hidden_populations_exc = []
spiketrains_hidden_populations_inh = []
spikes_hidden_populations = [ [] for x in range(num_hidden_populations)]

for idx, pop in enumerate(hidden_populations):
    spiketrains_hiddenPexc = pop[0].get_data('spikes').segments[0].spiketrains
    spiketrains_hidden_populations_exc.append(spiketrains_hiddenPexc)
    spikes_hiddenPexc = []
    for spiketrain in spiketrains_hiddenPexc:
        for spike in spiketrain:
            spikes_hiddenPexc.append(float(spike))
    spikes_hidden_populations[idx].append(np.array(spikes_hiddenPexc))

    spiketrains_hiddenPinh = pop[1].get_data('spikes').segments[0].spiketrains
    spiketrains_hidden_populations_inh.append(spiketrains_hiddenPinh)
    spikes_hiddenPinh = []
    for spiketrain in spiketrains_hiddenPinh:
        for spike in spiketrain:
            spikes_hiddenPinh.append(float(spike))
    spikes_hidden_populations[idx].append(np.array(spikes_hiddenPinh))
"""

if record_voltage_traces:
    v_hidden_populations_exc = PU.retrieve_voltage_data(list(np.array(hidden_populations)[:,0]))
    v_hidden_populations_inh = PU.retrieve_voltage_data(list(np.array(hidden_populations)[:,1]))
    v_hidden_populations = []
    for x,y in zip(v_hidden_populations_exc, v_hidden_populations_inh):
        v_hidden_populations.append(x)
        v_hidden_populations.append(y)
    v_input_populations = PU.retrieve_voltage_data(input_populations)
"""
if record_voltage_traces:
    v_hidden_populations = []
    for pop in hidden_populations:
        v_hiddenPexc = pop[0].get_data('v').segments[0].analogsignalarrays[0]
        v_hidden_populations.append(v_hiddenPexc)
        v_hiddenPinh = pop[1].get_data('v').segments[0].analogsignalarrays[0]
        v_hidden_populations.append(v_hiddenPinh)
    v_input_populations = []
    for input_pop in input_populations:
        v_input = input_pop.get_data('v').segments[0].analogsignalarrays[0]
        v_input_populations.append(v_input)
    # v_readout = readout_population.get_data('v').segments[0].analogsignalarrays[0]
"""

time0 = time.time()
input_population_activities = PU.get_population_activities(spikes_input_populations,timebin=1.0, start=0, stop=sim_duration,pop_size=hiddenPexc_size)
population_activities = PU.get_population_activities(spikes_hidden_populations_exc,timebin=1.0, start=0, stop=sim_duration,pop_size=hiddenPexc_size)
time1 = time.time()
print '%f s for calculating PAs'%(time1-time0)

input_mean_PAs = PU.get_population_activities(spikes_input_populations,timebin=FR_sim_dur, start=warmup_dur, stop=sim_duration-cooldown_dur,pop_size=hiddenPexc_size)
mean_PAs = PU.get_population_activities(spikes_hidden_populations_exc,timebin=FR_sim_dur, start=warmup_dur, stop=sim_duration-cooldown_dur,pop_size=hiddenPexc_size)
"""
# compute population activities
population_activities = []
for idx in range(num_hidden_populations):
    population_activity = []
    for t in np.arange(0,sim_duration,PA_timebin):
        spikes_hiddenP = spikes_hidden_populations[idx][0]
        Nspikes_pop = len(np.where(np.logical_and(spikes_hiddenP>=t,spikes_hiddenP<t+PA_timebin))[0])
        PA = (Nspikes_pop/float(hiddenPexc_size))/(0.001*PA_timebin)  # population activity in Hz
        population_activity.append(PA)
    population_activities.append(population_activity)

input_population_activities = []
for idx in range(input_size):
    input_population_activity = []
    for t in np.arange(0,sim_duration,PA_timebin):
        spikes_hiddenP = spikes_input_populations[idx]
        Nspikes_pop = len(np.where(np.logical_and(spikes_hiddenP>=t,spikes_hiddenP<t+PA_timebin))[0])
        PA = (Nspikes_pop/float(hiddenPexc_size))/(0.001*PA_timebin)  # population activity in Hz
        input_population_activity.append(PA)
    input_population_activities.append(input_population_activity)


# compute mean PAs
mean_PAs = []
for pop_idx in range(num_hidden_populations):
    mean_PA = []
    for idx in range(inputs.shape[1]):
        PA_count = 0
        N_count = 0
        t_start = idx*(FR_sim_dur+pause_dur)+warmup_dur+0.01
        t_stop = t_start+FR_sim_dur

        for idx, t in enumerate(np.arange(0, sim_duration, PA_timebin)):
            if np.logical_and(t>t_start, t<=t_stop):
                PA_count += population_activities[pop_idx][idx]
                N_count += 1
        mean_PA.append(PA_count/N_count)
    mean_PAs.append(mean_PA)

# compute input mean_PA
input_mean_PAs = []
for pop_idx in range(input_size):
    input_mean_PA = []
    for idx in range(inputs.shape[1]):
        PA_count = 0
        N_count = 0
        t_start = idx*(FR_sim_dur+pause_dur)+warmup_dur+0.01
        t_stop = t_start+FR_sim_dur

        for idx, t in enumerate(np.arange(0, sim_duration, PA_timebin)):
            if np.logical_and(t > t_start, t <= t_stop):
                PA_count += input_population_activities[pop_idx][idx]
                N_count += 1
        input_mean_PA.append(PA_count / N_count)
    input_mean_PAs.append(input_mean_PA)
"""

# =========================================================================
#                                 plot data
# =========================================================================

fig = plt.figure()
plt.plot(np.array(input_mean_PAs[0])/500.0,color='blue')
plt.plot(np.transpose(inputs),color='red')
plt.show()

PU.plot_simulation(spiketrains_in,spiketrains_hidden_exc,spiketrains_hidden_inh,population_activities,
                input_population_activities,PA_timebin,res_size,input_size,sim_duration,hiddenP_size,
                   hiddenPexc_size,plot_voltage_traces=False,v_hidden_populations=None)

"""
# change figure size
plt.rcParams["figure.figsize"][0] = 11.0
plt.rcParams["figure.figsize"][1] = 16.0

fig0 = plt.figure()

# plot input spikes
ax1 = fig0.add_subplot(411)
for idx in range(input_size):
    spiketrains_input = spiketrains_input_populations[idx]
    for st in spiketrains_input:
        y=np.ones((st.size)) * st.annotations['source_index'] + idx*(hiddenPexc_size+10)
        ax1.plot(st, y, '|', color='blue', mew=2, markersize=1.5)
ax1.set_xlim(0, sim_duration)
# ax1.set_ylim(-0.5, input_size - 0.5)
ax1.set_ylabel('Neuron ID')
ax1.set_title('Input Spikes')

# plot hiddenP spikes
if plot_voltage_traces:  # if voltage traces to be plotted use only one row, else two
    ax2 = fig0.add_subplot(412, sharex=ax1)
elif not(plot_voltage_traces):
    ax2 = plt.subplot2grid((4,1), (1, 0), rowspan=2, sharex=ax1)
for idx in range(num_hidden_populations):
    spiketrains_hiddenPexc = spiketrains_hidden_populations_exc[idx]
    spiketrains_hiddenPinh = spiketrains_hidden_populations_inh[idx]
    for st in spiketrains_hiddenPexc:
        y=np.ones((st.size)) * st.annotations['source_index'] + idx*(hiddenP_size + 10)
        ax2.plot(st, y, '|', color='black', mew=2, markersize=1.5)
    for st in spiketrains_hiddenPinh:
        y= np.ones((st.size)) * st.annotations['source_index'] + idx*(hiddenP_size + 10)+hiddenPexc_size
        ax2.plot(st, y, '|', color='red', mew=2, markersize=1.5)
# ax2.set_ylim(-readout_size-1,hiddenP_size)
ax2.set_xlim(0, sim_duration)
ax2.set_ylabel('Neuron ID')
ax2.set_title('Hidden Population Spikes')
ax2.legend()


# plot voltage of first X excitatory reservoir neurons
if plot_voltage_traces:
    Nvoltage_traces=3
    ax3 = fig0.add_subplot(413, sharex=ax1)
    for idx in range(num_hidden_populations):
        v_hiddenPexc = v_hidden_populations[0]
        for x in range(Nvoltage_traces):
            signal = v_hiddenPexc[:,x]
            signal = np.array([float(s) for s in signal])
            ax3.plot(v_hiddenPexc.times, signal+30*x+idx*30*Nvoltage_traces,color='black')
        v_hiddenPinh = v_hidden_populations[1]
        for x in range(Nvoltage_traces):
            signal = v_hiddenPinh[:,x]
            signal = np.array([float(s) for s in signal])
            ax3.plot(v_hiddenPinh.times, signal+30*x+2*idx*30*Nvoltage_traces,color='red')
    # for x in range(Nvoltage_traces):
    #     signal = v_input_populations[0][:, x]
    #     signal = np.array([float(s) for s in signal])
    #     ax3.plot(v_input_populations[0].times, signal + 30 * x + 2 * idx * 30 * Nvoltage_traces, color='blue')

    ax3.set_ylabel('membrane Voltage (mV)')
    ax3.set_title('membrane potential of x Hidden Population neurons')
    ax3.legend()


# plot population activity
ax4 = fig0.add_subplot(414, sharex=ax1)
for idx in range(num_hidden_populations):
    population_activity = np.array(population_activities[idx])
    if idx == 0: #  add label only once
        ax4.plot(np.arange(0,sim_duration,PA_timebin),population_activity+idx*1000,color='black',label='Population Activity')
    else :
        ax4.plot(np.arange(0,sim_duration,PA_timebin),population_activity+idx*1000,color='black')
summed_input_activities = [ sum(x) for x in zip(*input_population_activities)]
ax4.plot(np.arange(0, sim_duration, PA_timebin), np.array(summed_input_activities)+1000, color='blue',label='Input Population Activity (+1000)')
# ax4.set_ylim(-600,1000)
ax4.set_xlabel("time (ms)")
ax4.set_ylabel("Activity (Hz)")
ax4.legend()

# plot
plt.tight_layout()
plt.show()
# reset default plot window size
plt.rcParams["figure.figsize"][0] = 8.0
plt.rcParams["figure.figsize"][1] = 6.0
"""

PU.compare_states(ANN_states=states,SNN_states=mean_PAs,res_size=res_size,n_it=n_it)
"""
rse = 0
for i in range(res_size):
    rse += np.sum(np.sqrt(np.square(states[:,i][20:]-np.array(mean_PAs[i][21:])/500.0)))
mrse = rse/(res_size*(n_it-20))

# set plot window size
plt.rcParams["figure.figsize"][0] = 3.0
plt.rcParams["figure.figsize"][1] = 12.0
fig1 = plt.figure()
[plt.plot(range(n_it),[y/500+i*0.5 for y in mean_PAs[i]],color='black') for i in range(res_size)]
plt.plot(0,0,color='black', label="Population Activity")
[plt.plot(range(1,n_it),states[:,x]+x*0.5,color='green') for x in range(res_size)]
plt.plot(0,0,color='green',label="Rate-based neuron state")
plt.legend()
plt.xlabel("timesteps")
plt.ylabel("Neuron state/population activity")
plt.title("MRSE = "+str(mrse)+" (discarded first 20 timesteps)")
plt.show()
# reset default plot window size
plt.rcParams["figure.figsize"][0] = 8.0
plt.rcParams["figure.figsize"][1] = 6.0
"""

output_SNN = []
output_SNN_ridge = []
for idx in range(n_it):
    output_SNN.append(np.dot(np.array(mean_PAs)[:,idx]/500.0,output_weights)[0])
for weights in output_weights_ridge:
    out = []
    for idx in range(n_it):
        out.append(np.dot(np.array(mean_PAs)[:, idx] / 500.0, weights)[0])
    output_SNN_ridge.append(out)

fig2 = plt.figure()
[plt.plot(range(n_it),[x/500. for x in trace]) for trace in mean_PAs]
[plt.plot(x,linewidth=1.5,label="alpha: "+str(alphas[idx])) for idx, x in enumerate(output_SNN_ridge)]
plt.plot(np.transpose(inputs)[1:],linewidth=2, color='orange',label="target")
plt.plot(output_SNN,linewidth=2,color='pink',linestyle="dashed",label="OLS")
plt.title("Reservoir mean population activities (thin lines) + readout based on ANN-derived-weights (dashed line)")
plt.xlabel("timestep")
plt.ylabel("Mean population activity")
plt.legend()
# plt.ylim(0,1)
plt.show()


"""
plt.plot(range(len(mean_PA)), mean_PA, "o", markersize=5.0, label='mean population activity ('+str(FR_sim_dur)+') ms',color='black')
plt.plot(range(len(mean_RA)), mean_RA, "o", markersize=5.0, label='mean readout activity ('+str(FR_sim_dur)+') ms',color='red')
plt.plot(range(len(neuron.outputs)),[x*500 for x in neuron.outputs],'o',markersize=5.0,color='blue',label='ratebased output*500')
plt.suptitle('Populaton response to multiple ('+str(input_size)+') constant spiking inputs. Compared with rate-based neuron.')
plt.title('Population of '+str(hiddenP_size)+', unconnected noisy LIF neurons.')
plt.xlabel('Simulation N')
plt.ylabel('FR (Hz)')
plt.ylim(0,500)
plt.xlim(-1,len(mean_PA))
plt.legend()
plt.show()
"""

#############################################################################
#                           apply ridge regression
#############################################################################

# do ridge regression
fig = plt.figure()
plt.plot(np.transpose(inputs), color='blue', linewidth=2, label='Y')
# powers = np.hstack((-10., np.arange(-9.,-7.,0.25,)))
# powers = np.hstack((powers, np.arange(-7,-2)))
# powers = np.arange(-3.,1.)
powers = [-3., -2., -1.3, -1., 0.]
alphas = [np.power(10, x) for x in powers]
# self.alphas = [0.0001]
w_out_ridge = []
X = np.transpose(np.array(mean_PAs))
for alpha in alphas:
    ridge_regressor = sklearn.linear_model.Ridge(alpha, fit_intercept=False)
    rr = ridge_regressor.fit(X[:-1], np.transpose(inputs)[1:]) #one-step-ahead prediction
    readout_train = rr.predict(X)
    w_out_ridge.append(np.transpose(rr.coef_))
    if alpha == 0.01:
        w_out = np.transpose(rr.coef_)
        print(" -- Updating w_out using ridge regression , alpha = 0.01-- ")
    # calculate RMSE on train data
    # rmse_train = (np.sqrt((inputs - readout_train) ** 2)).mean()
    plt.plot(readout_train, label="alpha: " + str(alpha) )
    # print "alpa: " + str(alpha) + ", mse_train: " + str(rmse_train) + ", w_out_ridge:"
    # print w_out_ridge[-1]
plt.title("readout on spiking train data (w ridge regression)")
plt.legend()
plt.show()

#############################################################################
#                          one-step-ahead prediction
#############################################################################

run_iterations = 150
it_dur = FR_sim_dur
readout_list=[]

time0 = time.time()
print "running one-step-ahead iterations . . . "
for x in range(run_iterations):
    #### start and stop time of this iteration
    start = sim_duration+x*it_dur
    stop = start+it_dur
    #### get spikes
    # time0 = time.time()
    hiddenP_exc_spikes = PU.get_population_spikes(list(np.array(hidden_populations)[:,0]))
    # time1 = time.time()
    # print '%f s for retrieving spiketrains' % (time1 - time0)
    #### get readout
    # time0 = time.time()
    PA = PU.get_population_activities(hiddenP_exc_spikes,it_dur,start-it_dur,start,hiddenPexc_size)
    if max(PA) == 0:
        Tracer()()
    readout = np.dot(np.transpose(np.array(PA)),w_out)[0][0]
    readout_list.append(readout)
    # time1 = time.time()
    # print '%f s for calculating PAs and readout' % (time1 - time0)
    #### inject readout
    # time0 = time.time()
    amplitude = getPulseAmplitude(readout*500)
    pulse = sim.DCSource(amplitude=amplitude, start=start, stop=stop)
    pulse.inject_into(input_populations[0])
    amplitude_inverted = getPulseAmplitude(250-readout*500)
    pulse_inverted = sim.DCSource(amplitude=amplitude_inverted, start=start, stop=stop)
    pulse_inverted.inject_into(input_populations[1])
    # time1 = time.time()
    # print '%f s for calculating and injecting pulses' % (time1 - time0)
    #### run sim
    # time0 = time.time()
    sim.run(it_dur)
time1 = time.time()
print '%f s for simulating %i iterations' % (time1 - time0,run_iterations)

plt.plot(readout_list)
plt.title("spiking readouts during run")
plt.show()

time0 = time.time()
spikes_input_populations = PU.get_population_spikes(input_populations)
spikes_hidden_populations_exc = PU.get_population_spikes((list(np.array(hidden_populations)[:,0])))
spikes_hidden_populations_inh = PU.get_population_spikes((list(np.array(hidden_populations)[:,1])))
spikes_hidden_populations = [ [x,y] for x,y in zip(spikes_hidden_populations_exc,spikes_hidden_populations_inh) ]
time1 = time.time()
print '%f s for retrieving population spikes'%(time1-time0)
"""
spiketrains_input_populations = []
spikes_input_populations = [ [] for x in range(input_size)]
for idx, pop in enumerate(input_populations):
    spiketrains_input = pop.get_data('spikes').segments[0].spiketrains
    spiketrains_input_populations.append(spiketrains_input)
    spikes_input = []
    for spiketrain in spiketrains_input:
        for spike in spiketrain:
            spikes_input.append(float(spike))
    spikes_input_populations[idx].append(np.array(spikes_input))

spiketrains_hidden_populations_exc = []
spiketrains_hidden_populations_inh = []
spikes_hidden_populations = [ [] for x in range(num_hidden_populations)]

for idx, pop in enumerate(hidden_populations):
    spiketrains_hiddenPexc = pop[0].get_data('spikes').segments[0].spiketrains
    spiketrains_hidden_populations_exc.append(spiketrains_hiddenPexc)
    spikes_hiddenPexc = []
    for spiketrain in spiketrains_hiddenPexc:
        for spike in spiketrain:
            spikes_hiddenPexc.append(float(spike))
    spikes_hidden_populations[idx].append(np.array(spikes_hiddenPexc))

    spiketrains_hiddenPinh = pop[1].get_data('spikes').segments[0].spiketrains
    spiketrains_hidden_populations_inh.append(spiketrains_hiddenPinh)
    spikes_hiddenPinh = []
    for spiketrain in spiketrains_hiddenPinh:
        for spike in spiketrain:
            spikes_hiddenPinh.append(float(spike))
    spikes_hidden_populations[idx].append(np.array(spikes_hiddenPinh))
"""

time0 = time.time()
spiketrains_in = PU.get_spiketrains(input_populations)
spiketrains_hidden_exc = PU.get_spiketrains(list(np.array(hidden_populations)[:,0]))
spiketrains_hidden_inh = PU.get_spiketrains(list(np.array(hidden_populations)[:,1]))
time1 = time.time()
print '%f s for retrieving spiketrains'%(time1-time0)

total_time = sim_duration+run_iterations*it_dur
time0 = time.time()
input_population_activities = PU.get_population_activities(spikes_input_populations,timebin=1.0, start=0, stop=total_time,pop_size=hiddenPexc_size)
population_activities = PU.get_population_activities(spikes_hidden_populations_exc,timebin=1.0, start=0, stop=total_time,pop_size=hiddenPexc_size)
time1 = time.time()
print '%f s for calculating PAs'%(time1-time0)

input_mean_PAs = PU.get_population_activities(spikes_input_populations,timebin=FR_sim_dur, start=warmup_dur, stop=total_time,pop_size=hiddenPexc_size)
mean_PAs = PU.get_population_activities(spikes_hidden_populations_exc,timebin=FR_sim_dur, start=warmup_dur, stop=total_time,pop_size=hiddenPexc_size)

PU.plot_simulation(spiketrains_in,spiketrains_hidden_exc,spiketrains_hidden_inh,population_activities,
                input_population_activities,PA_timebin,res_size,input_size,total_time,hiddenP_size,
                   hiddenPexc_size,plot_voltage_traces=False,v_hidden_populations=None)

PU.compare_states(ANN_states=states,SNN_states=mean_PAs,res_size=res_size,n_it=n_it)

