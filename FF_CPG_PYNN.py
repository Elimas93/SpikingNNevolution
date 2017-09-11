"""

11/09/2017
Alexander Vandesompele, UGent

Create a FeedForward CPG (Central Pattern Generator)

receives periodic input
target = motor signals based on Auke Ijspeert parametrized cpg

saves network data to SNNfile, allows for rebuilding same network later on

* cubic reservoir (distance based connectivity)
* ridge regression on reservoir states
* reservoir states = monitor neuron membrane potentials
* monitor neuron 'listens' to one population (represent Population Activity) (infinite spiking threshold)

"""

import pyNN.nest as sim
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time

import RC_Utils
import Population_Utils as PU
import generate_cpg_control as gcpg

timestart = time.time()

# =========================================================================
#                           General Parameters
# =========================================================================

input_size = 2
res_size = 3*9                          # N hidden populations
output_size = 1
FR_sim_dur = 50.0                       # ms, duration of 1 (artificial net) timestep

total_duration= 15000                   # ms
learn_dur = 10000                       # ms
input_freq = 1.44                       # Hz
sample_freq = 1./(FR_sim_dur*0.001)*5   # Hz
samplesPerPeriod = int(1./input_freq/(FR_sim_dur*0.001))


fraction_inverted = 0.5  # fraction of reservoir units that receive inverted input signal
N_uninverted = int(res_size - res_size*fraction_inverted)

# =========================================================================
#                         create input and target signals
# =========================================================================

# ###################### Create input signal ##############################
input_signal, input_time = RC_Utils.createInputSine(input_freq=input_freq, input_duration=total_duration/1000., sample_freq=sample_freq)

# shift and scale input signal
inp = input_signal-min(input_signal)
inp = inp/max(inp)
inp = inp*0.4+0.1
inputs = inp.reshape(1,-1)

# ###################### Create target signals #############################
mu = [1247.8506096107708, 1271.4286080336062, 1350.704882789273, 1315.9368951279903] # amplitude
o = [-6.6, -6.6, -42.9, -42.9]                                                       # offset
omega = [8.966, 8.966, 8.966, 8.966]
d = [0.5946128975739761, 0.5946128975739761, 0.7639788381220521, 0.7639788381220521] # duty cycle
phase_offset = [0.0, 3.2, 3.2]
cpg = gcpg.CPGControl(mu, o, omega, d, phase_offset)

sig = np.transpose([ cpg.step_open_loop() for i in range(total_duration)]) #siebes cpg generator presumably with sample frequency of 1000 Hz

# Downsample target signal, we only need frequency of sample_freq
downfactor = 1000/sample_freq
sig2 = np.empty((sig.shape[0],sig.shape[1]/downfactor))
for idx, row in enumerate(sig):
    sig2[idx] = row[::downfactor]

# Normalize target signal
offset = []
multiplier = []
for idx, row in enumerate(sig2):
    offset.append(abs(min(row)))
    sig2[idx] = row+abs(min(row))
    multiplier.append(max(row))
    sig2[idx] = row/max(row)


y_signals = [list(x) for x in sig2]
N_readouts = len(y_signals)

# ################### Plot input and target signals #######################
plt.plot(inp,label="input")
[plt.plot([s + idx for s in y]) for idx,y in enumerate(y_signals)]
plt.xlabel("Timestep")
# plt.legend()
plt.show()

# =========================================================================
#                       Create Network Components
# =========================================================================

# initialize network data
weights_input, weights_res, Pconnect_input, Pconnect_res = PU.InitReservoirNet(n_in=input_size, n_res=res_size, spec_rad=2.0,
                         scale_in=1.5, negative_weights=True)# set input weights (default random)

# increase weights a bit
weights_input = weights_input+0.2
weights_res = weights_res+0.1

# print network info
print "========Network info==========="
print "average res weight: " + str(np.mean(weights_res)) + " " + u"\u00B1" + str(np.std(weights_res))
print "max(res weigths): "+str(np.max(weights_res))
print "min(res weigths): "+str(np.min(weights_res))
print "=============================="


# =========================================================================
#                           SNN Parameters
# =========================================================================

record_voltage_traces = False
plot_voltage_traces = False

N_res = res_size
hiddenP_size = 100
fraction_inhibitory = 0.2
hiddenPinh_size = int(np.round(hiddenP_size * fraction_inhibitory))
hiddenPexc_size = hiddenP_size - hiddenPinh_size
PA_timebin = 1.0  # ms, duration in which population activity is calculated
Pconnect_intra = 0.1  # connection probability between any two neurons within a population

# define input spiketrains
# input_size = 1
pause_dur = 0.0  # ms, time between input spiketrains of different frequency
warmup_dur = 51.0  # ms, warmup period
cooldown_dur = 0.0  # ms, cooldown period
input_spike_train = [ [] for input in range(input_size)]
input_FRs = inputs*500.0  # convert input for rate based neuron ([0.0,1.0]) to Firing Rate ([0,1000.0/msrefractoryperiod]),
weights_readout = [] #original readout weigths zero
for i in range(N_readouts):
    weights_readout.append(np.zeros((res_size,1)))

sim_duration = warmup_dur + total_duration+ cooldown_dur  # ms

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

readout_parameters = {
    'cm': 0.2,
    'v_reset': -75,
    'v_rest': -65,
    'v_thresh': 5000,
    'tau_m': 30.0,  # sim.RandomDistribution('uniform', (10.0, 15.0)),
    'tau_refrac': 0.1,
    'tau_syn_E': 5.5,  # np.linspace(0.1, 20, hiddenP_size),
    'tau_syn_I': 5.5,
    'i_offset': 0.0
}

####################### End Parameters ###########################

# setup simulation
timestep = 0.5
num_threads = 4
sim.setup(timestep=timestep, min_delay=1.0, max_delay=1001.0, threads=num_threads)

# ========================================================================= #
# ------------------------create pyNN network------------------------------ #
# ========================================================================= #

# ======================================================================================================
# create connectors
# ======================================================================================================

# connector_inp_hiddenP = sim.FixedProbabilityConnector(p_connect=Pconnect_intra)
connector_hiddenPexc_hiddenPinh = sim.FixedProbabilityConnector(p_connect=Pconnect_intra)
connector_hiddenPinh_hiddenPexc = sim.FixedProbabilityConnector(p_connect=Pconnect_intra)
#connector_hiddenP_hiddenP = sim.FixedProbabilityConnector(p_connect=Pconnect_intra)

# ======================================================================================================
# create populations
# ======================================================================================================

N_input = 2

input_populations = []
for idx in range(N_input):
    input_populations.append(sim.Population(hiddenPexc_size, sim.IF_curr_exp, neuron_parameters))

hidden_populations = [ [] for x in range(N_res)]
for idx in range(N_res):
    hiddenPexc = sim.Population(hiddenPexc_size, sim.IF_curr_exp, neuron_parameters)
    hiddenPinh = sim.Population(hiddenPinh_size, sim.IF_curr_exp, neuron_parameters)
    hidden_populations[idx].append(hiddenPexc)
    hidden_populations[idx].append(hiddenPinh)

monitor_population = sim.Population(N_res, sim.IF_curr_exp, readout_parameters)

readout_neuron_populations = []
for i in range(N_readouts):
    pop = sim.Population(1, sim.IF_curr_exp, readout_parameters)
    pop.set(tau_syn_E=5.5)  # ?improves linearity ...
    pop.set(v_thresh=5000)
    readout_neuron_populations.append(pop)

# ======================================================================================================
# create projections
# ======================================================================================================

# create interpopulation projections
projections_inp_hiddenP = []
for idx, pop in enumerate(hidden_populations[:N_uninverted]):
    projection_inp_hiddenP = sim.Projection(input_populations[0], pop[0], sim.FixedProbabilityConnector(p_connect=Pconnect_input[idx]),
                                        sim.StaticSynapse(weight=weights_input[idx,0])) #input_weights[idx,0] [neuron.weights for x in range(hiddenP_size)]
    projections_inp_hiddenP.append(projection_inp_hiddenP)  # ? is it necessary to store them in a list ?
for idx, pop in enumerate(hidden_populations[N_uninverted:]):
    projection_inp_hiddenP = sim.Projection(input_populations[1], pop[0], sim.FixedProbabilityConnector(p_connect=Pconnect_input[idx+N_uninverted]),
                                        sim.StaticSynapse(weight=weights_input[idx+N_uninverted,0])) #input_weights[idx,0] [neuron.weights for x in range(hiddenP_size)]
    projections_inp_hiddenP.append(projection_inp_hiddenP)  # ? is it necessary to store them in a list ?

#projections_hiddenP_monitor
for idx in range(N_res):
    connector = sim.FromListConnector(conn_list=[(x,idx) for x in range(hiddenPexc_size)])
    projection=sim.Projection(hidden_populations[idx][0], monitor_population, connector, sim.StaticSynapse(weight=1.0))

projections_hiddenP_readout = []
connector = sim.AllToAllConnector()  #FixedProbabilityConnector(p_connect=0.14) , delays=np.random.randint(6,72), rng=rng
for idx0 in range(N_readouts):
    readout_pop = readout_neuron_populations[idx0]
    projections = []
    for idx1 in range(N_res):
        pop0exc = hidden_populations[idx1][0]
        weight = weights_readout[idx0][idx1, 0]

        projection = sim.Projection(pop0exc, readout_pop, connector, sim.StaticSynapse(weight=weight))
        projections.append(projection)
    projections_hiddenP_readout.append(projections)

rng = sim.NumpyRNG(seed=2007200)
projections_hiddenP_hiddenP = []
for idx0 in range(N_res):
    pop0exc = hidden_populations[idx0][0]
    for idx1 in range(N_res):
        pop1exc = hidden_populations[idx1][0]
        projection = sim.Projection(pop0exc, pop1exc, sim.FixedProbabilityConnector(p_connect=Pconnect_res[idx1,idx0], rng=rng),sim.StaticSynapse(weight=weights_res[idx1,idx0],delay=np.random.randint(4,8)))
        projections_hiddenP_hiddenP.append(projection)

# create intrapopulation projections
for idx in range(N_res):
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

# ======================================================================================================
# inject noise/current and set recordings
# ======================================================================================================

white_noise_input = sim.NoisyCurrentSource(mean=0.0, stdev=2.0, start=0.0,dt=timestep)
input_populations[0].inject(white_noise_input)
input_populations[1].inject(white_noise_input)

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

for pop in readout_neuron_populations:
    pop.record(['spikes'])
    pop.record(['v'])

monitor_population.record(['v'], sampling_interval=1./sample_freq*1000)

# Create and inject input sines
sine = sim.ACSource(start=0, amplitude=1.5, offset=1.5, frequency=input_freq, phase=150.0)
sine.inject_into(input_populations[0])
#create inverted sine
sine2 = sim.ACSource(start=0, amplitude=1.5, offset=1.5,frequency=input_freq, phase=60.0)
sine2.inject_into(input_populations[1])

# ========================================================================= #
# -----------------------------run simulation------------------------------ #
# ========================================================================= #
sim_dur1 = learn_dur
print "simulating "+str(sim_dur1)+"ms network time"
start_time = time.time()
sim.run(sim_dur1)
sim_time = time.time() - start_time
print "%f s for simulation" % (sim_time)

timeend = time.time()
print '%f s for setting up and simulating on NEST'%(timeend-timestart)


#############################################################################
#                           apply ridge regression                          #
#############################################################################

#  retrieve some data needed for the regression
monitor_voltages = PU.retrieve_voltage_data([monitor_population])

#+++++++++++++++++++++++++Ridge Regression on Monitor neurons++++++++++++

print 'applying ridge regression ...'
powers = np.arange(-3., 5., 1)
alphas = 10 ** powers

X = np.array(monitor_voltages[0])
N = X.shape[0] # N samples
X_train = X[:N/2]  # don't use first cycles for learning
X_run = X[N/2:N]

ridgeResult = []
plt.figure()
for idx in range(N_readouts):
    Y = np.array(y_signals[idx]).reshape(-1,1)
    Y_train = Y[:N/2]  # don't use first cycles for learning
    Y_run = Y[N/2:N]

    ridgeResult.append(RC_Utils.doRidgeRegression(X_train,X_run,Y_train,Y_run,alphas))

    readout_list = ridgeResult[-1][1]
    mse_list = ridgeResult[-1][3]

    min_mse = min(mse_list)

    for readout, alpha, mse in zip(readout_list, alphas, mse_list):
        if alpha == min(alphas): #label only once
            plt.plot(readout+idx,color=cm.inferno(min_mse*10),label = str(min_mse))
        else:
            plt.plot(readout+idx,color=cm.inferno(min_mse*10))

    plt.plot(Y_run+idx, color='blue', linewidth=2)

plt.legend()
plt.title("readout on monitor run data")
plt.show()


print 'finished applying ridge regression on monitor data'
print 'setting readout weights ... '
weights_readout_previous = weights_readout
weights_readout = []
for idx in range(N_readouts):
    weights_readout.append(ridgeResult[idx][0][np.where(mse_list == min(mse_list))[0][0]]) #*25 for PA ridge reg

#change weights
for idx0 in range(N_readouts):
    mse_list = ridgeResult[idx0][3]
    for idx1,connection in enumerate(projections_hiddenP_readout[idx0]):

        weight = weights_readout[idx0][idx1, 0]

        weight_matrix = connection.get("weight",format="array")
        new_weight_matrix = weight_matrix

        new_weight_matrix[:] = weight

        connection.set(weight=new_weight_matrix)
print 'finished setting readout weights'

#############################################################################
#                           run simulation again
#############################################################################
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~SIMULATION~II~~~~~~~~~~~~~~~~~~~~~~~~~"
sim_dur2 = sim_duration-learn_dur
print "simulating "+str(sim_dur2)+"ms network time ..."
start_time = time.time()
sim.run(sim_dur2)
sim_time = time.time() - start_time
print "%f s for simulation" % (sim_time)

# ========================================================================= #
# ------------------------retrieve + process data-------------------------- #
# ========================================================================= #
"""
#collect readout neuron spiketrain
readout_neuron_spiketrain = PU.get_spiketrains(readout_neuron_populations)

spikes_readout = PU.get_population_spikes(readout_neuron_populations)
# compute readout activity
readout_activities = []
for idx0 in range(N_readouts):
    readout_activity = []
    for idx1 in range(spikes_readout[idx0].shape[0]):
        # calculate InterSpikeInterval (s)
        if idx1 == 0 : # first spike
            ISI = 1000
        else :
            ISI = (spikes_readout[idx0][idx1]-spikes_readout[idx0][idx1-1])*0.001
        RA = 1.0/ISI
        readout_activity.append(RA)
    readout_activities.append(readout_activity)
"""

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

if record_voltage_traces:
    v_hidden_populations_exc = PU.retrieve_voltage_data(list(np.array(hidden_populations)[:,0]))
    v_hidden_populations_inh = PU.retrieve_voltage_data(list(np.array(hidden_populations)[:,1]))
    v_hidden_populations = []
    for x,y in zip(v_hidden_populations_exc, v_hidden_populations_inh):
        v_hidden_populations.append(x)
        v_hidden_populations.append(y)
    v_input_populations = PU.retrieve_voltage_data(input_populations)

time0 = time.time()
input_population_activities = PU.get_population_activities(spikes_input_populations,timebin=1.0, start=0, stop=sim_duration,pop_size=hiddenPexc_size)
population_activities = PU.get_population_activities(spikes_hidden_populations_exc,timebin=1.0, start=0, stop=sim_duration,pop_size=hiddenPexc_size)
# population_activities = [[] for x in range(res_size)]
time1 = time.time()
print '%f s for calculating PAs'%(time1-time0)
"""
# get the 50ms PA every 10ms
time0 = time.time()
meansig = []
for PA in population_activities:
    meansig.append([sum(PA[t:t + 50]) / 50. for t in range(50,int(sim_dur1),10)])
time1 = time.time()
print '%f s for calculating mean PAs new style'%(time1-time0)
"""

input_mean_PAs = PU.get_population_activities(spikes_input_populations,timebin=FR_sim_dur, start=warmup_dur, stop=sim_duration-cooldown_dur,pop_size=hiddenPexc_size)
mean_PAs = PU.get_population_activities(spikes_hidden_populations_exc,timebin=FR_sim_dur, start=warmup_dur, stop=sim_duration-cooldown_dur,pop_size=hiddenPexc_size)

# =========================================================================
#                                 plot data
# =========================================================================

plt.figure()
readout_v = PU.retrieve_voltage_data(readout_neuron_populations)

for idx,v in enumerate(readout_v):
    plt.plot([float(x)+idx for x in v])
plt.title("readout neuron voltages")
plt.show()

fig = plt.figure()
plt.plot(np.array(input_mean_PAs[0])/500.0,color='blue',label='input population PA')
plt.plot(np.transpose(inputs[:,0::5]),color='red',label="input signal")
plt.legend()
plt.show()

PU.plot_simulation(spiketrains_in,spiketrains_hidden_exc,spiketrains_hidden_inh,population_activities,
                input_population_activities,PA_timebin,res_size,N_input,sim_duration,hiddenP_size,
                   hiddenPexc_size,plot_voltage_traces=False,v_hidden_populations_exc=None)


n_it = int(sim_dur1/1000.*sample_freq)-1 #N iteration simulated so far
PU.compare_states(ANN_states=np.zeros((n_it-1,res_size)),SNN_states=mean_PAs,res_size=res_size,n_it=n_it)

# =========================================================================
#                                 save SNNN
# =========================================================================

dic = {'networkStructure':[N_input,N_res,N_readouts],'N_excN_inh':[hiddenPexc_size,hiddenPinh_size],'weights':[weights_input,weights_res,weights_readout], 'Pconnect':[Pconnect_input,Pconnect_res,Pconnect_intra]}
PU.saveSNN2File(dic)