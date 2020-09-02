# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:49:45 2020

@author: Kellen Cheng
"""

##### Homework 3 : Problem 3
import numpy as np
import matplotlib.pyplot as plt
import nsp as nsp
import scipy.special

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (a)
r_0 = 35 # (spikes / s)
r_max = 60 # (spikes / s)
s_max = np.pi / 2 # (radians)
T = 1000 # trial length (ms)

np.random.exponential(1.0 / r_max * 1000)

num_trials = 100 # number of total spike trains
num_rasters_to_plot = 5 # number of spike trains to plot
#====================================================#
# YOUR CODE HERE:
#   Generate the spike times for 100 trials of an inhomogeneous
#   Poisson process.  Plot 5 example spike rasters.
#====================================================#
# Stores all 100 spike trains
inhom_spike_times = np.zeros((num_trials), dtype = list) 

for trial in range(num_trials):
    # Generate homogeneous Poisson Process, removing first time of 0.0
    homogeneous_spike_train = nsp.GeneratePoissonSpikeTrain(T, r_max)[1:] 
    inhom_spike_train = np.empty(0) # Stores inhomogeneous spike train
    
    for spike in homogeneous_spike_train:
        s = ((spike / 1000.0) ** 2.0) * np.pi # Calculate s(t)
        l = r_0 + (r_max - r_0) * np.cos(s - s_max)
        
        threshold = float(l / r_max) # Probability we keep the spike
        u = np.random.uniform(0, 1) # Sample from uniform distribution
        
        if (threshold > u): # In this case keep the spike
            inhom_spike_train = np.append(inhom_spike_train, spike)
    
    inhom_spike_times[trial] = inhom_spike_train
    
# Plot five example spike rasters
plt.figure(figsize = (10, 8))
nsp.PlotSpikeRaster(inhom_spike_times[0: num_rasters_to_plot])

plt.title("5 Example Spike Rasters")
plt.tight_layout()
#====================================================#
# END YOUR CODE
#====================================================#
    
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (b)
plt.figure(figsize = (10, 8))
bin_width = 12 # (ms)
#====================================================#
# YOUR CODE HERE:
#   Plot the spike histogram
#====================================================#
b = [i * 20 for i in range(50)] # Creates bin list
hist_values = np.zeros(49) # Creates empty numpy array

for trial in range(num_trials):
    hist, bin_edges = np.histogram(inhom_spike_times[trial], b)
    hist_values = np.add(hist_values, hist)

# Plot the tuning curve
x_axis = np.linspace(0, 1000, 1000)
tuning_x = np.empty(0) # Stores tuning curve parameters

for x in np.linspace(0, 1, 1000):
    tuning_x = np.append(tuning_x, (x ** 2) * np.pi)
tuning_y = r_0 + (r_max - r_0) * np.cos(tuning_x - s_max)

# Divide by 100 (trials), divide by 0.02 (20 milliseconds)
hist_values = np.divide(hist_values, 2.0) 
plt.bar(bin_edges[:-1], hist_values, width = 20, align = "edge")

# Plot the tuning curve
plt.plot(x_axis, tuning_y, c = "red")
plt.title("Firing Rate Histogram")
#====================================================#
# END YOUR CODE
#====================================================#
plt.ylabel('spikes/s')
plt.xlabel('Time(ms)')    

## Answer: It appears to loosely agree with the expected firing rate profile.
## The rate increases to a maximum around 600-800 ms, and then levels off.

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (c)
plt.figure(figsize = (10, 8))
#====================================================#
# YOUR CODE HERE:
#   Plot the normalized distribution of spike counts
#====================================================#
spike_counts = [] # Stores spike count for all trials

for trial in range(num_trials):
    spike_counts.append(len(inhom_spike_times[trial])) # Append trial count

mu = sum(spike_counts) / float(len(spike_counts)) # Calculate mean count
k = [i for i in range(70)]
poison = [((mu ** k) * np.exp(-mu)) / np.math.factorial(k) for k in k]

b = [i for i in range(70)]
plt.hist(spike_counts, bins = b, density = True) # Plot histogram of spike counts
plt.plot(k, poison) # Plot the Poisson distribution
#====================================================#
# END YOUR CODE
#====================================================#
plt.xlabel('spike count')
plt.ylabel('p(spikecount)')
plt.show()

## Answer: Yes, we should expect the spike counts to be Poisson-distributed,
## which is what the graph depicts. Even though the data derives from an 
## inhomogeneous Poisson Process, the spikes are still distributed according to
## the Poisson distribution.

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (d)
plt.figure(figsize = (10, 8))
#====================================================#
# YOUR CODE HERE:
#   Plot the normalized distribution of ISIs
#====================================================#
isi_times = [] # List to keep track of all ISI times
for trial in range(num_trials):
    train = inhom_spike_times[trial] # Obtains the spike train for one trial
    
    # Calculate ISI intervals from spike times
    for i in range(len(train)):
        if i == 0:
            isi_times.append(train[i]) # Append first time
        else:
            t_isi = train[i] - train[i - 1] # Calculate the interspike interval
            isi_times.append(t_isi) # Append ISI time to list
    
mean_isi = sum(isi_times) / len(isi_times) # Calculates average ISI time
l = 1.0 / mean_isi # Exponential lambda is reciprocal of the mean
x = np.linspace(0, 200, 400)
exponential = l * np.exp(-l * x) # Calculates exponential distribution

plt.hist(isi_times, density = "True") # Plot histogram of ISI times
plt.plot(x, exponential) # Plot exponential distribution
#====================================================#
# END YOUR CODE
#====================================================#
plt.xlabel('ISI (ms)')
plt.ylabel('P(ISI)')
plt.show()

## Answer: While we do see the empirical distribution to strongly resemble an
## exponential distribution, we know from our course concepts that for an
## inhomogeneous Poisson Process, the ISI times are not memoryless, not 
## independent and not exponentially distributed. As a result, even though the 
## graph bears strong resemblance to an exponential distribution, we know that
## the ISI's for an inhomogeneous Poisson Process cannot be exponentially 
## distributed.