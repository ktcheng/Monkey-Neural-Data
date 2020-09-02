# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:47:17 2020

@author: Kellen Cheng
"""

##### Homework 3 : Problem 2
import numpy as np
import matplotlib.pyplot as plt
import nsp as nsp # Includes helper functions
import scipy.special

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (a)
bin_width = 20 # (ms)
s = np.arange(8) * np.pi/4 # (radians)
num_cons = np.size(s) # num_cons = 8 in this case, number of directions
r_0 = 35 # (spikes/s)
r_max = 60 # (spikes/s)
s_max = np.pi / 2 # (radians)
T = 1000 # trial length (ms)
num_trials = 100 # number of spike trains to generate

tuning = r_0 + (r_max - r_0) * np.cos(s - s_max) # tuning curve
spike_times = np.empty((num_cons, num_trials), dtype = list)

for con in range(num_cons):
    s_0 = s[con] # Extracts s parameter on each iteration
    l = r_0 + (r_max - r_0) * np.cos(s_0 - s_max) # Calculates lambda rate
    
    for rep in range(num_trials):    
        #====================================================#
        # YOUR CODE HERE:
        #   Generate homogeneous Poisson process spike trains.
        #   You should populate the np.ndarray 'spike_times' according
        #   to the above description.
        #====================================================#
        # Generates spike train for each trial, excluding first time of 0.0
        spike_times[con, rep] = nsp.GeneratePoissonSpikeTrain(T, l)[1:] 
        #====================================================#
        # END YOUR CODE
        #====================================================#

s_labels = ['0', '$\pi$/4', '$\pi$/2', '3$\pi$/4', '$\pi$', '5$\pi$/4', '3$\pi$/2', '7$\pi$/4']
num_plot_rows = 5
num_plot_cols = 3
subplot_indx = [9, 6, 2, 4, 7, 10, 14, 12]
num_rasters_to_plot = 5 # per condition

# Generate and plot homogeneous Poisson process spike trains
plt.figure(figsize=(10,8))
for con in range(num_cons):

    # Plot spike rasters
    plt.subplot(num_plot_rows, num_plot_cols, subplot_indx[con])    
    nsp.PlotSpikeRaster(spike_times[con, 0:num_rasters_to_plot])
    
    plt.title('Spike trains, s= '+s_labels[con]+' radians')
    plt.tight_layout()
    
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (b)
plt.figure(figsize=(10,8))

b = [i * 20 for i in range(50)] # Creates bin list

for con in range(num_cons): # QUESTION: WHY SO MANY SPIKES UP FRONT
    plt.subplot(num_plot_rows,num_plot_cols,subplot_indx[con])
    #====================================================#
    # YOUR CODE HERE:
    #   Generate and plot spike histogram for this condition
    #====================================================#
    hist_values = np.zeros(49) # Creates empty numpy array

    for trial in range(num_trials):
        hist, bin_edges = np.histogram(spike_times[con, trial], b)
        hist_values = np.add(hist_values, hist)

    # Divide by 100 (trials), divide by 0.02 (20 milliseconds)
    hist_values = np.divide(hist_values, 2.0) 
    plt.bar(bin_edges[:-1], hist_values, width = 20, align = "edge")
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing Rate (spikes/s)")
    plt.ylim(0, r_max + 15)
    #====================================================#
    # END YOUR CODE
    #====================================================#
    plt.title('Spike trains, s= '+s_labels[con]+' radians')
    plt.tight_layout()
    
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (c)
plt.figure(figsize = (10, 8))
spike_counts = np.zeros((num_cons, num_trials)) # each element in spike_counts is the total spike count for this reach direction and trial 
#====================================================#
# YOUR CODE HERE:
#   Plot the single trial spike counts and the tuning curve
#   on top of each other.
#====================================================#
x = [] # List to hold x-coordinates of all points
y = [] # List to hold firing rates for all points
avg_x = [] # List to hold eight reach directions
avg_y = [] # List to hold mean firing rates for each reach angle

tuning_x = np.linspace(0, 6, 6000)
tuning_y = r_0 + (r_max - r_0) * np.cos(tuning_x - s_max) # Tuning curve

for con in range(num_cons):
    s_0 = [s[con] for i in range(num_trials)]
    x.append(s_0) # Populate x with 100 copies of each reach angle
    avg_x.append(s[con])
    
    counter = []
    mean_spikes = 0 # Stores mean firing rate
    
    for trial in range(num_trials):
        num_spikes = len(spike_times[con, trial])
        y.append(num_spikes) # This is the firing rate; time/trial is 1 second
        counter.append(num_spikes)
        mean_spikes += num_spikes
    
    spike_counts[con, :] = counter # Update spike_counts list accordingly
    
    # Calculate mean firing rate
    mean_spikes = np.divide(mean_spikes, float(num_trials)) 
    avg_y.append(mean_spikes) # Append mean firing rate
        
plt.scatter(x, y, s = 1) # Scatters all 800 points
plt.scatter(avg_x, avg_y, c = "red") # Scatteres mean firing rates
plt.plot(tuning_x, tuning_y, c = "green") # Plots tuning curve
#====================================================#
# END YOUR CODE
#====================================================#
plt.xlabel('Reach angle (radians)')
plt.ylabel('Firing rate (spikes / second)')
plt.title('Simulated spike counts (blue)\n'+
            'mean simulated spike counts (red),and\n'+
            'cosine tuning curve used in simulation (green)')
plt.xlim(0, 2*np.pi)

# Answer: The mean firing rates do lie near, or approximately near, the tuning curve.

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (d)
plt.figure(figsize=(10,8))
max_count = np.max(spike_counts)
spike_count_bin_centers =  np.arange(0,max_count,1)

for con in range(num_cons):
    plt.subplot(num_plot_rows,num_plot_cols,subplot_indx[con])
    
    #====================================================#
    # YOUR CODE HERE:
    #   Calculate the empirical mean for the Poisson spike
    #   counts, and then generate a curve reflecting the probability
    #   mass function of the Poisson distribution as a function
    #   of spike counts.
    #====================================================#
    # Plots normalized histogram
    b = [i * 2 for i in range(40)] # Create bin list
    plt.hist(spike_counts[con], bins = b, density = True) 
    
    trial = spike_counts[con, :] # Obtains data from each trial
    mean = sum(trial) / float(num_trials) # Equivalent to lambda
    k = [i for i in range(num_trials)]
    
    # List that holds Poisson distribution values
    poison = [((mean ** k) * np.exp(-mean)) / np.math.factorial(k) for k in k]
    #====================================================#
    # END YOUR CODE
    #====================================================#
    
    #====================================================#
    # YOUR CODE HERE:
    #   Plot the empirical count distribution, and on top of it 
    #   plot your fit Poisson distribution.
    #====================================================#
    plt.plot(k, poison) # Plots the Poisson distribution
    plt.xlabel("Spike Number")
    plt.ylabel("Probability")
    #====================================================#
    # END YOUR CODE
    #====================================================#
    plt.xlim([0, max_count])
    plt.title('Count distribution, s= '+ s_labels[con]+' radians')
    plt.tight_layout() 
    
## Answer: The empirical distributions are reasonably well-fit by the Poisson 
## distributions.

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (e)
plt.figure(figsize = (10, 8))
mean = [] # List to store mean spike counts
variance = [] # List to store variance of spike counts

for con in range(num_cons):
    # Append the average spike count
    mean.append(sum(spike_counts[con, :]) / float(num_trials))
    
    total = 0.0 # Tracks variance of spike counts
    for trial in range(num_trials):
        total += (spike_counts[con, trial] - mean[con]) ** 2.0
    
    total /= float(num_trials)
    variance.append(total)

# 45 Degree Diagonal
x = np.linspace(0, 60, 600)
y = x

# Plot Appropriately
plt.scatter(mean, variance)
plt.plot(x, y)
plt.xlabel("Mean of Spike Counts")
plt.ylabel("Variance of Spike Counts")
plt.title("Mean v. Variance of Spike Counts")

## Answer: The points do appear to lie near the 45 degree diagonal.

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (f)
plt.figure(figsize=(10,8))

for con in range(num_cons) : # num_cons
    plt.subplot(num_plot_rows,num_plot_cols,subplot_indx[con])
    #====================================================#
    # YOUR CODE HERE:
    #   Calculate the interspike interval (ISI) distribution
    #   by finding the empirical mean of the ISI's, which 
    #   is the inverse of the rate of the distribution.
    #====================================================#
    isi_dist_row = [] # List to keep track of ISI times
    for trial in range(num_trials):
        counter = spike_times[con, trial] # Obtains spike train for one trial
        
        for i in range(len(counter)): 
            if (i == 0):
                t_isi = counter[i]
            else: # Calculate the interspike interval
                t_isi = counter[i] - counter[i - 1] 
            isi_dist_row.append(t_isi)
    
    # Calculates average of each reach angle    
    mean_isi = sum(isi_dist_row) / len(isi_dist_row) 
    
    # Plots the normalized distribution
    b = [i * 3 for i in range(100)] # Create bin list
    plt.hist(isi_dist_row, bins = b, density = "True") 
    #====================================================#
    # END YOUR CODE
    #====================================================#  
    
    #====================================================#
    # YOUR CODE HERE:
    #   Plot Interspike interval (ISI) distribution
    #====================================================#    
    l = 1.0 / mean_isi # Calculates lambda for an exponential distribution
    x = np.linspace(0, 300, 600)
    exponential = l * np.exp(-l * x) # Exponential distribution equation

    plt.plot(x, exponential) # Plot the exponential distribution
    plt.xlabel("ISI Times (ms)")
    plt.ylabel("Probability")
    plt.xlim(0, 300)
    #====================================================#
    # END YOUR CODE
    #====================================================#   
    plt.title('ISI distribution, s= '+ s_labels[con]+' radians')
    plt.tight_layout() 
    
## Answer: The empirical distributions are reasonably well-fit by the 
## exponential distributions.
    
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (g)
plt.figure(figsize = (10, 8))
avg_isi = [] # List to hold average ISI values
c_v = [] # List to hold coefficient of variation values

for con in range(num_cons):
    isi_dist_row = [] # List to keep track of ISI times
    
    for trial in range(num_trials):
        counter = spike_times[con, trial] # Obtains spike train for one trial
        
        for i in range(len(counter)): 
            if (i == 0):
                t_isi = counter[i]
            else: # Calculate the interspike interval
                t_isi = counter[i] - counter[i - 1]
            isi_dist_row.append(t_isi)
      
    # Calculates the average of each reach angle
    mean_isi = sum(isi_dist_row) / len(isi_dist_row) 
    avg_isi.append(mean_isi)
    
    total = 0.0 # Keeps track of variance calculation
    
    for time in isi_dist_row:
        total += (time - mean_isi) ** 2
        
    total /= float(len(isi_dist_row))
    std_deviation = total ** 0.5
    c_v.append(std_deviation / mean_isi)
    
plt.scatter(avg_isi, c_v)
plt.xlabel("Average ISI (ms)")
plt.ylabel("Coefficient of Variation")
plt.title("ISI v. Coefficient of Variation for ISI")
plt.ylim(0.0, 1.05)

## Answer: From the scatter plot, we can clearly see that the coefficients of 
## variation lie around, and very close to, unity (or 1.00), as is expected of
## a Poisson Process.