# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:21:45 2020

@author: Kellen Cheng
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.io as sio
import math

data = sio.loadmat("ps4_simdata.mat") # Loat the .mat file
NumData = data["trial"].shape[0]
NumClass = data["trial"].shape[1]

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (a)
plt.figure(figsize=(7,7))
#====================================================#
# PLOTTING CODE BELOW
#====================================================#
dataArr =  np.zeros((NumClass,NumData ,2)) # dataArr contains the points
for classIX in range(NumClass):
    for dataIX in range(NumData):
        x = data['trial'][dataIX,classIX][0][0][0]
        y = data['trial'][dataIX,classIX][0][1][0]        
        dataArr[classIX,dataIX,0]=x
        dataArr[classIX,dataIX,1]=y
MarkerPat=np.array(['rx','g+','b*'])

for classIX in range(NumClass):
    for dataIX in range(NumData):
        plt.plot(dataArr[classIX,dataIX,0],dataArr[classIX,dataIX,1],MarkerPat[classIX])

#====================================================#
# END PLOTTING CODE
#====================================================# 
plt.axis([0,20,0,20])
plt.xlabel('x_1')
plt.ylabel('x_2')

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (b)
#====================================================#
# YOUR CODE HERE:
#   Find the parameters for each model you derived in problem 1 using
#   the simulated data, and print out the values of each parameter.
#
#   To facilitate plotting later on, we're going to ask you to 
#   format the data in the following way.
#   
#   (1) Keep three dictionaries, modParam1, modParam2, and modParam3
#       which contain the model parameters for model 1 (Gaussian, shared cov),
#       model 2 (Gaussian, class specific cov), and model 3 (Poisson).
#     
#       The Python dictionary is like a MATLAB struct. e.g., you can declare:
#       modParam1 = {} # declares the dictionary
#       modParam1['pi'] = np.array((0.33, 0.33, 0.34)) # sets the field 'pi' to be
#         an np.array of size (3,) containing the class probabilities.
#
#   (2) modParam1 has the following structure
#
#     modParam1['pi'] is an np.array of size (3,) containing the class probabilities.
#     modParam1['mean'] is an np.array of size (3,2) containing the class means.
#     modParam1['cov'] is an np.array of size (2,2) containing the shared cov.
#
#   (3) modParam2: 
#
#     modParam2['pi'] is an np.array of size (3,) containing the class probabilities.
#     modParam2['mean'] is an np.array of size (3,2) containing the class means.
#     modParam2['cov'] is an np.array of size (3,2,2) containing the cov for each of the 3 classes.
#
#   (4) modParam3:
#     modParam2['pi'] is an np.array of size (3,) containing the class probabilities.
#     modParam2['mean'] is an np.array of size (3,2) containing the Poisson parameters for each class.
#
#   These should be consistent with the print statement after this code block.
#
#   HINT: the np.mean and np.cov functions ought simplify the code.
#
#====================================================#
modParam1 = {} # Dictionary for Gaussian shared covariance
modParam2 = {} # Dictionary for Gaussian specific covariance
modParam3 = {} # Dictionary for Poisson distribution

# Class Prior Calculations
prior = float(NumData) / (NumData * NumClass)
modParam1["pi"] = np.array([prior for i in range(3)])
modParam3["pi"] = modParam2["pi"] = modParam1["pi"]

# Class Mean Calculations
mu_param1 = np.zeros((3, 2))
for classIX in range(NumClass):
    # For each class, calculate mean of neuron 1, neuron 2
    mu_param1[classIX][0] = np.mean(dataArr[classIX][:, 0]) # x - axis 
    mu_param1[classIX][1] = np.mean(dataArr[classIX][:, 1]) # y - axis
modParam3["mean"] = modParam2["mean"] = modParam1["mean"]= mu_param1

# Covariance Calculations (specific and shared)
specificov = np.zeros((3, 2, 2)) # Holds class specific covariance matrices
shared_cov = np.zeros((2, 2)) # Holds shared covariance matrix

for classIX in range(NumClass):
    x = dataArr[classIX][:, 0] # Holds data across entire class for neuron 1
    y = dataArr[classIX][:, 1] # Holds data across entire class for neuron 2
    
    cov_class = np.cov(np.stack((x, y), axis = 0))
    specificov[classIX] = cov_class # Sets the class specific covariance
    shared_cov += cov_class
    pass
shared_cov /= NumClass # Divide by NumClass to get weighted average

modParam1["cov"] = shared_cov
modParam2["cov"] = specificov
#====================================================#
# END YOUR CODE
#====================================================# 

# Print out the model parameters
print("Model 1:")
print("Class priors:")
print( modParam1['pi'])
print("Means:")
print( modParam1['mean'])
print("Cov:")
print( modParam1['cov'])

print("model 2:")
print("Class priors:")
print( modParam2['pi'])
print("Means:")
print( modParam2['mean'])
print("Cov1:")
print( modParam2['cov'][0])
print("Cov2:")
print( modParam2['cov'][1])
print("Cov3:")
print( modParam2['cov'][2])

print("model 3:")
print("Class priors:")
print( modParam3['pi'])
print("Lambdas:")
print( modParam3['mean'])

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (c)
plt.figure(2, figsize=(7,7)) # Made modification here
#====================================================#
# ML MEAN PLOT CODE HERE.
#====================================================#
colors = ['r.','g.','b.']
for classIX in range(NumClass):
    for dataIX in range(NumData):
        plt.plot(dataArr[classIX,dataIX,0],dataArr[classIX,dataIX,1],MarkerPat[classIX])
        plt.plot(modParam1['mean'][classIX,0],modParam1['mean'][classIX,1],colors[classIX],markersize=30)
#====================================================#
# END CODE
#====================================================# 
plt.axis([0,20,0,20])
plt.xlabel('x_1')
plt.ylabel('x_2')

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (d)
#====================================================#
# ML COV PLOT CODE HERE.
#====================================================#
colors2 = ['r','g','b']
modParam = [modParam1 , modParam2]
for modelIX in range(2):
    plt.figure(modelIX,figsize=(7,7))
    for classIX in range(NumClass):
        for dataIX in range(NumData):
            #plot the points and their means, just like before
            plt.plot(dataArr[classIX,dataIX,0],dataArr[classIX,dataIX,1],MarkerPat[classIX])
            plt.plot(modParam[modelIX]['mean'][classIX,0],modParam[modelIX]['mean'][classIX,1],colors[classIX],markersize=30)
        plt.axis([0,20,0,20])
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        MarkerCol=['r','g','b']
        
    #now begins plotting the elipse
    for classIX in range(NumClass):
        currMean=modParam[modelIX]['mean'][classIX ,:]
        if(modelIX == 0):
            currCov=modParam[modelIX]['cov']
        else:
            currCov=modParam[modelIX]['cov'][classIX]
        xl = np.linspace(0, 20, 201)
        yl = np.linspace(0, 20, 201)
        [X,Y] = np.meshgrid(xl,yl)

        Xlong = np.reshape(X-currMean[0],(np.prod(np.size(X))))
        Ylong = np.reshape(Y-currMean[1],(np.prod(np.size(X))))
        temp = np.row_stack([Xlong,Ylong])
        Zlong = []
        for i in range(np.size(Xlong)):
            Zlong.append(np.matmul(np.matmul(temp[:,i], np.linalg.inv(currCov)), temp[:,i].T))
        Zlong = np.matrix(Zlong)
        Zlong = np.exp(-Zlong/2)/np.sqrt((2*np.pi)*(2*np.pi)*np.linalg.det(currCov))
        Z = np.reshape(Zlong,X.shape)
        isoThr=[0.007]
        plt.contour(X,Y,Z,levels = isoThr,colors = colors2[classIX])
#====================================================#
# END CODE
#====================================================# 

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (e)
# Class 1 is red, Class 2 is green, Class 3 is blue
NNN = 1000 # Indicates step size, and thus accuracy of meshgrid classification
[X, Y] = np.meshgrid(np.linspace(0, 20, NNN), np.linspace(0, 20, NNN))
# Meshgrid Classifier Function
def z_classifier(X, Y, N, model):
	
    # General Parameters (pertinent to all three models)
    c = -1.0 / 2 # Common constant
    
    mu_k1 = modParam1["mean"][0] # Mean of class 1
    mu_k2 = modParam1["mean"][1] # Mean of class 2
    mu_k3 = modParam1["mean"][2] # Mean of class 3
    
    # 1 corresponds to model i), 2 to model ii), 3 to model iii)
    if (model == 1):
        # Define model i) parameters accordingly
        sigma_inv_shared = np.linalg.inv(modParam1["cov"])
        
        # w_k is our slope, w_k0 is our constant
        w_1 = sigma_inv_shared.dot(mu_k1)
        w_10 = (c * (mu_k1.T).dot(sigma_inv_shared)).dot(mu_k1)

        w_2 = sigma_inv_shared.dot(mu_k2)
        w_20 = (c * (mu_k2.T).dot(sigma_inv_shared)).dot(mu_k2)

        w_3 = sigma_inv_shared.dot(mu_k3)
        w_30 = (c * (mu_k3.T).dot(sigma_inv_shared)).dot(mu_k3)
        
        # Calculate alpha(x) probabilities
        z_1 = w_1.T[0] * X + w_1.T[1] * Y + w_10 # Class 1
        z_2 = w_2.T[0] * X + w_2.T[1] * Y + w_20 # Class 2
        z_3 = w_3.T[0] * X + w_3.T[1] * Y + w_30 # Class 3	
        
    elif (model == 2):
        # Define model ii) parameters accordingly
        sigma_k1_inv = np.linalg.inv(modParam2["cov"][0])
        sigma_k2_inv = np.linalg.inv(modParam2["cov"][1])
        sigma_k3_inv = np.linalg.inv(modParam2["cov"][2])
        sigma_k1 = np.linalg.inv(sigma_k1_inv)
        sigma_k2 = np.linalg.inv(sigma_k2_inv)
        sigma_k3 = np.linalg.inv(sigma_k3_inv)
        
        w_1 = sigma_k1_inv.dot(mu_k1)
        w_10 = c * (mu_k1.T).dot(sigma_k1_inv)
        w_10 = w_10.dot(mu_k1) + c * np.log(np.linalg.det(sigma_k1))

        w_2 = sigma_k2_inv.dot(mu_k2)
        w_20 = c * (mu_k2.T).dot(sigma_k2_inv)
        w_20 = w_20.dot(mu_k2) + c * np.log(np.linalg.det(sigma_k2))

        w_3 = sigma_k3_inv.dot(mu_k3)
        w_30 = c * (mu_k3.T).dot(sigma_k3_inv)
        w_30 = w_30.dot(mu_k3) + c * np.log(np.linalg.det(sigma_k3))
        
        # Calculate the quadratic terms
        w_1temp0 = X * sigma_k1_inv[0][0] + Y * sigma_k1_inv[1][0]
        w_1temp1 = X * sigma_k1_inv[0][1] + Y * sigma_k1_inv[1][1]
        w_temp1 = w_1temp0 * X + w_1temp1 * Y
       
        w_2temp0 = X * sigma_k2_inv[0][0] + Y * sigma_k2_inv[1][0]
        w_2temp1 = X * sigma_k2_inv[0][1] + Y * sigma_k2_inv[1][1]
        w_temp2 = w_2temp0 * X + w_2temp1 * Y
        
        w_3temp0 = X * sigma_k3_inv[0][0] + Y * sigma_k3_inv[1][0]
        w_3temp1 = X * sigma_k3_inv[0][1] + Y * sigma_k3_inv[1][1]
        w_temp3 = w_3temp0 * X + w_3temp1 * Y
        
        # Calculate alpha(x) probabilities
        z_1 = c * w_temp1 + w_1.T[0] * X + w_1.T[1] * Y + w_10
        z_2 = c * w_temp2 + w_2.T[0] * X + w_2.T[1] * Y + w_20
        z_3 = c * w_temp3 + w_3.T[0] * X + w_3.T[1] * Y + w_30
    
        Z = np.zeros((N, N)) # Resultant Matrix (aka heatmap)
                
    else:
        # Define model iii) parameters accordingly
        poissmean1 = modParam3["mean"][0]
        poissmean2 = modParam3["mean"][1]
        poissmean3 = modParam3["mean"][2]
        
        # Calculate alpha(x) probabilities
        z_1 = X * np.log(poissmean1[0]) + Y * np.log(poissmean1[1])
        z_1 = z_1 - poissmean1[0] - poissmean1[1]
        
        z_2 = X * np.log(poissmean2[0]) + Y * np.log(poissmean2[1])
        z_2 = z_2 - poissmean2[0] - poissmean2[1]
        
        z_3 = X * np.log(poissmean3[0]) + Y * np.log(poissmean3[1])
        z_3 = z_3 - poissmean3[0] - poissmean3[1]
       
    # Classify our points accordingly   
    Z = np.zeros((N, N)) # Resultant Matrix
        
    for i in range(N):
        for j in range(N):
            num = max(z_1[i][j], z_2[i][j], z_3[i][j])
            
            # Determine the class of num
            if (num == z_1[i][j]):
                num = 1
                pass
            elif (num == z_2[i][j]):
                num = 2
                pass
            else:
                num = 3
                pass
            
            Z[i, j] = num
            pass
        pass
    pass
    
    return(Z)

# Plot our classified meshgrid
# Color choices come from colors that are similar, but distinguishable to plot
colormap = plt.cm.colors.ListedColormap(["pink", "cyan", "lavender"])
bounds = [1, 2, 3] # Class 1 is pink, Class 2 is cyan, Class 3 is lavendar 

plt.figure(0)
h1 = z_classifier(X, Y, NNN, 1)
plt.pcolormesh(X, Y, h1, cmap = colormap, alpha = 0.5)

plt.figure(1)
h = z_classifier(X, Y, NNN, 2)
plt.pcolormesh(X, Y, h, cmap = colormap, alpha = 0.5)

plt.figure(2)
h2 = z_classifier(X, Y, NNN, 3)
plt.pcolormesh(X, Y, h2, cmap = colormap, alpha = 0.5)