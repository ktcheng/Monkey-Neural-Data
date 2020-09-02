# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:24:12 2020

@author: Kellen Cheng
"""

import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import scipy.special
import scipy.io as sio
import math

data = sio.loadmat("ps4_realdata.mat") # Load the .mat file
NumTrainData = data["train_trial"].shape[0]
NumClass = data["train_trial"].shape[1]
NumTestData = data["test_trial"].shape[0]

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (a)
# Calculate the firing rates
TrainData = data["train_trial"]
TestData = data["test_trial"]

# Contains the firing rates for all neurons on all 8 x 91 trials
trainDataArr =  np.zeros((NumClass,NumTrainData,97)) # for the training set
testDataArr =  np.zeros((NumClass,NumTestData,97)) # for the testing set
bin_width = 0.2 # 200 ms bin

for classIX in range(NumClass):
    for trainDataIX in range(NumTrainData): 
        total_train = (np.sum(TrainData[trainDataIX,classIX][1][:,350:550], 1))
        trainDataArr[classIX,trainDataIX,:] = total_train / bin_width
    for testDataIX in range(NumTestData):  
        total_test = (np.sum(TestData[testDataIX,classIX][1][:,350:550], 1))
        testDataArr[classIX,testDataIX,:]= total_test / bin_width
pass

# Class Prior Calculations
trainParam1 = {} # Dictionary for Gaussian shared covariance
training_prior = float(NumTrainData) / (NumClass * NumTrainData)
trainParam1["pi"] = np.array([training_prior for i in range(NumClass)])

# Mean and Covariance Calculations
neuron_num = trainDataArr.shape[2] # Extracts number of neurons
trainmu_param1 = np.zeros((NumClass, neuron_num)) # Holds means of all data

# n holds our class specific data, trained_shared_cov our class covariances
n = np.zeros((neuron_num, NumTrainData)) # A 97 x 91 matrix for each class
trained_shared_cov = np.zeros((neuron_num, neuron_num)) # A 97 x 97 matrix

for classIX in range(NumClass): 
    for neuron in range(neuron_num):
        neuron_data = trainDataArr[classIX][:, neuron]
        n[neuron] = neuron_data
        
        neuron_mu = np.mean(neuron_data)
        trainmu_param1[classIX][neuron] = neuron_mu
        pass
    trained_shared_cov += np.cov(n)
    pass

trainParam1["mean"] = trainmu_param1
trainParam1["cov"] = trained_shared_cov / NumClass

# Gather model i) parameters
sig_inv_shared = np.linalg.inv(trainParam1["cov"])
c = -1.0 / 2.0 # Common constant

# Retrieve parameters for each class
w_k_params = np.zeros((NumClass, 1, neuron_num)) # Stores w_k values
w_k_consts = np.zeros((NumClass)) 

for classIX in range(NumClass):
    class_mu = trainParam1["mean"][classIX] # Retrieve class mean
    
    w_k_params[classIX] = w_k = sig_inv_shared.dot(class_mu)
    
    w_const = c * ((class_mu.T).dot(sig_inv_shared)).dot(class_mu)
    w_k_consts[classIX] = w_const
    pass
    
# Classify test data, total of 8 x 91 trials
error_total = 0 # Keeps track of erroneously classified trials
total = 0

for classIX in range(NumClass):
    for dataIX in range(NumTestData):
        inputs = testDataArr[classIX][dataIX]
        
        # Calculate alpha(x) for all classes
        z_1 = (w_k_params[0][0].T).dot(inputs) + w_k_consts[0]
        z_2 = (w_k_params[1][0].T).dot(inputs) + w_k_consts[1]
        z_3 = (w_k_params[2][0].T).dot(inputs) + w_k_consts[2]
        z_4 = (w_k_params[3][0].T).dot(inputs) + w_k_consts[3]
        z_5 = (w_k_params[4][0].T).dot(inputs) + w_k_consts[4]
        z_6 = (w_k_params[5][0].T).dot(inputs) + w_k_consts[5]
        z_7 = (w_k_params[6][0].T).dot(inputs) + w_k_consts[6]
        z_8 = (w_k_params[7][0].T).dot(inputs) + w_k_consts[7]
        
        # Obtain the classified class number
        probs = [z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8]
        classified_num = probs.index(max(probs)) # Get the classified class
        
        # if the class is incorrectly classified, add 1 to the error number
        if (classified_num != classIX):
            error_total += 1
            pass
        total += 1
        pass
    pass

# Display our accuracy
msg = "This is our accuracy of classified test data points: "
accuracy = "{:.2f}".format((float(total - error_total) / total) * 100.0)
print(msg + accuracy + "%")

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (b): Generate Python Error- Singular Dimension Matrix

# # Calculate class specific covariances
trainParam2 = {} # Dictionary for Gaussian specific covariance
trainParam2["pi"] = trainParam1["pi"]
trainParam2["mean"] = trainParam1["mean"]

# # Specific Covariance Calculations
# n1 = np.zeros((NumClass, 97, 97)) # 8 x 97 x 97 matrix for all covariances
# n11 = np.zeros((neuron_num, NumTrainData)) # 97 x 91 matrix

# for classIX in range(NumClass):
#     for neuron in range(neuron_num):
#         nx = trainDataArr[classIX, :, neuron]
#         n11[neuron, :] = nx
#     cov_specific = np.cov(n11)
#     n1[classIX, :] = cov_specific
#     pass
# pass
# trainParam2["cov"] = n1

# w_k_params = np.zeros((NumClass, 1, neuron_num))
# w_k_consts = np.zeros((NumClass))

# for classIX in range(NumClass):
#     class_mu = trainParam2["mean"][classIX]
#     sig_inv_specific = np.linalg.inv(trainParam2["cov"][classIX])

#     w_k = sig_inv_specific.dot(class_mu)
#     w_k_params[classIX] = w_k
    
#     w_const = c * ((class_mu.T).dot(sig_inv_specific)).dot(class_mu)
#     w_k_consts[classIX] = w_const
#     pass

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (c)
neuronsToRemove = []

# Iterate through our data, finding our erroneous neurons
for classIX in range(NumClass):
    for neuron in range(neuron_num):
        access = trainDataArr[classIX][:, neuron]
        
        # If the neuron never fires in a class, append to neuronsToRemove
        if (all(access == 0) and (neuron not in neuronsToRemove)):
            neuronsToRemove.append(neuron)
            pass
        pass
    pass
neuronsToRemove.sort() # Sort the list from least to greatest
err_neurons = len(neuronsToRemove) # Number of defective neurons
err_n = err_neurons

# Calculate new means and covariances
new_specificov = np.zeros((NumClass, neuron_num - err_n, neuron_num - err_n))
new_specificlass = np.zeros((neuron_num - err_n, NumTrainData))
trainmu_param2 = np.zeros((NumClass, neuron_num - err_n))

for classIX in range(NumClass):
    counter = 0 # Index counter to account for defective neurons
    for neuron in range(neuron_num):
        if (neuron not in neuronsToRemove):
            valid_data = trainDataArr[classIX][:, neuron]
            new_specificlass[counter, :] = valid_data
            
            trainmu_param2[classIX][counter] = np.mean(valid_data)
            counter += 1
            pass
        pass
    new_specificov[classIX, :] = np.cov(new_specificlass)
    pass

trainParam2["cov"] = new_specificov
trainParam2["mean"] = trainmu_param2

# Retrieve the parameters
w_k_params = np.zeros((NumClass, 1, neuron_num - err_n))
w_k_consts = np.zeros((NumClass))

for classIX in range(NumClass):
    class_mu = trainParam2["mean"][classIX]
    sig_inv_specific = np.linalg.inv(trainParam2["cov"][classIX])

    w_k_params[classIX] = w_k = sig_inv_specific.dot(class_mu)
    
    w_const = c * ((class_mu.T).dot(sig_inv_specific)).dot(class_mu)
    w_const += c * np.log(np.linalg.det(trainParam2["cov"][classIX]))
    w_k_consts[classIX] = w_const
    pass

# Classify test data, total of 8 x 91 trials
error_total = 0 # Keeps track of erroneously classified trials
total = NumClass * NumTestData # Total number of trials

for classIX in range(NumClass):
    for dataIX in range(NumTestData):
        inputs = np.zeros((0))
        for i in range(neuron_num):
            if (i not in neuronsToRemove):
                inputs = np.append(inputs, testDataArr[classIX][dataIX][i])
                pass
            pass
        
        # Calculate quadratic terms
        quad1 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][0]))
        quad1 = quad1.dot(inputs)
        
        quad2 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][1]))
        quad2 = quad2.dot(inputs)
        
        quad3 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][2]))
        quad3 = quad3.dot(inputs)
        
        quad4 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][3]))
        quad4 = quad4.dot(inputs)
        
        quad5 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][4]))
        quad5 = quad5.dot(inputs)
        
        quad6 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][5]))
        quad6 = quad6.dot(inputs)
        
        quad7 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][6]))
        quad7 = quad7.dot(inputs)
        
        quad8 = (inputs.T).dot(np.linalg.inv(trainParam2["cov"][7]))
        quad8 = quad8.dot(inputs)
    
        # Calculate alpha(x) for all classes
        z_1 = (w_k_params[0][0].T).dot(inputs) + w_k_consts[0] + c * quad1
        z_2 = (w_k_params[1][0].T).dot(inputs) + w_k_consts[1] + c * quad2
        z_3 = (w_k_params[2][0].T).dot(inputs) + w_k_consts[2] + c * quad3
        z_4 = (w_k_params[3][0].T).dot(inputs) + w_k_consts[3] + c * quad4
        z_5 = (w_k_params[4][0].T).dot(inputs) + w_k_consts[4] + c * quad5
        z_6 = (w_k_params[5][0].T).dot(inputs) + w_k_consts[5] + c * quad6
        z_7 = (w_k_params[6][0].T).dot(inputs) + w_k_consts[6] + c * quad7
        z_8 = (w_k_params[7][0].T).dot(inputs) + w_k_consts[7] + c * quad8
    
        # Obtain the classified class number
        probs = [z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8]
        classified_num = probs.index(max(probs))
        
        # If the class is incorrectly classified, add 1 to the error number
        if (classified_num != classIX):
            error_total += 1
            pass
        pass
    pass

# Display our accuracy
msg = "This is our accuracy of classified test data points: "
accuracy = "{:.2f}".format((float(total - error_total) / total) * 100.0)
print(msg + accuracy + "%")

# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
# ========================================================================== #
### Part (d)
trainParam3 = {} # Dictionary for Poisson distribution
trainParam3["pi"] = trainParam2["pi"]
trainParam3["mean"] = trainParam2["mean"]

poiss_error_total = 0 # Keeps track of erroneously classified trials
poiss_total = NumClass * NumTestData # Total number of trials

for classIX in range(NumClass):
    for dataIX in range(NumTestData):
        inputs = np.zeros((0))
        for neuron in range(neuron_num):
            if (neuron not in neuronsToRemove):
                inputs = np.append(inputs, testDataArr[classIX][dataIX][neuron])
                pass
            pass
        
        # Retrieve Poisson parameters
        poissmean1 = trainParam3["mean"][0]
        poissmean2 = trainParam3["mean"][1]
        poissmean3 = trainParam3["mean"][2]
        poissmean4 = trainParam3["mean"][3]
        poissmean5 = trainParam3["mean"][4]
        poissmean6 = trainParam3["mean"][5]
        poissmean7 = trainParam3["mean"][6]
        poissmean8 = trainParam3["mean"][7]

        # Calculate alpha(x) for all classes
        z_1 = inputs.dot(np.log(poissmean1))
        z_2 = inputs.dot(np.log(poissmean2))
        z_3 = inputs.dot(np.log(poissmean3))
        z_4 = inputs.dot(np.log(poissmean4))
        z_5 = inputs.dot(np.log(poissmean5))
        z_6 = inputs.dot(np.log(poissmean6))
        z_7 = inputs.dot(np.log(poissmean7))
        z_8 = inputs.dot(np.log(poissmean8))
        
        for i in range(neuron_num - err_n): # Subtract lambda_{ki}
            z_1 -= poissmean1[i]
            z_2 -= poissmean2[i]
            z_3 -= poissmean3[i]
            z_4 -= poissmean4[i]
            z_5 -= poissmean5[i]
            z_6 -= poissmean6[i]
            z_7 -= poissmean7[i]
            z_8 -= poissmean8[i]
        
        # Obtain the classified class number
        probs = [z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8]
        classified_num = probs.index(max(probs))
        
        # If the class is incorrectly classified, add 1 to the error number
        if (classified_num != classIX):
            poiss_error_total += 1
            pass
        pass
    pass

# Display our accuracy
msg = "This is our accuracy of classified test data points: "
accuracy = "{:.2f}".format((float(total - poiss_error_total) / total) * 100.0)
print(msg + accuracy + "%")