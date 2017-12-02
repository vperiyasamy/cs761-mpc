from apm import *
import numpy as np
import random
import pickle
import apm_functions as apmfun

# Number of cycles to run
cycles = 100

# Initialize data arrays
y_store = np.zeros(cycles)
y_pred5 = np.zeros(cycles)

# Initialize simulater values
u           = 0.0
yp5         = 0.0
setpoints   = np.array([10])
sp_times    = np.array([0])
sim_Ks      = np.array([10, 5]) # dynamically change the process simulation model
sim_K_times = np.array([0, 50])

# Load the classifiers
clfs = []
for i in range(3):
    with open('LinearClassifier' + str(i) + '.pkl', 'rb') as f:
        clfs.append(pickle.load(f))

# Set machine learning parameters
models             = [1, 5, 10]
model_index        = 1
clf                = clfs[model_index]
mismatch_threshold = 0.0
clf_threshold      = 0.2
window_size        = 10
labels             = []
most_common_labels = np.ones(cycles).astype(int)

# Initialize the model and the controller
server = 'http://byu.apmonitor.com'
app1   = 'sim' + str(int(random.random() * 10000))
app2   = 'mpc' + str(int(random.random() * 10000))
print(apmfun.sim_init(server, app1))
print(apmfun.mpc_init(server, app2))

# Run cycles
for i in range(cycles):
    print('Cycle: ' + str(i + 1))
    
    ## Change the setpoint
    if np.any(sp_times == i):
        setpoint_index = np.where(sp_times == i)[0][0]
        setpoint = setpoints[setpoint_index]
    
    ## Process simulator
    if np.any(sim_K_times == i):
        sim_K = sim_Ks[sim_K_times == i][0]
    measurement_noise = 0.1 * (random.random() - 0.5)
    y_meas            = apmfun.sim(server, app1, u, sim_K) + measurement_noise
    
    ## Save data
    y_store[i] = y_meas
    y_pred5[i] = yp5
           
    ## Create and update the moving window
    if np.any(i - 4 > sp_times[setpoint_index:]):
        # Evaluate the feature values
        feature1 = abs(y_pred5[i - 5] - y_store[i])
        feature2 = abs(y_store[i] - y_store[i - 1])
        # Use the classifier and save the predicted labels in the moving window
        x = np.array([feature1, feature2, u])[None, :]
        if feature1 > mismatch_threshold:
            labels.append(clf.predict(x)[0])
        else:
            labels.append(most_common_labels[i - 1])
        # Update the classifier
        if feature1 < clf_threshold:
            clf = clfs[most_common_labels[i - 1]]
        # Enforce the window size constraint
        if len(labels) > window_size:
            labels = labels[1:]
    
    ## Decide which model should be used
    if len(labels) == window_size:
        most_common_labels[i] = np.argmax(np.bincount(np.array(labels)))
        
    ## Model Predictive Control (MPC)
    u, yp5 = apmfun.mpc(server, app2, [setpoint, y_meas, models[most_common_labels[i]]])
	
    if (i == 0):
       # Web viewers to see solution progression
       apm_web(server, app2) # Controller
#       apm_web(server, app1)

    
