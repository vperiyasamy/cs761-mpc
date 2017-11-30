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
setpoints   = np.array([10, 15])
change_time = np.array([0, 60])

# Load the classifier
with open('LinearClassifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Set machine learning parameters
models             = [1, 5, 10]
model_index        = 1
mismatch_threshold = 0.5
window_size        = 20
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
    if np.any(change_time == i):
        setpoint_index = np.where(change_time == i)[0][0]
        setpoint = setpoints[setpoint_index]
    
    ## Process simulator
    measurement_noise = 0.1 * (random.random() - 0.5)
    y_meas            = apmfun.sim(server, app1, u) + measurement_noise
    
    ## Save data
    y_store[i] = y_meas
    y_pred5[i] = yp5
           
    ## Create and update the moving window
    if np.any(i - 4 > change_time[setpoint_index:]):
#    if i > 4 and (i <= 60 or i > 65):
        # Evaluate the feature values
        feature1 = abs(y_pred5[i - 5] - y_store[i])
        feature2 = abs(y_store[i] - y_store[i - 1])
        # Use the classifier and save the predicted labels in the moving window
        x = np.array([feature1, feature2, u])[None, :]
        if feature1 > mismatch_threshold:
            labels.append(clf.predict(x)[0])
        else:
            labels.append(most_common_labels[i - 1])
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

    
