from apm import *
import numpy as np
import random
import pickle

def sim_init(s,a):
   apm(s,a,'clear all')
   apm_load(s,a,'process.apm')
   csv_load(s,a,'process.csv')
   apm_option(s,a,'nlc.imode',4)
   apm_option(s,a,'nlc.nodes',3)
   apm_info(s,a,'SV','y')
   apm_info(s,a,'FV','u')
   apm_option(s,a,'u.fstatus',1)
   msg = 'Successful simulator initialization'
   return msg

def mpc_init(s,a):
   apm(s,a,'clear all')
   apm_load(s,a,'model.apm')
   csv_load(s,a,'data.csv')
   apm_option(s,a,'nlc.imode',6)
   apm_option(s,a,'nlc.nodes',3)
   apm_option(s,a,'nlc.web_plot_freq',1)
   apm_info(s,a,'FV','K')
   apm_info(s,a,'FV','tau')
   apm_info(s,a,'MV','u')
   apm_info(s,a,'CV','y')
   # status, whether the optimizer can use it
   apm_option(s,a,'K.status',0)
   apm_option(s,a,'tau.status',0)
   apm_option(s,a,'u.status',1)
   apm_option(s,a,'y.status',1)
   # feedback status
   apm_option(s,a,'K.fstatus',1)
   apm_option(s,a,'tau.fstatus',1)
   apm_option(s,a,'u.fstatus',0)
   apm_option(s,a,'y.fstatus',1)
   # constraints
   apm_option(s,a,'u.upper',100)
   apm_option(s,a,'u.lower',0)
   # reference trajectory tuning
   apm_option(s,a,'nlc.traj_init',2)
   apm_option(s,a,'nlc.traj_open',0.5)
   apm_option(s,a,'y.tau',12)
   msg = 'Successful controller initialization'
   return msg
   
def sim(s,a,u):
   apm_meas(s,a,'u',u)
   apm(s,a,'solve')
   y = apm_tag(s,a,'y.model')
   return y
   
def mpc(s,a,inputs):
   sp = inputs[0]
   y_meas = inputs[1]
   k_pred = inputs[2]
   apm_meas(s,a,'y',y_meas)
   apm_meas(s,a,'k',k_pred)
#   apm_meas(s,a,'tau',tau_pred)
   sphi = sp + 0.1
   splo = sp - 0.1
   apm_option(s,a,'y.sphi',sphi)
   apm_option(s,a,'y.splo',splo)
   apm_option(s,a,'y.sp',sp)
   apm(s,a,'solve')
   u = apm_tag(s,a,'u.newval')
   y_pred5 = apm_tag(s,a,'y.pred[5]')
   return u, y_pred5

# Number of cycles to run
cycles = 100

# Data arrays
y_store = np.zeros(cycles)
y_pred5 = np.zeros(cycles)

# Pulse in the process setpoint
u = 0
sp = 10
yp5 = 0

# Load the classifier
with open('LinearClassifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Set machine learning parameters
models = [1, 5, 10]
model_index = 1
window_size = 20
labels = []
most_common_labels = np.ones(cycles).astype(int)

# Server
s = 'http://byu.apmonitor.com'
# Application names
a1 = 'sim' + str(int(random.random() * 10000))
a2 = 'mpc' + str(int(random.random() * 10000))
# Initialize applications
print(sim_init(s, a1))
print(mpc_init(s, a2))

# Run cycles
for i in range(cycles):
    print('Cycle: ' + str(i + 1))
    
    ## Change the setpoint
    
    
    ## Process simulator
    measurement_noise = 0.1 * (random.random() - 0.5)
    y_meas            = sim(s, a1, u) + measurement_noise
    
    ## Save data
    y_store[i] = y_meas
    y_pred5[i] = yp5
           
    ## Create and update the moving window
    if i > 4:
        # Evaluate the feature values
        feature1 = abs(y_pred5[i - 5] - y_store[i])
        feature2 = abs(y_store[i] - y_store[i - 1])
        # Use the classifier and save the predicted labels in the moving window
        x = np.array([feature1, feature2, u])[None, :]
#        clf_out = clf.predict(x)[0]
#        curr_label = most_common_labels[i - 1]
#        if clf_out == 2 and curr_label != 2:
#            labels.append(curr_label + 1) 
#        elif clf_out == 0 and curr_label != 0:
#            labels.append(curr_label - 1)
#        else:
#            labels.append(curr_label)
        if feature1 > 0.5:
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
    u, yp5 = mpc(s, a2, [sp, y_meas, models[most_common_labels[i]]])
	
    if (i == 0):
       # Web viewers to see solution progression
       apm_web(s, a2) # Controller

    
