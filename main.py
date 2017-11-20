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
   apm_meas(s,a,'y',y_meas)
#   apm_meas(s,a,'k',k_pred)
#   apm_meas(s,a,'tau',tau_pred)
   sphi = sp + 0.1
   splo = sp - 0.1
   apm_option(s,a,'y.sphi',sphi)
   apm_option(s,a,'y.splo',splo)
   apm_option(s,a,'y.sp',sp)
   apm(s,a,'solve')
   u = apm_tag(s,a,'u.newval')
   y_pred = apm_tag(s,a,'y.pred[1]')
   y_pred5 = apm_tag(s,a,'y.pred[5]')
   return u, y_pred, y_pred5


# Number of cycles to run
cycles = 100

# Data arrays
y_store = np.zeros(cycles)
u_store = np.zeros(cycles)
y_pred = np.zeros(cycles)
y_pred5 = np.zeros(cycles)

# Pulse in the process setpoint
u = 0
sp = 10
yp = 0
yp5 = 0

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
    if i == 40:
        sp = 15
    if i == 70:
        sp = 5
    
    ## Process simulator
    measurement_noise = 0.1 * (random.random() - 0.5)
    y_meas = sim(s, a1, u) + measurement_noise
    
    # Save data
    y_store[i] = y_meas
    u_store[i] = u
    y_pred[i]  = yp
    y_pred5[i]  = yp5
    
    ## Model Predictive Control (MPC)
    u, yp, yp5 = mpc(s, a2, [sp, y_meas])
	
    if (i == 0):
       # Web viewers to see solution progression
       apm_web(s, a2) # Controller

# Save data file
y_pred5 = np.insert(y_pred5, 0, (0,0,0,0))
y_pred5 = y_pred5[:cycles]
y_label = np.zeros(cycles).astype(int)
data = {'y_meas'   : y_store,
        'y_pred'   : y_pred,
        'y_pred_5' : y_pred5,
        'u'        : u_store,
        'y_label'  : y_label}

with open('TrainingData4.pkl', 'wb') as f:
    pickle.dump(data, f)
    
