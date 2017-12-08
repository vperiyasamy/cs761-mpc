from apm import *
import numpy as np
import random
import pickle
import apm_functions as apmfun
import matplotlib.pyplot as plt

# Define a function to generate dictionaries of different trials
def SetSimulationParams(**kwargs):
    params = {'setpoints'      : [10],
              'sp_times'       : [0],
              'sim_Ks'         : [5],
              'sim_K_times'    : [0],
              'cycles'         : 150,
              'window_size'    : 10,
              'mismatch_thres' : 0.2,
              'clf_thres'      : 0.2}
    params.update(kwargs)
    return params
              
# Define the various simulations
params1 = SetSimulationParams(cycles = 100) # startup
params2 = SetSimulationParams(sim_Ks = [10], cycles = 100) # startup
params3 = SetSimulationParams(sim_Ks = [1], cycles = 100) # startup
params4 = SetSimulationParams(sim_Ks = [5, 10], sim_K_times = [0, 70]) # disturbance
params5 = SetSimulationParams(sim_Ks = [10, 1], sim_K_times = [0, 70]) # disturbance
params6 = SetSimulationParams(setpoints = [10, 15], sp_times = [0, 70], sim_Ks = [10]) # setpoint robustness
params7 = SetSimulationParams(setpoints = [10, 15], sp_times = [0, 70], sim_Ks = [10, 5],
                              sim_K_times = [0, 80]) # transient setpoint change
params8 = SetSimulationParams(setpoints = [10, 12, 8, 10], sp_times = [0, 50, 110, 130],
                              sim_Ks = [5, 10], sim_K_times = [0, 70]) # craziness
simulations = [params1, params2, params3, params4, params5, params6, params7, params8]
simulations = [params8]
#%%
# Iterate over the different simulation cases
for sim in range(len(simulations)):
    # Load simulation parameters
    params             = simulations[sim]
    cycles             = params['cycles']
    setpoints          = np.array(params['setpoints'])
    sp_times           = np.array(params['sp_times'])
    sim_Ks             = np.array(params['sim_Ks'])
    sim_K_times        = np.array(params['sim_K_times'])
    window_size        = params['window_size']
    mismatch_threshold = params['mismatch_thres']
    clf_threshold      = params['clf_thres']
    
    # Load the classifiers
    clfs = []
    for i in range(3):
        with open('LinearClassifier' + str(i) + '.pkl', 'rb') as f:
            clfs.append(pickle.load(f))
            
    # Initialize simulater values
    u                  = 0.0
    yp5                = 0.0
    ym                 = 0.0
    models             = [1, 5, 10]
    model_index        = 1
    clf                = clfs[model_index]
    labels             = []
    most_common_labels = np.ones(cycles).astype(int)
    
    # Initialize data arrays
    y_store = np.zeros(cycles)
    y_pred5 = np.zeros(cycles)
    y_mpc   = np.zeros(cycles)
    setpts  = np.zeros(cycles)
    K_real  = np.zeros(cycles)
    K_pred  = np.zeros(cycles)
    
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
            setpoint       = setpoints[setpoint_index]
        
        ## Process simulator
        if np.any(sim_K_times == i):
            sim_K         = sim_Ks[sim_K_times == i][0]
        measurement_noise = 0.1 * (random.random() - 0.5)
        y_meas            = apmfun.sim(server, app1, u, sim_K) + measurement_noise
            
        ## Save data for SVM
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
            
        ## Save data for plotting
        y_mpc[i]   = ym
        setpts[i]  = setpoint
        K_real[i]  = sim_K
        K_pred[i]  = models[most_common_labels[i]]
            
        ## Model Predictive Control (MPC)
        u, yp5, ym = apmfun.mpc(server, app2, [setpoint, y_meas, models[most_common_labels[i]]])
    	
#        if (i == 0):
#           # Web viewers to see solution progression
#           apm_web(server, app2) # Controller
    #       apm_web(server, app1)
                      
    # Plot results
    plt.figure()
    plt.subplot(211)
    plt.plot(np.arange(0, cycles + 1, 1), np.insert(y_mpc, 0, 0.0))
    plt.plot(np.arange(0, cycles + 1, 1), np.insert(y_store, 0, 0.0), '--')
    plt.plot(np.arange(0, cycles + 1, 1), np.insert(setpts, 0, setpoints[0]), 'k-.')
    plt.legend(['MPC Model Prediction', 'Measured Value', 'Setpoint'], loc= 'best')
    plt.ylabel('Process Variable (Y)')
    
    plt.subplot(212)
    plt.plot(np.arange(0, cycles + 1, 1), np.insert(K_pred, 0, 5.0))
    plt.plot(np.arange(0, cycles + 1, 1), np.insert(K_real, 0, sim_Ks[0]), '--')
    plt.legend(['SVM Prediction', 'Actual Value'], loc= 'best')
    plt.xlabel('Time')
    plt.ylabel('Process Gain (K)')
