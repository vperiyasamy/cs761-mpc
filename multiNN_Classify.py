import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier

# Load in data
for i in range(3):
    data = []
    for j in range(3):
        with open('TrainingModel' + str(i) + 'Process' + str(j) + '.pkl', 'rb') as f:
            data.append(pickle.load(f))
    
    # Organize the data
    labels  = [data[0]['y_label'], data[1]['y_label'], data[2]['y_label']]
    y_meas  = [data[0]['y_meas'], data[1]['y_meas'], data[2]['y_meas']]
    y_pred  = [data[0]['y_pred'], data[1]['y_pred'], data[2]['y_pred']]
    y_pred5 = [data[0]['y_pred_5'], data[1]['y_pred_5'], data[2]['y_pred_5']]
    us      = [data[0]['u'], data[1]['u'], data[2]['u']]
    
    ## Create Features
    # Initialize the feature lists
    feature1s = []
    feature2s = []
    ys        = []
    u_values  = []
    # Delete the first 5 data points and the 4 data points following each setpoint change
    deletes1 = [0, 1, 2, 3, 4, 41, 42, 43, 44, 81, 82, 83, 84, 121, 122, 123, 124]
    deletes2 = [0, 1, 2, 3, 40, 41, 42, 43, 80, 81, 82, 83, 120, 121, 122, 123]
    # Evaluate the feature values
    for j in range(len(labels)):
        feature1s.append(np.abs(np.delete(y_pred5[j] - y_meas[j], deletes1)))
        feature2s.append(np.abs(np.delete(y_meas[j][1:] - y_meas[j][:159], deletes2)))
        ys.append(np.delete(labels[j], deletes1))
        u_values.append(np.delete(us[j], deletes1))
    
    # Prepare X and y
    feature1 = np.concatenate((feature1s[0], feature1s[1], feature1s[2]))
    feature2 = np.concatenate((feature2s[0], feature2s[1], feature2s[2]))
    u        = np.concatenate((u_values[0], u_values[1], u_values[2]))
    X        = np.column_stack((feature1, feature2, u))
    y        = np.concatenate((ys[0], ys[1], ys[2]))
     
    # Fit neural network with adam solver
    clf = MLPClassifier(activation = 'relu', solver = 'adam', alpha = 0.0001, learning_rate = 'adaptive', max_iter = 500)
    clf.fit(X, y)
    
    # Save the classifiers
    with open('NN_Classifier' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    # 3D plots of the 3D classifiers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature1[y == 0], feature2[y == 0], u[y == 0])
    ax.scatter(feature1[y == 1], feature2[y == 1], u[y == 1])
    ax.scatter(feature1[y == 2], feature2[y == 2], u[y == 2])
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Y Change')
    ax.set_zlabel('U')
    ax.text2D(0.05, 0.95, "Raw Data", transform=ax.transAxes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature1[clf.predict(X) == 0], feature2[clf.predict(X) == 0], u[clf.predict(X) == 0])
    ax.scatter(feature1[clf.predict(X) == 1], feature2[clf.predict(X) == 1], u[clf.predict(X) == 1])
    ax.scatter(feature1[clf.predict(X) == 2], feature2[clf.predict(X) == 2], u[clf.predict(X) == 2])
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Y Change')
    ax.set_zlabel('U')
    ax.text2D(0.05, 0.95, "Classified Data", transform=ax.transAxes)
    
plt.show()