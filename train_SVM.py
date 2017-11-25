import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

# Load in data
data = []
for i in range(3):
    with open('TrainingDataFull_' + str(i + 1) + '.pkl', 'rb') as f:
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
# Delete the first 5 data points and the 4 data points following each setpoint change
deletes1 = [0, 1, 2, 3, 4, 41, 42, 43, 44, 81, 82, 83, 84, 121, 122, 123, 124]
deletes2 = [0, 1, 2, 3, 40, 41, 42, 43, 80, 81, 82, 83, 120, 121, 122, 123]
# Evaluate the feature values
for i in range(len(labels)):
    feature1s.append(np.abs(np.delete(y_pred5[i] - y_meas[i], deletes1)))
    feature2s.append(np.abs(np.delete(y_meas[i][1:] - y_meas[i][:159], deletes2)))
    ys.append(np.delete(labels[i], deletes1))

# Prepare X and y
feature1 = np.concatenate((feature1s[0], feature1s[1], feature1s[2]))
feature2 = np.concatenate((feature2s[0], feature2s[1], feature2s[2]))
X = np.column_stack((feature1, feature2))
y = np.concatenate((ys[0], ys[1], ys[2]))
 
# Fit linear SVM
clf_lin = svm.SVC(kernel = 'linear')
clf_lin.fit(X, y)
w = clf_lin.coef_
w = w.flatten()

# The Gaussian SVM
clf_rbf = svm.SVC(kernel = 'rbf')
clf_rbf.fit(X, y)

# Save the classifiers
with open('LinearClassifier.pkl', 'wb') as f:
    pickle.dump(clf_lin, f)
    
with open('GaussianClassifier.pkl', 'wb') as f:
    pickle.dump(clf_rbf, f)

# Prepare the Linear decision lines for plotting
a1  = -w[2] / w[3]
xx1 = np.linspace(-.5, np.max(feature1) + .5)
yy1 = a1 * xx1 - (clf_lin.intercept_[1]) / w[3]
a2  = -w[4] / w[5]
xx2 = np.linspace(-.5, np.max(feature1) + .5)
yy2 = a2 * xx2 - (clf_lin.intercept_[2]) / w[5]

# Plot the features and the linear decision lines
plt.figure()
plt.scatter(feature1[y == 0], feature2[y == 0])
plt.scatter(feature1[y == 1], feature2[y == 1])
plt.scatter(feature1[y == 2], feature2[y == 2])
plt.plot(xx1, yy1, 'k-')
plt.legend(['Decision Line', 'Label = 0','Label = 1', 'Label = 2', 'support'])
plt.plot(xx2, yy2, 'k-')
plt.xlim([-0.2, np.max(feature1) + 0.5])
plt.ylim([-0.1, np.max(feature2) + 0.1])
plt.xlabel('Absolute Prediction Error')
plt.ylabel('Change in Process Variable')

# Plot a juxtaposation of the 2 different classifiers
plt.figure()
plt.subplot(131)
plt.scatter(feature1[y == 0], feature2[y == 0])
plt.scatter(feature1[y == 1], feature2[y == 1])
plt.scatter(feature1[y == 2], feature2[y == 2])
plt.ylabel('Change in Process Variable')
plt.title('Raw Data')

plt.subplot(132)
plt.scatter(feature1[clf_lin.predict(X) == 0], feature2[clf_lin.predict(X) == 0])
plt.scatter(feature1[clf_lin.predict(X) == 1], feature2[clf_lin.predict(X) == 1])
plt.scatter(feature1[clf_lin.predict(X) == 2], feature2[clf_lin.predict(X) == 2])
plt.xlabel('Absolute Prediction Error')
plt.title('Linear')

plt.subplot(133)
plt.scatter(feature1[clf_rbf.predict(X) == 0], feature2[clf_rbf.predict(X) == 0])
plt.scatter(feature1[clf_rbf.predict(X) == 1], feature2[clf_rbf.predict(X) == 1])
plt.scatter(feature1[clf_rbf.predict(X) == 2], feature2[clf_rbf.predict(X) == 2])
plt.title('Gaussian')

plt.show()