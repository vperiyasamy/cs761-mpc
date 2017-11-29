import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.model_selection import KFold

data = []

for i in range(4):
    with open('TrainingData' +str(i+1)+'.pkl', 'rb') as f:
        data.append(pickle.load(f))

labels = np.concatenate((data[0]['y_label'], data[1]['y_label'],
                          data[2]['y_label'], data[3]['y_label']))
y_meas = np.concatenate((data[0]['y_meas'], data[1]['y_meas'],
                          data[2]['y_meas'], data[3]['y_meas']))
y_pred = np.concatenate((data[0]['y_pred'], data[1]['y_pred'],
                          data[2]['y_pred'], data[3]['y_pred']))
y_pred5 = np.concatenate((data[0]['y_pred_5'], data[1]['y_pred_5'],
                          data[2]['y_pred_5'], data[3]['y_pred_5']))
us = np.concatenate((data[0]['u'], data[1]['u'],
                     data[2]['u'], data[3]['u']))

# Create features
cutoff = 200
deletes1 = [0, 1, 2, 3, 4, 100, 101, 102, 103, 104]
deletes2 = [0, 1, 2, 3, 99, 100, 101, 102, 103]
feature1 = np.abs(np.delete((y_pred5[0:cutoff] - y_meas[0:cutoff]), deletes1))
feature2 = np.abs(np.delete((y_meas[1:cutoff] - y_meas[0:cutoff-1]), deletes2))

# Prepare X and y
X = np.column_stack((feature1, feature2))
half = int(len(feature1) / 2)
y = np.ones(len(X))
y[half:] = 0

splits = 10
 
# Cross Validation
kf = KFold(n_splits=splits, shuffle=True)
clf = svm.SVC(kernel = 'linear')
avg_err = 0
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	# Fit SVM
	clf.fit(X_train, y_train)
	w = clf.coef_
	w = w.flatten()

	# calculate errors
	y_predict = clf.predict(X_test)
	err = 0.0
	for i, val in enumerate(y_predict):
		if val != y_test[i]:
			err += 1

	avg_err += (err / len(y_test))

# divide by n = 10
avg_err /= splits
print (avg_err)

# Prepare decision line for plotting
a = -w[0] / w[1]
xx = np.linspace(-.5, np.max(feature1))
yy = a * xx - (clf.intercept_[0]) / w[1]
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

# Plot results
plt.figure()
plt.scatter(feature1[:half], feature2[:half])
plt.scatter(feature1[half:], feature2[half:])
plt.plot(xx, yy, 'g-')
plt.scatter(feature1[clf.support_], feature2[clf.support_],
            facecolors = 'none', s =50, edgecolors='g')
plt.legend(['Decision Boundary','Label = 1', 'Label = 0', 'Support Vectors'])
plt.plot(xx, yy_up, 'g--')
plt.plot(xx, yy_down, 'g--')
plt.xlim([-0.2, np.max(feature1) + 0.5])
plt.ylim([-0.1, np.max(feature2) + 0.1])
plt.xlabel('Absolute Prediction Error')
plt.ylabel('Change in Process Variable')

plt.show()

