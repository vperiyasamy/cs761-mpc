import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
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
clf = MLPClassifier(activation = 'relu', solver = 'adam', alpha = 0.0001, learning_rate = 'adaptive', max_iter = 500)
avg_err = 0
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	# Train the neural network
	clf.fit(X_train, y_train)

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


# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.plot(len(datasets), len(classifiers) + 1)
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(-0.2, np.max(feature1) + 0.5)
ax.set_ylim(-0.1, np.max(feature2) + 0.1)
ax.set_xticks(())
ax.set_yticks(())


ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
score = clf.score(X_test, y_test)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='black', s=25)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           alpha=0.6, edgecolors='black', s=25)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(name)
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        size=15, horizontalalignment='right')

figure.subplots_adjust(left=.02, right=.98)
plt.show()
