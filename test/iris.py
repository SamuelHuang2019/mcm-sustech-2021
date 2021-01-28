from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn import svm

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)

# we can now use it like any other estimator
accu = accuracy_score(pipe.predict(X_test), y_test)
print(accu)
print(pipe.decision_function(X_train))

x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
grid_test = np.stack((xx.flat, yy.flat), axis=1)

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(X_train, y_train)

# grid_hat = pipe.predict(grid_test)
grid_hat = clf.predict(grid_test)
grid_hat = grid_hat.reshape(xx.shape)
plt.pcolormesh(xx, yy, grid_hat, shading='auto', cmap=plt.get_cmap('Pastel1'))

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50, cmap=plt.cm.Paired)
plt.scatter(X_test[:, 0], X_test[:, 1], s=120, facecolors='none', zorder=10)
plt.xlabel('calyx length')
plt.ylabel('calyx width')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Iris SVM Binary Classification')
plt.grid()
plt.show()
