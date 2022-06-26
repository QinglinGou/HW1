
import numpy as np
import matplotlib.pyplot as plt

class GuassianNB(object):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
    def GuassianProbability(self,x, mean, var):
        return np.array([1 / np.sqrt(2 * np.pi * var[i] * var[i]) * np.exp(-np.power(x[i] - mean[i], 2) / (2 * np.power(var[i], 2))) for i in range(len(x))])
    def fit(self):
        self.weight = np.array([[np.mean(self.data[self.labels == label],axis=0),np.var(self.data[self.labels == label],axis=0)] for label in np.unique(self.labels)])
    def predict(self,samples):
        return np.array([np.unique(self.labels)[np.argmax([self.GuassianProbability(sample,self.weight[:,0,:][i],self.weight[:,1,:][i]).prod()  for i in range(len(np.unique(self.labels)))])] for sample in samples])

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
x_test,y_test= make_blobs(100, 2, centers=2, random_state=2, cluster_std=3)
Gua = GuassianNB(X,y)
Gua.fit()
fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Naive Bayes Model', size=14)


xlim = (-8, 8)
ylim = (-15, 5)

xg = np.linspace(xlim[0], xlim[1], 200)
yg = np.linspace(ylim[0], ylim[1], 200)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)
    
Z = Gua.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) 
ax.contour(xx, yy, Z, [0.5], colors='red')  
ax.set(xlim=xlim, ylim=ylim)


def comp_confmat(actual, predicted):

    # extract the different classes
    classes = np.unique(actual)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat

y_predict = Gua.predict(x_test)
print("---Confusion Matrix---")
print(comp_confmat(y_test, y_predict))

conf_matrix= comp_confmat(y_test, y_predict)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
