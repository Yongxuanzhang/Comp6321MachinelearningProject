import numpy as np
import sklearn
from numpy import random
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,
                             mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,
                             mean_absolute_error, roc_curve, classification_report, auc)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
#from imblearn.ensemble import BalancedRandomForestClassifier
#import graphviz
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



data=pd.read_csv('RegressionData/1. Wine Quality/winequality-red.csv', header=0)
#data=pd.read_csv('2.Default of credit card clients/default of credit card clients.csv', header=0)
data=np.array(data)

X,y=data[:,0:-2], data[:, -1]
#print(X[0])
#print(y)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

#print(X_train)


pca = PCA(n_components=10)

pca.fit(X_train)
pcaX = pca.transform(X_train)
pcaT = pca.transform(X_test)



traindata = np.array(pcaX)
trainlabel = np.array(y_train)

testdata = np.array(pcaT)
testlabel = np.array(y_test)


def evaluate(y_test,y_pred):

    mse=mean_squared_error(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    return [mse,mae,r2]


def MLP(X_train,y_train,X_test,y_test):
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=0)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    np.savetxt('res/predictedMLP.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred)

def LinearRegression(X_train,y_train,X_test,y_test):
    clf = sklearn.linear_model.LinearRegression()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    np.savetxt('res/predictedLR.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred)

def GPR(X_train,y_train,X_test,y_test):
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state = 0).fit(X_train, y_train)
    gpr.score(X_test, y_test)

    y_pred = gpr.predict(X_test)
    return evaluate(y_test,y_pred)




def CART(X_train,y_train,X_test,y_test):
    #clf = tree.DecisionTreeClassifier(max_depth=5)
    #clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=5)
    clf = tree.DecisionTreeRegressor(random_state=0)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    np.savetxt('res/predictedDTbase.txt', y_pred, fmt='%01d')
#    tree.plot_tree(clf.fit(X_train,y_train))
    #tree.export_graphviz()
    #dot_data = tree.export_graphviz(clf, out_file=None)
    #graph = graphviz.Source(dot_data)
    #graph.render("tree")
    return evaluate(y_test,y_pred)



def RandomForest(X_train,y_train,X_test,y_test):
    clf = ensemble.RandomForestRegressor(max_features=None,random_state=0,n_estimators=50)
    #, max_depth = 20
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    np.savetxt('res/predictedRFbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred)




def SVM(X_train,y_train,X_test,y_test):

    clf = svm.SVR().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    np.savetxt('res/predictedSVMNormbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred)


def AdaBoost(X_train,y_train,X_test,y_test):
   # dt_stump = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
    dt_stump =tree.DecisionTreeRegressor(random_state=0,max_depth=3)
    dt_stump.fit(X_train, y_train)
    dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

    clf = ensemble.AdaBoostRegressor(base_estimator=dt_stump,learning_rate=0.1)
    clf=clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    np.savetxt('res/predictedABbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred)

#SVM-rbf
#knn odd num
a1=AdaBoost(traindata,trainlabel,testdata,testlabel)

a2=GPR(traindata,trainlabel,testdata,testlabel)

#a3=xgboost(traindata,trainlabel,testdata,testlabel)

#a4=kNN(traindata,trainlabel,testdata,testlabel)

a5=LinearRegression(traindata,trainlabel,testdata,testlabel)

a6=SVM(traindata,trainlabel,testdata,testlabel)

a7=MLP(traindata,trainlabel,testdata,testlabel)

a8=RandomForest(traindata,trainlabel,testdata,testlabel)


a10=CART(traindata,trainlabel,testdata,testlabel)


#a12=svm_one_class(traindata,trainlabel,testdata,testlabel)
print("AdaBoost")
print("[mean_squared_error,mean_absolute_error,r2_score]")
print(a1)

print("GPR")
print("[mean_squared_error,mean_absolute_error,r2_score]")
print(a2)

#print("xgboost")
#print("[accuracy,precision,recall,f1,auc]")
#print(a3)

#print("Knn")
#print("[accuracy,precision,recall,f1,auc]")
#print(a4)

print("LinearRegression")
print("[mean_squared_error,mean_absolute_error,r2_score]")
print(a5)

print("SVM")
print("[mean_squared_error,mean_absolute_error,r2_score]")
print(a6)


print("MLP")
print("[mean_squared_error,mean_absolute_error,r2_score]")
print(a7)

print("RandomForest")
print("[mean_squared_error,mean_absolute_error,r2_score]")
print(a8)

print("CART")
print("[mean_squared_error,mean_absolute_error,r2_score]")
print(a10)


def tsneShow(X,y,Xt,yt):
    yys = np.array(y)
    yyt = np.array(yt)
    for i in range(0, yyt.size):
        if yyt[i] == 0:
            yyt[i] = 2
        elif yyt[i] == 1:
            yyt[i] = 3

    H = np.vstack((X, Xt))
    Y = np.concatenate((yys, yyt), axis=0)

    #pca = PCA().fit_transform(H)
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(H)

    markers = ('.')
    plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=Y, marker=markers[0], cmap=plt.cm.gist_rainbow)

    plt.colorbar()
    plt.show()