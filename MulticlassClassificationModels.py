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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler



#data=pd.read_csv('ClassificationData1.Diabetic Retinopathy/messidor_features.csv', header=0)
#data=pd.read_csv('ClassificationData/2.Default of credit card clients/default of credit card clients.csv', header=0)
#data=pd.read_csv('ClassificationData/3.breast-cancer-wisconsin/breast-cancer-wisconsin.csv', header=None)
#data=pd.read_csv('ClassificationData/4. Statlog (Australian credit approval)/australian.csv', header=None)
#data=pd.read_csv('ClassificationData/5. Statlog (German credit data)/german.data-numeric.csv', header=None)
#data=pd.read_csv('ClassificationData/6.Steel Plates Faults/Faults.csv', header=None)
#data=pd.read_csv('ClassificationData/1.Diabetic Retinopathy/messidor_features.csv', header=None)
data=pd.read_csv('ClassificationData/8. Yeast/yeast.csv', header=None)
#data=pd.read_csv('ClassificationData/1.Diabetic Retinopathy/messidor_features.csv', header=None)
#data=pd.read_csv('ClassificationData/1.Diabetic Retinopathy/messidor_features.csv', header=None)



data.fillna('zero', inplace=True)
# Categorical boolean mask
categorical_feature_mask = data.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = data.columns[categorical_feature_mask].tolist()
#print(categorical_cols)
#Use LabelEncoder() to transfer categorical data to numurical data
le=LabelEncoder()
data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))


data=np.array(data)

#If first column is ID, then start from '1' to '-2' for X.
X,y=data[:,1:-2], data[:, -1]

'''
#Handle missing values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

X=np.array(X,dtype=float)
imp.fit(X)
imp.transform(X)
print(X)
'''

#Handle labels
for i in range (0,y.size):
    if y[i]==2:
        y[i]=0
    else:
        if y[i] == 4:
         y[i]=1


scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

#print(X_train)


pca = PCA(n_components=5)
# pca. fit_transform(trainX)
# print(pca.fit(trainX))
pca.fit(X_train)
pcaX = pca.transform(X_train)

pcaT = pca.transform(X_test)

#print(pcaX.shape)

traindata = np.array(pcaX)
trainlabel = np.array(y_train)

testdata = np.array(pcaT)
testlabel = np.array(y_test)



def MLP(X_train,y_train,X_test,y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=0)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedMLP.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

def LogisticRegression(X_train,y_train,X_test,y_test):
    clf = sklearn.linear_model.LogisticRegression(random_state=0, solver='lbfgs')
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedLR.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

def xgboost(X_train,y_train,X_test,y_test):
    clf = xgb.XGBClassifier(random_state=0,learning_rate=0.01)
    #, max_depth = 20
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedRFbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


def evaluate(y_test,y_pred,y_scores):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted',pos_label=1)
    recall = recall_score(y_test, y_pred, average='weighted',pos_label=1)
    f1 = f1_score(y_test, y_pred, average='weighted',pos_label=1)
#    auc = roc_auc_score(y_test, y_scores)
    return [accuracy,precision,recall,f1,auc]

def CART(X_train,y_train,X_test,y_test):
    #clf = tree.DecisionTreeClassifier(max_depth=5)
    #clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=5)
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedDTbase.txt', y_pred, fmt='%01d')
#    tree.plot_tree(clf.fit(X_train,y_train))
    #tree.export_graphviz()
    #dot_data = tree.export_graphviz(clf, out_file=None)
    #graph = graphviz.Source(dot_data)
    #graph.render("tree")
    return evaluate(y_test,y_pred,y_scores)

#randomforest

def RandomForest(X_train,y_train,X_test,y_test):
    clf = ensemble.RandomForestClassifier(max_features=None,random_state=0,n_estimators=50)
    #, max_depth = 20
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedRFbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)



def svm_one_class(X_train,y_train,X_test,y_test):
    _train = preprocessing.normalize(X_train, norm='l2')
    _test = preprocessing.normalize(X_test, norm='l2')
    lin_clf = svm.LinearSVC()
    lin_clf.fit(_train, y_train)
    LinearSVC(C=1000, class_weight=None, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
    y_pred = lin_clf.predict(_test)
    y_scores = lin_clf.decision_function(_test)
    np.savetxt('res/predictedSVMbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


def SVM(X_train,y_train,X_test,y_test):
    #_train = preprocessing.normalize(X_train, norm='l2')
    #_test = preprocessing.normalize(X_test, norm='l2')
    data_scaler = preprocessing.StandardScaler()
    _train = data_scaler.fit_transform(X_train)
    _test = data_scaler.transform(X_test)
    clf = svm.SVC(kernel='rbf',C=0.5,probability=True,random_state=0).fit(_train, y_train)
    #kernel='sigmoid',degree=3, C=2,probability=True
    y_pred = clf.predict(_test)
    y_scores = clf.decision_function(_test)
    np.savetxt('res/predictedSVMNormbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)


def Gaussian_NB(X_train,y_train,X_test,y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    y_scores  = gnb.fit(X_train, y_train).predict_proba(X_test)[:,1]

    np.savetxt('res/predictedNBbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

def kNN(X_train,y_train,X_test,y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    np.savetxt('res/predictedKNNbase.txt', y_pred, fmt='%01d')
    y_scores = clf.predict_proba(X_test)[:,1]
    return evaluate(y_test,y_pred,y_scores)

def AdaBoost(X_train,y_train,X_test,y_test):
   # dt_stump = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
    dt_stump = DecisionTreeClassifier(random_state=0)
    dt_stump.fit(X_train, y_train)
    dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

    clf = ensemble.AdaBoostClassifier(base_estimator=dt_stump,learning_rate=0.1,algorithm='SAMME')
    clf=clf.fit(X_train, y_train)
   # clf=clf.fit(X_train, y_train, sample_weight=0.5)
    #clf = clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    np.savetxt('res/predictedABbase.txt', y_pred, fmt='%01d')
    return evaluate(y_test,y_pred,y_scores)

#SVM-rbf
#knn odd num
a1=AdaBoost(traindata,trainlabel,testdata,testlabel)

a2=Gaussian_NB(traindata,trainlabel,testdata,testlabel)

#a3=xgboost(traindata,trainlabel,testdata,testlabel)

a4=kNN(traindata,trainlabel,testdata,testlabel)

a5=LogisticRegression(traindata,trainlabel,testdata,testlabel)

a6=SVM(traindata,trainlabel,testdata,testlabel)

a7=MLP(traindata,trainlabel,testdata,testlabel)

a8=RandomForest(traindata,trainlabel,testdata,testlabel)


a10=CART(traindata,trainlabel,testdata,testlabel)


#a12=svm_one_class(traindata,trainlabel,testdata,testlabel)
print("AdaBoost")
print("[accuracy,precision,recall,f1,auc]")
print(a1)

print("Naive_bayes")
print("[accuracy,precision,recall,f1,auc]")
print(a2)

#print("xgboost")
#print("[accuracy,precision,recall,f1,auc]")
#print(a3)

print("Knn")
print("[accuracy,precision,recall,f1,auc]")
print(a4)

print("LogisticRegression")
print("[accuracy,precision,recall,f1,auc]")
print(a5)

print("SVM")
print("[accuracy,precision,recall,f1,auc]")
print(a6)


print("MLP")
print("[accuracy,precision,recall,f1,auc]")
print(a7)

print("RandomForest")
print("[accuracy,precision,recall,f1,auc]")
print(a8)

print("CART")
print("[accuracy,precision,recall,f1,auc]")
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