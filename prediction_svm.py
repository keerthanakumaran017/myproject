import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.metrics


datas = pd.read_csv(open("C:\\Users\\ashis\\Desktop\\datat.csv"))

predictors = datas.values[:, 0:11]
targets = datas.values[:,12]


pred_train, pred_test, targ_train, targ_test = train_test_split(predictors, targets, test_size=0.33)

clf =svm.SVC(kernel='rbf')
clf.fit(pred_train,targ_train)

pred = clf.predict(pred_test)

#accuracy
print("Accuracy is",accuracy_score(targ_test, pred, normalize = True))
#classification error
print("Classification error is",1- accuracy_score(targ_test, pred, normalize = True))
