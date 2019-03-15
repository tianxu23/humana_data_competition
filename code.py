#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 09:26:39 2018

@author: zhengdongzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 22:39:24 2018

@author: zhengdongzhang
"""
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# data cleanning
path = '/Users/zhengdongzhang/Documents/大学/研究生/health care analytics/TAMU_FINAL_DATASET_2018.csv'
df = pd.read_csv(path)
#check = df.count()
#check.to_csv('checknull.csv')
#drop na
humana = df.dropna(axis=0, how='any', subset = ['Online_User'])
humana.loc[humana['Diab_Type'].isnull(),'Diab_Type'] = 0
humana.loc[humana['Diab_Type']=='Diabetes Type I','Diab_Type'] = 1
humana.loc[humana['Diab_Type']=='Diabetes Type II','Diab_Type'] = 2
humana.loc[humana['Diab_Type']=='Diabetes Unspeci','Diab_Type'] = 3
humana.loc[humana['Decile_struggle_Med_lang'].isnull(),'Decile_struggle_Med_lang'] = 99
humana['Decile_struggle_Med_lang'] = humana['Decile_struggle_Med_lang']+1
humana.loc[humana['Decile_struggle_Med_lang']==100,'Decile_struggle_Med_lang'] = 0
humana = humana.dropna(axis=0, how='any', subset = ['MCO_HLVL_PLAN_CD'])
humana = humana.dropna(axis=0, how='any', subset = ['MCO_PROD_TYPE_CD'])
humana = humana.dropna(axis=0, how='any', subset = ['ESRD_IND'])
humana = humana.dropna(axis=0, how='any', subset = ['HOSPICE_IND'])
humana = humana.dropna(axis=0, how='any', subset = ['SEX_CD'])
humana['Diab_Type'].head(50)
#生成哑变量
dummy_fields = ["SEX_CD","ESRD_IND","HOSPICE_IND","ORIG_REAS_ENTITLE_CD",
                "PCP_ASSIGNMENT","DUAL","INSTITUTIONAL","LIS",
                "MCO_HLVL_PLAN_CD","MCO_PROD_TYPE_CD","Diab_Type","Dwelling_Type"]

for each in dummy_fields:
    dummies = pd.get_dummies(humana.loc[:,each],prefix = each)
    humana = pd.concat([humana,dummies],axis = 1)
    
field_to_drop = ["SEX_CD","ESRD_IND","HOSPICE_IND","ORIG_REAS_ENTITLE_CD",
                 "PCP_ASSIGNMENT","DUAL","INSTITUTIONAL","LIS",
                 "MCO_HLVL_PLAN_CD","MCO_PROD_TYPE_CD","Diab_Type","Dwelling_Type"]
humana2 = humana.drop(field_to_drop,axis = 1)

#合并列
test2 = pd.DataFrame()
test3 = pd.DataFrame()

list1 = []
list2 = []

column_names_x = humana2.columns
for column in column_names_x:
    if column.find("POT_VISIT_") == 0:
        list1.append(column)
list1.sort()
#print(list1)        #test1[column] = humana[column]
for column in list1:
    name1 = column.split("_")[0]
    name2 = column.split("_")[1]
    number = column.split("_")[2]
    quarter = column.split("_")[3]
    new_column = name1 + '_' + name2 + '_' + number + '_score'
    if quarter == 'Q01':
        c = 0.1
    elif quarter == 'Q02':
        c = 0.2
    elif quarter == 'Q03':
        c = 0.3
    else:
        c = 0.4
    try:
        test2[new_column] = test2[new_column] + c * humana2[column]
    except:
        test2[new_column] = c * humana2[column]

for column in column_names_x:
    if column.find("CON_VISIT_") == 0:
        list2.append(column)
list2.sort()
#print(list2)        #test1[column] = humana[column]
for column in list2:
    name1 = column.split("_")[0]
    name2 = column.split("_")[1]
    number = column.split("_")[2]
    quarter = column.split("_")[3]
    new_column = name1 + '_' + name2 + '_' + number + '_score'
    if quarter == 'Q01':
        c = 0.1
    elif quarter == 'Q02':
        c = 0.2
    elif quarter == 'Q03':
        c = 0.3
    else:
        c = 0.4
    try:
        test3[new_column] = test3[new_column] + c * humana2[column]
    except:
        test3[new_column] = c * humana2[column]
        
humana3 = pd.concat([humana2,test2,test3],axis = 1)
humana3 = humana3.drop(list1,axis = 1).drop(list2,axis=1)

#STACKING
#------------------------------------------------------------------------------------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import sklearn as sk
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold;
from sklearn.cross_validation import train_test_split


humana3 = humana2.drop(['Home_value','Est_income','Est_Net_worth'],axis=1)
humana3.describe()
humana3.shape

train = humana3.ix[0:70000,:] #这里随机取，要改一下
test = humana3.ix[70000:,:]

humana3.to_csv('sss.csv') #147-150不用看
train.describe()
train.shape
test.shape
# Some useful parameters which will come in handy later on
ntrain = train.shape[0]  #输入了一下行数，目前没什么用
ntest = test.shape[0]
SEED = 0 # for reproducibility 随机种子标识
NFOLDS = 5 # set folds for out-of-fold prediction 折叠数
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED) #把样本随机本成5个

# Class to extend the Sklearn classifier

class SklearnHelper(object):
    def __init__(self, clf, model_name, seed=0, params=None):
        if seed != None:
            params['random_state'] = seed
        self.clf = clf(**params) #看不懂
        self.name = model_name

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def predict_proba(self, x):
        return self.clf.predict_proba(x)
    
    def score(self, x, y):
        return self.clf.score(x, y)
    
    def fit(self,x,y):
        try:
            return self.clf.fit(x,y)
        except AttributeError:
            return self.clf.train(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        
    def model_name(self):
        return self.name
    
# 训练方法，不用管
np.zeros((ntrain,)).shape
np.empty((NFOLDS, ntest)).shape


#这个函数很重要，很复杂
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))  ##是把样本去了个平均放进去
    oof_test_skf = np.empty((NFOLDS, ntest))
    train_accuracy = 0
    test_accuracy = 0

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]       #把train和test的每一行都存进去了
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)[:, 0]
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 0]
        train_accuracy += clf.score(x_tr, y_tr)
        test_accuracy += clf.score(x_te, y_te)
    
    train_accuracy = train_accuracy/len(kf)
    test_accuracy = test_accuracy/len(kf)
    print('模型%s训练准确率为%f'%(clf.model_name(), train_accuracy))
    print('模型%s测试准确率为%f'%(clf.model_name(), test_accuracy))
    oof_test[:] = oof_test_skf.mean(axis=0)#这里加个mean不知道什么意思
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 3,
    'subsample':0.5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'poly',
    'C' : 0.025 ,
    'probability' : True
    }
gbm_params = {
    'learning_rate' : 0.4,
    'n_estimators' : 500,
    'max_depth' : 4,
    'min_child_weight': 2,
    #gamma=1,
    'gamma':0.9,                        
    'subsample':0.5,
    'colsample_bytree':0.8,
    'objective': 'binary:logistic',
    'reg_lambda':5,
    'nthread':-1,
    'scale_pos_weight' :1
}
lglr_params = {
    'C' : 0.5,
    'max_iter' : 1000
}
knn_params = {
    'n_neighbors' : 10,
    'weights' : 'uniform'
}

rf = SklearnHelper(RandomForestClassifier, 'RandomForest', seed=SEED, params=rf_params) # 
et = SklearnHelper(ExtraTreesClassifier, 'ExtraTrees',seed=SEED, params=et_params)
ada = SklearnHelper(AdaBoostClassifier, 'adaboost', seed=SEED, params=ada_params)
gb = SklearnHelper(GradientBoostingClassifier, 'GradientBoosting', seed=SEED, params=gb_params)
svc = SklearnHelper(SVC, 'SVM',seed=SEED, params=svc_params)
gbm = SklearnHelper(xgb.XGBClassifier, 'XGB', seed=SEED, params=gbm_params)
lglr = SklearnHelper(sk.linear_model.LogisticRegression, 'logistic', seed=SEED, params=lglr_params)
knn = SklearnHelper(KNeighborsClassifier, 'KNN', seed=None, params=knn_params)

try:
    y_train = train['AMI_FLAG'].ravel() #自己查一下这个函数的意思
    train = train.drop(['AMI_FLAG'], axis=1)
except KeyError:
    print('no need')

Y_oos_test = test['AMI_FLAG'].ravel()
test = test.drop(['AMI_FLAG'], axis = 1)
    
#np.isnan(x_test).sum()
#print(a.isnull().any())
#a = pd.DataFrame(x_train)
#a.to_csv('???.csv')
#b = pd.DataFrame(y_train)
#b.to_csv('!!!.csv')
#c = pd.DataFrame(x_test)
#c.to_csv('bbb.csv')
#d = c.drop([0], axis = 1)
#d.to_csv('@@@.csv')

x_train = train.values # Creates an array of the train data
x_test = test.values #test.values # Creats an array of the test data

#print(np.isfinite(x_train).all())

#dfa = df.apply(lambda a: pd.to_numeric(a,errors='ignore'))
#dfb = df.apply(lambda b: pd.to_numeric(b,errors='ignore'))
#dfc = df.apply(lambda c: pd.to_numeric(c,errors='ignore'))

#aaa = pd.DataFrame(y_train)
#aaa.head(20)
#bbb = pd.DataFrame(x_train)
#bbb.head(5)
#x_train
#x_test
#ccc = pd.DataFrame(x_test)
#ccc.head(5)

#ddd=ccc.drop([0], axis = 1)
##ddd.head(5)
#x_test = ddd.values
#x_test
#x_train = dfa.values
#y_train = dfb.values
#x_test = dfc.values
#dfa
#np.isnan(x_train).sum()


et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
gbm_oof_train, gbm_oof_test = get_oof(gbm, x_train, y_train, x_test)
lglr_oof_train, lglr_oof_test = get_oof(lglr, x_train, y_train, x_test)
knn_oof_train, knn_oof_test= get_oof(knn, x_train, y_train, x_test)

print("Training is complete")

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
       'SVC' : svc_oof_train.ravel(),
     'gbm' :gbm_oof_train.ravel(),
    'lglr' : lglr_oof_train.ravel(),
    'knn' : knn_oof_train.ravel()
    })
base_predictions_train.head()

sns.heatmap(base_predictions_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,  linecolor='white', annot=True)

sec_x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, svc_oof_train, lglr_oof_train,knn_oof_train), axis=1)
sec_x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, svc_oof_test,lglr_oof_test,knn_oof_test), axis=1)

gbm2_params = {
    'learning_rate' : 0.2,
    'n_estimators' : 500,
    'max_depth' : 4,
    'min_child_weight': 3,
    #gamma=1,
    'gamma':0.9,                        
    'subsample':0.4,
    'colsample_bytree':0.8,
    'objective': 'binary:logistic',
    'reg_lambda':2,
    'nthread':-1,
    'scale_pos_weight' :1
}
lglr2_params = {
     'C' : 0.5,
    'max_iter' : 1000
}
svc2_params = {
    'kernel' : 'linear',
    'C' : 0.025 ,
    'probability' : True
}
ada2_params = {
    'n_estimators': 30,
    'learning_rate' : 0.9
}

gbm2 = SklearnHelper(xgb.XGBClassifier, 'XGB', seed=SEED, params=gbm2_params)
lglr2 = SklearnHelper(sk.linear_model.LogisticRegression, 'lglr', seed=SEED, params=lglr2_params)
svc2 = SklearnHelper(SVC, 'svc', seed=SEED, params=svc2_params)
adaboost2 = SklearnHelper(AdaBoostClassifier, 'adaboost', seed=SEED, params=ada2_params)

x_train2, x_dev2, y_train2,y_dev2 = train_test_split(sec_x_train, y_train, test_size = 0.2)

_,_ = get_oof(gbm2, sec_x_train, y_train, sec_x_test)
_,_ = get_oof(lglr2, sec_x_train, y_train, sec_x_test)
_,_ = get_oof(svc2, sec_x_train, y_train, sec_x_test)
_,_ = get_oof(adaboost2, sec_x_train, y_train, sec_x_test)


#adaboost2.fit(x_train2, y_train2)
print ("在训练集上的准确率为%f"%(adaboost2.score(x_train2, y_train2)))
print ("在测试集上的准确率为%f"%(adaboost2.score(x_dev2, y_dev2)))

num_tree = [1,2, 5, 10, 15, 20, 30 , 40, 50, 100, 150 ,200, 300,500]
train_accuracy = []
test_accuracy = []
ada2_params['learning_rate'] = 1
for num in num_tree:
    ada2_params['n_estimators'] = num
    adaboost2 = SklearnHelper(AdaBoostClassifier, 'adaboost', seed=SEED, params=ada2_params)
    adaboost2.fit(x_train2, y_train2)
    train_accuracy.append(adaboost2.score(x_train2, y_train2))
    test_accuracy.append(adaboost2.score(x_dev2, y_dev2))
plt.plot(num_tree, train_accuracy, 'r')
plt.plot(num_tree, test_accuracy, 'b')
plt.show()

models = [gbm2, lglr2, svc2, adaboost2]
for model in models:
    predictions = model.predict(sec_x_test)
    Submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
    Submission.to_csv(model.name + ".csv", index = False)
    
    
    
    

#处理unbalance样本
#查一查他这个score是什么情况
#搞一下评价标准
#调参数
#怎么加可解释性
#rf, et , gb 可以跑，svc跑不动，剩下的试试看











