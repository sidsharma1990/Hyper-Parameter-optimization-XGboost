# Hyper Parameter optimization XGboost
## Hyperparameter optimization using RandomizedSearchCV

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import xgboost

df=pd.read_csv('Hyper Parameter optimization XGboost.csv')

# Correlation
corr = df.corr()

# Dependent and independent variable
X=df.iloc[:, :-1]
Y=df.iloc[:, -1]

# Dummy Variables
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

# Drop Categorical Features
X=X.drop(['Geography','Gender'],axis=1)

# concatenation
X=pd.concat([X,geography,gender],axis=1)

# Hyper Parameter Optimization
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}

# Classifier
classifier=xgboost.XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,
                                 scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X,Y)

# Best estimator and Parameter
random_search.best_estimator_
random_search.best_params_

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.4,
              learning_rate=0.2, max_delta_step=0, max_depth=8,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,Y,cv=10)
print (score)
score.mean()