import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter('error', SettingWithCopyWarning)

train=pd.read_csv(os.getcwd()+'\\Dataset\\train.csv')
test=pd.read_csv(os.getcwd()+'\\Dataset\\test.csv')

print(train.head())
print(test.head())

print(train.info())
print(test.info())

for x in train.columns:
    print(train[x].unique())
for x in test.columns:
    print(test[x].unique())

# Perform EDA to create hypotheses

non_risks=['area_assesed', 'district_id', 'has_repair_started','vdcmun_id','has_geotechnical_risk']

# Create x variables for risk counts for bar plots
count_risks=train.drop(non_risks, axis=1)
count_risks.info()
count_risks['total_risks']=count_risks.sum(axis=1)

damage_vs_num_risks=count_risks.groupby(['damage_grade','total_risks']).size()
dvr=damage_vs_num_risks.reset_index()
dvr.columns=['damage_grade','total_risks','counts']
dvr_p=pd.pivot_table(dvr,'counts','damage_grade','total_risks')

plt.figure(figsize=[9,4])
for x in range(0,7):
    plt.bar(dvr_p[x].index,dvr_p[x].values)
plt.savefig('test.png')

# Plotted damage grade vs count of number of risks per building.
# Seems damage is correlated negatively to the number of geotechnical risks in the area. Perhaps damage grade is rated from grade 5 being low damage to grade 1 being high damage.

# Plot damage grade vs district and municipality
non_locations=['district_id', 'vdcmun_id', 'damage_grade']
location_df=train[non_locations]

corr = location_df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#knowing the district and municipality are definitely correlated, we can drop the more complex column

train.drop('vdcmun_id', axis=1, inplace=True)
test.drop('vdcmun_id', axis=1, inplace=True)

# has_repair_started is a leakage column of data because repairs will start after the damage has occured. Our task is to predict the level of damage in a building in a particular area given several geotechnical risks, so it is a form of leaky predictor.

train.drop('has_repair_started', axis=1, inplace=True)
test.drop('has_repair_started', axis=1, inplace=True)

# building_id does not matter for training purposes so we should drop this column from both sets but retain to add to submission data later

test_id=test.building_id
train_id=train.building_id

train.drop('building_id', axis=1, inplace=True)
test.drop('building_id', axis=1, inplace=True)

# as long as we do not sort values, we should be able to concatenate the two later on

# create dummy variables for area_assesed and district_id
train = pd.concat([train.drop('area_assesed', axis=1), pd.get_dummies(train['area_assesed'])], axis=1)
train= pd.concat([train.drop('district_id', axis=1), pd.get_dummies(train['district_id'])], axis=1)

test= pd.concat([test.drop('area_assesed', axis=1), pd.get_dummies(test['area_assesed'])], axis=1)
test= pd.concat([test.drop('district_id', axis=1), pd.get_dummies(test['district_id'])], axis=1)


train['damage_grade']=list(map(lambda x: x.split(' ')[1], train['damage_grade']))
train['damage_grade']=train['damage_grade'].astype(int)
train.info()
# now we build a simple model with logisticregression and onevsrestclassifier

clf=OneVsRestClassifier(LogisticRegression())
x_train=train.drop('damage_grade', axis=1)
y_train=pd.get_dummies(train['damage_grade'])
clf.fit(x_train,y_train)

print("Accuracy: {}".format(clf.score(x_train, y_train))) #not accuracy 30% only.
x_test=test
predictions = clf.predict_proba(x_test)
pred=pd.DataFrame(predictions, columns=y_train.columns)

#return later when bderiel





# now build model with xgboost using objective:softmax and testing for best CV params
#cv_params = {'max_depth':[3,5]}
#clf_params={'objective':'multi:softmax','nthread':2,'max_depth':3,'n_estimators':1000,'subsample':0.5}
#
#clf=XGBClassifier(**clf_params)
#
#model=GridSearchCV(clf, cv_params, scoring='accuracy', cv=3, n_jobs=3)
#
#model.fit(x_train, train['damage_grade'])

params={}
params['objective'] = 'multi:softmax'
# scale weight of positive examples
params['eta'] = 0.1
params['max_depth'] = 7
params['silent'] = 1
params['nthread'] = 4
params['num_class'] = 6
params['subsample']=0.8
params['colsample_bytree']=0.7
params['min_child_weight']=3

xgb_train=xgb.DMatrix(x_train, label=train['damage_grade'])
xgb_test=xgb.DMatrix(x_test,label =np.zeros(len(x_test)))

watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
num_round = 25
bst = xgb.train(params, xgb_train, num_round, watchlist)
# get prediction
train_pred=bst.predict(xgb_train)
preds = bst.predict(xgb_test)
error_rate = np.sum(train_pred != train['damage_grade']) / train['damage_grade'].shape[0]
print('Accuracy using softmax = {}'.format(1-error_rate))
preds=pd.DataFrame(preds)
submission=pd.concat([test_id, preds], axis=1)

submission.iloc[:,1]=list(map(lambda x: str(x), submission.iloc[:,1]))
submission.iloc[:,1]=list(map(lambda x: x.replace('.0',''), submission.iloc[:,1]))
submission.iloc[:,1]=list(map(lambda x: 'Grade '+(x), submission.iloc[:,1]))
submission.columns=['building_id', 'damage_grade']
submission.set_index('building_id')
submission.to_csv('sub.csv',index=False)
