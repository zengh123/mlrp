import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from math import log
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sys

# arg1 is n_estimator is int(sys.argv[1])


CV = True
cv_size = 0.2
np.random.seed(0)
le = LabelEncoder()

def NDCG_eval(X, y):
	assert X.shape[0] == y.shape[0]
	denominator = np.zeros( (5,1) )
	# print denominator;
	for i in range(0,5):
		#print i
		denominator[i] = 1 / log(i+2, 2);
		#print denominator[i]
	ind = np.dot((X == y),denominator)
	#print ind.shape
	return np.mean(ind)

################## STEP1: Data Preprossesing
#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')

if CV:
    df_train, df_test = train_test_split(df_train, test_size=cv_size)
    cv_answer = df_test['country_destination'].values
    df_test.drop(['country_destination'], axis=1, inplace=True)
    y_test = le.fit_transform(cv_answer)
else:
    df_test = pd.read_csv('../input/test_users.csv')

labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'	], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

################## STEP2: Feature engineering
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

#Classifier
#xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=int(sys.argv[1]),
#                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)   
#rf = RandomForestClassifier(n_estimators=100,criterion='gini', max_features = int(sys.argv[1]))         
rf = RandomForestClassifier(n_estimators=int(sys.argv[1]),criterion='gini') 

# Here control the number of training samples
X_part = X[:int(0.2*X.shape[0]),:]
y_part = y[:int(0.2*X.shape[0])]

print 'begin training'
# ------- gbdt: choose 1 and comment another ---------
#xgb.fit(X_part, y_part)
#xgb.fit(X, y)
# ------- ---------------------------- ---------

# ------- rf: choose 1 and comment another ---------
rf = rf.fit(X, y)
# ------- ---------------------------- ---------

################## STEP3: test or Cross Validation

#y_pred = xgb.predict_proba(X_test)
y_pred = rf.predict_proba(X_test)
if CV:
	top5_cv = np.argsort(y_pred)[:,::-1][:,:5]
	y_test2 = y_test.reshape(y_test.shape[0],1)
	print 'cross_validation score:'
	print NDCG_eval(top5_cv,y_test2)
	#y_train_pred = xgb.predict_proba(X)
	y_train_pred = rf.predict_proba(X)
	top5 = np.argsort(y_train_pred)[:,::-1][:,:5]
	y_reshape = y.reshape(y.shape[0],1)
	print 'train score:'
	print NDCG_eval(top5, y_reshape)
else:
	#y_pred_rf = rf.predict_proba(X_test)
	#Taking the 5 classes with highest probabilities
	ids = []  #list of ids
	cts = []  #list of countries
	for i in range(len(id_test)):
	    idx = id_test[i]
	    ids += [idx] * 5
	    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
	    #cts += le.inverse_transform(np.argsort(y_pred_rf[i])[::-1])[:5].tolist()
	    #cts += le.inverse_transform(np.argsort(y_pred_combine[i])[::-1])[:5].tolist()
	#Generate submission
	sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
	# Name the file differently in order to draw the figure
	sub.to_csv('sub_combine_' + sys.argv[1] + '.csv',index=False)


################## STEP4: Other processing

'''
# training err
y_train_pred = xgb.predict_proba(X)
top5 = np.argsort(y_train_pred)[:,::-1][:,:5]
y_reshape = y.reshape(y.shape[0],1)
train_err = NDCG_eval(top5, y_reshape)
print train_err

# test err
# [1 1 1 1 1], [1 1 1 1 1], ...
#np.argsort(y_pred)[::-1][:,:5]

# Now pay attention to each class 
for k in range(12):
	ind = np.where(y_reshape == k)
	top5_part = top5[ind[0]]
	y_part = y_reshape[ind[0]]
	print k
	print le.inverse_transform(k)
	print NDCG_eval(top5_part, y_part)
'''
