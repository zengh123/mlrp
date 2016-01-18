import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import operator

# Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

# Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

# date_account_created
dac = np.vstack(
    df_all.date_account_created.astype(str).apply(
        lambda x: list(map(int, x.split('-')))
        ).values
    )
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

# timestamp_first_active
tfa = np.vstack(
    df_all.timestamp_first_active.astype(str).apply(
        lambda x: list(map(int, [x[:4], x[4:6], x[6:8],
                                 x[8:10], x[10:12],
                                 x[12:14]]))
        ).values
    )
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

# One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
             'affiliate_channel', 'affiliate_provider', 
             'first_affiliate_tracked', 'signup_app', 
             'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

# Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

# Classifier
params = {'eta': 0.2,
          'max_depth': 6,
          'subsample': 0.5,
          'colsample_bytree': 0.5,
          'objective': 'multi:softprob',
          'num_class': 12}
num_boost_round = 2
dtrain = xgb.DMatrix(X, y)
clf1 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

# Plot feature importance
xgb.plot_importance(clf1)

# Function to store feature map (required by XGBoost.get_fscore)
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    print i
    outfile.close()

# Get feature scores and store in DataFrame
create_feature_map(list(df_all.columns.values))
importance = clf1.get_fscore(fmap='xgb.fmap')
importance_df = pd.DataFrame(
    sorted(importance.items(), key=operator.itemgetter(1)), 
    columns=['feature','fscore']
    )

# Save Importance
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 25))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')


# Only select features w/ a feature score (can also specify min fscore)
# Retrain model with reduced feature set
df_all = df_all[importance_df.feature.values]
vals = df_all.values
X_test = vals[piv_train:]
dtrain = xgb.DMatrix(X, y)
clf2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

y_pred = clf2.predict(xgb.DMatrix(X_test)).reshape(df_test.shape[0],12)

# Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)