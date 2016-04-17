
# coding: utf-8

# In[49]:

####################################################################################################################

# Predictions will be made using exponential decay from the number of repairs in the previous month, using the
# formula p = k ** n, where p is the prediction, k is a constant to be determined, and n is the number of months
# elapsed since the last known repair number. The following code will find the k for each component that minimizes
# the error on a local test set, which is from 2009-09-01 - 2009-12-01 inclusive.

# BEST SCORE: 2.61782

####################################################################################################################

import pandas as pd
import numpy as np

# Load files created in preprocesser
df_repair_train = pd.read_pickle('./data/pp_repair_train.pkl')
df_repair_test = pd.read_pickle('./data/pp_repair_test.pkl')

# Split the training file into two for local testing without having to submit to the leaderboard
df_local_train = df_repair_train[df_repair_train.date_repair <= '2009-08-01']
df_local_test = df_repair_train[df_repair_train.date_repair > '2009-08-01']

# Set up the local_file with columns - item_id,date_repair,months_elapsed,truth,prediction,error
# Current prediction is the number of repairs in 2009-08-01. Error is absolute
df_local_test = df_local_test.rename(columns={'number_repair':'truth'})
df_local_test['startdate'] = pd.to_datetime('2009-07-15')
df_local_test['months_elapsed'] = (df_local_test['date_repair'] - df_local_test['startdate'])\
                                        .astype('timedelta64[M]')
df_local_test = df_local_test.drop('startdate',axis=1)
df_local_test = df_local_train[df_local_train.date_repair=='2009-08-01'].drop('date_repair',axis=1)\
                                        .merge(df_local_test, on='item_id')
df_local_test = df_local_test.rename(columns={'number_repair':'prediction'})
df_local_test['error'] = abs(df_local_test.prediction-df_local_test.truth)

# Reorder columns
df_local_test = df_local_test[['item_id','date_repair','months_elapsed','truth','prediction','error']]


# List of Ks to try
Ks = np.arange(0.87,0.96,0.01)

# Initialise best k for this component
df_local_test['best_k'] = 0

# Iterate through each item and value of k. Calculate the prediction and error, and select the k that minimises
# the error
for item in df_local_test.item_id.unique():
    best_k = -1
    best_error = float("inf")
    for k in Ks:
        df_local_test['expon_prediction'] = df_local_test.prediction * k**df_local_test.months_elapsed
        df_local_test['expon_error'] = abs(df_local_test.expon_prediction - df_local_test.truth )
        this_error = df_local_test[df_local_test.item_id==item].expon_error.mean()
        if this_error < best_error:
            best_error = this_error
            best_k = k

    df_local_test.loc[df_local_test['item_id']==item,'best_prediction'] = \
                                                         df_local_test.prediction * best_k**df_local_test.months_elapsed

    df_local_test.loc[df_local_test['item_id']==item,'best_error'] = \
                                                         abs(df_local_test.best_prediction - df_local_test.truth )

    df_local_test.loc[df_local_test['item_id']==item,'best_k'] = best_k

df_local_test = df_local_test.drop(['expon_prediction','expon_error'],axis=1)

# A dataframe with the best k value found for each item
best_ks = df_local_test.groupby('item_id').best_k.mean()
best_ks = best_ks.reset_index()

# Get the number of months elapsed in the testing set
df_repair_test['startdate'] = pd.to_datetime('2009-11-15')
df_repair_test['months_elapsed'] = (df_repair_test['date'] - df_repair_test['startdate']).astype('timedelta64[M]')
df_repair_test = df_repair_test.drop('startdate',axis=1)

# Get the number of repairs in the last month available (2009-12)
df_repair_test = df_repair_train[df_repair_train.date_repair=='2009-12-01'].drop('date_repair',axis=1)\
                                        .merge(df_repair_test, on='item_id')

# Merge that with the best values of k
df_repair_test = df_repair_test.merge(best_ks,on='item_id')

# Make the predictions
df_repair_test.number_repair = df_repair_test.number_repair * df_repair_test.best_k**df_repair_test.months_elapsed
predictions = df_repair_test['number_repair'].reset_index()
predictions = predictions.rename(columns = {'number_repair': 'target','index':'id'})
predictions.id = predictions.id +1

# Save to file
predictions.to_csv('submission.csv',index=False)
