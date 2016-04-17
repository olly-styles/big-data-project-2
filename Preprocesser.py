####################################################################################################################

# Reads in all data except the sale file, and outputs two pandas dataframes, one for training and one for testing.
# These dataframes use datatime rather than the two separate columns for month and year, and the module id/component
# id are concatenated into a single item_id column. Each file contains a single entry for each month and module
# component pair, with a 0 entry if no repairs were made in the training file.
#
# These are then saved to disk in the /data folder as 'df_repair_train.pkl' and 'df_repair_test.pkl'.

####################################################################################################################

import pandas as pd

# Read in the data
df_repair = pd.read_csv('./data/RepairTrain.csv')
df_sample = pd.read_csv('./data/SampleSubmission.csv')
df_mapping = pd.read_csv('./data/Output_TargetID_Mapping.csv')

# Create an item id in each dataframe. The id is the concatenation of the module category and component category
df_repair['item_id'] = df_repair['module_category']+df_repair['component_category']
df_mapping['item_id'] = df_mapping['module_category']+df_mapping['component_category']

# Rename columns into something more sensible
df_repair = df_repair.rename(columns={'year/month(sale)': 'date_sale', 'year/month(repair)': 'date_repair'})

# Convert date columns in repair into pandas datetime
df_repair['date_sale'] = pd.to_datetime(df_repair['date_sale'],format='%Y/%m')
df_repair['date_repair'] = pd.to_datetime(df_repair['date_repair'],format='%Y/%m')

#Group the repairs by item and date
df_repair = df_repair[['item_id','date_repair','number_repair']]\
                .groupby(['item_id','date_repair']).sum().reset_index()


# Convert date column in the mapping dataframe into pandas datatime
df_mapping['slash'] = '/'
df_mapping['date'] = df_mapping['year'].apply(str) + df_mapping['slash'] + df_mapping['month'].apply(str)
df_mapping.date = pd.to_datetime(df_mapping.date,format='%Y/%m')

# Create the testing dataframe for in a better format that the sample
df_repair_test = df_mapping[['item_id','date']]


####################################################################################################################

# This block creates a repair training dateframe with all possible module-component pairs and dates from 02/2005 to
# 12/2009. The module-component pair is transformed to 'item_id' and the date is in pandas datetime format.
# There is certainly a cleaner way to do this, but it works and runs quickly

####################################################################################################################

# Create a dateframe for each year
df_2009 = df_mapping[df_mapping.year<2011]
df_2008 = df_mapping[df_mapping.year<2011]
df_2007 = df_mapping[df_mapping.year<2011]
df_2006 = df_mapping[df_mapping.year<2011]
df_2005 = df_mapping[df_mapping.year<2011]

# Set the year for each
df_2009.year = 2009
df_2008.year = 2008
df_2007.year = 2007
df_2006.year = 2006
df_2005.year = 2005

# Records from 2005 start in February
df_2005 = df_2005[df_2005.month > 1]

# Concatenate
df_repair_train = pd.concat([df_2005,df_2006,df_2007,df_2008,df_2009])
df_repair_train = df_repair_train.sort_values(by='item_id')

# Convert the month/year format into pandas datetime
df_repair_train['slash'] = '/'
df_repair_train['date_repair'] = df_repair_train['year'].apply(str)\
                                  + df_repair_train['slash'] + df_repair_train['month'].apply(str)
df_repair_train.date_repair = pd.to_datetime(df_repair_train.date_repair,format='%Y/%m')

# Get the number of repairs, drop the unimportant columns, and fill null repairs with 0s
df_repair_train = df_repair_train.merge(df_repair,on=['item_id','date_repair'],how='left')
df_repair_train = df_repair_train[['item_id','date_repair','number_repair']]\
                    .sort_values(by=['item_id','date_repair'])
df_repair_train = df_repair_train.fillna(0)

# Reset the index so it is ordered
df_repair_train = df_repair_train.reset_index().drop('index',axis=1)


####################################################################################################################

# Check that nothing has broken

####################################################################################################################

# Same number of unique items in training and testing
assert len(df_repair_train.item_id.unique()) == len(df_repair_test.item_id.unique())

# Same number of entries in the testing dateframe and sample submission
assert len(df_repair_test) == len(df_sample)

# One entry in training file for each month for each component
num_months = 12 * 5 - 1
assert len(df_repair_train) == (num_months * len(df_repair_train.item_id.unique()))


####################################################################################################################

# Save to disk

####################################################################################################################

df_repair_train.to_pickle('./data/pp_repair_train.pkl')
df_repair_test.to_pickle('./data/pp_repair_test.pkl')
