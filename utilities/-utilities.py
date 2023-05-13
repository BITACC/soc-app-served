# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:

# import pandas as pd
# from datetime import datetime, timedelta
# import numpy as np
# import os
# import uuid

# UPLOAD_DIRECTORY = "Uploads"
# SESSION_ID= str(uuid.uuid1())


# # remove empty columns

# def strip_df_colnames(_df):
#     _df = _df.rename(columns={c: str.strip(c)  for c in _df.columns})
#     return _df


# def get_filename_wo_extention(_fname):
#     return os.path.splitext(_fname)[0]


# # remove empty columns
# def drop_empty_cols(_df, spec_cols ):
#     empty_cols = _df.columns[_df.isnull().mean()>0.5]
#     _df.drop(empty_cols, axis = 1, inplace = True)
#     for c in spec_cols:
#         print (c)
#         _df.dropna(subset=[c], inplace=True)
    
#     return _df

# def keep_columns(_df, colnames_kept):
#     for e in _df.columns:
#         if not  e in colnames_kept:
#             _df.drop(e, inplace = True, axis = 1)
#     return _df

# def keept_cols(_df, _colnames_kept):
#     kc= [e for e in _df.columns if  e in _colnames_kept]
#     return kc

# def convert_non_numeric_to_string(_df):
#     cols = list(_df.select_dtypes(include=['object', 'string']).columns)
#     #print(_df.dtypes)
#     for c in cols: 
#         _df[c] = _df[c].astype(str)
#         _df[c] = _df[c].apply(lambda x:str.strip (x)  ) 
#     return _df


# from datetime import datetime
# def format_timestamp(timestamp_colname, _df):
#     if (timestamp_colname in _df.select_dtypes('datetime').columns ):
#         return _df
        
#     _df[timestamp_colname] = _df[timestamp_colname].apply(lambda x: 
#                                     datetime.strptime(x,"%d/%m/%Y %H:%M:%S"))
#     return _df

# # return the list of numerical columns
# def get_numeric_columns(_df):
#     #numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     #newdf = df.select_dtypes(include=numerics)
#     return _df.select_dtypes(include=np.number).columns.tolist()    

# def rows_with_k_consecutive_nans(df, k):
#     """This is exactly like the above but using pandas functions instead of
#     numpys. (see also Scott Boston answer). The approach is completly identical!
#     """
#     return df.isnull().rolling(window=k, axis=1).sum().ge(k).any(axis=1)

# def cols_with_k_consecutive_nans(df, k):
#     """This is exactly like the above but using pandas functions instead of
#     numpys. (see also Scott Boston answer). The approach is completly identical!
#     """
#     return df.isnull().rolling(window=k, axis=0).sum().ge(k).any(axis=0)



# # def floor_dt(tm):
# #     #type(tm)
# #     #print(tm)
# #     tm = tm - timedelta(minutes=tm.minute % 30,
# #                              seconds=tm.second,
# #                              microseconds=tm.microsecond)
# #     return tm

# # def datetime_column(_df, datetime_col):
# #     _df[datetime_col] = _df[datetime_col].astype(str)
# #     _df[datetime_col] = _df[datetime_col].apply(lambda x:str.strip (x)   ) 
# #     _df = format_timestamp(datetime_col, _df)

# #     if(datetime_col in  _df.columns):
# #         _df.set_index(datetime_col, inplace = True)
# #     return _df
# # def get_numeric_columns(_df):
# #     #numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# #     #newdf = df.select_dtypes(include=numerics)
# #     return _df.select_dtypes(include=np.number).columns.tolist()    


# # def df_to_cnn_rnn_format(df, train_size=0.5, look_back=5, target_column='target', scale_X=True):
# #     """
# #     TODO: output train and test datetime
# #     Input is a Pandas DataFrame. 
# #     Output is a np array in the format of (samples, timesteps, features).
# #     Currently this function only accepts one target variable.

# #     Usage example:

# #     # variables
# #     df = data # should be a pandas dataframe
# #     test_size = 0.5 # percentage to use for training
# #     target_column = 'c' # target column name, all other columns are taken as features
# #     scale_X = False
# #     look_back = 5 # Amount of previous X values to look at when predicting the current y value
# #     """
# #     df = df.copy()

# #     # Make sure the target column is the last column in the dataframe
# #     df['target'] = df[target_column] # Make a copy of the target column
# #     df = df.drop(columns=[target_column]) # Drop the original target column
    
# #     target_location = df.shape[1] - 1 # column index number of target
# #     split_index = int(df.shape[0]*train_size) # the index at which to split df into train and test
    
# #     # ...train
# #     X_train = df.values[:split_index, :target_location]
# #     y_train = df.values[:split_index, target_location]

# #     # ...test
# #     X_test = df.values[split_index:, :target_location] # original is split_index:-1
# #     y_test = df.values[split_index:, target_location] # original is split_index:-1

# #     # Scale the features
# #     if scale_X:
# #         scalerX = StandardScaler(with_mean=True, with_std=True).fit(X_train)
# #         X_train = scalerX.transform(X_train)
# #         X_test = scalerX.transform(X_test)
        
# #     # Reshape the arrays
# #     samples = len(X_train) # in this case 217 samples in the training set
# #     num_features = target_location # All columns before the target column are features

# #     samples_train = X_train.shape[0] - look_back
# #     X_train_reshaped = np.zeros((samples_train, look_back, num_features))
# #     y_train_reshaped = np.zeros((samples_train))

# #     for i in range(samples_train):
# #         y_position = i + look_back
# #         X_train_reshaped[i] = X_train[i:y_position]
# #         y_train_reshaped[i] = y_train[y_position]


# #     samples_test = X_test.shape[0] - look_back
# #     X_test_reshaped = np.zeros((samples_test, look_back, num_features))
# #     y_test_reshaped = np.zeros((samples_test))

# #     for i in range(samples_test):
# #         y_position = i + look_back
# #         X_test_reshaped[i] = X_test[i:y_position]
# #         y_test_reshaped[i] = y_test[y_position]
    
# #     return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped


# # In[ ]:

# def get_filename_wo_extention(_fname):
#     return os.path.splitext(_fname)[0]
    

# # remove empty columns
# def drop_empty_or_sparse_cols(_df, spec_cols ):
#     #delete all cols containing Unnamed
#     for e in _df.columns:
#         if 'Unnamed' in e:
#             #print(e)
#             _df.drop(e, axis = 1, inplace = True)
#     empty_cols = _df.columns[_df.isnull().mean()>0.5]
#     _df.drop(empty_cols, axis = 1, inplace = True)
#     for c in intersection(spec_cols, _df.columns):
#         _df.dropna(subset=[c], inplace=True)
    
#     return _df, empty_cols



# def create_datatime_col(_df, spec_cols):
#     if ('Date'in _df.columns and 'Time' in _df.columns):
        
#         _df['Date']= pd.to_datetime(_df['Date'], errors='coerce', dayfirst=True)
#         _df['Date'] = _df['Date'].dt.strftime('%d/%m/%Y')
#         _df['Time'] = _df['Time'].apply(pd.to_datetime, format='%H:%M:%S')
#         _df['Time'] = _df['Time'].dt.strftime('%H:%M:%S')
#         _df['DateTime'] = _df['Date'].astype(str) + ' ' + _df['Time'].astype(str)
#         _df['DateTime'] = _df['DateTime'].apply(lambda x: 
#                                        datetime.strptime(x,"%d/%m/%Y %H:%M:%S"))
#         for e in spec_cols:
#             if  e in  _df.columns:
#                 #print(e)
#                 _df.drop(e, axis = 1, inplace = True)                                       
#     else:
#         print('Date and/or Time columns do not found ...')
#     return _df

    
# # set indez
# def set_df_col_as_index(_df, idx_col = 'DateTime'):
#     if('DateTime' in _df.columns):
#         _df.set_index('DateTime', inplace = True)
#         return _df
#     else:
#         print("The column " + idx_col + " does not exist or it is alreay as index...")
#         return _df    

# def intersection(lst1, lst2):
#     lst3 = [value for value in lst1 if value in lst2]
#     return lst3        


# def get_filename_wo_extention(_fname):
#     import os
#     return os.path.splitext(_fname)[0]    


# # set indez
# def set_df_col_as_index(_df, idx_col = 'DateTime'):
#     if('DateTime' in _df.columns):
#         _df.set_index('DateTime', inplace = True)
#         return _df
#     else:
#         print("The column " + idx_col + " does not exist or it is alreay as index...")
#         return _df
