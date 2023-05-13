# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:


# # importing module
# import sys, os
# # appending a path
# APP_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'app'))
# sys.path.append(APP_DIR)
# DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'latest')) #"/.../.../.../Data/lates" #..\\..\\..\\Data\\latest
# NOTEBOOKS_FOLDER = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks')) #"/.../.../.../Data/lates" #..\\..\\..\\Data\\latest
# sys.path.append(NOTEBOOKS_FOLDER)


# from params import parameters
# casefolded_train_col_names = [str.casefold(x) for x in parameters.cols_for_training]
# casefolded_train_col_names_except_date_time = [str.casefold(x) for x in parameters.cols_for_training_except_date_time]
# casefolded_train_col_names_except_date_time_target = [str.casefold(x) for x in parameters.cols_for_training_except_date_time_CorrosionRate]


# import pandas as pd
# from datetime import datetime, timedelta
# import numpy as np
# import os
# import copy
# import re

# target = 'corrosion rate (lpr)'




# def data_import( Multifile = True):
#     FILES = [  "Laboratory_data_KUL_MBBR1_BVE(processed2).xlsx", 
#             "Laboratory_data_KUL_MBBR3_BVE(processed2).xlsx"]#, "Laboratory_data_AMA_All_Data.xlsx"  , "Ugent_20220216_till_20220420.xlsx" ] # ,  "Laboratory_data_KUL.xlsx"  , "Laboratory_data_AMA_All_Data.xlsx"   ,  "Laboratory_data_UGENT.xlsx"
#     train_datafile = "Laboratory_data_KUL_MBBR3_BVE(processed2).xlsx" 
#     # 
#     df = None
#     if not Multifile :
#         pass
#     else:
#         df = [pd.read_excel(DATA_DIR + "\\" + f)  for f in FILES]

#         for d in df:
#             print("# records: ", d.shape[0])
    
#     return df, train_datafile

# def prepare_data(df, Multifile = True):
#     if not Multifile :
#         df = strip_remove_multiplespace_df_colnames(df, to_casefold=True)
#         df = df[casefolded_train_col_names]
#         df = drop_empty_rows_cols(df)
#         df = create_datatime_col(df)
#         df = drop_empty_or_sparse_cols(df, ['date', 'time'])
#         df = strip_no_numeric_cols(df)
#         set_df_col_as_index(df, 'DateTime') 
#     else:
#         df = [strip_remove_multiplespace_df_colnames(f, to_casefold=True) for f in df]
#         #{str.casefold(x) for x in parameters.cols_for_training} & {str.casefold(x) for x in a2}
#         df = [f[casefolded_train_col_names] for f in df]
#         df = [drop_empty_rows_cols(f) for f in df]
#         df = [create_datatime_col(f) for f in df]
#         df = [drop_empty_or_sparse_cols(f, ['date', 'time']) for f in df]  #
#         df = [strip_no_numeric_cols(f) for f in df]
#         df = [set_df_col_as_index(f, 'DateTime') for f in df] 
    
#     return df









# def drop_empty_rows_cols(_df):
#     _df = _df.dropna(how='all')
#     return _df

# # remove empty columns

# def strip_remove_multiplespace_df_colnames(_df, to_casefold = False):
#     _df = _df.rename(columns={c: str.strip(c)  for c in _df.columns})
#     _df = _df.rename(columns={c: re.sub(' +', ' ', c)  for c in _df.columns})
#     if to_casefold:
#         _df = _df.rename(columns={c: str.casefold(c)  for c in _df.columns})
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


# def strip_no_numeric_cols(_df):
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
#     #_df.drop(empty_cols, axis = 1, inplace = True)
    
    
#     for c in spec_cols:
#         type(_df)
#         #_df.drop(subset=[c], axis=1, inplace=True)
#         _df = _df[_df[c].notna()]
#     return _df

# def create_datatime_col(_df ):
#     if ('date' in _df.columns and 'time' in _df.columns):
        
#         _df['date']= pd.to_datetime(_df['date'], errors='coerce', dayfirst=True)
#         _df['date'] = _df['date'].dt.strftime('%d/%m/%Y')
#         #print(_df['Date'])
        
#         #_df['Time']= pd.to_datetime(df['Time'])
#         #
#         #df['Date'].ffill(axis= 0)
#         _df['time'] = _df['time'].apply(pd.to_datetime, format='%H:%M:%S')
#         _df['time'] = _df['time'].dt.strftime('%H:%M:%S')
        
        
#         #df['Time']=pd.to_datetime(df['Time'], errors='coerce')
#         #df['Time'] = df['Time'].dt.strftime('%H:%M:%S')
#         #df['Time'].ffill(axis= 0)
#         #_df['Date'] = pd.to_datetime(df['Date']).dt.date
#         #_df['Time'] = pd.to_datetime(df['Time']).dt.time
        
#         _df['DateTime'] = _df['date'].astype(str) + ' ' + _df['time'].astype(str)
#         _df['DateTime'] = _df['DateTime'].apply(lambda x: 
#                                        datetime.strptime(x,"%d/%m/%Y %H:%M:%S"))
        

#         #_df.loc[:,'DateTime'] = pd.to_datetime(df.Date.astype(str)+' '+df.Time.astype(str))
        
#         return _df
#     else:
#         print('Date and/or Time columns do not found ...')
#         return _df

    

    
# # set indez
# # set indez
# def set_df_col_as_index(_df, idx_col = 'DateTime'):
#     if('DateTime' in _df.columns):
#         _df.set_index('DateTime', inplace = True)
#         return _df
#     else:
#         print("The column " + idx_col + " does not exist or it is alreay as index...")
#         return _df
    
# # remove empty columns
# def drop_empty_or_sparse_cols(_df, spec_cols ):
#     #delete all cols containing Unnamed
#     for e in _df.columns:
#         if 'Unnamed' in e:
#             #print(e)
#             _df.drop(e, axis = 1, inplace = True)
#     empty_cols = _df.columns[_df.isnull().mean()>0.5]
#     _df.drop(empty_cols, axis = 1, inplace = True)
#     for c in spec_cols:
#         if c in _df.columns:
#             print(c)
#             _df.drop(columns=[c], inplace=True)
    
#     return _df    
# def drop_empty_rows(_df):
#     _df.dropna( axis = 0, inplace=True)

    
# from sklearn.metrics import *
# def report_metrics(y_true_test, y_pred_test):
#     dec = 4
#     try: 
#         print("mean_absolute_percentage_error: ", round(mean_absolute_percentage_error(y_true_test, y_pred_test), dec))
#     except:
#           print("ERR: mean_absolute_percentage_error") 
#     try: 
#         print("max_error: ", round(max_error(y_true_test, y_pred_test), dec)) #max_error metric calculates the maximum residual error.
#     except:
#           print("ERR: max_error") 
#     try: 
#         print("mean_absolute_error: ", round(mean_absolute_error(y_true_test, y_pred_test), dec)) #Mean absolute error regression loss.
#     except:
#           print("ERR: mean_absolute_error")             
#     try: 
#         print("mean_squared_error: ", round(mean_squared_error(y_true_test, y_pred_test), dec)) #Mean squared error regression loss.
#     except:
#           print("ERR: mean_squared_error")   
#     try: 
#         print("mean_squared_log_error: ", round(mean_squared_log_error(y_true_test, y_pred_test), dec)) #Mean squared logarithmic error regression loss.
#     except:
#           print("ERR: mean_squared_log_error")   
#     try: 
#         print("median_absolute_error: ", round(median_absolute_error(y_true_test, y_pred_test), dec)) #Median absolute error regression loss.
#     except:
#           print("ERR: median_absolute_error")   
#     # try: 
#     #     print("MAPE: ", metrics.mean_absolute_percentage_error(…)  #Mean absolute percentage error regression loss.
#     # except:
#     #       print("ERR: MAPE")    
#     try: 
#         print("r2_score: ", round(r2_score(y_true_test, y_pred_test), dec))   # (coefficient of determination) regression score function.
#     except:
#           print("ERR: r2_score")                
#     try: 
#         print("mean_poisson_deviance: ", round(mean_poisson_deviance(y_true_test, y_pred_test), dec)) #Mean Poisson deviance regression loss.
#     except:
#           print("ERR: mean_poisson_deviance")                
#     try: 
#         print("mean_gamma_deviance: ", round(mean_gamma_deviance(y_true_test, y_pred_test), dec)) #Mean Gamma deviance regression loss.
#     except:
#           print("ERR: mean_gamma_deviance")   
#     try: 
#         print("mean_tweedie_deviance: ", round(mean_tweedie_deviance(y_true_test, y_pred_test), dec)) #Mean Tweedie deviance regression loss.
#     except:
#           print("ERR: mean_tweedie_deviance")   
#     try: 
#         print("d2_tweedie_score: ", round(d2_tweedie_score(y_true_test, y_pred_test), dec))  # D^2  -*097653TWIOP]'.CXFJM;14 ion, percentage of Tweedie deviance explained.
#     except:
#           print("ERR: d2_tweedie_score")                 
#     try: 
#         print("mean_pinball_loss: ", round(mean_pinball_loss(y_true_test, y_pred_test), dec)) #Pinball loss for q/FVtile regression.
#     except:
#           print("ERR: mean_pinball_loss")          
              

# def mean_absolute_percentage_error(y_true, y_pred): 
#     """Calculates MAPE given y_true and y_pred"""
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
    
    
    
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.utils import indexable
# from sklearn.utils.validation import _num_samples

# class TimeSeriesSplitCustom(TimeSeriesSplit):
#     def __init__(self, n_splits=5, max_train_size=None,
#                  test_size=1,
#                  min_train_size=1):
#         super().__init__(n_splits=n_splits, max_train_size=max_train_size)
#         self.test_size = test_size
#         self.min_train_size = min_train_size

#     def overlapping_split(self, X, y=None, groups=None):
#         min_train_size = self.min_train_size
#         test_size = self.test_size

#         n_splits = self.n_splits
#         n_samples = _num_samples(X)

#         if (n_samples - min_train_size) / test_size >= n_splits:
#             print('(n_samples -  min_train_size) / test_size >= n_splits')
#             print('default TimeSeriesSplit.split() used')
#             yield from super().split(X)

#         else:
#             shift = int(np.floor(
#                 (n_samples - test_size - min_train_size) / (n_splits - 1)))

#             start_test = n_samples - (n_splits * shift + test_size - shift)

#             test_starts = range(start_test, n_samples - test_size + 1, shift)

#             if start_test < min_train_size:
#                 raise ValueError(
#                     ("The start of the testing : {0} is smaller"
#                      " than the minimum training samples: {1}.").format(start_test,
#                                                                         min_train_size))

#             indices = np.arange(n_samples)

#             for test_start in test_starts:
#                 if self.max_train_size and self.max_train_size < test_start:
#                     yield (indices[test_start - self.max_train_size:test_start],
#                            indices[test_start:test_start + test_size])
#                 else:
#                     yield (indices[:test_start],
#                            indices[test_start:test_start + test_size]) 
          
