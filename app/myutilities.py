import numpy as np
def get_numeric_cols_subdf(df_):
    #numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #newdf = df_.select_dtypes(include=numerics)
    newdf = df_.select_dtypes(include=np.number)
    return newdf
def get_columns_dic(df_):
    # = get_numeric_cols_subdf(df_)
    #option_list = []
    #for col in newdf.columns:
    #    option_list.append({'label': col, 'value': col})
    option_list = df_.select_dtypes(include=np.number).columns.tolist()
    option_dic = []
    for col in option_list:
        option_dic.append({'label': col, 'value': col})
    #print(option_dic)
    #return option_dic

# def get_columns_list(df_):
#     option_list = df_.select_dtypes(include=np.number).columns.tolist()
#     #newdf = get_numeric_cols_subdf(df_)
#     #option_list = []
#     #for col in newdf.columns:
#     #    option_list.append( col)
#     return option_list

# def get_datetime_columns_list(df_):
#     datetime = ['datetime64', 'datetime64[ns]']
#     newdf = df_.select_dtypes(include=datetime)
#     option_list = []
#     for col in newdf.columns:
#         option_list.append( col)
#     return option_list





## COLORS

# convert plotly hex colors to rgba to enable transparency adjustments
def hex_rgba(hex, transparency):
    col_hex = hex.lstrip('#')
    col_rgb = list(int(col_hex[i:i + 2], 16) for i in (0, 2, 4))
    col_rgb.extend([transparency])
    areacol = tuple(col_rgb)
    return areacol

# Make sure the colors run in cycles if there are more lines than colors
def next_col(cols):
    while True:
        for col in cols:
            yield col
def normalize_col(_df, _colname):
    _df[_colname]=(_df[_colname]-_df[_colname].min())/(_df[_colname].max()-_df[_colname].min())
    return _df