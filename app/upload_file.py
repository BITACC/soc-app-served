import pandas as pd
import base64
import datetime
import io
import dash_html_components as html
import dash_table

path = '..\\..\\Phase 1\\data\\'
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        #print(e)
        return [html.Div([
            'There was an error processing this file.'
        ]), None]

    return [ html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),


        #dash_table.DataTable(
        #    data=df.to_dict('records'),
        #    columns=[{'name': i, 'id': i} for i in df.columns]
        #),

        #html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #})
    ]), df]
''''
def parse_contents( filename):
    #content_type, content_string = contents.split(',')
    print("in  ", filename[0] , [ 'csv' in filename[0]])
    #decoded = base64.b64decode(content_string)
    try:
        if  'csv' in filename[0]:
            print("read csv")
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(path + filename[0]
               # io.StringIO(decoded.decode('utf-8'))
               )
        elif 'xls' in filename[0]:
            # Assume that the user uploaded an excel file
            print("read excel1", path + filename[0])
            df = pd.read_excel(path + filename[0] , header=0, parse_dates=[0], index_col=0)#
            #print("df")
            #print(df)
            #df.head()
            print("read excel2")
    except Exception as e:
        print("read except")
        return html.Div([
            'There was an error processing this file.'
        ])
    print("parse content return")
    return df, path + filename[0]
'''

