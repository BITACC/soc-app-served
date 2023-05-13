import pandas as pd
import base64
import datetime
import io
import os
import dash_html_components as html
import dash_table
from  utilities import * #SESSION_ID, UPLOAD_DIRECTORY

#path = '..\\..\\Phase 1\\data\\'
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

    return [ html.Div(filename), df]
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

def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    #SESSION_ID
    #dirname = os.path.join(UPLOAD_DIRECTORY, SESSION_ID)
    dirname = UPLOAD_DIRECTORY
    file_path = os.path.join(dirname, name)

    # isfile = ""
    # if os.path.exists(file_path):
    #     isfile = "Exists"
    # else:
    #     isfile = "No Exists"

    # # if os.path.isdir(UPLOAD_DIRECTORY): 
    # #     isfile = "   i s" + file_path
    # # else:
    # #     isfile = "  isnot" + file_path


    # return file_path, "Good join" + file_path + isfile
    line = 0
    with open(file_path, "wb") as fp:
        line = fp.write(base64.decodebytes(data))
    
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(io.BytesIO(decoded))

        

    return file_path, "Good join" + file_path  +  "   "  + str(line)

def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


# def file_download_link(filename):
#     """Create a Plotly Dash 'A' element that downloads a file from the app."""
#     location = "/download/{}".format(urlquote(filename))
#     return html.A(filename, href=location)

def parse_data1(contents, filename):
    content_type, content_string = contents.split(",")

    msg = "None"
    decoded = base64.b64decode(content_string)
    msg = "Decode"
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            msg = "csv"
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=';', parse_dates = {'DateTime' : ['Date', 'Time']}, index_col = ['DateTime'], dayfirst=True)
        elif "xlsx" in filename:
            # Assume that the user uploaded an excel file
            msg = "xlsx"
            _decoded = io.BytesIO(decoded)
            msg = "_decoded"
            df = pd.read_excel(_decoded)
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            msg = "txt"
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return [ None, html.Div(["There was an error processing this file." + msg ])]

    return [df,  html.Div(["processing this file Done."])]