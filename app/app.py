# Import required libraries
import pickle
import copy
import pathlib
import dash
import math
import datetime as dt
#from numpy.lib.function_base import percentile
#import dash_snapshot_engine
#import dash_design_kit
# import dask.dataframe as dd
import pandas as pd
# import modin.pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash import dcc
from dash import dash_table
import dash_html_components as html
import dash_core_components  as core
from plotly.subplots import make_subplots

# from plotly.offline import plot
import plotly.graph_objs as go
from plotly import tools
import plotly

#import dash_bootstrap_components as dbc

import random
# Multi-dropdown options
from utilities.spg_color import COUNTIES, STATUSES, WELL_TYPES, COLORS
from utilities.spg_color import *
import utilities.upload_file as upload
import plotly.express as px
import numpy as np


import params.parameters as params


for e in params.models:
    print( e, params.models[e])

import dash_bootstrap_components as dbc

from utilities import *
import joblib

#import scipy.signal as scp
#from scipy import stats

from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go, pandas as pd, plotly.express as px

import pickle
from sklearn.linear_model import LinearRegression

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
    #,  external_stylesheets =[dbc.themes.BOOTSTRAP]
    #, assets_folder="static"
)
app.title = "SOCORRO APP"

#app._favicon = ("static/favicon.ico")
# from settings import config
# app = dash.Dash(name=config.app_name, assets_folder="static"#, meta_tags=[{"name": "viewport", "content": "width=device-width"}]

# )#, external_stylesheets=[ dbc.themes.LUX,config.fontawesome]
# app.title = config.app_name

#df_current = pd.read_excel( "data/data-03122020-manuallymodified-SG.xlsx", header=0)#, parse_dates=[0], index_col=0
default_data_file = "data/AllMeas(Apr01-08).xlsx"

server = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# sample data in a pandas dataframe
np.random.seed(1)

# define colors as a list
colors = px.colors.qualitative.Plotly
rgba = [hex_rgba(c, transparency=0.8) for c in colors]
colCycle = ['rgba' + str(elem) for elem in rgba]
line_color = next_col(cols=colCycle)

import uuid
import dash_uploader  as du
UPLOAD_FOLDER_ROOT = "./Uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

from datetime import date

datetime = ['datetime64[ns]']
target = "Meas_Rate"
	

colnames_kept = [    'Temperature', 'Dissolved Oxygen', #"Meas_Rate", #'DDate',
                    'pH',  'ORP'#, #################################################  =>'AUX2:Chloride'  # FIX: 'AUX2:Turrbidity'
                ,   'Conductivity'
                ]

                    #"Chloride [mg/l]", "Conductivity at 25°C", "Dissolved oxygen [mg/l]", "Meas_Rate", "ORP Redox potential", "Corrosion rate (LPR)"
colnames_model = [
                    ["Date", "Time", "Temperature", "pH", "Dissolved oxygen", "Conductivity", "ORP", "Chloride"], 
                    ["Date", "Time", "Temperature", "pH", "Dissolved oxygen", "Conductivity", "ORP"]
                ]
model_name = ["HZS", "UGKLU"]
def check_model_type(_cols):
    
    _name = "None"
    _cols = [x.lower() for x in _cols] 
    for lst, name  in zip(colnames_model, model_name):
        _lst = [x.lower() for x in lst] 
        flag = 0
        if(all(x in _cols for x in _lst)):
            flag = 1
        if flag == 1:
            return name



# ml_models = [   {'label': 'Model 1', 'value':'Model 1'},
#                 {'label':'Models 2' , 'value': 'Model 2'},
#                 {'label': 'Model 3', 'value':'Model 3'}
#             ]



ml_models = [   #{'label': 'Model 1', 'value':'Model 1'},
                {'label':'Sea Water (Lab)' , 'value': 'Sea Water (Lab)'},
                {'label': 'Waste Water', 'value':'Waste Water'},
                {'label': 'Sea Water (Demo)', 'value':'Sea Water (Demo)'}
            ]


# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)


# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id='df'),
        dcc.Store(id='model-name'),
        # main body

   
    dbc.Row(html.Div(dbc.Alert("This is one column", color="primary"))),



        html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("socorro.png"),
                        id="logo-spg",
                        style={
                            "height": "60px",
                            "width": "auto",
                            "margin-bottom": "25px",
                        },
                    )
                ],
                className="one-third column",
            ),
            html.Div(
                [
                    # html.H3(
                    #     "SOCORRO",
                    #     style={"margin-bottom": "0px"},
                    # ),
                    # html.H5(
                    #     "Seeking out corrosion, before it is too late.",
                    #     style={"margin-top": "0px"}
                    # ),
                ],
                className="one-third column",
            ),
            # html.Div(
            #     [
            #     html.Img(
            #         src=app.get_asset_url("logo-itba.png"),
            #         id="logo-none",
            #         style={
            #             "height": "60px",
            #             "width": "500px",
            #             "align: right"
            #             "margin-bottom": "25px",
            #         },
            #     ),  
            #     ],
            #     className="one-third column",
            #     id="logos",
            # ),
        ],
        id="logos-div",
        className="row flex-display",
        style={"margin-bottom": "25px"},
    ),
        
        dcc.Tabs(
            id="tabs-with-classes",
            value='tab-1',
            parent_className='custom-tabs',
            className='custom-tabs-container',
            children=[  # tabs
                dcc.Tab(label='Monitoring',
                        value='tab-1',
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        children=[

                            #dcc.Store(id="aggregate_data"),
                            # empty Div to trigger javascript file for graph resizing
                            html.Div(id="output-clientside"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Img(
                                                src=app.get_asset_url("logo-SPG.png"),
                                                id="logo-SOCORRO",
                                                style={
                                                    "height": "60px",
                                                    "width": "auto",
                                                    "margin-bottom": "25px",
                                                },
                                            )
                                        ],
                                        className="one-third column",
                                    ),
                                    # html.Div(
                                    #     [
                                    #         html.Img(
                                    #             src=app.get_asset_url("newlogo-spg.png"),
                                    #             id="logo-SPG",
                                    #             style={
                                    #                 "height": "60px",
                                    #                 "width": "auto",
                                    #                 "margin-bottom": "25px",
                                    #             },
                                    #         )
                                    #     ],
                                    #     className="one-third column",
                                    # ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3(
                                                        "SOCORRO",
                                                        style={"margin-bottom": "0px"},
                                                    ),
                                                    html.H5(
                                                        "Seeking out corrosion, before it is too late.",
                                                        style={"margin-top": "0px"}
                                                    ),
                                                ]
                                            )
                                        ],
                                        className="one-third column",
                                        id="title",
                                    ),
                                    html.Div(
                                        [
                                            html.A(
                                                html.Button("SOCORRO Home", id="learn-more-button"),
                                                href="https://www.socorro.eu/",
                                            )
                                        ],
                                        className="one-third column",
                                        id="button",
                                    ),
                                ],
                                id="header",
                                className="row flex-display",
                                style={"margin-bottom": "25px"},
                            ),
                            html.Div(
                                [

                                    html.Div(
                                        [
                                            html.Div([

                                                #     du.Upload(
                                                #     id='dash-uploader',
                                                #     max_file_size=1800,  # 1800 Mb
                                                #     filetypes=['csv', 'xlsx'],
                                                #     upload_id=uuid.uuid1(),  # Unique session id
                                                #     # Allow multiple files to be uploaded
                                                #     max_files=1
                                                # ),
                                                html.Div(id="callback-output"),

                                                dcc.Upload(
                                                    id='upload-data',
                                                    children=html.Div([
                                                        'Drag and Drop or ',
                                                        html.A('Select Files'),
                                                        html.Ul(id="file-list"),
                                                        # dcc.ConfirmDialog(
                                                        #     id='confirm-danger',
                                                        #     message='Danger danger! Are you sure you want to continue?',
                                                        # ),
                                                    ]),
                                                    style={
                                                        'width': '95%',
                                                        'height': '60px',
                                                        'lineHeight': '60px',
                                                        'borderWidth': '1px',
                                                        'borderStyle': 'dashed',
                                                        'borderRadius': '5px',
                                                        'textAlign': 'center',
                                                        'margin': '10px'
                                                    },
                                                    # Allow multiple files to be uploaded
                                                    multiple=True
                                                ),
                                                html.Div(id='output-data-upload')

                                            ]),

                                            html.P(
                                                "Select the predictive model:",
                                                className="control_label",
                                            ),
                                            dcc.Dropdown(
                                                id="select-model-ddown",
                                                options=ml_models,#[{"label": "RF based method", "value":"RF based method"},{"label":"Convolutional based" , "value":"CNN"},{"label": "LSTM based", "value":"LSTM"} ],
                                                ## multi=True,
                                                value=ml_models[0]['value'],
                                                className="dcc_control",
                                            ),                                          
                                            html.P("Choose y-axis:", className="control_label"),
                                            dcc.Dropdown(
                                                id="xaxis-column",
                                                #options=[{'label': k, 'value': k} for k in df_current.select_dtypes(include=np.number).columns.tolist()] + ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_current)]),# get_options_dic(),
                                                # multi=True,
                                                #value=datetime_col[0],
                                                className="dcc_control",
                                            ),
                                            dcc.Dropdown(
                                                id="yaxis-column",
                                                #options=[{'label': k, 'value': k} for k in df_current.select_dtypes(include=np.number).columns.tolist()]+ ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_current)]),
                                                # multi=True,
                                                #value=numerical_col[0],
                                                className="dcc_control",
                                            ),
                                        ],
                                        className="pretty_container four columns",
                                        id="cross-filter-options",
                                    ),
                                    html.Div(
                                        [
                                            
                                            html.Div(
                                                [
                                                    dcc.Graph(id="graph_tab1_fig1")
                                                ],
                                                id="graph_tab1_fig1_Container",
                                                className="pretty_container",
                                            ),
                                        ],
                                        id="right-column",
                                        className="eight columns",
                                    ),
                                ],
                                className="row flex-display",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        children=[
                                           

                                            dcc.Dropdown(
                                                id="y-axix-ddown",
                                                #options= [{'label': k, 'value': k} for k in colnames_kept], #get_options_dic(),
                                                multi=True,
                                                #value=[k for k in colnames_kept],
                                                className="dcc_control",
                                            ),
                                            dcc.Graph(id="graph_tab1_fig2")
                                        ],
                                        className="pretty_container twelve columns",
                                    ),
                                ],
                                className="row flex-display",
                            ),

                        ]),



                dcc.Tab(label='Analysis',
                        value='tab-2',
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        children=[

                            dcc.Store(id="tab2-data"),
                            dcc.Store(id = "tab2-data-preprocessed"),
                            dcc.Store(id = "tab2-data-preprocessed-predicted"),


                            # empty Div to trigger javascript file for graph resizing
                            html.Div(id="tab2-output-clientside"),
                            
                            html.Div(
                                [

                                    html.Div(
                                        [
                                            # dcc.Checklist(
                                            #     options=[
                                            #         {'label': 'Preprocessing', 'value': 'Preprocessing'}
                                            #     ],
                                            #     id='check-preprocess',
                                            #     value=[]
                                            # ),

                                             html.P(
                                                "Select data range:",
                                                className="control_label",
                                            ),
                                                html.Div(
                                                    html.P("From:", className="control_label"),
                                                        #style={'display': 'inline-block', 'padding': '10px'},
                                                ),
                                                html.Div(
                                                     dcc.DatePickerSingle(
                                                    id='datepicker-from-tab2',
                                                    #min_date_allowed=date(1995, 8, 5),
                                                    #max_date_allowed=date(2017, 9, 19),
                                                    #initial_visible_month=date(2017, 8, 5),
                                                    #date=date(2017, 8, 25)
                                                    ), 
                                                    style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'middle', 'padding: ': '10px', 
                                                    #"float": 'left',
                                                    },
                                                ),
                                                html.Div(id='output-data-none'),
                                                html.Div(
                                                    html.P("To:", className="control_label"),
                                                        #style={'display': 'inline-block', 'padding': '10px'},
                                                ),
                                                html.Div(
                                                     dcc.DatePickerSingle(
                                                    id='datepicker-to-tab2',
                                                    #min_date_allowed=date(1995, 8, 5),
                                                    #max_date_allowed=date(2017, 9, 19),
                                                    #initial_visible_month=date(2017, 8, 5),
                                                    #date=date(2017, 8, 25)
                                                    ), 
                                                    style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'middle', 'padding: ': '10px', 
                                                    #"float": 'left',
                                                    },
                                                ),
                                            
                                        ],
                                        className="pretty_container four columns",
                                        id="tab2-cross-filter-options",
                                    ),
                                    html.Div(
                                        [
                                            
                                            html.Div(
                                                [
                                                                                                       
                                                ],
                                                id="tab2-report1",
                                                className="pretty_container",
                                            ),
                                        ],
                                        id="tab2-right-column",
                                        className="eight columns",
                                    ),
                                ],
                                className="row flex-display",
                            ),
                            
                                           
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            # input box for corrosion    
                                            html.P("Max Corrosion Risk      ", className="control_label"),
                                            dcc.Input(
                                                id="tab2-input-max-corrosion-risk",
                                                type="number",
                                                #placeholder="Max Corrosion Risk"
                                            ),

                                            dcc.Dropdown(
                                                id="tab2y-axix-ddown",
                                                #options= [{'label': k, 'value': k} for k in df_current.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                #value=get_columns_list(df_current),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-preprocessed-graph1"

                                                           )],
                                                id="tab2-row-preprocessing",
                                                className="pretty_container",
                                            ),
                                            # html.Div(
                                            # [
                                                
                                                html.Div(
                                                    [
                                                                                                        
                                                    ],
                                                    id="tab2-fig1-report",
                                                    className="pretty_container",
                                                ),
                                            #],
                                            # id="tab2-section-input-report",
                                            # className="eight columns",
                                            # ),
                                        ],
                                        className="pretty_container twelve columns",
                                    ),
                                ],
                                className="row flex-display",
                            ),
                             
                            # html.Div(
                            #             [
                            #                 html.Div(
                            #                     children=[
                            #                         # html.P("Max Corrosion Risk      ", className="control_label"),
                            #                         # dcc.Input(
                            #                         #     id="tab2-input-max-corrosion-risk",
                            #                         #     type="number",
                            #                         #     #placeholder="Max Corrosion Risk"
                            #                         # ),
                            #                         # html.P("Accumulated Corrosion Risk", className="control_label"),
                            #                         # dcc.Input(
                            #                         #     id="tab2-input-accumulated-corrosion-risk",
                            #                         #     type="number",
                            #                         #     #placeholder="Accumulated Corrosion Risk"
                            #                         # ),
                            #                         # html.P("Max Average over time period", className="control_label"),
                            #                         # dcc.Input(
                            #                         #     id="tab2-input-max-average-corrosion-risk",
                            #                         #     type="number",
                            #                         #     #placeholder="Max Average over time period"
                            #                         # ),
                            #                         # html.P("Time Window for Average      ", className="control_label"),
                            #                         # dcc.Dropdown(
                            #                         #     id="select-period",
                            #                         #     options=[{"label": str(i), "value":str(i)} for i in range(10, 101)],
                            #                         #     ## multi=True,
                            #                         #     value='10',
                            #                         #     className="dcc_control",
                            #                         # ),                                          

                            #                     ],
                            #                         className="pretty_container four columns",
                            #                         id="tab2-inputs",
                            #                 ),
                            #                 html.Div(
                            #                             [
                                                            
                            #                                 html.Div(
                            #                                     [
                                                                                                                    
                            #                                     ],
                            #                                     id="tab2-fig1-report",
                            #                                     className="pretty_container",
                            #                                 ),
                            #                             ],
                            #                             id="tab2-section-input-report",
                            #                             className="eight columns",
                            #                          ),
                                           
                            #             ],
                            #            className="row flex-display",
                            #         ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P("Calculated Corrosion Risk      ", className="control_label"),
                                            html.P("Max Average over time period", className="control_label"),
                                            dcc.Input(
                                                id="tab2-input-max-average-corrosion-risk",
                                                type="number",
                                                #placeholder="Max Average over time period"
                                            ),
                                            html.P("Time Window for Average      ", className="control_label"),
                                            dcc.Dropdown(
                                                id="select-period",
                                                options=[{"label": str(i), "value":str(i)} for i in range(10, 101)],
                                                ## multi=True,
                                                value='10',
                                                className="dcc_control",
                                            ),                                          

                                            dcc.Dropdown(
                                                id="tab2-section3-y-axix-ddown",
                                                #options= [{'label': k, 'value': k} for k in df_current.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                #value=get_columns_list(df_current),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-preprocessed-graph2"

                                                            )],
                                                id="tab2-section3-row-preprocessing",
                                                className="pretty_container",
                                            ),
                                            # html.Div(
                                            # [
                                                
                                                html.Div(
                                                    [
                                                                                                        
                                                    ],
                                                    id="tab2-fig2-report",
                                                    className="pretty_container",
                                                ),
                                            #],
                                            # id="tab2-section-input-report",
                                            # className="eight columns",
                                            # ),
                                        ],
                                        className="pretty_container twelve columns",
                                    ),
                                ],
                                className="row flex-display",
                            ),
                            
                            html.Div(
                                [
                                    
                                    html.Div(
                                        [
                                            html.P("Accumulated Corrosion Risk", className="control_label"),
                                            dcc.Input(
                                                id="tab2-input-accumulated-corrosion-risk",
                                                type="number",
                                                #placeholder="Accumulated Corrosion Risk"
                                            ),


                                            dcc.Dropdown(
                                                id="tab2-section4-y-axix-ddown",
                                                #options= [{'label': k, 'value': k} for k in df_current.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                #value=get_columns_list(df_current),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-preprocessed-graph3"

                                                            )],
                                                id="tab2-section4-row-preprocessing",
                                                className="pretty_container",
                                            ),
                                            # html.Div(
                                            # [
                                                
                                                html.Div(
                                                    [
                                                                                                        
                                                    ],
                                                    id="tab2-fig3-report",
                                                    className="pretty_container",
                                                ),
                                            #],
                                            # id="tab2-section-input-report",
                                            # className="eight columns",
                                            # ),
                                        ],
                                        className="pretty_container twelve columns",
                                    ),
                                ],
                                className="row flex-display",
                            ),
                            #  html.Div(
                            #         [
                            #             html.Div(
                            #                 children=[ 
                            #                     html.Div(
                            #                         html.P("Export results as     ", className="control_label"),
                            #                         style={'display': 'inline-block', 'padding': '10px'},
                            #                     ),
                            #                     html.Div(
                            #                         dcc.Dropdown(
                            #                             id="tab2-export-ddown",
                            #                             options= [
                            #                                         {'label': 'html', 'value': 'html'},
                            #                                         {'label': 'pdf', 'value': 'pdf'}
                            #                                     ], #get_options_dic(),
                            #                             multi=False,
                            #                             value="pdf",
                            #                             className="dcc_control"
                            #                         ), 
                            #                         style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                            #                     ),
                            #                     html.Button('Print', id='print', n_clicks=0),
                            #                 ],
                            #                     #
                            #                     className="pretty_container twelve columns",
                            #                     id="tab2-section4-inputs",
                            #             )
                            #         ],
                            #         className="pretty_container twelve columns",
                            #     ),
 

                        ]),
                       
                dcc.Tab(label='About SOCORRO',
                        value='tab-3',
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Comming soon",
                                        style={"margin-bottom": "0px"},
                                    ),
                                    html.H5(
                                        "functionalities to be added", style={"margin-top": "0px"}
                                    ),
                                ]
                            )

                        ]),
            ])  # end of tabs

    ],  # end of main body

    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


#READ THE XLSS FILE, DO PREPROCESSING, IMPUTING, OUTLIER REMOVAL AND SCALING OF MEAS_RATE


@app.callback(Output('df',  'data'),
              Output('callback-output', 'children'),
              Output("file-list", "children"),
               #Output('xaxis-column', 'options'), Output('yaxis-column', 'options'),
               # Output('xaxis-column', 'value'), Output('yaxis-column', 'value'),
              Output('datepicker-from-tab2', 'min_date_allowed'), Output('datepicker-from-tab2', 'max_date_allowed'),Output('datepicker-from-tab2', 'date'),
              Output('datepicker-to-tab2', 'min_date_allowed'), Output('datepicker-to-tab2', 'max_date_allowed'),Output('datepicker-to-tab2', 'date'),

 
            [Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified')
            ])

def update_output(uploaded_file_contents, uploaded_filenames, list_of_dates):
    df_current = None
    msg =  ""# [html.Li("Nothing!")]
    uploaded_file_path = None
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            df_current, msg = upload.parse_data1(data, name)
            msg = ""
            break

    if uploaded_filenames is  None or uploaded_file_contents is  None:
        
        df_current = pd.read_excel( default_data_file, header=0)#, parse_dates=[0], index_col=0
        msg = "" #"Loaded with default data file"



    df_current = strip_df_colnames(df_current)
    df_current, empties = drop_empty_or_sparse_cols(df_current, ['Date', 'Time'])
    df_current = create_datatime_col(df_current, ['Date', 'Time'])    
    df_current = strip_no_numeric_cols(df_current)
    df_current = set_df_col_as_index(df_current, 'DateTime')   
    #from indexed df select onlu numerical columns (DateTime already used in index)
    df_current = df_current.select_dtypes(include=np.number)

    
 
    #print(df_current.dtypes)
    #print(df_current.columns)
    #mdl_tag = check_model_type(df_current.columns)
    #print(df_current.head)
    print(df_current.head())
 
    df_current = keep_columns(df_current, colnames_kept)
    # more generally, this line would be
    # json.dumps(cleaned_df)
    return [df_current.to_json(date_format='iso', orient='split'), html.Ul( uploaded_file_path ), msg,
            df_current.index.min(), df_current.index.max(), df_current.index.min(),
            df_current.index.min(), df_current.index.max(), df_current.index.max(),
                ]
               



    #return [None, html.Ul( "Problem in loading data file."),   msg]




@app.callback(

    Output('xaxis-column', 'options'),  Output('yaxis-column', 'options'),  Output('y-axix-ddown', 'options'),
    Output('xaxis-column', 'value'),    Output('yaxis-column', 'value'),    Output('y-axix-ddown', 'value'), 


    [
        Input('df',  'data')
    ]  # ,
    # [State("lock_selector", "value"), State("graph_tab1_fig2", "relayoutData")],
)
def update_dropdowns(
        jsonified_cleaned_data
):
    print("uuid.uuid1(): ", uuid.uuid1())

    if jsonified_cleaned_data is not None:
        df = pd.read_json(jsonified_cleaned_data, orient='split')

        return  [   [{'label': k, 'value': k} for k in ['index'] + keept_cols(df, colnames_kept) ],
                    [{'label': k, 'value': k} for k in ['index'] + keept_cols(df, colnames_kept) ],
                    [{'label': k, 'value': k} for k in ['index'] + keept_cols(df, colnames_kept) ],
                    'index', colnames_kept[0], [k for k in keept_cols(df, colnames_kept)]
                ]
    return [ [ ],  [], [], None, None , None  ]


# SET THE MODEL
@app.callback(
    Output(component_id='model-name', component_property='data'),
    [
        Input("select-model-ddown", "value"),
    ],
  
)
def set_model_name(
        y
):  
    #print("model is", y)
    return {"model": y}


# UPDATE FIRST PLOT BASED ON NAY CHANGE IN DROP BOX OPTIONS

# Selectors -> main graph
@app.callback(
    Output(component_id='graph_tab1_fig1', component_property='figure'),
    [
        Input("xaxis-column", "value"),
        Input("yaxis-column", "value"),
        Input('df', 'data')
    ],
  
)
def make_count_figure(
        x_col, y_col, jsonified_cleaned_data
):
    if jsonified_cleaned_data is not None:
        # more generally, this line would be
        # json.loads(jsonified_cleaned_data)
        df = pd.read_json(jsonified_cleaned_data, orient='split')

        layout_graph_tab1_fig1 = copy.deepcopy(layout)

        if x_col == 'index':
            xx = df.index
        else:
            xx = df[x_col]
        if y_col == 'index':
            yy = df.index
        else: 
            yy = df[y_col]

        data = [
            dict(
                type="Scattergl",
                #name="Gas Produced (mcf)",
                x=xx,
                y=yy,
                #line=dict(shape="spline", smoothing=2, width=1),  # , color="#fac1b7"

                opacity=0.5,
                hoverinfo="skip",
                # marker=dict(color=colors),

                mode="markers",  # lines+
                # line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                marker=dict(symbol="circle-open"),
                marker_size=2,

            )]
        layout_graph_tab1_fig1["title"] = y_col
        layout_graph_tab1_fig1["dragmode"] = "select"
        layout_graph_tab1_fig1["showlegend"] = False
        layout_graph_tab1_fig1["autosize"] = True
        figure = dict(data=data, layout=layout_graph_tab1_fig1)


        return figure
    return {'layout': {'title': 'No input specified, please fill in an input.'}}




@app.callback(
    Output(component_id='graph_tab1_fig2', component_property='figure'),
    [
        Input("y-axix-ddown", "value"),
        Input('df', 'data')
    ]  # ,
    # [State("lock_selector", "value"), State("graph_tab1_fig2", "relayoutData")],
)
def make_figure(
        y_col, jsonified_cleaned_data # , selector, graph_tab1_fig2_layout
):
    if jsonified_cleaned_data is not None:
        # more generally, this line would be
        # json.loads(jsonified_cleaned_data)
        df = pd.read_json(jsonified_cleaned_data, orient='split')

        layout_graph_tab1_fig1 = copy.deepcopy(layout)

        n_rows = len(y_col)
        data = []
        for i in range(n_rows):
            data.append(df[y_col[i]].tolist())
        labels = y_col

    

        plotly_data = []
        plotly_layout = plotly.graph_objs.Layout()
        colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'gray']
        # your layout goes here
        layout_kwargs = {
                        'title': 'Sensor Data:',
                        'xaxis': {'domain': [0, 0.8]}
                        }
        for i, d in enumerate(data):
            # we define our layout keys by string concatenation
            # * (i > 0) is just to get rid of the if i > 0 statement
            axis_name = 'yaxis' + str(i + 1) * (i > 0)
            yaxis = 'y' + str(i + 1) * (i > 0)
            plotly_data.append(plotly.graph_objs.Scatter(x= df.index, y=d, 
                                                        name=labels[i]))
          
            layout_kwargs[axis_name] = {
                                        'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1],
                                        'position': 1 - i * 0.04,
                                    
                                        'title' : labels[i],
                                        'titlefont' : dict(
                                          #color=colors[i],# "#9467bd"
                                            color=px.colors.qualitative.D3[i]
                                        ),
                                    
                                        'tickfont' : dict(
                                            #color=colors[i],# "#9467bd"
                                            color=px.colors.qualitative.D3[i]
                                        ),
                                        'anchor' : "free",
                                        #'overlaying' : "y",
                                        'side' : "right",
                                        "showline": True
                                    
                                    }

            plotly_data[i]['yaxis'] = yaxis
            if i > 0:
                layout_kwargs[axis_name]['overlaying'] = 'y'

            
            
        fig = go.Figure(data=plotly_data, layout=plotly.graph_objs.Layout(**layout_kwargs))
        
        fig.layout.plot_bgcolor = '#fff'
        fig.layout.paper_bgcolor = '#fff'

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        return fig
    return {'layout': {'title': 'No input specified, please fill in an input.'}}



#########################################################################################################################

###################################################### TAB 2

#########################################################################################################################

#----------------------------------------------------------

# READ THE XLSS FILE, DO PREPROCESSING, IMPUTING, OUTLIER REMOVAL AND SCALING OF MEAS_RATE
#----------------------------------------------------------

@app.callback(Output('tab2-data',  'data'),
            #    Output('datepicker-from-tab2', 'min_date_allowed'), Output('datepicker-from-tab2', 'max_date_allowed'),Output('datepicker-from-tab2', 'date'),
            #    Output('datepicker-to-tab2', 'min_date_allowed'), Output('datepicker-to-tab2', 'max_date_allowed'),Output('datepicker-to-tab2', 'date'),
               Output('tab2-report1', 'children'),
               Output('tab2y-axix-ddown', 'options'),Output('tab2y-axix-ddown', 'value'),
               Output('tab2-section3-y-axix-ddown', 'options'),Output('tab2-section3-y-axix-ddown', 'value'),
               Output('tab2-section4-y-axix-ddown', 'options'),Output('tab2-section4-y-axix-ddown', 'value'),
               
               Output('tab2-data-preprocessed', 'data'),
                Output("tab2-input-max-corrosion-risk", "value"),
                Output("tab2-input-accumulated-corrosion-risk", "value"),
                Output("tab2-input-max-average-corrosion-risk", "value"),
                
            [Input('df',  'data'),
            #Input('check-preprocess', 'value'),
            Input("datepicker-from-tab2", "date"),
            Input("datepicker-to-tab2", "date"),

            ])

def preprocess_df(jsonified_cleaned_data,  date_from, date_to):#checkedProcess,
     df_current = None
    
     if jsonified_cleaned_data is not None: #and  'Preprocessing' in checkedProcess:
        # more generally, this line would be
        # json.loads(jsonified_cleaned_data)
        df_current = pd.read_json(jsonified_cleaned_data, orient='split')

        # filter by date
 #, '%d/%m/%y %H:%M:%S'  2020-11-12T17:00:00+00:00
        #print("test", date_from, date_to)
        if date_from is not None and date_to is not None:
            #print(date_from, date_to)
            date_from = date_from.split('T')[0]
            #print(date_from)
            date_to = date_to.split('T')[0]
            print("date time picker modified: ", date_to, "\t", date_from )
            #print(date_to)
            ##print("=>", type(datetime.strptime(date_from, "%Y-%m-%d")))
            #print("=>", type(datetime.strptime(date_from, "%Y-%m-%d")+1))

        
            df_current = df_current.loc[date_from: date_to]  

            

        #df_current = None
        #df_current = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[1]

        #remove the outliers
        #df_current['savgol'] = scp.savgol_filter(x = df_current['Meas_Rate'], window_length=21, polyorder = 1, deriv=0)#, delta=1.0, axis=- 1, mode='interp', cval=0.0
        #df_current['savgol_deriv'] = scp.savgol_filter(x = df_current['Meas_Rate'], window_length=21, polyorder = 1, deriv=1)#, delta=1.0, axis=- 1, mode='interp', cval=0.0
        #df_current['savgol_stable'] = df_current['savgol' ][abs(df_current['savgol_deriv']) < 0.00063]
        #df_current['savgol_stable'] = df_current['savgol' ]

        #  Scale Meas_Rate
        #features = df_current['savgol_stable'] 
        # Use scaler of choice; here Standard scaler is used
        #scaler = StandardScaler().fit(features.values.reshape(-1,1))
        #features = scaler.transform(features.values.reshape(-1,1))
        #df_current['Meas_Rate'] = features

        #print("df_current updated")
        #print("=============")
        #print(df_current.info(null_counts=True))
        #nan_dfa_loop1 = pd.DataFrame([(col, n, n/df_current.shape[0]) for col in df_current.columns for n in (df_current[col].isna().sum(),) if n], columns=df_current.columns)
        
        #count_nan = df_current.isna().sum()/df_current.shape[0]*100        
        count_nan = df_current.isnull().mean() * 100
        
         
        

        nbObs = 0 #df_current['savgol_stable'].shape[0]
        nbAfterSavGol = 0 #nbObs -  df_current['savgol_stable'].isnull().sum()
        percentage = 0 #(float (nbAfterSavGol) / float (nbObs))*100
        #percentage = "{:.2f}".format(percentage)

        return_divs = [html.P("Percentage of missing values:\n {missing_percentages}", style={'color': 'red'} )]

        print("----------test:\n")
        K=3
        conseq_nulls = cols_with_k_consecutive_nans(df_current, K)
        #print(conseq_nulls)
        mes_pieces = {}
        for col in df_current.columns:
            if(conseq_nulls[col]):
                mes_pieces[col]="and contain consecutive nulls of the length "  + str(K) 


        return_divs.append (	html.P(f"total number of observations: {df_current.shape[0]}"))
        cols = df_current.columns
        df_current = df_current.dropna(subset=cols)

        print(df_current.head())

        for val, col in zip(count_nan, df_current.columns):
            print(col, col, "\t", len(df_current[col].values), "\t", len([i for i in range(df_current.shape[0])]))
            
            reg = LinearRegression().fit(np.array([i for i in range(df_current.shape[0])]).reshape(-1,1), list(df_current[col].values))
            slope = (reg.coef_[0])
            #return_divs.append(	html.P())
            msg = col  + " contains " + str("{:.2f}".format(val)) + " percent missing values "

            if(mes_pieces.get(col)):
                msg = msg + mes_pieces[col]

            msg = msg + ". Linear regression slope for is : " +  str("{:.3f}".format(slope))
            print(msg)
            return_divs.append(	html.P(msg ))

        return_divs.append (html.P("Consecutive missing values (at least 3)\n", style={'color': 'red'}))
        missing_seqs = [ df_current[a].isnull().astype(int).groupby(df_current[a].notnull().astype(int).cumsum()).sum() for a in df_current.columns]



        # has_missing = False
        # for a, col  in zip(missing_seqs, df_current.columns):
        #     if len(list(a[a > 2].index)) > 0:
        #         has_missing = True
        #         return_divs.append(	html.P(col  + " contains missing sequences at " + str(list(a[a > 2].index)) ))
        

       
        return_divs.append(	html.P("The computations continue by elliminating the rows containing missing values.", style={'color': 'red'}))
        #return_divs.append(	html.P("Existing trends: ", style={'color': 'red'}))

        # if has_missing:
        #     df_current.dropna(inplace=True, axis='rows')

        
        #print(count_nan)


        # for  col  in df_current.columns:
        #     reg = LinearRegression().fit(np.array([i for i in range(df_current.shape[0])]).reshape(-1,1), list(df_current[col].values))
        #     slope = (reg.coef_[0])
        #     return_divs.append(	html.P(" Linear regression slope for " + col  + " is : " +  str("{:.3f}".format(slope)) ))

        #return_divs.append(	html.P("End of reporting.", style={'color': 'red'}))

        
        #print("=============")


        
        return [
                df_current.to_json(date_format='iso', orient='split'), 
                # df_current.index.min(), df_current.index.max(), df_current.index.min(),
                # df_current.index.min(), df_current.index.max(), df_current.index.max(),
                return_divs,
                [{'label': k, 'value': k} for k in  colnames_kept ], [ k for k in  keept_cols(df_current, colnames_kept) ],
                [{'label': k, 'value': k} for k in  colnames_kept ], [ k for k in  keept_cols(df_current, colnames_kept) ],
                [{'label': k, 'value': k} for k in  colnames_kept ], [ k for k in  keept_cols(df_current, colnames_kept) ],
                
                df_current.to_json(date_format='iso', orient='split'), 0.6, 100,0.6
        ]
                #
                #df_current.df.index.min(), df_current.df.index.max(), df_current.df.index.max()]
               
     #return [None , None, None, None, None, None, None]
     return [None , None , None, None, None , None, None, [], [], [], [], [], [], [], None, 0.6, 100,0.6]



#----------------------------------------------------------
# FIRST PLOT IN TAB2
#----------------------------------------------------------

# @app.callback(
#     Output(component_id='tab2-preprocessed_graph1', component_property='figure'),
#     Output("tab2-input-max-corrosion-risk", "value"),
#     Output("tab2-input-accumulated-corrosion-risk", "value"),
#     Output("tab2-input-max-average-corrosion-risk", "value"),
#     [
#         Input("tab2y-axix-ddown", "value"),
#         Input("tab2-data-preprocessed", "data"),

 
#     ]  
# )
# def make_figure_tab2_fig1(
#         y_col, jsonified_cleaned_data # , selector, graph_tab1_fig2_layout
# ):

#     if jsonified_cleaned_data is not None:
#         # more generally, this line would be
#         # json.loads(jsonified_cleaned_data)
#         df = pd.read_json(jsonified_cleaned_data, orient='split')

#         layout_graph_tab1_fig1 = copy.deepcopy(layout)

#         n_rows = len(y_col)
#         data = []
#         for i in range(n_rows):
#             data.append(df[y_col[i]].tolist())
#         labels = y_col

#         plotly_data = []
#         plotly_layout = plotly.graph_objs.Layout()
#         # your layout goes here
#         layout_kwargs = {
#                         'title': 'Sensor Data:',
#                         'xaxis': {'domain': [0, 0.8]}
#                         }
#         cntr = 0
#         for i, d in enumerate(data):

#             # we define our layout keys by string concatenation
#             # * (i > 0) is just to get rid of the if i > 0 statement
#             axis_name = 'yaxis' + str(i + 1) * (i > 0)
#             yaxis = 'y' + str(i + 1) * (i > 0)
#             plotly_data.append(plotly.graph_objs.Scatter(x= df.index, y=d, name=labels[i], mode='markers', #opacity=0.5, 
#                     marker={'size': 2 , 'symbol': 'diamond'},

#                      ))
            
#             layout_kwargs[axis_name] = {
#                                             'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] ,
#                                             'position': 1 - i * 0.04,
                                        
#                                             'title' : labels[i],
#                                             'titlefont' : dict(
#                                             #color=colors[i],# "#9467bd"
#                                                 color=px.colors.qualitative.D3[i] 
#                                             ),
                                        
#                                             'tickfont' : dict(
#                                                 #color=colors[i],# "#9467bd"
#                                                 color=px.colors.qualitative.D3[i]
#                                             ),
#                                             'anchor' : "free",
#                                             #'overlaying' : "y",
#                                             'side' : "right",
#                                             "showline": True,
#                                             'showgrid': False,
                                            
#                                         }

#             plotly_data[i]['yaxis'] = yaxis
#             if i > 0:
#                 layout_kwargs[axis_name]['overlaying'] = 'y'

            
      
            

         
#         fig = go.Figure(data=plotly_data, layout=plotly.graph_objs.Layout(**layout_kwargs))
#         fig.layout.plot_bgcolor = '#fff'
#         fig.layout.paper_bgcolor = '#fff'

#         fig.update_layout(legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ))


#         return [fig, 0.6, 100, 100]
#     return [{'layout': {'title': 'No input specified, please fill in an input.'}}, 0.6,100,100]

#----------------------------------------------------------
# PLOT  CALCULATED CORROSION RISK
#----------------------------------------------------------
@app.callback(
    Output(component_id = 'tab2-preprocessed-graph1', component_property='figure'),
    Output(component_id = 'tab2-preprocessed-graph2', component_property='figure'),
    Output(component_id = 'tab2-preprocessed-graph3', component_property='figure'),
    Output(component_id = 'tab2-data-preprocessed-predicted',   component_property='data'),
    Output('tab2-fig1-report', 'children'),
    Output('tab2-fig2-report', 'children'),
    Output('tab2-fig3-report', 'children'),
    [
        Input("tab2-section3-y-axix-ddown", "value"),
        Input("tab2-input-max-corrosion-risk", "value"),
        Input("tab2-input-accumulated-corrosion-risk", "value"),
        Input("tab2-input-max-average-corrosion-risk", "value"),
        Input("tab2-data-preprocessed", "data"),
        Input("model-name",'data'),
        Input('select-period', 'value')
        
    ]  
)
def make_figure_tab2(
        y_col, max_corrosion_risk, accumulated_corrosion_risk, 
        max_average_corrosion_risk, jsonified_cleaned_data, ml_model_name, _time_window
):
    if jsonified_cleaned_data is not None:
        # more generally, this line would be
        # json.loads(jsonified_cleaned_data)
        df = pd.read_json(jsonified_cleaned_data, orient='split')

        model_loaded = None
        if ml_model_name['model'] == 'Sea Water (Lab)':
            #model_loaded = pickle.load(open('socorro_model_11.pkl', "rb"))
            model_loaded = joblib.load('socorro_model_1.sav')
            df_tmp = df[['Temperature', 'pH', 'ORP', 'Conductivity']]
            df_tmp.columns = ['f0', 'f1', 'f2', 'f3' ]
            model_loaded.predict(df_tmp)
        elif ml_model_name['model'] == 'Sea Water (Demo)':
            model_loaded = joblib.load('SH1NH2_best.sav')
            df_tmp = df[['Temperature', 'pH', 'dissolved oxygen (mg/l)', 'conductivity at 25°c', 'orp redox potential']]
            
        else: 
            print("module is not available... program exits")




        df['Corrosion Risk' ] = model_loaded.predict(df_tmp)
        #print( df['Corrosion Risk' ])
        df['Critical Corrosion Risk' ] = df['Corrosion Risk'][df['Corrosion Risk'] > max_corrosion_risk]#   > threshold).any(1)
        print("max_corrosion_risk", max_corrosion_risk)
        #df['Average Over Time Period' ] = df['Corrosion Risk' ].ewm(span=max_average_corrosion_risk, adjust=False).mean()
        df['Average Over Time Period' ] = df['Corrosion Risk' ].rolling(window=int(_time_window)).mean()
        df['Critical Average Over Time Period' ] = df['Average Over Time Period' ][df['Average Over Time Period' ] > max_average_corrosion_risk ]

        df['Acc. Corrosion Risk'] = df['Corrosion Risk' ].cumsum()
        #df['Acc. Corrosion Risk'] = df['Acc. Corrosion Risk'][df['Acc. Corrosion Risk'] >= accumulated_corrosion_risk]


        return_divs1 = None
        return_divs2 = None
        return_divs3 = None



        number_of_exceedings = df['Critical Corrosion Risk' ].isnull().sum()
        exceeding_percentage = float(number_of_exceedings) / float(df['Critical Corrosion Risk' ].shape[0])
        return_divs1 = [html.P(f"Number of observations above threshold: {number_of_exceedings} i.e. {(1- exceeding_percentage)*100:.2f} percent")]

        if( exceeding_percentage*100 > 30):
            return_divs1.append(html.P("Perhap try incresing the threshold. ", style={'color': 'red'}))
        if (df['Acc. Corrosion Risk'][df['Acc. Corrosion Risk'] >= accumulated_corrosion_risk].shape[0] > 0 ):
            timestamp = df['Acc. Corrosion Risk'][df['Acc. Corrosion Risk'] >= accumulated_corrosion_risk].index[0]
            return_divs3 = [html.P("The first occurence of Acc. Corrosion Risk: " + str(timestamp), style={'color': 'red'})]
        
        return_divs2 = [html.P("Average.", style={'color': 'red'})]

        #layout_graph_tab1_fig1 = copy.deepcopy(layout)
        n_rows = len(y_col)
        data = []
        for i in range(n_rows):
            data.append(df[y_col[i]].tolist())
        data.append(df['Corrosion Risk' ].tolist())
        data.append(df['Critical Corrosion Risk' ].tolist())
        data.append(df['Corrosion Risk' ].tolist())

        added_cols = ['Corrosion Risk' , 'Critical Corrosion Risk', 'Average Over Time Period']
        labels = y_col + added_cols


        fig1 = make_figure_tab2_fig1(df, data, y_col, ['Corrosion Risk' , 'Critical Corrosion Risk'])
        fig2 = make_figure_tab2_fig2(df, data, y_col, ['Average Over Time Period', 'Critical Average Over Time Period'])
        fig3 = make_figure_tab2_fig3(df, data, y_col, ['Acc. Corrosion Risk'], accumulated_corrosion_risk)

        return [fig1, fig2, fig3, df.to_json(date_format='iso', orient='split'), return_divs1, return_divs2, return_divs3] #, , 
    return [None, None, None, {'layout': {'title': 'No input specified, please fill in an input.'}},  None, None, None] #, None, None

def make_figure_tab2_fig1(df, data, _labels, added_cols):
        print("labels", _labels)
        print("added cols", added_cols)
        print(len(_labels+added_cols))
        labels = _labels+added_cols
        N = 20
        plotly_data = []
        #plotly_layout = plotly.graph_objs.Layout()
        # your layout goes here
        layout_kwargs = {
                        'title': 'Predicted Corrosion Risk:',
                        'xaxis': {'domain': [0, 0.8]}
                        }
        cntr = 0
        for i in range(len(labels)):
            # we define our layout keys by string concatenation
            # * (i > 0) is just to get rid of the if i > 0 statement
            axis_name = 'yaxis' + str(i + 1) * (i > 0)
            yaxis = 'y' + str(i + 1) * (i > 0)
            plotly_data.append(plotly.graph_objs.Scatter(x= df.index,y=df[labels[i]], name=labels[i], mode='markers' if labels[i] not in added_cols + ['Corrosion Risk'] else None, #opacity=0.8, 

                                                        visible='legendonly'if not (labels[i]== 'Critical Corrosion Risk' or labels[i] == 'Corrosion Risk'or labels[i] == 'Average Over Time Period' ) else None ,
                                                        marker={    'size': 2 if labels[i] != "Critical Corrosion Risk" else 6, 
                                                                    'symbol': 'diamond',  
                                                                    'color' : 'red' if labels[i] == 'Critical Corrosion Risk' else next(line_color),
                                                                    'colorscale':'jet',
                                                                },
                                                                            
                     )
                     
                     )
            
            layout_kwargs[axis_name] = {
                                            'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] if labels[i] != "Critical Corrosion Risk" else  [df["Corrosion Risk"].min()*0.9, df["Corrosion Risk"].max()*1.1],
                                            'position': 1 - i * 0.04,
                                        
                                            'title' : labels[i],
                                            'titlefont' : dict(
                                            #color=colors[i],# "#9467bd"
                                                color=px.colors.qualitative.D3[i] 
                                            ),
                                        
                                            'tickfont' : dict(
                                                #color=colors[i],# "#9467bd"
                                                color=px.colors.qualitative.D3[i]
                                            ),
                                            'anchor' : "free",
                                            #'overlaying' : "y",
                                            'side' : "right",
                                            "showline": True,
                                            'showgrid': False,
                                                                                    }

            plotly_data[i]['yaxis'] = yaxis
            if i > 0:
                layout_kwargs[axis_name]['overlaying'] = 'y'

        fig = go.Figure(data=plotly_data, layout=plotly.graph_objs.Layout(**layout_kwargs))
        fig.layout.plot_bgcolor = '#fff'
        fig.layout.paper_bgcolor = '#fff'
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        return fig

def make_figure_tab2_fig2(df, data, _labels, added_cols):
        print("labels", _labels)
        print("added cols", added_cols)
        print(len(_labels+added_cols))
        labels = _labels+added_cols
        N = 20
        plotly_data = []
        #plotly_layout = plotly.graph_objs.Layout()
        # your layout goes here
        layout_kwargs = {
                        'title': 'Predicted Corrosion Risk:',
                        'xaxis': {'domain': [0, 0.8]}
                        }
        cntr = 0
        for i in range(len(labels)):
            # we define our layout keys by string concatenation
            # * (i > 0) is just to get rid of the if i > 0 statement
            axis_name = 'yaxis' + str(i + 1) * (i > 0)
            yaxis = 'y' + str(i + 1) * (i > 0)
            plotly_data.append(plotly.graph_objs.Scatter(x= df.index,y=df[labels[i]], name=labels[i], mode='markers' if labels[i] not in added_cols + ['Average Over Time Period'] else None, #opacity=0.8, 

                                                        visible='legendonly'if not (labels[i]== 'Critical Average Over Time Period' or labels[i] == 'Average Over Time Period'or labels[i] == 'Average Over Time Period' ) else None ,
                                                        marker={    'size': 2 if labels[i] != "Critical Average Over Time Period" else 6, 
                                                                    'symbol': 'diamond',  
                                                                    'color' : 'red' if labels[i] == 'Critical Average Over Time Period' else next(line_color),
                                                                    'colorscale':'jet',
                                                                },
                                                                            
                     )
                     
                     )
            
            layout_kwargs[axis_name] = {
                                            'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] if labels[i] != "Critical Average Over Time Period" else  [df["Average Over Time Period"].min()*0.9, df["Average Over Time Period"].max()*1.1],
                                            'position': 1 - i * 0.04,
                                        
                                            'title' : labels[i],
                                            'titlefont' : dict(
                                            #color=colors[i],# "#9467bd"
                                                color=px.colors.qualitative.D3[i] 
                                            ),
                                        
                                            'tickfont' : dict(
                                                #color=colors[i],# "#9467bd"
                                                color=px.colors.qualitative.D3[i]
                                            ),
                                            'anchor' : "free",
                                            #'overlaying' : "y",
                                            'side' : "right",
                                            "showline": True,
                                            'showgrid': False,
                                                                                    }

            plotly_data[i]['yaxis'] = yaxis
            if i > 0:
                layout_kwargs[axis_name]['overlaying'] = 'y'

        fig = go.Figure(data=plotly_data, layout=plotly.graph_objs.Layout(**layout_kwargs))
        fig.layout.plot_bgcolor = '#fff'
        fig.layout.paper_bgcolor = '#fff'
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        return fig

def make_figure_tab2_fig3(df, data, _labels, added_cols, _accumulated_corrosion_risk):
    #layout_graph_tab1_fig1 = copy.deepcopy(layout)
    labels = _labels+added_cols
    plotly_data = []
    #plotly_layout = plotly.graph_objs.Layout()
    # your layout goes here
    layout_kwargs = {
                    'title': 'Accumulated Corrosion Risk:',
                    'xaxis': {'domain': [0, 0.8]}
                    }
    cntr = 0
   


    cntr = 0
    for i in range(len(labels)):

        # we define our layout keys by string concatenation
        # * (i > 0) is just to get rid of the if i > 0 statement
        axis_name = 'yaxis' + str(i + 1) * (i > 0)
        yaxis = 'y' + str(i + 1) * (i > 0)

        if(labels[i] != 'Acc. Corrosion Risk'):
            plotly_data.append(plotly.graph_objs.Scatter(x= df.index, y=df[labels[i]], name=labels[i], mode='markers' if labels[i] not in added_cols else None,  
                                                            fill='tozeroy'if (labels[i]== 'Acc. Corrosion Risk'  ) else None ,
                                                            visible='legendonly'if not (labels[i]== 'Acc. Corrosion Risk'  ) else None ,

                                                            marker={    'size': 2 , 
                                                                        'symbol': 'diamond',  
                                                                        'color' :  next(line_color),
                                                                        'colorscale':'jet',
                                                                        'opacity':  0.9
                                                                    },
                    
                        ))
            layout_kwargs[axis_name] = {
                                            'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] if labels[i] != ("Corrosion Risk"  ) else  [df["Acc. Corrosion Risk"].min()*0.9, df["Acc. Corrosion Risk"].max()*1.1],
                                            'position': 1 - i * 0.04,
                                        
                                            'title' : labels[i],
                                            'titlefont' : dict(
                                                color=px.colors.qualitative.D3[i] 
                                            ),
                                            'tickfont' : dict(
                                                color=px.colors.qualitative.D3[i]
                                            ),
                                            'anchor' : "free",
                                            #'overlaying' : "y",
                                            'side' : "right",
                                            "showline": True,
                                            'showgrid': False,
                                            
                                        }      

        else:
            plotly_data.append(plotly.graph_objs.Scatter(x= df.index,y=df[labels[i]], name=labels[i], mode='markers' if labels[i] not in added_cols else None,  
                                                                fill='tozeroy' ,
                                                                visible= None ,

                                                                marker={    'size': 6, 
                                                                            'symbol': 'diamond',  
                                                                            'color' : 'orange' ,
                                                                            'colorscale':'jet',
                                                                            'opacity': 0.1 
                                                                        },

                        
                            ))

            layout_kwargs[axis_name] = {
                                            'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] if labels[i] != ("Corrosion Risk"  ) else  [df["Acc. Corrosion Risk"].min()*0.9, df["Acc. Corrosion Risk"].max()*1.1],
                                            'position': 1 - i * 0.04,
                                        
                                            'title' : labels[i],
                                            'titlefont' : dict(
                                                color=px.colors.qualitative.D3[i] 
                                            ),
                                            'tickfont' : dict(
                                                color=px.colors.qualitative.D3[i]
                                            ),
                                            'anchor' : "free",
                                            #'overlaying' : "y",
                                            'side' : "right",
                                            "showline": True,
                                            'showgrid': False,
                                            
                                        }  
        plotly_data[i]['yaxis'] = yaxis
        if i > 0:
            layout_kwargs[axis_name]['overlaying'] = 'y'

        print(axis_name)
        
#######################################################
    print(str(i+1) + "X thereshold")
    nbLevels = math.floor(df["Acc. Corrosion Risk"].max() /_accumulated_corrosion_risk)
    for i in range(nbLevels):
        cntr = len(labels) + i
        col = [_accumulated_corrosion_risk*(i+1) for e in range(df.shape[0])]
        plotly_data.append(plotly.graph_objs.Line(x= df.index,y=col, mode='markers', name=str(i+1) + "x thereshold",
                                marker={    'size': 6, 
                                    #'symbol': 'diamond',  
                                    'color' : 'red' ,
                                    'colorscale':'jet',
                                    'opacity': 0.1 
                                },
                        ))
        axis_name = 'yaxis' + str(cntr + 1) 
        print(axis_name)
        layout_kwargs[axis_name] = {
                                        
                                    }  
        plotly_data[cntr]['yaxis'] = yaxis
        layout_kwargs[axis_name]['overlaying'] = 'y'








        

        
    fig = go.Figure(data=plotly_data, layout=plotly.graph_objs.Layout(**layout_kwargs))
    #fig.add_hline(y=100, line_width=3, line_dash="dash", line_color="red")
    

    fig.layout.plot_bgcolor = '#fff'
    fig.layout.paper_bgcolor = '#fff'

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))


    # # ## plot accumulated levels
    # nbLevels = math.floor(df["Acc. Corrosion Risk"].max() /_accumulated_corrosion_risk)
    # for level in range(1, nbLevels+1):
    #     #fig.add_hline(y=_accumulated_corrosion_risk*level, line_width=2)
    #     fig.add_hline(y=100, line_width=2)
    #     #,  line_dash="dot", row=1, col=1, line_color="#000000"
    #     print(_accumulated_corrosion_risk*level)



    return fig

#----------------------------------------------------------
# PRINT LAYOUT
#----------------------------------------------------------







if __name__ == "__main__":
    import os

    debug =  True
    app.run_server(host="0.0.0.0", port=8050, debug=debug)


#if __name__ == "__main__":
#    # print("Data Types: " , df_current.dtypes)
#    # print("Index Name: ", df_current.index.name)
#    app.run_server(debug=True, threaded=False)


