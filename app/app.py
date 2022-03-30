# Import required libraries
import pickle
import copy
import pathlib
import dash
import math
import datetime as dt
from numpy.lib.function_base import percentile
#import dash_snapshot_engine
#import dash_design_kit
# import dask.dataframe as dd
import pandas as pd
# import modin.pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash import dcc
from dash import dash_table
import dash_html_components as html
from plotly.subplots import make_subplots

# from plotly.offline import plot
import plotly.graph_objs as go
from plotly import tools
import plotly

from myutilities import *
import random
# Multi-dropdown options
from color import COUNTIES, STATUSES, WELL_TYPES, COLORS
import upload_file as upload
import plotly.express as px
import numpy as np
import vaex

from parameters import *
from myutilities import *
from utilities import *

import scipy.signal as scp
from scipy import stats

from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go, pandas as pd, plotly.express as px

import pickle
from sklearn.linear_model import LinearRegression

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# sample data in a pandas dataframe
np.random.seed(1)

# define colors as a list
colors = px.colors.qualitative.Plotly
rgba = [hex_rgba(c, transparency=0.2) for c in colors]
colCycle = ['rgba' + str(elem) for elem in rgba]
line_color = next_col(cols=colCycle)




import data_cleaning as dc
import imputers as imp
from datetime import date

datetime = ['datetime64[ns]']
target = "Meas_Rate"

colnames_kept = [    'Temperature', 'Dissolved Oxygen', "Meas_Rate", #'DDate',
                    'pH',  'ORP'#, #################################################  =>'AUX2:Chloride'  # FIX: 'AUX2:Turrbidity'
                ,   'Conductivity'
                ]

ml_models = [   {'label': 'GRF', 'value':'GRF'},
                {'label':'CNN' , 'value': 'CNN'},
                {'label': 'LSTM', 'value':'LSTM'}
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

   
        html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("logo-spg.png"),
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
                dcc.Tab(label='Moniroting',
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
                                                src=app.get_asset_url("socorro.png"),
                                                id="plotly-image",
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
                                                dcc.Upload(
                                                    id='upload-data',
                                                    children=html.Div([
                                                        'Drag and Drop or ',
                                                        html.A('Select Files')
                                                    ]),
                                                    style={
                                                        'width': '100%',
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
                                            dcc.Checklist(
                                                options=[
                                                    {'label': 'Preprocessing', 'value': 'Preprocessing'}
                                                ],
                                                id='check-preprocess',
                                                value=[]
                                            ),
                                             html.P(
                                                "Select data range:",
                                                className="control_label",
                                            ),
                                                html.Div(
                                                    html.P("From:", className="control_label"),
                                                        style={'display': 'inline-block', 'padding': '10px'},
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
                                                        style={'display': 'inline-block', 'padding': '10px'},
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
                                           

                                            dcc.Dropdown(
                                                id="tab2y-axix-ddown",
                                                #options= [{'label': k, 'value': k} for k in df_current.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                #value=get_columns_list(df_current),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-preprocessed_graph1"

                                                           )],
                                                id="tab2-row-preprocessing",
                                                className="pretty_container",
                                            ),
                                        ],
                                        className="pretty_container twelve columns",
                                    ),
                                ],
                                className="row flex-display",
                            ),
                            html.Div(
                                        [
                                            html.Div(
                                                children=[ 
                                                    html.P("Max Corrosion Risk      ", className="control_label"),
                                                    dcc.Input(
                                                        id="tab2-input-max-corrosion-risk",
                                                        type="number",
                                                        #placeholder="Max Corrosion Risk"
                                                    ),
                                                    html.P("Accumulated Corrosion Risk", className="control_label"),
                                                    dcc.Input(
                                                        id="tab2-input-accumulated-corrosion-risk",
                                                        type="number",
                                                        #placeholder="Accumulated Corrosion Risk"
                                                    ),
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

                                                ],
                                                    className="pretty_container four columns",
                                                    id="tab2-inputs",
                                            ),
                                            html.Div(
                                                        [
                                                            
                                                            html.Div(
                                                                [
                                                                                                                    
                                                                ],
                                                                id="tab2-report2",
                                                                className="pretty_container",
                                                            ),
                                                        ],
                                                        id="tab2-section-input-report",
                                                        className="eight columns",
                                                     ),
                                           
                                        ],
                                       className="row flex-display",
                                    ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P("Calculated Corrosion Risk      ", className="control_label"),

                                            dcc.Dropdown(
                                                id="tab2-section3-y-axix-ddown",
                                                #options= [{'label': k, 'value': k} for k in df_current.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                #value=get_columns_list(df_current),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-preprocessed_graph2"

                                                            )],
                                                id="tab2-section3-row-preprocessing",
                                                className="pretty_container",
                                            ),
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
                                            
                                            html.P("Accumulated Corrosion Risk      ", className="control_label"),

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



# READ THE XLSS FILE, DO PREPROCESSING, IMPUTING, OUTLIER REMOVAL AND SCALING OF MEAS_RATE

@app.callback(Output('df',  'data'),
               #Output('xaxis-column', 'options'), Output('yaxis-column', 'options'),
               # Output('xaxis-column', 'value'), Output('yaxis-column', 'value'),
 
            [Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified')
            ])

def update_output(list_of_contents, list_of_names, list_of_dates):
    df_current = None
    if list_of_contents is not None:
        df_current = None
        df_current = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[1]
        #upload_cont = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[0]

        #numerical_col = df_current.select_dtypes(include=np.number).columns.tolist()
        #datetime_col = df_current.select_dtypes(include=datetime).columns.tolist()

        # preprocess
        
        df_current = strip_df_colnames(df_current)
        df_current, empties = drop_empty_or_sparse_cols(df_current, ['Date', 'Time'])
        df_current = create_datatime_col(df_current)    
        df_current = strip_no_numeric_cols(df_current)
        df_current = set_df_col_as_index(df_current, 'DateTime')   
        #from indexed df select onlu numerical columns (DateTime already used in index)
        df_current = df_current.select_dtypes(include=np.number)
        df_current = keep_columns(df_current, colnames_kept)
        
        #print(keept_cols(df_current, colnames_kept))

        #imputing
        # Impute the missing values in 'MeasRate' with average/mean
        #df_current['Meas_Rate']=df_current['Meas_Rate'].mask(df_current['Meas_Rate']==0).fillna(df_current['Meas_Rate'].mean())


        # #remove the outliers
        # df_current['savgol'] = scp.savgol_filter(x = df_current['Meas_Rate'], window_length=21, polyorder = 1, deriv=0)#, delta=1.0, axis=- 1, mode='interp', cval=0.0
        # df_current['savgol_deriv'] = scp.savgol_filter(x = df_current['Meas_Rate'], window_length=21, polyorder = 1, deriv=1)#, delta=1.0, axis=- 1, mode='interp', cval=0.0
        # df_current['savgol_stable'] = df_current['savgol' ][abs(df_current['savgol_deriv']) < 0.00063]


        # #  Scale Meas_Rate
        # features = df_current['Meas_Rate']
        # # Use scaler of choice; here Standard scaler is used
        # scaler = StandardScaler().fit(features.values.reshape(-1,1))
        # features = scaler.transform(features.values.reshape(-1,1))
        # df_current['Meas_Rate'] = features


        
        # more generally, this line would be
        # json.dumps(cleaned_df)
        return df_current.to_json(date_format='iso', orient='split')
               
                #, 
                #[{'label': k, 'value': k} for k in ['index'] + colnames_kept ],
                #    [{'label': k, 'value': k} for k in ['index'] + colnames_kept ],
                #'index', colnames_kept[0]
    return None #[None,[], [], None, None]




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
    print("model is", y)
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
               Output('datepicker-from-tab2', 'min_date_allowed'), Output('datepicker-from-tab2', 'max_date_allowed'),Output('datepicker-from-tab2', 'date'),
               Output('datepicker-to-tab2', 'min_date_allowed'), Output('datepicker-to-tab2', 'max_date_allowed'),Output('datepicker-to-tab2', 'date'),
               Output('tab2-report1', 'children'),
               Output('tab2y-axix-ddown', 'options'),Output('tab2y-axix-ddown', 'value'),
               Output('tab2-section3-y-axix-ddown', 'options'),Output('tab2-section3-y-axix-ddown', 'value'),
               Output('tab2-section4-y-axix-ddown', 'options'),Output('tab2-section4-y-axix-ddown', 'value'),
               
               Output('tab2-data-preprocessed', 'data'),
                
            [Input('df',  'data'),
            Input('check-preprocess', 'value'),
            Input("datepicker-from-tab2", "date"),
            Input("datepicker-to-tab2", "date"),

            ])

def preprocess_df(jsonified_cleaned_data, checkedProcess, date_from, date_to):
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
        
        count_nan = df_current.isna().sum()/df_current.shape[0]*100        
        
         
        

        nbObs = 0 #df_current['savgol_stable'].shape[0]
        nbAfterSavGol = 0 #nbObs -  df_current['savgol_stable'].isnull().sum()
        percentage = 0 #(float (nbAfterSavGol) / float (nbObs))*100
        #percentage = "{:.2f}".format(percentage)

        return_divs = [html.P("Percentage of missing values:\n", style={'color': 'red'})]
        return_divs.append (	html.P(f"total number of observations: {df_current.shape[0]}"))

        for val, col in zip(count_nan, df_current.columns):
            return_divs.append(	html.P(col  + " contains " + str("{:.2f}".format(val)) + " percent missing values.\n"))

        return_divs.append (html.P("Consecutive missing values (at least 3)\n", style={'color': 'red'}))
        missing_seqs = [ df_current[a].isnull().astype(int).groupby(df_current[a].notnull().astype(int).cumsum()).sum() for a in df_current.columns]

        has_missing = False
        for a, col  in zip(missing_seqs, df_current.columns):
            if len(list(a[a > 2].index)) > 0:
                has_missing = True
                return_divs.append(	html.P(col  + " contains missing sequences at " + str(list(a[a > 2].index)) ))
       
        return_divs.append(	html.P("The computations continue by elliminating the rows containing missing values.", style={'color': 'red'}))
        return_divs.append(	html.P("Existing trends: ", style={'color': 'red'}))

        if has_missing:
            df_current.dropna(inplace=True, axis='rows')

        for  col  in df_current.columns:
            
            reg = LinearRegression().fit(np.array([i for i in range(df_current.shape[0])]).reshape(-1,1), list(df_current[col].values))
            slope = (reg.coef_[0])
            return_divs.append(	html.P(" Linear regression slope for " + col  + " is : " +  str("{:.3f}".format(slope)) ))

        return_divs.append(	html.P("End of reporting.", style={'color': 'red'}))

        
        #print("=============")


        
        return [
                df_current.to_json(date_format='iso', orient='split'), 
                df_current.index.min(), df_current.index.max(), df_current.index.min(),
                df_current.index.min(), df_current.index.max(), df_current.index.max(),
                return_divs,
                [{'label': k, 'value': k} for k in  colnames_kept ], [ k for k in  keept_cols(df_current, colnames_kept) ],
                [{'label': k, 'value': k} for k in  colnames_kept ], [ k for k in  keept_cols(df_current, colnames_kept) ],
                [{'label': k, 'value': k} for k in  colnames_kept ], [ k for k in  keept_cols(df_current, colnames_kept) ],
                
                df_current.to_json(date_format='iso', orient='split')
        ]
                #
                #df_current.df.index.min(), df_current.df.index.max(), df_current.df.index.max()]
               
     #return [None , None, None, None, None, None, None]
     return [None , None , None, None, None , None, None, [], [], [], [], [], [], [], None]



#----------------------------------------------------------
# FIRST PLOT IN TAB2
#----------------------------------------------------------

@app.callback(
    Output(component_id='tab2-preprocessed_graph1', component_property='figure'),
    Output("tab2-input-max-corrosion-risk", "value"),
    Output("tab2-input-accumulated-corrosion-risk", "value"),
    Output("tab2-input-max-average-corrosion-risk", "value"),
    [
        Input("tab2y-axix-ddown", "value"),
        Input("tab2-data-preprocessed", "data"),

 
    ]  
)
def make_figure_tab2_fig1(
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
        # your layout goes here
        layout_kwargs = {
                        'title': 'Sensor Data:',
                        'xaxis': {'domain': [0, 0.8]}
                        }
        cntr = 0
        for i, d in enumerate(data):

            # we define our layout keys by string concatenation
            # * (i > 0) is just to get rid of the if i > 0 statement
            axis_name = 'yaxis' + str(i + 1) * (i > 0)
            yaxis = 'y' + str(i + 1) * (i > 0)
            plotly_data.append(plotly.graph_objs.Scatter(x= df.index, y=d, name=labels[i], mode='markers', #opacity=0.5, 
                    marker={'size': 2 , 'symbol': 'diamond'}
                     ))
            
            layout_kwargs[axis_name] = {
                                            'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] ,
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


        return [fig, 0.6, 100, 100]
    return [{'layout': {'title': 'No input specified, please fill in an input.'}}, 0.6,100,100]

#----------------------------------------------------------
# PLOT  CALCULATED CORROSION RISK
#----------------------------------------------------------
@app.callback(
    Output(component_id = 'tab2-preprocessed_graph2',           component_property='figure'),
    Output(component_id = 'tab2-data-preprocessed-predicted',   component_property='data'),
    Output('tab2-report2', 'children'),
    Output(component_id='tab2-preprocessed-graph3', component_property='figure'),
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
def make_figure_tab2_3(
        y_col, max_corrosion_risk, accumulated_corrosion_risk, 
        max_average_corrosion_risk, jsonified_cleaned_data, ml_model_name, _time_window
):
    if jsonified_cleaned_data is not None:
        # more generally, this line would be
        # json.loads(jsonified_cleaned_data)
        df = pd.read_json(jsonified_cleaned_data, orient='split')

        model_loaded = None
        
        if ml_model_name['model'] == "GRF":
            model_loaded = pickle.load(open('socorro_model_1.pkl', "rb"))
        else:
            print("module is not available... program exits")




        df_tmp = df[['Temperature', 'pH', 'ORP', 'Conductivity']]
        df_tmp.columns = ['f0', 'f1', 'f2', 'f3' ]
        model_loaded.predict(df_tmp)
        df['Corrosion Risk' ] = model_loaded.predict(df_tmp)
        #print( df['Corrosion Risk' ])
        df['Critical Corrosion Risk' ] = df['Corrosion Risk'][df['Corrosion Risk'] > max_corrosion_risk]#   > threshold).any(1)
        #df['Average Over Time Period' ] = df['Corrosion Risk' ].ewm(span=max_average_corrosion_risk, adjust=False).mean()
        df['Average Over Time Period' ] = df['Corrosion Risk' ].rolling(window=int(_time_window)).mean()


        df['Acc. Corrosion Risk'] = df['Corrosion Risk' ].cumsum()
        df['Acc. Corrosion Risk'] = df['Acc. Corrosion Risk'][df['Acc. Corrosion Risk'] > accumulated_corrosion_risk]

        
        number_of_exceedings = df['Critical Corrosion Risk' ].isnull().sum()
        return_divs = [html.P(f"Number of observations above threshold: {number_of_exceedings}")]
        
        if(number_of_exceedings > 30):
            return_divs.append(html.P("Perhap try incresing the threshold. ", style={'color': 'red'}))
        if (df['Acc. Corrosion Risk'].shape[0] > 0 ):
            timestamp = df['Acc. Corrosion Risk'].index[0]
            return_divs.append(html.P("The first occurence of Acc. Corrosion Risk: " + str(timestamp), style={'color': 'red'}))


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

        # print(labels)
        # print(df.columns)
        # fig = px.scatter(df, x=df.index, y=labels,  title="Long-Form Input")
        # fig.for_each_trace(
        #     lambda trace: 
        #     trace.update(marker_symbol="diamond" ) if trace.name == "Critical Corrosion Risk" else (),
        # )
        # fig.update_xaxes(showgrid=False)


        fig = make_figure_tab2_fig2(df, data, labels, added_cols)
        fig2 = make_figure_tab2_fig3(df, data, labels, added_cols, y_col)


        return [fig, df.to_json(date_format='iso', orient='split'), return_divs, fig2]
    return [{'layout': {'title': 'No input specified, please fill in an input.'}}, None, None, None]

def make_figure_tab2_fig2(df, data, labels, added_cols):
        plotly_data = []
        #plotly_layout = plotly.graph_objs.Layout()
        # your layout goes here
        layout_kwargs = {
                        'title': 'Predicted Corrosion Risk:',
                        'xaxis': {'domain': [0, 0.8]}
                        }
        cntr = 0
        for i, d in enumerate(data):

            # we define our layout keys by string concatenation
            # * (i > 0) is just to get rid of the if i > 0 statement
            axis_name = 'yaxis' + str(i + 1) * (i > 0)
            yaxis = 'y' + str(i + 1) * (i > 0)
            plotly_data.append(plotly.graph_objs.Scatter(x= df.index,y=d, name=labels[i], mode='markers' if labels[i] not in added_cols + ['Corrosion Risk'] else None, #opacity=0.5, 
                    marker={'size': 2 if labels[i] != "Critical Corrosion Risk" else 6, 'symbol': 'diamond'}
                     ))
            
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

def make_figure_tab2_fig3(df, data, labels, added_cols, y_col):
    layout_graph_tab1_fig1 = copy.deepcopy(layout)
    n_rows = len(y_col)
    data = []
    for i in range(n_rows):
        data.append(df[y_col[i]].tolist())
    data.append(df['Acc. Corrosion Risk'].tolist())


    added_cols = [ 'Acc. Corrosion Risk'  ]
    labels = y_col + added_cols



    plotly_data = []
    plotly_layout = plotly.graph_objs.Layout()
    # your layout goes here
    layout_kwargs = {
                    'title': 'Accumulated Corrosion Risk:',
                    'xaxis': {'domain': [0, 0.8]}
                    }
    cntr = 0
    for i, d in enumerate(data):

        # we define our layout keys by string concatenation
        # * (i > 0) is just to get rid of the if i > 0 statement
        axis_name = 'yaxis' + str(i + 1) * (i > 0)
        yaxis = 'y' + str(i + 1) * (i > 0)
        plotly_data.append(plotly.graph_objs.Scatter(x= df.index,y=d, name=labels[i], mode='markers' if labels[i] not in added_cols else None,
                marker={'size': 2 if labels[i] != ("Critical Corrosion Risk"  or "Acc. Corrosion Risk") else 6, 'symbol': 'diamond'}
                    ))
        
        layout_kwargs[axis_name] = {
                                        'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] if labels[i] != ("Critical Corrosion Risk"  or "Acc. Corrosion Risk") else  [df["Corrosion Risk"].min()*0.9, df["Corrosion Risk"].max()*1.1],
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



# #----------------------------------------------------------
# # PLOT  ACCUMULATED CORROSION RISK
# #----------------------------------------------------------
# @app.callback(
#     Output(component_id='tab2-preprocessed-graph3', component_property='figure'),
#     #Output(component_id = 'tab2-data-preprocessed-predicted', , component_property='data')
#     [
#         Input("tab2-section4-y-axix-ddown", "value"),
#         Input("tab2-input-max-corrosion-risk", "value"),
#         Input("tab2-input-accumulated-corrosion-risk", "value"),
#         Input("tab2-input-max-average-corrosion-risk", "value"),
#         Input("tab2-data-preprocessed-predicted", "data"),
#     ]  
# )
# def make_figure_tab2_4(
#         y_col, max_corrosion_risk, accumulated_corrosion_risk, 
#         max_average_corrosion_risk, jsonified_cleaned_data 
# ):
#     if jsonified_cleaned_data is not None:
#         # more generally, this line would be
#         # json.loads(jsonified_cleaned_data)
#         df = pd.read_json(jsonified_cleaned_data, orient='split')

#         # df['Acc. Corrosion Risk'] = df['Corrosion Risk' ].cumsum()
#         # df['Acc. Corrosion Risk'] = df['Acc. Corrosion Risk'][df['Acc. Corrosion Risk'] > accumulated_corrosion_risk]
   

        
#         layout_graph_tab1_fig1 = copy.deepcopy(layout)
#         print(y_col)
#         n_rows = len(y_col)
#         data = []
#         for i in range(n_rows):
#             data.append(df[y_col[i]].tolist())
#         data.append(df['Acc. Corrosion Risk'].tolist())


#         added_cols = [ 'Acc. Corrosion Risk'  ]
#         labels = y_col + added_cols



#         plotly_data = []
#         plotly_layout = plotly.graph_objs.Layout()
#         # your layout goes here
#         layout_kwargs = {
#                         'title': 'Accumulated Corrosion Risk:',
#                         'xaxis': {'domain': [0, 0.8]}
#                         }
#         cntr = 0
#         for i, d in enumerate(data):

#             # we define our layout keys by string concatenation
#             # * (i > 0) is just to get rid of the if i > 0 statement
#             axis_name = 'yaxis' + str(i + 1) * (i > 0)
#             yaxis = 'y' + str(i + 1) * (i > 0)
#             plotly_data.append(plotly.graph_objs.Scatter(x= df.index,y=d, name=labels[i], mode='markers' if labels[i] not in added_cols else None,
#                     marker={'size': 2 if labels[i] != ("Critical Corrosion Risk"  or "Acc. Corrosion Risk") else 6, 'symbol': 'diamond'}
#                      ))
            
#             layout_kwargs[axis_name] = {
#                                             'range': [df[labels[i]].min()*0.9, df[labels[i]].max()*1.1 ] if labels[i] != ("Critical Corrosion Risk"  or "Acc. Corrosion Risk") else  [df["Corrosion Risk"].min()*0.9, df["Corrosion Risk"].max()*1.1],
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

#         return fig
#     return {'layout': {'title': 'No input specified, please fill in an input.'}}


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


