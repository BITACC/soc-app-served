# Import required libraries
import pickle
import copy
import pathlib
import dash
import math
import datetime as dt
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


#current data source
df_uploaded = []

# Load data
#df_soc = pd.read_excel('../../data/data until 2020-12-3.xlsx' ,header=0)# , parse_dates=[0],  squeeze=True , index_col=0
df_soc = pd.read_excel('data/data-03122020-manuallymodified-SG.xlsx' ,header=0)# , parse_dates=[0],  squeeze=True , index_col=0

df_soc['DDate'] = pd.to_datetime(df_soc['DDate'], dayfirst=True)
print(df_soc['DDate'])
#print(df_soc.dtypes)
#df_soc['DDate'] = pd.to_datetime(df_soc['DDate'],format="%d/%m/%Y %H:%M:%S")#.dt.date

print (df_soc.DDate.min())
print (df_soc.DDate.max())
print("mmmmmmmmmmmmmmmmmmmmmmmmm")

datetime = ['datetime64[ns]']
numerical_col = df_soc.select_dtypes(include=np.number).columns.tolist()
datetime_col = df_soc.select_dtypes(include=datetime).columns.tolist()

x_col = datetime_col[0]
print(x_col)

# df.index = pd.to_datetime(df['Date'])


print(df_soc.dtypes)



### preprocessing


# Imputation

# Impute Series


print("start  cleaning")

import data_cleaning as dc

#df_soc = dc.od(df_soc, numerical_col, 'naive')


get_columns_dic(df_soc)


#df_soc['Cleaned Date'] = pd.to_datetime(df_soc.iloc[:, 0],format="%d/%m/%Y %H:%M:%S").dt.date


import imputers as imp
from datetime import date

# df_soc = imp.impute_naive(df_soc, numerical_col)
df_soc = imp.impute_mean(df_soc, numerical_col)

cols_with_missing = (col for col in df_soc.columns
                     if df_soc[col].isnull().any())
print("columns with NAN after imputing ...", cols_with_missing)

# pp = pdvis()


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
        # main body

        html.Img(
            src=app.get_asset_url("logo.png"),
            id="plotly-image-SPG",
            style={
                "height": "60px",
                "width": "300px",
                "margin-bottom": "25px",
            },
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

                            dcc.Store(id="aggregate_data"),
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
                                        className="one-half column",
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
                                                "Filter by construction date (or select range in histogram):",
                                                className="control_label",
                                            ),
                                            dcc.RangeSlider(
                                                id="year_slider",
                                                min=2019,
                                                max=2021,
                                                step=None,
                                                marks={
                                                    2019: {'label': '2019', 'style': {'color': '#77b0b1'}},
                                                    2020: '2020',
                                                    2021: {'label': '2021', 'style': {'color': '#f50'}}
                                                },
                                                value=[2019, 2021],
                                                className="dcc_control",

                                            ),
                                            html.P("Choose y-axis:", className="control_label"),
                                            dcc.Dropdown(
                                                id="xaxis-column",
                                                options=[{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()] + ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_soc)]),# get_options_dic(),
                                                # multi=True,
                                                value=datetime_col[0],
                                                className="dcc_control",
                                            ),
                                            dcc.Dropdown(
                                                id="yaxis-column",
                                                options=[{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()]+ ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_soc)]),
                                                # multi=True,
                                                value=numerical_col[0],
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
                                                    dcc.Graph(id="count_graph")
                                                ],
                                                id="countGraphContainer",
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
                                                options= [{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                value=get_columns_list(df_soc),
                                                className="dcc_control",
                                            ),
                                            dcc.Graph(id="main_graph")
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
                            # empty Div to trigger javascript file for graph resizing
                            html.Div(id="tab2-output-clientside"),
                            
                            html.Div(
                                [

                                    html.Div(
                                        [
                                            
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
                                                    id='my-date-picker-single-from',
                                                    min_date_allowed=date(1995, 8, 5),
                                                    max_date_allowed=date(2017, 9, 19),
                                                    initial_visible_month=date(2017, 8, 5),
                                                    date=date(2017, 8, 25)
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
                                                    id='my-date-picker-single-to',
                                                    min_date_allowed=date(1995, 8, 5),
                                                    max_date_allowed=date(2017, 9, 19),
                                                    initial_visible_month=date(2017, 8, 5),
                                                    date=date(2017, 8, 25)
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
                                                    html.P("Report:"),
                                                    html.P("Warning:", style={'color': 'red'}),
                                                    html.P("Preprocessing output statistics:"),
                                                    html.P("Report:"),
                                                    html.P("Report:"),
                                                    html.P("Report:")
                                                ],
                                                id="countGraphContainer2",
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
                                                options= [{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                value=get_columns_list(df_soc),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-preprocessed-count_graph"

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
                                                    )
                                                ],
                                                    className="pretty_container four columns",
                                                    id="tab2-inputs",
                                            ),
                                            html.Div(
                                                        [
                                                            
                                                            html.Div(
                                                                [
                                                                    html.P("Warning:", style={'color': 'red'}),
                                                                    html.P("Report:"),
                                                                    html.P("Report:"),
                                                                    html.P("Report:"),
                                                                    html.P("Event:", style={'color': 'blue'}),
                                                                    html.P("Report:"),
                                                                    html.P("Report:"),
                                                                    html.P("Report:")
                                                                ],
                                                                id="tab2-input-report",
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
                                                options= [{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                value=get_columns_list(df_soc),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-section3-preprocessed-count_graph"

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
                                                options= [{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),
                                                multi=True,
                                                value=get_columns_list(df_soc),
                                                className="dcc_control",
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="tab2-section4-preprocessed-count_graph"

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
                             html.Div(
                                    [
                                        html.Div(
                                            children=[ 
                                                html.Div(
                                                    html.P("Export results as     ", className="control_label"),
                                                    style={'display': 'inline-block', 'padding': '10px'},
                                                ),
                                                html.Div(
                                                    dcc.Dropdown(
                                                        id="tab2-export-ddown",
                                                        options= [
                                                                    {'label': 'html', 'value': 'html'},
                                                                    {'label': 'pdf', 'value': 'pdf'}
                                                                ], #get_options_dic(),
                                                        multi=False,
                                                        value="pdf",
                                                        className="dcc_control"
                                                    ), 
                                                    style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                                                ),
                                                html.Button('Submit', id='submit-val', n_clicks=0),
                                            ],
                                                #
                                                className="pretty_container twelve columns",
                                                id="tab2-section4-inputs",
                                        )
                                    ],
                                    className="pretty_container twelve columns",
                                ),
 

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


# # Helper functions
# def human_format(num):
#     if num == 0:
#         return "0"

#     magnitude = int(math.log(num, 1000))
#     mantissa = str(int(num / (1000 ** magnitude)))
#     return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]


def filter_dataframe(df, year_slider):
    print("call: filter_dataframe")
    # return df

    if not year_slider:
        print("return:1 filter_dataframe was empty")
        return df
    else:
        print("else:2 filter_dataframe")
        dff = df

        #dff = df[
        #    (df[get_datetime_col()[0]] > dt.datetime(year_slider[0], 1, 1))
        #    & (df[get_datetime_col()[0]] < dt.datetime(year_slider[1], 1, 1))
        #    ]
        print("return:2 filter_dataframe")
        return dff


# Selectors -> main graph
@app.callback(
    Output(component_id='count_graph', component_property='figure'),
    [
        Input("xaxis-column", "value"),
        Input("yaxis-column", "value"),
        Input("year_slider", "value"),

        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified')
    ],
    # [#State("lock_selector", "value"),
    # State("main_graph", "relayoutData")],
)
def make_count_figure(
        x_col, y_col, year_slider  , list_of_contents, list_of_names, list_of_dates# , selector, main_graph_layout
):
    print(" called: make_main_figure")
    dff = df_soc

    upload_cont = None
    if list_of_contents is not None:
        dff = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[1]
        upload_cont = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[0]
        # print("This is a upload test******************************\n")
        # print(list_of_contents[0], list_of_names[0], list_of_dates[0])
        # print(dff)
        numerical_col = dff.select_dtypes(include=np.number).columns.tolist()
        datetime_col = dff.select_dtypes(include=datetime).columns.tolist()

    #dff = filter_dataframe(df_soc, year_slider)
    print("********************")
    df_soc.describe()
    print(df_soc[x_col])
    print(df_soc[y_col])
    colors = []
    for i in range(2019, 2021):
        if i >= int(year_slider[0]) and i < int(year_slider[1]):
            colors.append("rgb(123, 199, 255)")
        else:
            colors.append("rgba(123, 199, 255, 0.2)")

    layout_count_graph = copy.deepcopy(layout)

    data = [
        dict(
            type="Scattergl",
            name="Gas Produced (mcf)",
            x=dff[x_col],
            y=dff[y_col],
            #line=dict(shape="spline", smoothing=2, width=1),  # , color="#fac1b7"

            opacity=0.5,
            hoverinfo="skip",
            # marker=dict(color=colors),

            mode="markers",  # lines+
            # line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
            marker=dict(symbol="circle-open"),
            marker_size=2,

        )]
    layout_count_graph["title"] = y_col
    layout_count_graph["dragmode"] = "select"
    layout_count_graph["showlegend"] = False
    layout_count_graph["autosize"] = True
    figure = dict(data=data, layout=layout_count_graph)

    print(" returned: make_main_figure")  # ,  figure
    return figure


@app.callback(
    Output(component_id='main_graph', component_property='figure'),
    Output('output-data-upload', 'children'),
    [
        Input("y-axix-ddown", "value"),
        Input("year_slider", "value"),
        Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified')
    ]  # ,
    # [State("lock_selector", "value"), State("main_graph", "relayoutData")],
)
def make_main_figure(
        y_col, year_slider, list_of_contents, list_of_names, list_of_dates  # , selector, main_graph_layout
):
    print(" called: make_main_figure")


    dff = df_soc
    dff = filter_dataframe(df_soc, year_slider)

    upload_cont = None
    if list_of_contents is not None:
        dff = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[1]
        upload_cont= upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[0]
        #print("This is a upload test******************************\n")
        #print(list_of_contents[0], list_of_names[0], list_of_dates[0])
        #print(dff)
        numerical_col = dff.select_dtypes(include=np.number).columns.tolist()
        datetime_col = dff.select_dtypes(include=datetime).columns.tolist()



    layout_main_graph = copy.deepcopy(layout)
    # 'Cond µS/cm', 'SpCond µS/cm', 'nLF Cond µS/cm'
    layout_main_graph["title"] = y_col
    layout_main_graph["dragmode"] = "select"
    layout_main_graph["showlegend"] = True
    layout_main_graph["autosize"] = True

    n_rows = len(y_col)  # make_subplots breaks down if rows > 70
    fig = tools.make_subplots(rows=n_rows, cols=1)
    fig['layout'].update(height=3000, autosize=True, showlegend=True, dragmode="select",
                         title='Second Class of Parameters')  # , title = y_col , width=1000

    # print(fig['layout'])
    for i in range(n_rows):
        new_col = next(line_color)
        trace = go.Scattergl(x=dff[get_datetime_columns_list(dff)[0]], y=dff[y_col[i]],
                             name=y_col[i],
                             mode="markers",  # lines+
                             # line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                             marker=dict(symbol="circle-open"),
                             marker_size=2,
                             hoverinfo="skip", )
        fig.append_trace(trace, i + 1, 1)
        fig['layout']['yaxis' + str(i + 1)].update(title=y_col[i])

        trace1 = go.Scattergl(x=dff[get_datetime_columns_list(dff)[0]], y=dff[y_col[i]].rolling(period1).mean(),
                              name="MA(" + str(period1) + ") " + y_col[i],
                              mode="lines",  # lines+
                              # fill=None,
                              # fillcolor=new_col,
                              line=dict(color=new_col, width=2.5),
                              # line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                              marker=dict(symbol="circle-open"),
                              hoverinfo="skip", )
        fig.append_trace(trace1, i + 1, 1)
        new_col = next(line_color)
        trace2 = go.Scattergl(x=dff[get_datetime_columns_list(dff)[0]], y=dff[y_col[i]].rolling(period2).mean(),
                              name="MA(" + str(period2) + ") " + y_col[i],
                              mode="lines",  # lines+
                              # fill='tonexty',
                              line=dict(color=new_col, width=2.5),
                              # line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                              marker=dict(symbol="circle-open"),
                              hoverinfo="skip", )
        fig.append_trace(trace2, i + 1, 1)

        temp_y = dff[y_col[i]].rolling(period2).mean()
        temp_x = dff[get_datetime_columns_list(dff)[0]]
        idx = np.argwhere(np.diff(np.sign(dff[y_col[i]].rolling(period2).mean() - dff[y_col[i]].rolling(period1).mean()))).flatten()
        new_col = next(line_color)
        trace3 = go.Scattergl(x=temp_x[idx], y=temp_y[idx],
                              name="Event " + y_col[i],
                              # fill='tonexty',
                              line=dict(color=(list(COLORS.values()))[i], width=2.5),
                              mode="markers",
                              marker_color = "black",
                              hoverinfo="skip", )
        fig.append_trace(trace3, i + 1, 1)



    print(" returned: make_main_figure")  # ,  fig
    return [fig,upload_cont]




@app.callback(
    Output(component_id='tab2-preprocessed-count_graph', component_property='figure'),
    [
        Input("y-axix-ddown", "value"),
        Input("year_slider", "value"),
        Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified')
    ]  # ,
    # [State("lock_selector", "value"), State("main_graph", "relayoutData")],
)
def make_figure(
        y_col, year_slider, list_of_contents, list_of_names, list_of_dates  # , selector, main_graph_layout
):
                # Imports
                #import matplotlib.pyplot as plt
                #from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    dff = df_soc
    dff = filter_dataframe(df_soc, year_slider)

    upload_cont = None
    if list_of_contents is not None:
        dff = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[1]
        upload_cont= upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[0]
        #print("This is a upload test******************************\n")
        #print(list_of_contents[0], list_of_names[0], list_of_dates[0])
        #print(dff)
        numerical_col = dff.select_dtypes(include=np.number).columns.tolist()
        datetime_col = dff.select_dtypes(include=datetime).columns.tolist()


    # generate dummy data, taken from question
    import plotly
    n_rows = len(y_col) 
    data = []
    for i in range(n_rows):
        data.append(dff[y_col[i]].tolist())
        #print(dff[y_col[i]])
        #print(dff[y_col[i]].values)
        #print(dff[y_col[i])
    labels = y_col

    plotly_data = []
    plotly_layout = plotly.graph_objs.Layout()

    # your layout goes here
    layout_kwargs = {'title': 'y-axes in loop',
                    'xaxis': {'domain': [0, 0.8]}
                    }
    for i, d in enumerate(data):
        # we define our layout keys by string concatenation
        # * (i > 0) is just to get rid of the if i > 0 statement
        print("=================================")
        axis_name = 'yaxis' + str(i + 1) * (i > 0)
        yaxis = 'y' + str(i + 1) * (i > 0)
        plotly_data.append(plotly.graph_objs.Scatter(y=d, 
                                                    name=labels[i]))
        layout_kwargs[axis_name] = {'range': [0, i + 0.1],
                                'position': 1 - i * 0.04}

        plotly_data[i]['yaxis'] = yaxis
        if i > 0:
            layout_kwargs[axis_name]['overlaying'] = 'y'

        print(plotly_data[i])

    fig = plotly.graph_objs.Figure(data=plotly_data, layout=plotly.graph_objs.Layout(**layout_kwargs))

    return fig



@app.callback(

    Output('xaxis-column', 'options'), Output('yaxis-column', 'options'),
    Output('xaxis-column', 'value'), Output('yaxis-column', 'value'),
    [
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified')
    ]  # ,
    # [State("lock_selector", "value"), State("main_graph", "relayoutData")],
)
def update_dropdowns(
        list_of_contents, list_of_names, list_of_dates
):
    print(" called: update on upload")

    dff = df_soc
    if list_of_contents is not None:
        dff = upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[1]
        #upload_cont= upload.parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])[0]
    numerical_col = dff.select_dtypes(include=np.number).columns.tolist()
    datetime_col = dff.select_dtypes(include=datetime).columns.tolist()
    print("DateTIme Column")
    print(datetime_col[0], datetime_col[1])
    print(dff.columns)
    print(dff)


    return [[{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()] + ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_soc)]),
            [{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()] + ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_soc)]),
            datetime_col[0], numerical_col[0]
            ]






if __name__ == "__main__":
    import os

    debug =  True
    app.run_server(host="0.0.0.0", port=8050, debug=debug)


#if __name__ == "__main__":
#    # print("Data Types: " , df_soc.dtypes)
#    # print("Index Name: ", df_soc.index.name)
#    app.run_server(debug=True, threaded=False)
