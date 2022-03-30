def print_html_tab_analysis():

	return "                dcc.Tab(label='Moniroting',		\
                        value='tab-1',		\
                        className='custom-tab',		\
                        selected_className='custom-tab--selected',		\
                        children=[		\
                            dcc.Store(id="aggregate_data"),		\
                            # empty Div to trigger javascript file for graph resizing		\
                            html.Div(id="output-clientside"),		\
                            html.Div(		\
                                [			\
                                    html.Div(		\
                                        [		\
                                            html.Img(		\
                                                src=app.get_asset_url("socorro.png"),		\
                                                id="plotly-image",		\
                                                style={		\
                                                    "height": "60px",		\
                                                    "width": "auto",		\
                                                    "margin-bottom": "25px",		\
                                                },		\
                                            )		\
                                        ],		\
                                        className="one-third column",		\
                                    ),		\
                                    html.Div(		\
                                        [		\
                                            html.Div(		\
                                                [		\
                                                    html.H3(		\
                                                        "SOCORRO",		\
                                                        style={"margin-bottom": "0px"},		\
                                                    ),		\
                                                    html.H5(		\
                                                        "Seeking out corrosion, before it is too late.",		\
                                                        style={"margin-top": "0px"}		\
                                                    ),		\
                                                ]		\
                                            )		\
                                        ],		\
                                        className="one-half column",		\
                                        id="title",		\
                                    ),		\
                                    html.Div(		\
                                        [		\
                                            html.A(		\
                                                html.Button("SOCORRO Home", id="learn-more-button"),		\
                                                href="https://www.socorro.eu/",		\
                                            )		\
                                        ],		\
                                        className="one-third column",		\
                                        id="button",		\
                                    ),		\
                                ],		\
                                id="header",		\
                                className="row flex-display",		\
                                style={"margin-bottom": "25px"},		\
                            ),		\
                            html.Div(		\
                                [		\
                                    html.Div(		\
                                        [		\
                                            html.Div([		\
                                                dcc.Upload(		\
                                                    id='upload-data',		\
                                                    children=html.Div([		\
                                                        'Drag and Drop or ',		\
                                                        html.A('Select Files')		\
                                                    ]),		\
                                                    style={		\
                                                        'width': '100%',		\
                                                        'height': '60px',		\
                                                        'lineHeight': '60px',		\
                                                        'borderWidth': '1px',		\
                                                        'borderStyle': 'dashed',		\
                                                        'borderRadius': '5px',		\
                                                        'textAlign': 'center',		\
                                                        'margin': '10px'		\
                                                    },		\
                                                    # Allow multiple files to be uploaded		\
                                                    multiple=True		\
                                                ),		\
                                                html.Div(id='output-data-upload')		\
                                            ]),		\
                                            html.P(		\
                                                "Filter by construction date (or select range in histogram):",		\
                                                className="control_label",		\
                                            ),		\
                                            dcc.RangeSlider(		\
                                                id="year_slider",		\
                                                min=2019,		\
                                                max=2021,		\
                                                step=None,		\
                                                marks={		\
                                                    2019: {'label': '2019', 'style': {'color': '#77b0b1'}},		\
                                                    2020: '2020',		\
                                                    2021: {'label': '2021', 'style': {'color': '#f50'}}		\
                                                },		\
                                                value=[2019, 2021],		\
                                                className="dcc_control",		\
                                            ),		\
                                            html.P("Choose y-axis:", className="control_label"),		\
                                            dcc.Dropdown(		\
                                                id="xaxis-column",		\
                                                options=[{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()] + ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_soc)]),# get_options_dic(),		\
                                                # multi=True,		\
                                                value=datetime_col[0],		\
                                                className="dcc_control",		\
                                            ),		\
                                            dcc.Dropdown(		\
                                                id="yaxis-column",		\
                                                options=[{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()]+ ([{'label': a, 'value': a} for a in get_datetime_columns_list(df_soc)]),		\
                                                # multi=True,		\
                                                value=numerical_col[0],		\
                                                className="dcc_control",		\
                                            ),		\
                                        ],		\
                                        className="pretty_container four columns",		\
                                        id="cross-filter-options",		\
                                    ),		\
                                    html.Div(		\
                                        [		\
                                            html.Div(		\
                                                [		\
                                                ],		\
                                                id="info-container",		\
                                                className="row container-display",		\
                                            ),		\
                                            html.Div(		\
                                                [dcc.Graph(id="count_graph"		\
                                                           )],		\
                                                id="countGraphContainer",		\
                                                className="pretty_container",		\
                                            ),		\
                                        ],		\
                                        id="right-column",		\
                                        className="eight columns",		\
                                    ),		\
                                ],		\
                                className="row flex-display",		\
                            ),		\
                            html.Div(		\
                                [		\
                                    html.Div(		\
                                        children=[		\
                                            dcc.Dropdown(		\
                                                id="y-axix-ddown",		\
                                                options= [{'label': k, 'value': k} for k in df_soc.select_dtypes(include=np.number).columns.tolist()], #get_options_dic(),		\
                                                multi=True,		\
                                                value=get_columns_list(df_soc),		\
                                                className="dcc_control",		\
                                            ),		\
                                            dcc.Graph(id="main_graph")		\
                                        ],		\
                                        className="pretty_container twelve columns",		\
                                    ),		\
                                ],		\
                                className="row flex-display",		\
                            ),		\
                        ]),"