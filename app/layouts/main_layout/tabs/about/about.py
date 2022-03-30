from dash.dependencies import Output, Input
from app import df_soc
#Selectors -> main graph
@app.callback(
    Output(component_id='count_graph', component_property='figure'),
    [
        Input("xaxis-column", "value"),
        Input("yaxis-column", "value"),
        Input("year_slider", "value"),
    ],
    #[#State("lock_selector", "value"), 
    #State("low_range_graph", "relayoutData")],
)
def make_main_figure(
    x_col,y_col,  year_slider#, selector, low_range_graph_layout
):
    print(" called: make_main_figure" )
    dff = df_soc
    dff =  filter_dataframe(df_soc,  year_slider)


    colors = []
    for i in range(2012, 2020):
        if i >= int(year_slider[0]) and i < int(year_slider[1]):
            colors.append("rgb(123, 199, 255)")
        else:
            colors.append("rgba(123, 199, 255, 0.2)")

    layout_count_graph = copy.deepcopy(layout)

    data = [
        dict(
            type="Scattergl",
            mode="markers",#lines+
            name="Gas Produced (mcf)",
            x=dff[x_col],
            y=dff[y_col],
            line=dict(shape="spline", smoothing=2, width=1),#, color="#fac1b7"
            
            #marker=dict(symbol="diamond-open"),
            opacity=0.5,
            hoverinfo="skip",
            #marker=dict(color=colors),
            marker=dict(symbol="circle-open"),
        )]
    layout_count_graph["title"] = y_col
    layout_count_graph["dragmode"] = "select"
    layout_count_graph["showlegend"] = False
    layout_count_graph["autosize"] = True
    figure = dict(data=data, layout=layout_count_graph)

    print(" returned: make_main_figure")#,  figure
    return figure


@app.callback(
    Output(component_id='low_range_graph', component_property='figure'),
    [
        Input("yaxis-column-low", "value"),
        Input("year_slider", "value"),
    ]#,
    #[State("lock_selector", "value"), State("low_range_graph", "relayoutData")],
)
def make_low_range_figure(
    y_col,  year_slider#, selector, low_range_graph_layout
):
    print(" called: make_low_range_figure" )
    dff = df_soc
 
    dff =  filter_dataframe(df_soc,  year_slider)

    layout_low_range_graph = copy.deepcopy(layout)
    #'Cond µS/cm', 'SpCond µS/cm', 'nLF Cond µS/cm'
    layout_low_range_graph["title"] = y_col
    layout_low_range_graph["dragmode"] = "select"
    layout_low_range_graph["showlegend"] = True
    layout_low_range_graph["autosize"] = True
 

    n_rows = len(y_col) # make_subplots breaks down if rows > 70

    fig = make_subplots(rows=n_rows, cols=1)


    fig['layout'].update(height=3000, autosize = True, showlegend = True, dragmode = "select", title='Second Class of Parameters')#, title = y_col , width=1000
        
    #print(fig['layout'])

    for i in range(n_rows):
        new_col = next(line_color)

        trace = go.Scattergl(x = dff[get_datetime_col()[0]], y = dff[y_col[i]], 
                            name=y_col[i], 
                            mode="markers",#lines+
                            #line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                            marker=dict(symbol="circle-open"),
                            hoverinfo="skip",)
        fig.append_trace(trace, i+1, 1)
        fig['layout']['yaxis' + str(i+1)].update(title=y_col[i])

        trace1 = go.Scattergl(x = dff[get_datetime_col()[0]], y = dff[y_col[i]].rolling(period1).mean(), 
                            name="MA(" + str(period1) + ") " + y_col[i], 
                            mode="lines",#lines+
                            #fill=None,
                            #fillcolor=new_col,
                            line=dict(color=new_col, width=2.5),
                            #line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                            marker=dict(symbol="circle-open"),
                            hoverinfo="skip",)
        fig.append_trace(trace1, i+1, 1)
        new_col = next(line_color)
        trace2 = go.Scattergl(x = dff[get_datetime_col()[0]], y = dff[y_col[i]].rolling(period2).mean(), 
                            name="MA(" + str(period2) + ") " + y_col[i], 
                            mode="lines",#lines+
                            #fill='tonexty',
                            line=dict(color=new_col, width=2.5),
                            #line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                            marker=dict(symbol="circle-open"),
                            hoverinfo="skip",)
        fig.append_trace(trace2, i+1, 1)


    print(" returned: make_low_range_figure")#,  fig
    return fig


    data = [ dict(
            type="scatter",
            mode="lines",#lines+
            name=y_col[i],
            x=dff[get_datetime_col()[0]],
            y=dff[y_col[i]],
            line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
            marker=dict(symbol="diamond-open"),
            #opacity=0,
            hoverinfo="skip",
            #marker=dict(color=colors),
        ) 
            for i in range(len(y_col))]
    layout_low_range_graph["title"] = y_col
    layout_low_range_graph["dragmode"] = "select"
    layout_low_range_graph["showlegend"] = True
    layout_low_range_graph["autosize"] = True
    figure = dict(data=data, layout=layout_low_range_graph)


    return figure


@app.callback(
    Output(component_id='high_range_graph', component_property='figure'),
    [
        Input("yaxis-column-high", "value"),
        Input("year_slider", "value"),
    ]#,
    #[State("lock_selector", "value"), State("low_range_graph", "relayoutData")],
)
def make_high_range_figure(
    y_col,  year_slider#, selector, low_range_graph_layout
):
    print(" called: make_high_range_figure")

    dff = df_soc
    dff =  filter_dataframe(df_soc,  year_slider)

    layout_high_range_graph = copy.deepcopy(layout)
    #'Cond µS/cm', 'SpCond µS/cm', 'nLF Cond µS/cm'


    n_rows = len(y_col) # make_subplots breaks down if rows > 70

    fig = make_subplots(rows=n_rows, cols=1)


    fig['layout'].update(height=1000, autosize = True, showlegend = True, dragmode = "select", title='First Class of Parameters')#, title = y_col , width=1000

    #print(fig['layout'])

    for i in range(n_rows):
        new_col = next(line_color)
        trace = go.Scattergl(x = dff[get_datetime_col()[0]], y = dff[y_col[i]], 
                            name=y_col[i], 
                            mode="markers",#lines+
                            #line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                            marker=dict(symbol="circle-open"),
                            hoverinfo="skip",)
        fig.append_trace(trace, i+1, 1)
        fig['layout']['yaxis' + str(i+1)].update(title=y_col[i])

        trace1 = go.Scattergl(x = dff[get_datetime_col()[0]], y = dff[y_col[i]].rolling(period1).mean(), 
                            name="MA(" + str(period1) + ") " + y_col[i], 
                            mode="lines",#lines+
                            #line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                            #fill=None,
                            #fillcolor=new_col,
                            line=dict(color=new_col, width=2.5),
                            marker=dict(symbol="circle-open"),
                            hoverinfo="skip",)
        fig.append_trace(trace1, i+1, 1)
        new_col = next(line_color)
        trace2 = go.Scattergl(x = dff[get_datetime_col()[0]], y = dff[y_col[i]].rolling(period2).mean(), 
                            name="MA(" + str(period2) + ") " + y_col[i], 
                            mode="lines",#lines+
                            #fill='tonexty',
                            line=dict(color=new_col, width=2.5),
                            #line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
                            marker=dict(symbol="circle-open"),
                            hoverinfo="skip",)
        fig.append_trace(trace2, i+1, 1)

    print(" returned: make_high_range_figure")#,  fig
    return fig



    data = [ dict(
            type="scatter",
            mode="lines",#lines+
            name=y_col[i],
            x=dff[get_datetime_col()[0]],
            y=dff[y_col[i]],
            line=dict(shape="spline", smoothing=2, width=1, color=(list(COLORS.values()))[i]),
            marker=dict(symbol="diamond-open"),
            #opacity=0,
            hoverinfo="skip",
            #marker=dict(color=colors),
        ) 
            for i in range(len(y_col))]
    layout_high_range_graph["title"] = y_col
    layout_high_range_graph["dragmode"] = "select"
    layout_high_range_graph["showlegend"] = True
    layout_high_range_graph["autosize"] = True

    # layout_pie["title"] = "Production Summary: {} to {}".format(
    #     year_slider[0], year_slider[1]


    figure = dict(data=data, layout=layout_high_range_graph)
    return figure

# # Main graph -> individual graph
# @app.callback(Output("high_range_graph", "figure"), [Input("low_range_graph", "hoverData")])
# def make_individual_figure(low_range_graph_hover):

#     layout_individual = copy.deepcopy(layout)

#     if low_range_graph_hover is None:
#         low_range_graph_hover = {
#             "points": [
#                 {"curveNumber": 4, "pointNumber": 569, "customdata": 31101173130000}
#             ]
#         }

#     chosen = [point["customdata"] for point in low_range_graph_hover["points"]]
#     index, gas, oil, water = produce_individual(chosen[0])

#     if index is None:
#         annotation = dict(
#             text="No data available",
#             x=0.5,
#             y=0.5,
#             align="center",
#             showarrow=False,
#             xref="paper",
#             yref="paper",
#         )
#         layout_individual["annotations"] = [annotation]
#         data = []
#     else:
#         data = [
#             dict(
#                 type="scatter",
#                 mode="lines+markers",
#                 name="Gas Produced (mcf)",
#                 x=index,
#                 y=gas,
#                 line=dict(shape="spline", smoothing=2, width=1, color="#fac1b7"),
#                 marker=dict(symbol="diamond-open"),
#             ),
#             dict(
#                 type="scatter",
#                 mode="lines+markers",
#                 name="Oil Produced (bbl)",
#                 x=index,
#                 y=oil,
#                 line=dict(shape="spline", smoothing=2, width=1, color="#a9bb95"),
#                 marker=dict(symbol="diamond-open"),
#             ),
#             dict(
#                 type="scatter",
#                 mode="lines+markers",
#                 name="Water Produced (bbl)",
#                 x=index,
#                 y=water,
#                 line=dict(shape="spline", smoothing=2, width=1, color="#92d8d8"),
#                 marker=dict(symbol="diamond-open"),
#             ),
#         ]
#         layout_individual["title"] = dataset[chosen[0]]["Well_Name"]

#     figure = dict(data=data, layout=layout_individual)
#     return figure


# # Selectors, main graph -> aggregate graph
# @app.callback(
#     Output("aggregate_graph", "figure"),
#     [
#         Input("STATUSES", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#         Input("low_range_graph", "hoverData"),
#     ],
# )
# def make_aggregate_figure(STATUSES, well_types, year_slider, low_range_graph_hover):

#     layout_aggregate = copy.deepcopy(layout)

#     if low_range_graph_hover is None:
#         low_range_graph_hover = {
#             "points": [
#                 {"curveNumber": 4, "pointNumber": 569, "customdata": 31101173130000}
#             ]
#         }

#     chosen = [point["customdata"] for point in low_range_graph_hover["points"]]
#     well_type = dataset[chosen[0]]["Well_Type"]
#     dff = filter_dataframe(df, STATUSES, well_types, year_slider)

#     selected = dff[dff["Well_Type"] == well_type]["API_WellNo"].values
#     index, gas, oil, water = produce_aggregate(selected, year_slider)

#     data = [
#         dict(
#             type="scatter",
#             mode="lines",
#             name="Gas Produced (mcf)",
#             x=index,
#             y=gas,
#             line=dict(shape="spline", smoothing="2", color="#F9ADA0"),
#         ),
#         dict(
#             type="scatter",
#             mode="lines",
#             name="Oil Produced (bbl)",
#             x=index,
#             y=oil,
#             line=dict(shape="spline", smoothing="2", color="#849E68"),
#         ),
#         dict(
#             type="scatter",
#             mode="lines",
#             name="Water Produced (bbl)",
#             x=index,
#             y=water,
#             line=dict(shape="spline", smoothing="2", color="#59C3C3"),
#         ),
#     ]
#     layout_aggregate["title"] = "Aggregate: " + WELL_TYPES[well_type]

#     figure = dict(data=data, layout=layout_aggregate)
#     return figure


# # Selectors, main graph -> pie graph
# @app.callback(
#     Output("pie_graph", "figure"),
#     [
#         Input("STATUSES", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#     ],
# )
# def make_pie_figure(STATUSES, well_types, year_slider):

#     layout_pie = copy.deepcopy(layout)

#     dff = filter_dataframe(df, STATUSES, well_types, year_slider)

#     selected = dff["API_WellNo"].values
#     index, gas, oil, water = produce_aggregate(selected, year_slider)

#     aggregate = dff.groupby(["Well_Type"]).count()

#     data = [
#         dict(
#             type="pie",
#             labels=["Gas", "Oil", "Water"],
#             values=[sum(gas), sum(oil), sum(water)],
#             name="Production Breakdown",
#             text=[
#                 "Total Gas Produced (mcf)",
#                 "Total Oil Produced (bbl)",
#                 "Total Water Produced (bbl)",
#             ],
#             hoverinfo="text+value+percent",
#             textinfo="label+percent+name",
#             hole=0.5,
#             marker=dict(colors=["#fac1b7", "#a9bb95", "#92d8d8"]),
#             domain={"x": [0, 0.45], "y": [0.2, 0.8]},
#         ),
#         dict(
#             type="pie",
#             labels=[WELL_TYPES[i] for i in aggregate.index],
#             values=aggregate["API_WellNo"],
#             name="Well Type Breakdown",
#             hoverinfo="label+text+value+percent",
#             textinfo="label+percent+name",
#             hole=0.5,
#             marker=dict(colors=[COLORS[i] for i in aggregate.index]),
#             domain={"x": [0.55, 1], "y": [0.2, 0.8]},
#         ),
#     ]
#     layout_pie["title"] = "Production Summary: {} to {}".format(
#         year_slider[0], year_slider[1]
#     )
#     layout_pie["font"] = dict(color="#777777")
#     layout_pie["legend"] = dict(
#         font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
#     )

#     figure = dict(data=data, layout=layout_pie)
#     return figure


# # Selectors -> count graph
# @app.callback(
#     Output("count_graph", "figure"),
#     [
#         Input("STATUSES", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#     ],
# )
# def make_count_figure(STATUSES, well_types, year_slider):

#     layout_count = copy.deepcopy(layout)

#     dff = filter_dataframe(df, STATUSES, well_types, [1960, 2017])
#     g = dff[["API_WellNo", "Date_Well_Completed"]]
#     g.index = g["Date_Well_Completed"]
#     g = g.resample("A").count()

#     colors = []
#     for i in range(1960, 2018):
#         if i >= int(year_slider[0]) and i < int(year_slider[1]):
#             colors.append("rgb(123, 199, 255)")
#         else:
#             colors.append("rgba(123, 199, 255, 0.2)")

#     data = [
#         dict(
#             type="scatter",
#             mode="markers",
#             x=g.index,
#             y=g["API_WellNo"] / 2,
#             name="All Wells",
#             opacity=0,
#             hoverinfo="skip",
#         ),
#         dict(
#             type="bar",
#             x=g.index,
#             y=g["API_WellNo"],
#             name="All Wells",
#             marker=dict(color=colors),
#         ),
#     ]

#     layout_count["title"] = "Completed Wells/Year"
#     layout_count["dragmode"] = "select"
#     layout_count["showlegend"] = False
#     layout_count["autosize"] = True

#     figure = dict(data=data, layout=layout_count)
#     return figure



@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        [df_soc, fnamepath] = upload.parse_contents(list_of_names) 
        return 
# Main
if __name__ == "__main__":

    #print("Data Types: " , df_soc.dtypes)
    #print("Index Name: ", df_soc.index.name)
    app.run_server(debug=True, threaded=False)
    