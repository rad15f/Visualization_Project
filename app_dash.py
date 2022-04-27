import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px
import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from datetime import datetime as dt
import os
# -- Import and clean data (importing csv into pandas)
# Reading columns
df = pd.read_csv('DataCoSupplyChainDataset.csv', sep=',', encoding='latin-1')
#Fixing Dates
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)']).dt.date
df = df.sort_values(by = 'order date (DateOrders)')
#Selecting Columns
df_col = df[['Type','Sales per customer', 'Delivery Status','order date (DateOrders)',
       'Late_delivery_risk',  'Category Name', 'Customer City','Customer Country', 'Customer Segment',
       'Customer State',  'Department Name','Market','Order City', 'Order Country','Order Item Discount','Order Item Product Price',
       'Order Item Quantity', 'Sales', 'Order Item Total','Order Profit Per Order', 'Order Region', 'Order State', 'Order Status',
       'Product Name', 'Product Price', 'Shipping Mode']]
df_col = df_col[df_col['Order Status'] != 'SUSPECTED_FRAUD']
df_col = df_col[~df_col['Customer State'].isin(["91732","95758"])]
#Checking for NA
df_col.isnull().sum(axis=0)
#Gettig only Numerical
df_col['Profit'] = df_col['Order Item Total'] - df_col['Product Price']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical = df_col.select_dtypes(include=numerics)
numerical.drop('Late_delivery_risk',axis=1,inplace= True)
col_numerical = ['Sales per customer','Order Item Discount',
       'Order Item Product Price', 'Sales','Order Item Total', 'Order Profit Per Order', 'Product Price']
col = numerical.columns
#SEPARATE DATES
df_col['year'] = pd.DatetimeIndex(df_col['order date (DateOrders)']).year
df_col['month'] = pd.DatetimeIndex(df_col['order date (DateOrders)']).month
df_col['order date (DateOrders)']= pd.to_datetime(df_col['order date (DateOrders)'])
# SPECIALY ROW FOR THIS
df_col2 = df_col.copy()
df_col2['ym'] = df_col2['order date (DateOrders)'].dt.strftime('%Y-%m')
df_col2['ym-date'] = df_col2['order date (DateOrders)'].dt.strftime('%Y-%m-%d')
df_col2.set_index('ym-date', inplace= True)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server

# ------------------------------------------------------------------------------
# App layout

app.layout = html.Div([
    html.H1("Warehouse Analysis with Interactive Dashboards by Ricardo Diaz", style={'text-align': 'center'}),
            html.Br(),
            dcc.Tabs( id = 'hw-questions',
                      children = [

                        dcc.Tab(label = 'Barplot & Pie Chart', value  = 'q1'),
                        dcc.Tab(label='Lineplot & Histogram', value='q2'),
                        dcc.Tab(label='Heatmap', value='q3'),
                        dcc.Tab(label='Countplot & Violin', value='q4'),
                        dcc.Tab(label='Scatterplot & Regression', value='q5'),
                        dcc.Tab(label='Boxplot and Distribution', value='q6'),

                      ], value = 'q1'),
            html.Div(id = 'layout')

])
#TAB1
app_1layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.Div([
            html.H1(children='Barplot'),

            html.Div(children='''
                Profit per Department by Market
            '''),

            dcc.Graph(
                id='graph1',
                figure={}
            ),
        ], className='six columns'),
        html.Div([
            html.H1(children='Piechart'),

            html.Div(children='''
                Sales % divided by Segment
            '''),

            dcc.Graph(
                id='graph2',
                figure={}
            ),
        ], className='six columns'),
    ], className='row'),
    html.Div(['Pick the Market name',
              dcc.Dropdown(id="market",
                           options=[
                               {"label": "LATAM", "value": 'LATAM'},
                               {"label": "Europe", "value": 'Europe'},
                               {"label": "Pacific Asia", "value": 'Pacific Asia'},
                               {"label": "USCA", "value": 'USCA'},
                               {"label": "Africa", "value": 'Africa'},],
                           multi=False,
                           value='LATAM',
                           style={'width': "40%"}
                           )
              ]),
    html.Div(['Pick the year',
              dcc.Slider(id='year', min=2015, max=2018,value=2015,step=3,
               marks={2015: '2015', 2016: '2016', 2017: '2017', 2018: '2018'})
                 ]),
    html.Br(),
    html.Div(id='container', children=[]),
])
#TAB2

app_2layout = html.Div([
    dcc.DatePickerRange(
        id='my-date-picker-range',  # ID to be used for callback
        calendar_orientation='horizontal',  # vertical or horizontal
        day_size=39,  # size of calendar image. Default is 39
        end_date_placeholder_text="Return",  # text that appears when no end date chosen
        with_portal=False,  # if True calendar will open in a full screen overlay portal
        first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
        reopen_calendar_on_clear=True,
        is_RTL=False,  # True or False for direction of calendar
        clearable=True,  # whether or not the user can clear the dropdown
        number_of_months_shown=1,  # number of months shown when calendar is open
        min_date_allowed=dt(2015, 1, 1),  # minimum date allowed on the DatePickerRange component
        max_date_allowed=dt(2018, 1, 31),  # maximum date allowed on the DatePickerRange component
        initial_visible_month=dt(2015, 1, 1),  # the month initially presented when the user opens the calendar
        start_date=dt(2015, 1, 1).date(),
        end_date=dt(2016, 1, 1).date(),
        display_format='MMM Do, YY',  # how selected dates are displayed in the DatePickerRange component.
        month_format='MMMM, YYYY',  # how calendar headers are displayed when the calendar is opened.
        minimum_nights=10,  # minimum number of days between start and end date
        persistence=True,
        persisted_props=['start_date'],
        persistence_type='session',  # session, local, or memory. Default is 'local'

        updatemode='singledate'  # singledate or bothdates. Determines when callback is triggered
    ),
        html.H3("Profit vs Time Lineplot Analysis", style={'textAlign': 'center'}),
        dcc.Graph(id='line'),
        html.Br(),
        html.H3("Histogram Profit Analysis per Segment", style={'textAlign': 'center'}),
        dcc.Graph(id='hist'),
])

#TAB3

app_3layout = html.Div([
        html.Div([
              html.H1("Heatmap", style={'textAlign': 'center'}),
        ]),
        html.Div([
            dcc.Checklist(
                id='my_checklist',                      # used to identify component in callback
                options=[
                         {'label': x, 'value': x, 'disabled':False}
                         for x in col
                ],
                value=['Sales per customer', 'Order Item Discount', 'Order Item Product Price',
       'Order Item Quantity', 'Sales', 'Order Item Total',
       'Order Profit Per Order', 'Product Price', 'Profit'],    # values chosen by default
                className='my_box_container',           # class of the container (div)
                inputClassName='my_box_input',          # class of the <input> checkbox element
                labelClassName='my_box_label',          # class of the <label> that wraps the checkbox input and the option's label
                inline = True
            ),
        ]),
        html.Div([
            dcc.Graph(id='the_graph')
    ]),
])

#TAB4
app_4layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.Div([
            html.H1(children='Countplot'),

            html.Div(children='''
                Customer Size by State
            '''),

            dcc.Graph(
                id='countplot',
                figure={}
            ),
        ], className='six columns'),
        html.Div([
            html.H1(children='Violin Plot'),

            html.Div(children='''
                Statistically Representation of Sales by Segment
            '''),

            dcc.Graph(
                id='violinplot',
                figure={}
            ),
        ], className='six columns'),
    ], className='row'),
       html.Div([
              dcc.Checklist(
                     id='segment',  # used to identify component in callback
                     options=[
                            {'label': x, 'value': x, 'disabled': False}
                            for x in df_col['Customer Segment'].unique()
                     ],
                     value=['Corporate', 'Consumer','Home Office'],  # values chosen by default
                     className='my_box_container',  # class of the container (div)
                     inputClassName='my_box_input',  # class of the <input> checkbox element
                     labelClassName='my_box_label',
                     # class of the <label> that wraps the checkbox input and the option's label
                     inline=True
              ),
       ]),
       html.H5('Compare Customers from 2 US States'),
       html.Div([
              "Please enter 1 US STATE Abbreviations:",
              dcc.Input(id='state', value='FL', type='text')
       ]),
       html.Div([
              "Please enter 2 US STATE Abbreviations:",
              dcc.Input(id='state2', value='CA', type='text')
       ]),
])

#TAB5
app_5layout = html.Div([
        html.Div([
              html.H1("Regresion and Scatter Plot", style={'textAlign': 'center'}),
        ]),
        html.H5('Select your first feature:'),
        html.Div([
            dcc.RadioItems(
                id='radio1',                      # used to identify component in callback
                options=[
                         {'label': x, 'value': x, 'disabled':False}
                         for x in col
                ],
                className='my_box_container',           # class of the container (div)
                inputClassName='my_box_input',          # class of the <input> checkbox element
                labelClassName='my_box_label',          # class of the <label> that wraps the checkbox input and the option's label
                inline = True
            ),
        ]),
        html.H5('Select your second feature:'),
        html.Div([
            dcc.RadioItems(
                id='radio2',  # used to identify component in callback
                options=[
                    {'label': x, 'value': x, 'disabled': False}
                    for x in col
                ],
                className='my_box_container',  # class of the container (div)
                inputClassName='my_box_input',  # class of the <input> checkbox element
                labelClassName='my_box_label',  # class of the <label> that wraps the checkbox input and the option's label
                inline=True
            ),
        ]),
        html.Div([
            dcc.Graph(id='regression')
    ]),
])

#TAB6

app_6layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.Div([
            html.H1(children='Distribuiton Plot'),

            html.Div(children='''
                Distribution of the selected feature per year
            '''),

            dcc.Graph(
                id='boxplot',
                figure={}
            ),
        ], className='six columns'),
        html.Div([
            html.H1(children='Boxplot'),

            html.Div(children='''
                Boxplot of user chosen feature per year
            '''),

            dcc.Graph(
                id='distribution',
                figure={}
            ),
        ], className='six columns'),
    ], className='row'),
    html.Div([
        dcc.Checklist(
            id='year_checklist',  # used to identify component in callback
            options=[
                {'label': x, 'value': x, 'disabled': False}
                for x in df_col['year'].unique()
            ],
            value=[2015,2016,2017,2018],  # values chosen by default
            className='my_box_container',  # class of the container (div)
            inputClassName='my_box_input',  # class of the <input> checkbox element
            labelClassName='my_box_label',  # class of the <label> that wraps the checkbox input and the option's label
            inline=True
        ),
    ]),
    html.Br(),
    html.Div(id='container', children=[]),
    html.Div(['Pick the numerical feature for the histogram and boxplot:',
              dcc.Dropdown(id="seg",
                           options=[
                               {"label": "Sales per customer", "value": 'Sales per customer'},
                               {"label": "Order Item Product Price", "value": 'Order Item Product Price'},
                               {"label": "Sales", "value": 'Sales'},
                               {"label": "Order Item Total", "value": 'Order Item Total'},
                               {"label": "Order Profit Per Order", "value": 'Order Profit Per Order'},
                               {"label": "Product Price", "value": 'Product Price'},
                           ],
                           multi=False,
                           value='Sales per customer',
                           style={'width': "40%"}
                           )
              ]),
])



@app.callback(
    Output(component_id= 'layout', component_property='children'),
    [Input(component_id='hw-questions', component_property='value')]
)

def update_layout(ques):
    if ques == 'q1':
        return app_1layout
    elif ques == 'q2':
        return app_2layout
    elif ques == 'q3':
        return app_3layout
    elif ques == 'q4':
        return app_4layout
    elif ques == 'q5':
        return app_5layout
    elif ques == 'q6':
        return app_6layout

#TAB1
@app.callback(
    [dash.dependencies.Output(component_id='graph1', component_property='figure'),
    dash.dependencies.Output(component_id='graph2', component_property='figure'),
    dash.dependencies.Output(component_id='container', component_property='children')],
    [dash.dependencies.Input(component_id='market', component_property='value'),
     dash.dependencies.Input(component_id='year', component_property='value')]
)

def display_color(market,year):
    df = df_col[(df_col['year'] == year) & (df_col['Market'] == market)]
    container = "The Market chosen by user was: {}".format(market)
    fig = px.bar(df, x = 'Department Name' , y = 'Profit', color = 'Department Name')
    fig.update_layout({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
    })
    fig2 = px.pie(df, values = 'Sales', names = 'Customer Segment')

    return fig,fig2,container

#TAB2
@app.callback(
    dash.dependencies.Output('line', 'figure'),
    dash.dependencies.Output('hist', 'figure'),
    [dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date')]
)
def update_output(start_date, end_date):
    dff = df_col2.loc[start_date:end_date]
    dff2 = dff.groupby(['ym'], as_index = 'False').sum()
    fig = px.line(dff2, y = 'Profit', x = dff2.index)
    fig2 = px.histogram(dff, x = 'Profit', color = 'Customer Segment', nbins =40)
    return fig,fig2

#TAB3
@app.callback(
    dash.dependencies.Output(component_id='the_graph', component_property='figure'),
    [dash.dependencies.Input(component_id='my_checklist', component_property='value')]
)

def update_graph(options_chosen):
    fig = px.imshow(numerical[options_chosen].corr().round(2), text_auto=True, aspect = 'auto')
    return fig

#TAB4
@app.callback(
    [dash.dependencies.Output(component_id='countplot', component_property='figure'),
    dash.dependencies.Output(component_id='violinplot', component_property='figure'),],
    [dash.dependencies.Input(component_id='segment', component_property='value'),
     dash.dependencies.Input(component_id='state', component_property='value'),
     dash.dependencies.Input(component_id='state2', component_property='value')]
)

def display_color(segment,state,state2):
    dff = df_col[df_col['Customer Segment'].isin(segment)].copy()
    dff = dff[(dff['Customer State'] == state) | (dff['Customer State'] == state2)]
    bar = dff.groupby('Customer State', as_index=False).size()
    #COUNTPLOT
    fig = px.bar(bar, x = 'Customer State' , y = 'size')
    #VIOLIN PLOT
    fig2 = px.violin(dff, y = 'Sales', color = 'Customer Segment', box = True, points = 'all')
    return fig,fig2

#TAB5
@app.callback(
    dash.dependencies.Output(component_id='regression', component_property='figure'),
    [dash.dependencies.Input(component_id='radio1', component_property='value'),
     dash.dependencies.Input(component_id='radio2', component_property='value')]
)
def update_graph(x,y):
    print(x)
    print(type(x))
    fig = px.scatter(numerical, x= x, y= y, trendline = 'ols')
    return fig


@app.callback(
    dash.dependencies.Output(component_id='boxplot', component_property='figure'),
    dash.dependencies.Output(component_id='distribution', component_property='figure'),
    [dash.dependencies.Input(component_id='year_checklist', component_property='value'),
     dash.dependencies.Input(component_id='seg', component_property='value')]
)

def display_color(y_check,seg):
    dfh = df_col[(df_col['year'].isin(y_check))]
    fig = px.histogram(dfh, x = seg , color = 'year', nbins =80,barmode = 'overlay')
    fig2 = px.box(dfh, x = 'year', y = seg)
    return fig,fig2


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, host = "0.0.0.0",port = 8080)

