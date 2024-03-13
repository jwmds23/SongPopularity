import altair as alt
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from helper import handle_select_all

tab3 =  html.Div(id='tab-3-content', children=[
        dbc.Row([
        dbc.Col([
        dbc.Container([
            html.H3('Set Features for Your Song'),
            dbc.Row([
                dbc.Col([html.Img(src='assets/dance.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Danceability', style={'margin-left': '-90%', 'margin-top': '1%'})]),
                dbc.Col([html.Img(src='assets/energy.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Energy', style={'margin-left': '-90%', 'margin-top': '1%'})])
            ], style={'border-top': '3px solid lightgrey', 'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px', 'border-top-left-radius':'10px', 'border-top-right-radius':'10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='danceability',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-5.2%', 'margin-top': '-5.8%', 'width': '120%'}),
                ], width=5),
                dbc.Col([
                    html.Div(id='danceability-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='energy',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )],  style={'margin-left': '-5.5%', 'margin-top': '-5.8%', 'width': '120%'})
                ]),
                dbc.Col([
                    html.Div(id='energy-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/speech.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Speechiness', style={'margin-left': '-90%', 'margin-top': '1%'})]),
                dbc.Col([html.Img(src='assets/acoustic.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Acousticness', style={'margin-left': '-90%', 'margin-top': '1%'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='speechiness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-5.2%', 'margin-top': '-5.8%', 'width': '120%'})
                ], width=5),
                dbc.Col([
                    html.Div(id='speechiness-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='acousticness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-5.5%', 'margin-top': '-5.8%', 'width': '120%'})
                ]),
                dbc.Col([
                    html.Div(id='acousticness-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/instrumental.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Instrumentalness', style={'margin-left': '-90%', 'margin-top': '1%'})]),
                dbc.Col([html.Img(src='assets/liveness.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Liveness', style={'margin-left': '-90%', 'margin-top': '1%'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='instrumentalness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-5.2%', 'margin-top': '-5.8%', 'width': '120%'})
                    ], width=5),
                dbc.Col([
                    html.Div(id='instrumentalness-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                dbc.Col([
                        html.Div([
                            dcc.Slider(
                                id='liveness',
                                min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                updatemode='drag'
                            )], style={'margin-left': '-5.5%', 'margin-top': '-5.8%', 'width': '120%'})
                    ]),
                dbc.Col([
                    html.Div(id='liveness-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                html.Br()
                ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/valence.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Valence', style={'margin-left': '-90%', 'margin-top': '1%'})]),
                dbc.Col([html.Img(src='assets/loudness.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Loudness', style={'margin-left': '-90%', 'margin-top': '1%'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='valence',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag')
                    ], style={'margin-left': '-5.2%', 'margin-top': '-5.8%', 'width': '120%'})
                ], width=5),
                dbc.Col([
                    html.Div(id='valence-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='loudness',
                            min=-47, max=1.5, value=-47, marks={-47: '-47', -40: '-40', -30: '-30', -20: '-20', -10: '-10', 1.5: '1.5'},
                            updatemode='drag'
                        )], style={'margin-left': '-5.5%', 'margin-top': '-5.8%', 'width': '120%'})
                ]),
                dbc.Col([
                    html.Div(id='loudness-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/tempo.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Tempo', style={'margin-left': '-90%', 'margin-top': '1%'})]),
                dbc.Col([html.Img(src='assets/mode.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Mode', style={'margin-left': '-90%', 'margin-top': '1%'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                        html.Div([
                            dcc.Slider(
                                id='tempo',
                                min=0, max=240, value=0, marks={0: '0', 30: '30', 60: '60', 90: '90', 120: '120', 150: '150', 180: '180', 210: '210', 240: '240'},
                                updatemode='drag'
                            )], style={'margin-left': '-5.2%', 'margin-top': '-5.8%', 'width': '120%'})
                    ], width=5),
                dbc.Col([
                    html.Div(id='tempo-output')],
                    style={'margin-top': '-3.2%', 'margin-left': '-0.8%'},
                    width=1),
                dbc.Col([
                    html.Div([  
                        dcc.Dropdown(
                            id='mode',
                            options=[{'label': '0', 'value': 0}, {'label': '1', 'value': 1}],
                            multi=False,
                            clearable=True,
                            style = {'color': '#16E536'}
                        )], style={'margin-top': '-6.5%', 'margin-left': '0%', 'width': '101%'}
                    )
                ]),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/genre.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Genre', style={'margin-left': '-90%', 'margin-top': '1%'})]),
                dbc.Col([html.Img(src='assets/key.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Key', style={'margin-left': '-90%', 'margin-top': '1%'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='genre',
                            options=[
                                {'label': 'EDM', 'value': 'edm'},
                                {'label': 'Latin', 'value': 'latin'},
                                {'label': 'Pop', 'value': 'pop'},
                                {'label': 'R&B', 'value': 'r&b'},
                                {'label': 'Rap', 'value': 'rap'},
                                {'label': 'Rock', 'value': 'rock'}
                            ],
                            multi=False,
                            clearable=True,
                            searchable=True,
                            style={'color': '#16E536'})
                    ], style={'margin-top': '-6.5%', 'width': '100%'})
                ]),
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='key',
                            options=[{'label': str(i), 'value': i} for i in range(12)],
                            multi=False,
                            clearable=True,
                            style={'color': '#16E536'}
                    )], style={'margin-top': '-6.5%', 'margin-left': '-1%', 'width': '102%'})
                ]),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                html.Label('Duration'),
                dbc.Col([
                    html.Div([
                        dcc.Input(
                            id='minutes',
                            type='number',
                            placeholder='Enter minutes...',
                            min=0,
                            style={'border': '2px solid rgb(26, 24, 24)', 'border-radius': '10px', 'background-color': 'rgb(26, 24, 24)', 'color': '#16E536'})
                        ])
                ]),
                dbc.Col([html.Label('minutes')], style={'margin-left': '-1.5%'}),
                dbc.Col([
                    html.Div([
                        dcc.Input(
                            id='seconds',
                            type='number',
                            placeholder='Enter seconds...',
                            min=0,
                            max=59,
                            style={'border': '2px solid rgb(26, 24, 24)', 'border-radius': '10px', 'background-color': 'rgb(26, 24, 24)', 'color': '#16E536'})
                    ])
                ]),
                dbc.Col([html.Label('seconds')], style={'margin-left': '-1.5%', 'color': '#16E536'})
            ],style={'border-bottom': '3px solid lightgrey', 'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px', 'border-bottom-left-radius':'10px', 'border-bottom-right-radius':'10px'}),
            html.Br(),
            dbc.Row([
                dbc.Col([
                html.Button('Apply', id='apply', n_clicks=0, style={'border': '2px solid #1f1e1e', 'background-color': '#1f1e1e', 'color': 'rgb(4, 184, 4)', 'fontweight': 'bold', 'border-radius': '10px'})]),
                dbc.Col([
                    html.Button('Reset', id='reset-button-3', n_clicks=0,
                                style={'border': '2px solid #1f1e1e',
                                        'background-color': '#1f1e1e',
                                        'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                                        'border-radius': '10px',
                                        'align': 'left'})], width=3),
                dbc.Col([]),
                dbc.Col([])]),
        ])
        ], width='6'),
        dbc.Col([
        dbc.Container([
            dbc.Col([
                html.H3('The Predicted Popularity'),
                dbc.Row([
                html.Iframe(id='pred-result',width=400,height=350)], style={'margin': '0 auto -1% 15%', 'margin-left': '23%'}),
                dbc.Row([
                dcc.Graph(id='pred-radar', style={'margin': '0 auto', 'width':'600px', 'height':'420px', 'margin-top': '-10%'})])
            ], style={'align':'center'}, width='12')
        ])
    ])
])])