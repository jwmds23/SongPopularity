import altair as alt
import pandas as pd
import numpy as np
import os
import plotly.graph_objs as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import predict_func

# load the data set
df = pd.read_csv('../data/processed/spotify_songs_processed.csv', parse_dates = ['track_album_release_date'], index_col=0)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dcc.Tab(label='Prediction', children=[
    dbc.Row([
        dbc.Col([
        dbc.Container([
            html.H3('Set Features for Your Song'),
            dbc.Row([
                dbc.Col([html.Img(src='assets/dance.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Danceability', style={'margin-left': '-130px'})]),
                dbc.Col([html.Img(src='assets/energy.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Energy', style={'margin-left': '-130px'})])
            ], style={'border-top': '3px solid lightgrey', 'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px', 'border-top-left-radius':'10px', 'border-top-right-radius':'10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='danceability',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'}),
                ], width=5),
                dbc.Col([
                    html.Div(id='danceability-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='energy',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )],  style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                ]),
                dbc.Col([
                    html.Div(id='energy-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/speech.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Speechiness', style={'margin-left': '-130px'})]),
                dbc.Col([html.Img(src='assets/acoustic.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Acousticness', style={'margin-left': '-130px'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='speechiness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                ], width=5),
                dbc.Col([
                    html.Div(id='speechiness-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='acousticness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                ]),
                dbc.Col([
                    html.Div(id='acousticness-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/instrumental.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Instrumentalness', style={'margin-left': '-130px'})]),
                dbc.Col([html.Img(src='assets/liveness.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Liveness', style={'margin-left': '-130px'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='instrumentalness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                    ], width=5),
                dbc.Col([
                    html.Div(id='instrumentalness-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                dbc.Col([
                        html.Div([
                            dcc.Slider(
                                id='liveness',
                                min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                updatemode='drag'
                            )], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                    ]),
                dbc.Col([
                    html.Div(id='liveness-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                html.Br()
                ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/valence.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Valence', style={'margin-left': '-130px'})]),
                dbc.Col([html.Img(src='assets/loudness.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Loudness', style={'margin-left': '-130px'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='valence',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag')
                    ], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                ], width=5),
                dbc.Col([
                    html.Div(id='valence-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='loudness',
                            min=-47, max=1.5, value=-47, marks={-47: '-47', -40: '-40', -30: '-30', -20: '-20', -10: '-10', 1.5: '1.5'},
                            updatemode='drag'
                        )], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                ]),
                dbc.Col([
                    html.Div(id='loudness-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/tempo.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Tempo', style={'margin-left': '-130px'})]),
                dbc.Col([html.Img(src='assets/mode.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Mode', style={'margin-left': '-130px'})])
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([
                        html.Div([
                            dcc.Slider(
                                id='tempo',
                                min=0, max=240, value=0, marks={0: '0', 30: '30', 60: '60', 90: '90', 120: '120', 150: '150', 180: '180', 210: '210', 240: '240'},
                                updatemode='drag'
                            )], style={'margin-left': '-15px', 'margin-top': '-15px', 'width': '120%'})
                    ], width=5),
                dbc.Col([
                    html.Div(id='tempo-output')],
                    style={'margin-left': '-5px', 'margin-top': '-20px'},
                    width=1),
                dbc.Col([
                    html.Div([  
                        dcc.Dropdown(
                            id='mode',
                            options=[{'label': '0', 'value': 0}, {'label': '1', 'value': 1}],
                            multi=False,
                            clearable=True,
                            placeholder='Please select the mode...'
                        )], style={'margin-top': '-20px', 'width': '100%'}
                    )
                ]),
                html.Br()
            ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
            dbc.Row([
                dbc.Col([html.Img(src='assets/genre.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Genre', style={'margin-left': '-130px'})]),
                dbc.Col([html.Img(src='assets/key.png', style={'width': '25px', 'height': '25px', 'background-color':'transparent'})]),
                dbc.Col([html.Label('Key', style={'margin-left': '-130px'})])
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
                            placeholder='Please select the genre of the song...')
                    ], style={'margin-top': '-20px', 'width': '100%'})
                ]),
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='key',
                            options=[{'label': str(i), 'value': i} for i in range(12)],
                            multi=False,
                            clearable=True,
                            placeholder='Please select the key...'
                    )], style={'margin-top': '-20px', 'margin-left': '-5px', 'width': '102%'})
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
                            style={'border': '2px solid lightgrey', 'border-radius': '10px'})
                        ])
                ]),
                dbc.Col([html.Label('minutes')], style={'margin-left': '-10px'}),
                dbc.Col([
                    html.Div([
                        dcc.Input(
                            id='seconds',
                            type='number',
                            placeholder='Enter seconds...',
                            min=0,
                            max=59,
                            style={'border': '2px solid lightgrey', 'border-radius': '10px'})
                    ])
                ]),
                dbc.Col([html.Label('seconds')], style={'margin-left': '-10px'})
            ],style={'border-bottom': '3px solid lightgrey', 'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px', 'border-bottom-left-radius':'10px', 'border-bottom-right-radius':'10px'}),
            html.Br(),
            dbc.Row([
                dbc.Col([
                html.Button('Apply', id='apply', n_clicks=0, style={'border': '2px solid white', 'background-color': 'white', 'color': 'black', 'fontweight': 'bold', 'border-radius': '10px'})])]),
                dbc.Col([])
        ])
        ]),
        dbc.Col([
        dbc.Container([
            dbc.Col([
                html.H3('The Predicted Popularity'),
                dbc.Row([
                html.Iframe(id='pred-result',width=400,height=350)], style={'margin': '0 auto -70px 140px'}),
                dbc.Row([
                dcc.Graph(id='pred-radar', style={'margin': '0 auto', 'width':'600px', 'height':'420px'})])
            ], style={'align':'center'})
        ])
        ])
    ])
])

@app.callback(
    Output('danceability-output', 'children'),
    Input('danceability', 'value') 
)
def update_output(value):
    return round(value, 2)

@app.callback(
    Output('energy-output', 'children'),
    Input('energy', 'value') 
)
def update_output(value):
    return round(value, 2)

@app.callback(
    Output('speechiness-output', 'children'),
    Input('speechiness', 'value') 
)
def update_output(value):
    return round(value, 2)

@app.callback(
    Output('acousticness-output', 'children'),
    Input('acousticness', 'value') 
)
def update_output(value):
    return round(value, 2)

@app.callback(
    Output('instrumentalness-output', 'children'),
    Input('instrumentalness', 'value') 
)
def update_output(value):
    return round(value, 2)

@app.callback(
    Output('liveness-output', 'children'),
    Input('liveness', 'value') 
)
def update_output(value):
    return round(value, 2)

@app.callback(
    Output('valence-output', 'children'),
    Input('valence', 'value') 
)
def update_output(value):
    return round(value, 2)

@app.callback(
    Output('loudness-output', 'children'),
    Input('loudness', 'value') 
)
def update_output(value):
    return round(value, 1)

@app.callback(
    Output('tempo-output', 'children'),
    Input('tempo', 'value') 
)
def update_output(value):
    return round(value, 0)

@app.callback(
    [Output('pred-result', 'srcDoc'),
     Output('pred-radar', 'figure')],
    [Input('apply', 'n_clicks')],
    [State('genre', 'value'),
     State('danceability', 'value'),
     State('energy', 'value'),
     State('key', 'value'),
     State('loudness', 'value'),
     State('mode', 'value'),
     State('speechiness', 'value'),
     State('acousticness', 'value'),
     State('instrumentalness', 'value'),
     State('liveness', 'value'),
     State('valence', 'value'),
     State('tempo', 'value'),
     State('minutes', 'value'),
     State('seconds', 'value')
    ]
)
def update_output(n_clicks, genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, minutes, seconds):
    if n_clicks > 0:
        duration_ms = minutes * 60000 + seconds * 1000   # transfer the time to miliseconds
        result = predict_func.pred_chart(round(predict_func.pop_predict(genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms), 0))
        radar = predict_func.track_radar(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms)
        return result, radar
    else:
        return predict_func.pred_chart(0), predict_func.track_radar(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)