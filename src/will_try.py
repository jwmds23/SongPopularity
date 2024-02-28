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
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Danceability'),
                        dcc.Slider(
                            id='danceability',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                            updatemode='drag'
                        )], style={'width': '100%'})
                ]),
                dbc.Col([
                    html.Div([
                        html.Label('Energy'),
                        dcc.Slider(
                            id='energy',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
                        )], style={'width': '100%'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Speechiness'),
                        dcc.Slider(
                            id='speechiness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
                        )], style={'width': '100%'}
                    )
                ]),
                dbc.Col([
                    html.Div([
                        html.Label('Acousticness'),
                        dcc.Slider(
                            id='acousticness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
                        )], style={'width': '100%'}
                    )
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Instrumentalness'),
                        dcc.Slider(
                            id='instrumentalness',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
                        )], style={'width': '100%'})
                    ]),
                dbc.Col([
                        html.Div([
                            html.Label('Liveness'),
                            dcc.Slider(
                                id='liveness',
                                min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
                            )], style={'width': '100%'}
                        )
                    ])
                ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Valence'),
                        dcc.Slider(
                            id='valence',
                            min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'})
                    ], style={'width': '100%'})
                ]),
                dbc.Col([
                    html.Div([
                        html.Label('Loudness'),
                        dcc.Slider(
                            id='loudness',
                            min=-47, max=1.5, value=-47, marks={-47: '-47', -40: '-40', -30: '-30', -20: '-20', -10: '-10', 1.5: '1.5'}
                        )], style={'width': '100%'}
                    )
                ])
            ]),
            dbc.Row([
                dbc.Col([
                        html.Div([
                            html.Label('Tempo'),
                            dcc.Slider(
                                id='tempo',
                                min=0, max=240, value=0, marks={0: '0', 30: '30', 60: '60', 90: '90', 120: '120', 150: '150', 180: '180', 210: '210', 240: '240'}
                            )], style={'width': '100%'}
                        )
                    ]),
                dbc.Col([
                    html.Div([  
                        html.Label('Mode'),
                        dcc.Dropdown(
                            id='mode',
                            options=[{'label': '0', 'value': 0}, {'label': '1', 'value': 1}],
                            multi=False,
                            clearable=True,
                            placeholder='Please select the mode...'
                        )], style={'width': '100%'}
                    )
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Genre'),
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
                    ], style={'width': '100%'})
                ]),
                dbc.Col([
                    html.Div([
                        html.Label('Key'),
                        dcc.Dropdown(
                            id='key',
                            options=[{'label': str(i), 'value': i} for i in range(12)],
                            multi=False,
                            clearable=True,
                            placeholder='Please select the key...'
                    )], style={'width': '100%'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label('Minutes:'),
                        dcc.Input(
                            id='minutes',
                            type='number',
                            placeholder='Enter minutes...',
                            min=0)
                        ])
                ]),
                dbc.Col([
                    html.Div([
                        html.Label('Seconds:'),
                        dcc.Input(
                            id='seconds',
                            type='number',
                            placeholder='Enter seconds...',
                            min=0,
                            max=59)
                    ])
                ])
            ]),
            dbc.Row([
                html.Button('Apply', id='apply', n_clicks=0)])
        ])
        ]),
        dbc.Col([
        dbc.Container([
            dbc.Col([
                html.Div(id='pred-result'),
                dcc.Graph(id='pred-radar')
            ])
        ])
        ])
    ])
])

@app.callback(
    [Output('pred-result', 'children'),
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
        duration_ms = minutes * 6e+5 + seconds * 1000   # transfer the time to miliseconds
        result = round(predict_func.pop_predict(genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms), 0)
        radar = predict_func.track_radar(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms)
        return result, radar
    else:
        return '', predict_func.track_radar(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)