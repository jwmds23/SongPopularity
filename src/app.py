# python
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import altair as alt
import dash
import plotly.graph_objs as go
from dash import dcc, html, Input, Output,State
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helper import handle_select_all, create_feature_distribution_charts, pred_chart, track_radar, pop_predict
alt.data_transformers.disable_max_rows()
# alt.data_transformers.enable("vegafusion")

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "Spotify Song Popularity"

server = app.server
app.config.suppress_callback_exceptions = True

# set the background of all altair graphs to transparent
def transparent_bg():
    d = {
        'config': {
            'background':'#191A19',
            'view': {
                'height': 300,
                'width': 300,
            }
        }
    }
    return d

def grey_bg():
    d = {
        'config': {
            'background':'#888A87',
            'view': {
                'height': 300,
                'width': 300,
            }
        }
    }
    return d

alt.themes.register('transparent_bg', transparent_bg)
alt.themes.register('grey_bg', grey_bg)

# Read data
df = pd.read_csv('data/processed/spotify_songs_processed.csv', index_col=0)
df.dropna(inplace=True)
object_columns = df.select_dtypes(include=['object']).columns
for column in object_columns:
    df[column] = df[column].astype('string')
df['duration_min'] = df['duration_ms'] / 60000
feature_list=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'duration_min']
genre_list = df['playlist_genre'].unique().tolist()
subgenre_list = df['playlist_subgenre'].unique().tolist()
artist_list = df['track_artist'].unique().tolist()

# Format release date
def parse_date(x):
    try:
        if len(x)==10:
            return dt.strptime(x, "%Y-%m-%d")
        elif len(x)==7:
            return dt.strptime(x, "%Y-%m")
        elif len(x)==4:
            return dt.strptime(x, "%Y")
    except ValueError:
        return None

df["track_album_release_date"] = df["track_album_release_date"].apply(parse_date)

# Add decade column
def calculate_decade(date):
    if isinstance(date, pd.Timestamp):
        decade = 10 * (date.year // 10)
        return str(decade) + 's'
    else:
        return None

df["decade"] = df["track_album_release_date"].apply(calculate_decade)

def summary_description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card-1",
        children=[
            html.H3("Filter Menu"),
            html.Br(),    
            ],
    )

def feature_description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card-2",
        children=[
            html.H3("Filter Menu"),
            html.Br(),    
            ],
    )

tab1 =  html.Div(id='tab-1-content', children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    summary_description_card(),
                    html.P("Release Date"),
                    dcc.DatePickerRange(
                            id='date-picker-range-1',
                            start_date_placeholder_text='Start Date',
                            end_date_placeholder_text='End Date'
                        ),
                    html.Br(),
                    html.Br(),
                    html.P("Genre"),
                    dcc.Dropdown(
                            id='genre-dropdown-1',
                            options=[{'label': 'Select All', 'value': 'all'}] + [{'label': genre, 'value': genre} for genre in genre_list],
                            value=['all'],
                            multi=True,
                            style={'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
                        ),
                    html.Br(),
                    html.P("SubGenre"),
                    dcc.Dropdown(
                            id='subgenre-dropdown-1',
                            options=[{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list],
                            value=['all'],
                            multi=True,
                            style={'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
                        ),
                    html.Br(),
                    html.P("Artist"),
                    dcc.Dropdown(
                            id='artist-dropdown-1',
                            options=[{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list],
                            value=['all'],
                            multi=True,
                            style={'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
                        ),
                    html.Br(),
                    dbc.Row([
                                dbc.Col(html.Button('Apply', id='apply-button-1', n_clicks=0,
                                                    style={'border': '2px solid #1f1e1e',
                                                            'background-color': '#1f1e1e',
                                                            'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                                                            'border-radius': '10px','margin-right': '5px'}), width=3),
                                dbc.Col(html.Button('Reset', id='reset-button-1', n_clicks=0,
                                                    style={'border': '2px solid #1f1e1e',
                                                            'background-color': '#1f1e1e',
                                                            'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                                                            'border-radius': '10px'}), width=3)
                            ])
                    ])],width="4"
                ),
            dbc.Col([
                # The first plot column 
                html.Div(
                    id="popularity-level-distribution-chart",
                    children=[
                        html.H4("Popularity and Genre Distribution"),
                        html.Iframe(
                            id="popularity-level-distribution-chart-iframe",
                            style={'border-width': '0', 'width': '100%', 'height': '400px'},
                        ),
                    ],
                ),
                html.Div(
                    id="top-10-popularity-songs-artists-chart",
                    children=[
                        html.H4("Top 10 Popularity Songs & Artists"),
                        html.Iframe(
                            id="top-10-popularity-songs-artists-chart-iframe",
                            style={'border-width': '0', 'width': '100%', 'height': '1000px'},
                        ),
                    ],
                ),                    
                ],width="4"),
            dbc.Col([
                # The second plot column
                html.Div(
                    id="decade-trend-line-chart",
                    children=[
                        html.H4("Decade Trend Line Chart"),
                        html.Iframe(
                            id="decade-trend-line-chart-iframe",
                            style={'border-width': '0', 'width': '100%', 'height': '400px'},
                        ),
                    ],
                    ),
                html.Div(
                    id="feature_scatter-chart",
                    children=[
                        html.H4("Two-Feature Scatter Plot"),
                        "Release Year",
                        dbc.Row([
                            dbc.Col([
                                dcc.Slider(id='year-slider', 
                                    min=1957, max=2021, 
                                    value=2000,
                                    marks={1960: '1960', 1970: '1970', 1980: '1980', 1990: '1990', 2000: '2000', 2010: '2010'},
                                    updatemode='drag',
                                    step=1),
                                ], width=9),
                            dbc.Col([
                                html.Div(id='year-output')],
                                style={'margin-left': '-30px', 'margin-top': '-5px'},
                                width=1)]),
                        dbc.Row([
                            dbc.Col([
                                "Feature 1",
                                dcc.Dropdown(
                                        id='feature1-dropdown',
                                        options=[{'label': feature, 'value': feature} for feature in feature_list],
                                        value='danceability',
                                        multi=False,
                                        style={'width': '200px','backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'} 
                                    ),]),
                            dbc.Col([
                                "Feature 2",
                                dcc.Dropdown(
                                        id='feature2-dropdown',
                                        options=[{'label': feature, 'value': feature} for feature in feature_list],
                                        value='liveness',
                                        multi=False,
                                        style={'width': '200px','backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
                                    ),]),
                        ]),
                        html.Iframe(
                            id="feature_scatter-chart-iframe",
                            style={'border-width': '0', 'width': '100%', 'height': '800px'},
                        ),     
                    ],
                ),
                ],width="4")]
            )])

tab2 =  html.Div(id='tab-2-content', children=[
        dbc.Row([
            dbc.Col(
                html.Div([
                    feature_description_card(),
                    html.P("Release Date"),
                    dcc.DatePickerRange(
                        id='date-picker-range-2',
                        start_date_placeholder_text='Start Date',
                        end_date_placeholder_text='End Date',
                    ),
                    html.Br(),
                    html.Br(),
                    html.P("Genre"),
                    dcc.Dropdown(
                        id='genre-dropdown-2',
                        options=[{'label': 'Select All', 'value': 'all'}] + [{'label': genre, 'value': genre} for genre in genre_list],
                        value=['all'],  # Default value
                        multi=True,
                        style={'backgroundColor': 'black', 'color': '#16E536'},
                        className='custom-dropdown'
                    ),
                    html.Br(),
                    html.P("SubGenre"),
                    dcc.Dropdown(
                        id='subgenre-dropdown-2',
                        options=[{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list],
                        value=['all'],  # Default value
                        multi=True,
                        style={'backgroundColor': 'black', 'color': '#16E536'}
                    ),
                    html.Br(),
                    html.P("Artist"),
                    dcc.Dropdown(
                        id='artist-dropdown-2',
                        options=[{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list],
                        value=['all'],  # Default value
                        multi=True,
                        style={'backgroundColor': 'black', 'color': '#16E536'}
                    ),
                    html.Br(),
                    html.P("Feature"),
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=[{'label': 'Select All', 'value': 'all'}] + [{'label': feature, 'value': feature} for feature in feature_list],
                        value=['all'],  # Default value
                        multi=True,
                        style={'backgroundColor': 'black', 'color': '#16E536'}
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.Button('Apply', id='apply-button-2', n_clicks=0,
                                            style={'border': '2px solid #1f1e1e',
                                                    'background-color': '#1f1e1e',
                                                    'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                                                    'border-radius': '10px','margin-right': '5px'}), width=3),
                        dbc.Col(html.Button('Reset', id='reset-button-2', n_clicks=0,
                                            style={'border': '2px solid #1f1e1e',
                                                    'background-color': '#1f1e1e',
                                                    'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                                                    'border-radius': '10px'}), width=3)
                            ])
                ]),width="4",
            ),
            dbc.Col(
                [html.H3("Popularity Distribution"),
                    html.Div(id='feature-charts-container')
                ],width="8"
            )
        ])])

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
        ]),
        dbc.Col([
        dbc.Container([
            dbc.Col([
                html.H3('The Predicted Popularity'),
                dbc.Row([
                html.Iframe(id='pred-result',width=400,height=350)], style={'margin': '0 auto -1% 15%', 'margin-left': '23%'}),
                dbc.Row([
                dcc.Graph(id='pred-radar', style={'margin': '0 auto', 'width':'600px', 'height':'420px', 'margin-top': '-10%'})])
            ], style={'align':'center'})
        ])
    ])
])])

tabs =  dcc.Tabs([
        dcc.Tab(label='Summary', style={'backgroundColor': 'rgba(0,0,0,0)', 'border': '0px', 'fontSize': 24}, 
                selected_style={'backgroundColor': '#116307', 'border': '0px', 'color': '#F9FCF9', 'fontSize': 24},
                value='tab-1'
        ),
        dcc.Tab(label='Explore', style={'backgroundColor': 'rgba(0,0,0,0)', 'border': '0px', 'fontSize': 24, 'white-space': 'nowrap'}, 
                selected_style={'backgroundColor': '#116307', 'border': '0px', 'color': '#F9FCF9', 'fontSize': 24, 'white-space': 'nowrap'},
                value='tab-2'
        ),   
        dcc.Tab(label='Prediction', style={'backgroundColor': 'rgba(0,0,0,0)', 'border': '0px', 'fontSize': 24}, 
                selected_style={'backgroundColor': '#116307', 'border': '0px', 'color': '#F9FCF9', 'fontSize': 24},
                value='tab-3')
    ], id='tabs-label')

# Define the layout of the Dash app
app.layout = html.Div(style = {'backgroundColor': '#060606', 'color':'#16E536'}, children=[
    dbc.Row([
        dbc.Col([html.Img(src='assets/spotify.png', style={'width': '40px', 'height': '40px', 'background-color':'transparent', 'margin-top': '15px'})], width='auto'),
        dbc.Col([html.H2('Spotify Music Marketing', style={'fontSize': 36, 'textAlign': 'left', 'margin-top': '10px'})], width=True),
        dbc.Col(tabs, width='auto')]),
    html.Div([
        dbc.Row(html.Div(id='tabs-content'))],
        style={'background-color':'#191A19'})
])

# Define callback to reset all filters
@app.callback(
    [Output('date-picker-range-1', 'start_date'),
     Output('date-picker-range-1', 'end_date'),
     Output('genre-dropdown-1', 'value'),
     Output('subgenre-dropdown-1', 'value'),
     Output('artist-dropdown-1', 'value'),],
    [Input('reset-button-1', 'n_clicks')]
)
def reset_all_filters(n_clicks):
    start_date = None
    end_date = None
    genre_value = ['all']
    subgenre_value = ['all']
    artist_value = ['all']
    return start_date, end_date, genre_value,subgenre_value,artist_value

@app.callback(
    [Output('date-picker-range-2', 'start_date'),
     Output('date-picker-range-2', 'end_date'),
     Output('genre-dropdown-2', 'value'),
     Output('subgenre-dropdown-2', 'value'),
     Output('artist-dropdown-2', 'value'),
     Output('feature-dropdown','value'),],
    [Input('reset-button-2', 'n_clicks')]
)
def reset_all_filters(n_clicks):
    start_date = None
    end_date = None
    genre_value = ['all']
    subgenre_value = ['all']
    artist_value = ['all']
    feature_value = ['all']
    return start_date, end_date, genre_value,subgenre_value,artist_value,feature_value

@app.callback(
    [Output('genre', 'value'),
     Output('danceability', 'value'),
     Output('energy', 'value'),
     Output('key', 'value'),
     Output('loudness', 'value'),
     Output('mode', 'value'),
     Output('speechiness', 'value'),
     Output('acousticness', 'value'),
     Output('instrumentalness', 'value'),
     Output('liveness', 'value'),
     Output('valence', 'value'),
     Output('tempo', 'value'),
     Output('minutes', 'value'),
     Output('seconds', 'value')],
    [Input('reset-button-3', 'n_clicks')]
)
def reset_all_filters(n_clicks):
    genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, minutes, seconds = None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    return genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, minutes, seconds

# Define callback to update charts

def update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists):
    genres_list = [{'label': genre, 'value': genre} for genre in genre_list]
    subgenres_list = [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list]
    artists_list = [{'label': artist, 'value': artist} for artist in artist_list]
    # Handle 'Select All' selections for each dropdown
    selected_genres = handle_select_all(selected_genres, genres_list)
    selected_subgenres = handle_select_all(selected_subgenres, subgenres_list)
    selected_artists = handle_select_all(selected_artists, artists_list)

    selected_genres = [] if 'all' in selected_genres else selected_genres
    selected_subgenres = [] if 'all' in selected_subgenres else selected_subgenres
    selected_artists = [] if 'all' in selected_artists else selected_artists

    # Filter the DataFrame based on the selected genres, subgenres, and artists
    # For genres
    if selected_genres:
        filtered_df = df[df['playlist_genre'].isin(selected_genres)]
    else:
        filtered_df = df

    # For subgenres
    if selected_subgenres:
        filtered_df = filtered_df[filtered_df['playlist_subgenre'].isin(selected_subgenres)]

    # For artists
    if selected_artists:
        filtered_df = filtered_df[filtered_df['track_artist'].isin(selected_artists)]
    
    # For dates
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['track_album_release_date'] >= start_date) & (filtered_df['track_album_release_date'] <= end_date)]
    
    return filtered_df

@app.callback(
    Output('feature-charts-container', 'children'),
    [Input('apply-button-2', 'n_clicks')],
    [State('date-picker-range-2', 'start_date'),
     State('date-picker-range-2', 'end_date'),
     State('genre-dropdown-2', 'value'),
     State('subgenre-dropdown-2', 'value'),
     State('artist-dropdown-2', 'value'),
     State('feature-dropdown', 'value')]
)

def update_output(n_clicks, start_date, end_date, selected_genres, selected_subgenres, selected_artists, selected_features):
    features_list = [{'label': feature, 'value': feature} for feature in feature_list]
    selected_features = handle_select_all(selected_features, features_list)
    selected_features = feature_list if 'all' in selected_features else selected_features
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    # Create Altair charts based on the selected features
    chart_html = create_feature_distribution_charts(filtered_df, selected_features)
    return html.Iframe(srcDoc=chart_html, style={"width": "100%", "height": "800px"})


@app.callback(
    Output('decade-trend-line-chart-iframe', 'srcDoc'),
    [Input('apply-button-1', 'n_clicks')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]
)
def update_decade_trend_line(n_clicks, start_date, end_date, selected_genres, selected_subgenres, selected_artists):
    alt.themes.enable('grey_bg')
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    filtered_df_trend = filtered_df[['decade','track_popularity']].groupby('decade').mean().reset_index()
    chart=alt.Chart(filtered_df_trend).mark_line(color='darkgreen',strokeWidth=5).encode(
        x=alt.X('decade',type='ordinal',title=None
                ,axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y('mean(track_popularity)',scale=alt.Scale(zero=False),title='Average Popularity',
                axis=alt.Axis(labelColor='white', titleColor='white')),
        tooltip = [alt.Tooltip('decade', title='Decade'), alt.Tooltip('mean(track_popularity)', title='Average Popularity', format='.2f')]
    ).properties(height=300, width=400)
    return chart.to_html()


@app.callback(
    Output('popularity-level-distribution-chart-iframe', 'srcDoc'),
    [Input('apply-button-1', 'n_clicks')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]
)
def update_popularity_level_distribution(n_clicks, start_date, end_date, selected_genres, selected_subgenres, selected_artists):
    alt.themes.enable('grey_bg')
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    sel1 = alt.selection_point(fields = ['playlist_genre'])
    sel2 = alt.selection_point(fields = ['nominal_popularity'])
    filtered_df_popularity_level = filtered_df[['playlist_genre','nominal_popularity','track_id']].groupby(['playlist_genre','nominal_popularity']).count().reset_index()
    chart1=alt.Chart(filtered_df_popularity_level).mark_bar().encode(
        x=alt.X('nominal_popularity',type='ordinal',title=None,
                sort=['low','medium','high'],
                axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y('sum(track_id)',title='Count of Records',
                axis=alt.Axis(labelColor='white', titleColor='white')),
        color= alt.Color('nominal_popularity', legend=None),
        tooltip = [alt.Tooltip('nominal_popularity', title='Popularity'), alt.Tooltip('sum(track_id)', title='Count of Records')]
        ).properties(height=300,width=100).transform_filter(sel1).add_params(sel2)
    chart2=alt.Chart(filtered_df_popularity_level).mark_arc().encode(
            color=alt.Color('playlist_genre',legend=alt.Legend(title=None,labelColor='white')),
            theta='sum(track_id)',
            tooltip = [alt.Tooltip('playlist_genre', title='Genre'), alt.Tooltip('sum(track_id)', title='Count of Records')]
        ).properties(height=300,width=150).add_params(sel1).transform_filter(sel2)
    return (alt.hconcat(chart1, chart2).resolve_scale(color='independent')).to_html()


@app.callback(
    Output('top-10-popularity-songs-artists-chart-iframe', 'srcDoc'),
    [Input('apply-button-1', 'n_clicks')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]
)
def update_top_10_popularity_songs_artists(n_clicks, start_date, end_date,selected_genres, selected_subgenres, selected_artists):
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    df_new=filtered_df[['track_name','track_artist','track_popularity']].drop_duplicates()
    popularity_by_songs = df_new[['track_name','track_popularity']].groupby('track_name').mean('track_popularity').reset_index()
    top10_songs=popularity_by_songs.nlargest(10,"track_popularity")
    popularity_min=top10_songs['track_popularity'].min()-5
    popularity_by_artists = df_new[['track_artist','track_popularity']].groupby('track_artist').mean('track_popularity').reset_index()
    top10_artists=popularity_by_artists.nlargest(10,"track_popularity")
    artist_min=top10_artists['track_popularity'].min()-5
    chart1 = alt.Chart(top10_songs).mark_bar(clip=True,color='darkgreen').encode(
        x=alt.X("track_popularity",scale=alt.Scale(domain=[popularity_min,100]),title='Popularity',
                axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y("track_name", sort='-x',title=None,
                axis=alt.Axis(labelColor='white', titleColor='white')),
        tooltip = [alt.Tooltip('track_name', title='Track Name'), alt.Tooltip('track_popularity', title='Popularity')]
    ).properties(title= alt.Title('Top 10 Songs',color='white'
                                  ))
    chart2 = alt.Chart(top10_artists).mark_bar(clip=True,color='darkgreen').encode(
        x=alt.X("track_popularity",scale=alt.Scale(domain=[artist_min,100]),title='Average Popularity',
                axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y("track_artist", sort='-x',title=None,
                axis=alt.Axis(labelColor='white', titleColor='white')),
        tooltip = [alt.Tooltip('track_artist', title='Artist Name'), alt.Tooltip('track_popularity', title='Average Popularity')]
    ).properties(title= alt.Title('Top 10 Artists',color='white'
                                  ))
    return (chart1&chart2).to_html()


@app.callback(
    Output('feature_scatter-chart-iframe', 'srcDoc'),
    [Input('apply-button-1', 'n_clicks'),
     Input('year-slider','value'),
     Input('feature1-dropdown', 'value'),
     Input('feature2-dropdown', 'value')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]   
)
def update_feature_scatter(n_clicks, select_year, f1, f2, start_date, end_date, selected_genres, selected_subgenres, selected_artists):
    alt.themes.enable('grey_bg')
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    df_new = filtered_df[filtered_df['track_album_release_date'].dt.year == select_year][['track_name', 'track_artist', 'track_album_name', 'track_popularity', 'playlist_genre', 'nominal_popularity', f1, f2]]
    brush = alt.selection_interval()
    sel = alt.selection_point(fields=['playlist_genre'])
    graph = alt.Chart(df_new).mark_circle().encode(
        x = alt.X(f1,axis=alt.Axis(labelColor='white', titleColor='white')),
        y = alt.Y(f2,axis=alt.Axis(labelColor='white', titleColor='white')),
        color = alt.condition(
            brush,
            'nominal_popularity',
            alt.value('grey'),
            title='Popularity',
            legend=alt.Legend(title=None,labelColor='white')),
        tooltip = [alt.Tooltip('track_name',title='Track Name'), alt.Tooltip('track_artist',title='Artist'), alt.Tooltip('track_album_name',title='Album Name'), alt.Tooltip('track_popularity',title='Popularity')]
    ).properties(height=300).add_params(brush).transform_filter(sel)

    graph1 = alt.Chart(df_new).mark_bar().encode(
    x = alt.X('count()',axis=alt.Axis(labelColor='white', titleColor='white')),
    y = alt.Y('playlist_genre', title=None,axis=alt.Axis(labelColor='white', titleColor='white')),
    color = alt.Color('playlist_genre',legend=None),
    tooltip = [alt.Tooltip('playlist_genre',title='Genre'),alt.Tooltip('count()',title='Count of Records')]
    ).properties(height=100).add_params(
        sel
    ).transform_filter(
        brush
    )
    return (alt.vconcat(graph, graph1).resolve_scale(color='independent')).to_html()

@app.callback(
    Output('year-output', 'children'),
    Input('year-slider', 'value') 
)
def update_output(value):
    return value

@app.callback(
    Output('subgenre-dropdown-1', 'options'),
    [Input('genre-dropdown-1', 'value')]
)
def update_subgenre_options(selected_genres):
    if 'all' not in selected_genres:
        available_subgenres = df[df['playlist_genre'].isin(selected_genres)]['playlist_subgenre'].unique()
        options = [{'label': subgenre, 'value': subgenre} for subgenre in available_subgenres]
        return options
    else: 
        options = [{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list]
        return options
    

@app.callback(
    Output('subgenre-dropdown-2', 'options'),
    [Input('genre-dropdown-2', 'value')]
)
def update_subgenre_options(selected_genres):
    if 'all' not in selected_genres:
        available_subgenres = df[df['playlist_genre'].isin(selected_genres)]['playlist_subgenre'].unique()
        options = [{'label': subgenre, 'value': subgenre} for subgenre in available_subgenres]
        return options
    else: 
        options = [{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list]
        return options
    


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
        result = pred_chart(round(pop_predict(genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms), 0))
        radar = track_radar(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms)
        return result, radar
    else:
        return pred_chart(0), track_radar(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs-label', 'value')]
)
def render_tab_content(tab):
    if tab == 'tab-2':
        return tab2
    elif tab == 'tab-3':
        return tab3
    else:
        return tab1

if __name__ == "__main__":
    app.run_server(debug=True)