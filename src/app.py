# python
import os
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
from joblib import load
#from .predict_func import handle_select_all, create_feature_distribution_charts, pred_chart, track_radar, pop_predict
alt.data_transformers.disable_max_rows()
# alt.data_transformers.enable("vegafusion")

# Construct the absolute path for the CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'spotify_songs_processed.csv')

# Load the CSV file

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
    #  external_stylesheets=["assets/sb-admin-2.css"]
)
app.title = "Spotify Song Popularity"

server = app.server
app.config.suppress_callback_exceptions = True

# set the background of all altair graphs to transparent
def transparent_bg():
    d = {
        'config': {
            'background':'none',
            'view': {
                'height': 300,
                'width': 300,
            }
        }
    }
    return d

alt.themes.register('transparent_bg', transparent_bg)
alt.themes.enable('transparent_bg')


# Read data
#df = pd.read_csv('../data/processed/spotify_songs_processed.csv', index_col=0)
df = pd.read_csv(data_path, index_col=0)
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


# Define the layout of the Dash app
app.layout = html.Div(style = {'backgroundColor': '#060606', 'color':'#16E536'}, children=[
    dbc.Container(
    dcc.Tabs([
        dcc.Tab(label='Summary', style={'backgroundColor': '#201F1F', 'border': '0px', 'fontSize': 24}, 
                selected_style={'backgroundColor': '#116307', 'border': '0px', 'color': '#F9FCF9', 'fontSize': 24}, children=[
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
                            html.Button('Apply', id='apply-button-1', n_clicks=0,
                                        style={'border': '2px solid #1f1e1e', 'background-color': '#1f1e1e', 'color': 'rgb(4, 184, 4)', 'fontweight': 'bold', 'border-radius': '10px'}),
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
                ),]            
        ),
        dcc.Tab(label='By Feature', style={'backgroundColor': '#201F1F', 'border': '0px', 'fontSize': 24}, 
                selected_style={'backgroundColor': '#116307', 'border': '0px', 'color': '#F9FCF9', 'fontSize': 24}, children=[  
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
                            html.Button('Apply', id='apply-button-2', n_clicks=0,
                                        style={'border': '2px solid #1f1e1e', 'background-color': '#1f1e1e', 'color': 'rgb(4, 184, 4)', 'fontweight': 'bold', 'border-radius': '10px'}),
                        ]),width="4",
                ),
                dbc.Col(
                    [html.H3("Popularity Distribution"),
                     html.Div(id='feature-charts-container')
                    ],width="8"
                )
            ]),            
            ]),   
        dcc.Tab(label='Prediction', style={'backgroundColor': '#201F1F', 'border': '0px', 'fontSize': 24}, 
                selected_style={'backgroundColor': '#116307', 'border': '0px', 'color': '#F9FCF9', 'fontSize': 24}, children=[
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
                                style={'border': '2px solid rgb(26, 24, 24)', 'border-radius': '10px', 'background-color': 'rgb(26, 24, 24)'})
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
                                style={'border': '2px solid rgb(26, 24, 24)', 'border-radius': '10px', 'background-color': 'rgb(26, 24, 24)'})
                        ])
                    ]),
                    dbc.Col([html.Label('seconds')], style={'margin-left': '-10px'})
                ],style={'border-bottom': '3px solid lightgrey', 'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px', 'border-bottom-left-radius':'10px', 'border-bottom-right-radius':'10px'}),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                    html.Button('Apply', id='apply', n_clicks=0, style={'border': '2px solid #1f1e1e', 'background-color': '#1f1e1e', 'color': 'rgb(4, 184, 4)', 'fontweight': 'bold', 'border-radius': '10px'})])]),
                    dbc.Col([])
            ])
            ]),
            dbc.Col([
            dbc.Container([
                dbc.Col([
                    html.H3('The Predicted Popularity'),
                    dbc.Row([
                    html.Iframe(id='pred-result',width=400,height=350)], style={'margin': '0 auto -7px 140px', 'margin-left': '120px'}),
                    dbc.Row([
                    dcc.Graph(id='pred-radar', style={'margin': '0 auto', 'width':'600px', 'height':'420px', 'margin-top': '-50px'})])
                ], style={'align':'center'})
            ])
        ])
    ])
            ]),
    ]),
    fluid=True
)
])



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
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    chart=alt.Chart(filtered_df).mark_line(color='darkgreen',strokeWidth=5).encode(
        x=alt.X('decade',type='ordinal',title=None
                ,axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y('mean(track_popularity)',scale=alt.Scale(zero=False),title='Average Popularity',
                axis=alt.Axis(labelColor='white', titleColor='white')),
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
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    sel1 = alt.selection_point(fields = ['playlist_genre'])
    sel2 = alt.selection_point(fields = ['nominal_popularity'])
    chart1=alt.Chart(filtered_df).mark_bar(color='darkred',opacity=0.7).encode(
        x=alt.X('nominal_popularity',type='ordinal',title=None,
                sort=['low','medium','high'],
                axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y('count()',title='Count of Records',
                axis=alt.Axis(labelColor='white', titleColor='white')),
        color= alt.Color('nominal_popularity', legend=None)).properties(height=300,width=100).transform_filter(sel1).add_params(sel2)
    chart2=alt.Chart(filtered_df).mark_arc().encode(
            color=alt.Color('playlist_genre',legend=alt.Legend(title=None,labelColor='white')),
            theta='count()'
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
    popularity_min=top10_artists['track_popularity'].min()-5
    chart1 = alt.Chart(top10_songs).mark_bar(clip=True,color='darkgreen',opacity=0.8).encode(
        x=alt.X("track_popularity",scale=alt.Scale(domain=[popularity_min,100]),title='Popularity',
                axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y("track_name", sort='-x',title=None,
                axis=alt.Axis(labelColor='white', titleColor='white')) # sort the x value in descent order
    ).properties(title='Top 10 Songs')
    chart2 = alt.Chart(top10_artists).mark_bar(clip=True,color='darkgreen').encode(
        x=alt.X("track_popularity",scale=alt.Scale(domain=[popularity_min,100]),title='Average Popularity',
                axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y("track_artist", sort='-x',title=None,
                axis=alt.Axis(labelColor='white', titleColor='white')) # sort the x value in descent order
    ).properties(title='Top 10 Artists')
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
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    df_new = filtered_df[filtered_df['track_album_release_date'].dt.year == select_year]
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
        tooltip = ['track_name', 'track_artist', 'track_album_name', 'track_popularity']
    ).properties(height=300).add_params(brush).transform_filter(sel)

    graph1 = alt.Chart(df_new).mark_bar().encode(
    x = alt.X('count()',axis=alt.Axis(labelColor='white', titleColor='white')),
    y = alt.Y('playlist_genre', title=None,axis=alt.Axis(labelColor='white', titleColor='white')),
    color = alt.Color('playlist_genre',legend=None),
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




# load the prediction model and feature scalar
model = load('support_model/spotify_model.joblib')
key_scalar = load('support_model/key_scalar.joblib')
loud_scalar = load('support_model/loud_scalar.joblib')
dur_scalar = load('support_model/duration_scalar.joblib')
tempo_scalar = load('support_model/tempo_scalar.joblib')
key_scalar_mm = load('support_model/key_scalar_mm.joblib')
loud_scalar_mm = load('support_model/loud_scalar_mm.joblib')
dur_scalar_mm = load('support_model/duration_scalar_mm.joblib')
tempo_scalar_mm = load('support_model/tempo_scalar_mm.joblib')

def track_radar(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, key_scalar_mm=key_scalar_mm, loud_scalar_mm=loud_scalar_mm, dur_scalar_mm=dur_scalar_mm, tempo_scalar_mm=tempo_scalar_mm):
    categories = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration']
    
    # Apply scaling
    key_scaled = key_scalar_mm.transform(pd.DataFrame({'key': [key]}))[0][0]
    loudness_scaled = loud_scalar_mm.transform(pd.DataFrame({'loudness': [loudness]}))[0][0]
    tempo_scaled = tempo_scalar_mm.transform(pd.DataFrame({'tempo': [tempo]}))[0][0]
    duration_scaled = dur_scalar_mm.transform(pd.DataFrame({'duration_ms': [duration]}))[0][0]
    
    # Organize values
    values_scaled = [danceability, energy, key_scaled, loudness_scaled, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo_scaled, duration_scaled]
    values = values_scaled + [values_scaled[0]]  # Close the radar chart
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Track Features',
            line=dict(color='#0AF52F'),
            marker=dict(color='#0AF52F'),
            textfont=dict(color='#0AF52F')
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',  # Set background color to transparent
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0AF52F')
    )
    
    return fig

def pop_predict(type, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, key_scalar=key_scalar, loud_scalar=loud_scalar, tempo_scalar=tempo_scalar, dur_scalar=dur_scalar, model=model):
    # standardize key, loud, tempo and duartion
    key = key_scalar.transform(pd.DataFrame({'key': [loudness]}))[0][0]
    loudness = loud_scalar.transform(pd.DataFrame({'loudness': [loudness]}))[0][0]
    tempo = tempo_scalar.transform(pd.DataFrame({'tempo': [tempo]}))[0][0]
    duration = dur_scalar.transform(pd.DataFrame({'duration_ms': [duration]}))[0][0]

    # initialize the prediction df
    pred_df = pd.DataFrame(
        np.array([danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration]).reshape(1, 12),
        columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    )
    
    # define the track genre variable for prediction
    dummy_df = pd.DataFrame({
        'edm': [0], 
        'latin': [0], 
        'pop': [0], 
        'r&b': [0], 
        'rap': [0], 
        'rock': [0]
    })
    # set the selected genre to 1
    for each in dummy_df.columns:
        if each == type:
            dummy_df.loc[:, each][0] = 1
            break

    # create the X for prediction
    pred_df = pd.concat([dummy_df, pred_df], axis=1)

    return model.predict(pred_df)[0]

def pred_chart(result):    
    df_pred = pd.DataFrame({'result': [result, 100-result],
                        'category': [1, 2]})
    circle_chart = alt.Chart(df_pred).mark_arc(innerRadius=80).encode(
        theta=alt.Theta('result'),
        color=alt.condition(
            alt.datum.category==1,
            alt.value('green'),
            alt.value('grey')
        )
    )

    text_chart = alt.Chart(df_pred.iloc[0:1, ]).mark_text(size=60).encode(
        text = 'result',
        color= alt.Color('result', scale=alt.Scale(scheme='greens'), legend=None)
    )

    pred = circle_chart+text_chart
    
    pred.configure_view(strokeWidth=0)

    return pred.to_html()

# handle select all options for feature tab
def handle_select_all(selected_values, options_list):
    if 'all' in selected_values:
        # Exclude 'Select All' option itself
        return [option['value'] for option in options_list if option['value'] != 'all']
    return selected_values

# plot function for feature tab
def create_feature_distribution_charts(df, selected_features):
    charts = []
    
    # Define a single selection that binds to the legend and allows toggling
    selection = alt.selection_point(fields=['nominal_popularity'], bind='legend')

    popularity_colors = {
        'high': '#69A053',  
        'medium':  '#E8CC52',  
        'low': '#5777A5', 
    }

    # Determine the layout based on the number of selected features
    layout_columns = 3 if len(selected_features) > 1 else 1
    
    for feature in selected_features:
        # Check if the feature is 'key' or 'mode' for categorical encoding, else treat as numerical
        if feature in ['key', 'mode']:  # Categorical features
            chart = alt.Chart(df).mark_bar(tooltip=True).encode(
                alt.X(f"{feature}:N", sort='-y', title=feature.capitalize(),axis=alt.Axis(labelColor='white', titleColor='white')),
                alt.Y('count()',axis=alt.Axis(labelColor='white', titleColor='white')),
                alt.Color('nominal_popularity:N', legend=alt.Legend(title="Select Popularity Level", symbolSize= 200, labelColor='white'), scale=alt.Scale(domain=list(popularity_colors.keys()), range=list(popularity_colors.values()))),
                tooltip=[alt.Tooltip(f"{feature}:N"), alt.Tooltip('count()', title='Count')]
            ).add_params(
                selection
            ).transform_filter(
                selection
            )
        else:  # Numerical features
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(f"{feature}:Q", bin=True, title=feature.capitalize().replace(' (binned)', ''),axis=alt.Axis(labelColor='white', titleColor='white')),
                alt.Y('count()', title=None,axis=alt.Axis(labelColor='white', titleColor='white')),
                alt.Color('nominal_popularity:N', legend=alt.Legend(title="Select Popularity Level", symbolSize= 200,labelColor='white'), scale=alt.Scale(domain=list(popularity_colors.keys()), range=list(popularity_colors.values()))),
                tooltip=[alt.Tooltip(f"{feature}:Q", bin=True), alt.Tooltip('count()', title='Count')]
            ).add_params(
                selection
            ).transform_filter(
                selection
            )
        
        charts.append(chart)
    
    # Combine all charts into a single chart, adjusting the layout based on the number of charts
    if len(charts) > 1:
        combined_chart = alt.hconcat(*[alt.vconcat(*charts[i::layout_columns]).resolve_scale(y='independent') for i in range(layout_columns)])
    else:
        combined_chart = charts[0]  # If only one chart, just use it directly
    
    # Apply global configurations
    combined_chart = combined_chart.configure_view(
        strokeWidth=0
    )
    
    return combined_chart.to_html()


if __name__ == "__main__":
    app.run_server(debug=True)