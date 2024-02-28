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
import predict_func
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


# Read data
df = pd.read_csv('../data/processed/spotify_songs_processed.csv', index_col=0)
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
            html.H3("Spotify Song Popularity Summary"),
            html.Div(
                id="intro-1",
                children="In this section, users can find visual representations of the average popularity trending, the general distribution of all songs' popularity, as well as breakdowns by different genres and rankings of songs and artists.",
            ),
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
            html.H3("Spotify Song Popularity by Feature"),
            html.Div(
                id="intro-2",
                children="In this section, users can explore how song popularity is distributed across the technical features of the songs, including danceability, duration, key, modality, and more. Users can select and customize filters via a dropdown menu for their own purposes.",
            ),
            html.Br(),    
            ],
    )


# Define the layout of the Dash app
app.layout = dbc.Container(
    dcc.Tabs([
        dcc.Tab(label='Summary', children=[
            dbc.Row([
                dbc.Col([
                    html.Div([
                            summary_description_card(),
                            html.P("Release Date"),
                            dcc.DatePickerRange(
                                    id='date-picker-range-1',
                                    start_date_placeholder_text='Start Date',
                                    end_date_placeholder_text='End Date',
                                ),
                            html.Br(),
                            html.Br(),
                            html.P("Genre"),
                            dcc.Dropdown(
                                    id='genre-dropdown-1',
                                    options=[{'label': 'Select All', 'value': 'all'}] + [{'label': genre, 'value': genre} for genre in genre_list],
                                    value=['all'],
                                    multi=True
                                ),
                            html.Br(),
                            html.P("SubGenre"),
                            dcc.Dropdown(
                                    id='subgenre-dropdown-1',
                                    options=[{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list],
                                    value=['all'],
                                    multi=True
                                ),
                            html.Br(),
                            html.P("Artist"),
                            dcc.Dropdown(
                                    id='artist-dropdown-1',
                                    options=[{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list],
                                    value=['all'],
                                    multi=True
                                ),
                            html.Br(),
                            html.Button('Apply', id='apply-button-1', n_clicks=0),
                            ])],width="4"
                        ),
                dbc.Col([
                    # The first plot column 
                    html.Div(
                        id="decade-trend-line-chart",
                        children=[
                            html.H6("Decade Trend Line Chart"),
                            html.Iframe(
                                id="decade-trend-line-chart-iframe",
                                style={'border-width': '0', 'width': '100%', 'height': '400px'},
                            ),
                        ],
                        ),
                    html.Div(
                        id="top-10-popularity-songs-chart",
                        children=[
                            html.H6("Top 10 Popularity Songs"),
                            html.Iframe(
                                id="top-10-popularity-songs-chart-iframe",
                                style={'border-width': '0', 'width': '100%', 'height': '400px'},
                            ),
                        ],
                    ),                    
                    ],width="4"),
                dbc.Col([
                    # The second plot column
                    html.Div(
                        id="popularity-level-distribution-chart",
                        children=[
                            html.H6("Popularity and Genre Distribution"),
                            html.Iframe(
                                id="popularity-level-distribution-chart-iframe",
                                style={'border-width': '0', 'width': '100%', 'height': '400px'},
                            ),
                        ],
                    ),
                    html.Div(
                        id="top-10-average-popularity-artists-chart",
                        children=[
                            html.H6("Top 10 Average Popularity Artists"),
                            html.Iframe(
                                id="top-10-average-popularity-artists-chart-iframe",
                                style={'border-width': '0', 'width': '100%', 'height': '400px'},
                            ),
                        ],
                    ),
                    ],width="4")]
                ),]            
        ),
        dcc.Tab(label='By Feature', children=[  
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
                                multi=True
                            ),
                            html.Br(),
                            html.P("SubGenre"),
                            dcc.Dropdown(
                                id='subgenre-dropdown-2',
                                options=[{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list],
                                value=['all'],  # Default value
                                multi=True
                            ),
                            html.Br(),
                            html.P("Artist"),
                            dcc.Dropdown(
                                id='artist-dropdown-2',
                                options=[{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list],
                                value=['all'],  # Default value
                                multi=True
                            ),
                            html.Br(),
                            html.P("Feature"),
                            dcc.Dropdown(
                                id='feature-dropdown',
                                options=[{'label': 'Select All', 'value': 'all'}] + [{'label': feature, 'value': feature} for feature in feature_list],
                                value=['all'],  # Default value
                                multi=True
                            ),
                            html.Br(),
                            html.Button('Apply', id='apply-button-2', n_clicks=0),
                        ]),width="4",
                ),
                dbc.Col(
                    [html.Div(id='feature-charts-container')],width="8"
                )
            ]),            
            ]),   
        dcc.Tab(label='Prediction', children=[
            dbc.Row([
                dbc.Col([
                dbc.Container([
                    html.H3('Set Features for Your Song'),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label('Danceability'),
                                dcc.Slider(
                                    id='danceability',
                                    min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                    updatemode='drag'
                                )], style={'width': '110%'}),
                        ], width=5),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='danceability-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        dbc.Col([
                            html.Div([
                                html.Label('Energy'),
                                dcc.Slider(
                                    id='energy',
                                    min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                    updatemode='drag'
                                )], style={'width': '110%'})
                        ]),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='energy-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        html.Br()
                    ], style={'border-top': '3px solid lightgrey', 'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px', 'border-top-left-radius':'10px', 'border-top-right-radius':'10px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label('Speechiness'),
                                dcc.Slider(
                                    id='speechiness',
                                    min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                    updatemode='drag'
                                )], style={'width': '110%'}
                            )
                        ], width=5),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='speechiness-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        dbc.Col([
                            html.Div([
                                html.Label('Acousticness'),
                                dcc.Slider(
                                    id='acousticness',
                                    min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                    updatemode='drag'
                                )], style={'width': '110%'}
                            )
                        ]),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='acousticness-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        html.Br()
                    ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label('Instrumentalness'),
                                dcc.Slider(
                                    id='instrumentalness',
                                    min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                    updatemode='drag'
                                )], style={'width': '110%'})
                            ], width=5),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='instrumentalness-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        dbc.Col([
                                html.Div([
                                    html.Label('Liveness'),
                                    dcc.Slider(
                                        id='liveness',
                                        min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                        updatemode='drag'
                                    )], style={'width': '110%'}
                                )
                            ]),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='liveness-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        html.Br()
                        ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label('Valence'),
                                dcc.Slider(
                                    id='valence',
                                    min=0, max=1, value=0, marks={0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
                                    updatemode='drag')
                            ], style={'width': '110%'})
                        ], width=5),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='valence-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        dbc.Col([
                            html.Div([
                                html.Label('Loudness'),
                                dcc.Slider(
                                    id='loudness',
                                    min=-47, max=1.5, value=-47, marks={-47: '-47', -40: '-40', -30: '-30', -20: '-20', -10: '-10', 1.5: '1.5'},
                                    updatemode='drag'
                                )], style={'width': '110%'}
                            )
                        ]),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='loudness-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        html.Br()
                    ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}),
                    dbc.Row([
                        dbc.Col([
                                html.Div([
                                    html.Label('Tempo'),
                                    dcc.Slider(
                                        id='tempo',
                                        min=0, max=240, value=0, marks={0: '0', 30: '30', 60: '60', 90: '90', 120: '120', 150: '150', 180: '180', 210: '210', 240: '240'},
                                        updatemode='drag'
                                    )], style={'width': '110%'}
                                )
                            ], width=5),
                        dbc.Col([
                            html.Br(),
                            html.Div(id='tempo-output')],
                            style={'margin-left': '-5px', 'margin-top': '4.5px'},
                            width=1),
                        dbc.Col([
                            html.Div([  
                                html.Label('Mode'),
                                dcc.Dropdown(
                                    id='mode',
                                    options=[{'label': '0', 'value': 0}, {'label': '1', 'value': 1}],
                                    multi=False,
                                    clearable=True,
                                    placeholder='Please select the mode...'
                                )], style={'width': '100%', 'margin-left':'5px'}
                            )
                        ]),
                        html.Br()
                    ], style={'border-left':'3px solid lightgrey', 'border-right':'3px solid lightgrey', 'padding': '10px'}), 
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
                        html.Button('APPLY', id='apply', n_clicks=0, style={'border': '2px solid lightgrey', 'background-color': 'lightgrey', 'color': 'black', 'fontweight': 'bold', 'border-radius': '10px'})])]),
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
            ]),
    ]),
    fluid=True
)

def create_feature_distribution_charts(df, selected_features):
    charts = []
    
    # Determine the layout based on the number of selected features
    layout_columns = 2 if len(selected_features) > 1 else 1
    
    for feature in selected_features:
        # Check if the feature is 'key' or 'mode' for categorical encoding, else treat as numerical
        if feature in ['key', 'mode']:  # Categorical features
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(f"{feature}:N", sort='-y'),
                alt.Y('count()', stack=None),
                alt.Color('nominal_popularity:N', legend=alt.Legend(title="Popularity"), scale=alt.Scale(scheme='set2'))
            )
        else:  # Numerical features
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(f"{feature}:Q", bin=True),
                alt.Y('count()', stack=None),
                alt.Color('nominal_popularity:N', legend=alt.Legend(title="Popularity"), scale=alt.Scale(scheme='set2'))
            )
        
        charts.append(chart)
    
    # Combine all charts into a single chart, adjusting the layout based on the number of charts
    if len(charts) > 1:
        combined_chart = alt.hconcat(*[alt.vconcat(*charts[i::layout_columns]).resolve_scale(y='independent') for i in range(layout_columns)]).configure_view(
            strokeWidth=0
        ).configure_range(category={'scheme': 'set2'})
    else:
        combined_chart = charts[0].configure_range(category={'scheme': 'set2'})  # If only one chart, just use it directly
    
    return combined_chart.to_html()

# Define callback to update charts
def handle_select_all(selected_values, options_list):
    if 'all' in selected_values:
        # Exclude 'Select All' option itself
        return [option['value'] for option in options_list if option['value'] != 'all']
    return selected_values

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
    chart=alt.Chart(filtered_df).mark_line(color='red',opacity=0.4).encode(
        x=alt.X('decade',type='ordinal',title=None),
        y=alt.Y('mean(track_popularity)',scale=alt.Scale(zero=False),title='average popularity'),
    ).properties(width=300)
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
        x=alt.X('nominal_popularity',type='ordinal',title=None,sort=['low','medium','high']),
        y=alt.Y('count()',title='Count of Records'),
    ).properties(height=300,width=100).transform_filter(sel1).add_params(sel2)
    chart2=alt.Chart(filtered_df).mark_arc().encode(
            color=alt.Color('playlist_genre',legend=alt.Legend(title=None)),
            theta='count()',
        ).properties(height=300,width=150).add_params(sel1).transform_filter(sel2)
    return (chart1|chart2).to_html()


@app.callback(
    Output('top-10-popularity-songs-chart-iframe', 'srcDoc'),
    [Input('apply-button-1', 'n_clicks')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]
)
def update_top_10_popularity_songs(n_clicks, start_date, end_date,selected_genres, selected_subgenres, selected_artists):
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    popularity_by_songs = filtered_df[['track_name','track_popularity']].groupby('track_name').mean('track_popularity').reset_index()
    top10_songs=popularity_by_songs.nlargest(10,"track_popularity")
    popularity_min=top10_songs['track_popularity'].min()-5
    chart = alt.Chart(top10_songs).mark_bar(clip=True).encode(
        x=alt.X("track_popularity",scale=alt.Scale(domain=[popularity_min,100]),title='Average Popularity'),
        y=alt.Y("track_name", sort='-x') # sort the x value in descent order
    )
    return chart.to_html()

@app.callback(
    Output('top-10-average-popularity-artists-chart-iframe', 'srcDoc'),
    [Input('apply-button-1', 'n_clicks')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]
)
def update_top_10_popularity_artists(n_clicks, start_date, end_date, selected_genres, selected_subgenres, selected_artists):
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    popularity_by_artists = filtered_df[['track_artist','track_popularity']].groupby('track_artist').mean('track_popularity').reset_index()
    top10_artists=popularity_by_artists.nlargest(10,"track_popularity")
    popularity_min=top10_artists['track_popularity'].min()-5
    chart = alt.Chart(top10_artists).mark_bar(clip=True).encode(
        x=alt.X("track_popularity",scale=alt.Scale(domain=[popularity_min,100]),title='Average Popularity'),
        y=alt.Y("track_artist", sort='-x') # sort the x value in descent order
    )
    return chart.to_html()

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

if __name__ == "__main__":
    app.run_server(debug=True)