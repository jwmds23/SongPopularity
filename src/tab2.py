import altair as alt
import pandas as pd
import datetime
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from helper import handle_select_all

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

tab2_content_bivariate_scatter = html.Div(
    id='tab-2-bivariate-scatter-content', children=[    
                        html.H5("Bivariate-Feature Scatter Plot with Interactive Genre Distribution"),
                        html.Iframe(
                            id="feature_scatter-chart-iframe",
                            style={'border-width': '0', 'width': '100%', 'height': '800px'},
                        )     
                    ]
                )

bivariate_scatter_filters = html.Div(children=[
    dbc.Row([
        dbc.Col(html.Label("Release Year"), width=6),
        dbc.Col(html.Div(id='year-output'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Slider(
            id='year-slider',
            min=1957,
            max=2021,
            value=2000,
            marks={i: str(i) for i in range(1960, 2021, 10)},
            updatemode='drag',
            step=1
        ), width=12), # Set to 12 to take full width of the grid
    ]),
    html.Br(),
    html.P("Genre"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
                id='genre-dropdown-2a',
                options=[{'label': 'Select All', 'value': 'all'}] + [{'label': genre, 'value': genre} for genre in genre_list],
                value=['all'],
                multi=True,
                style={'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
            ), width=12),
    ]),
    html.Br(),
    html.P("SubGenre"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
                id='subgenre-dropdown-2a',
                options=[{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list],
                value=['all'],
                multi=True,
                style={'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
            ), width=12),
    ]),
    html.Br(),
    html.P("Artist"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
                id='artist-dropdown-2a',
                options=[{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list],
                value=['all'],
                multi=True,
                style={'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
            ), width=12),
    ]),
    html.Br(),
    
    dbc.Row([
        dbc.Col([
            html.Label("Feature 1"),
            dcc.Dropdown(
                id='feature1-dropdown',
                options=[{'label': feature, 'value': feature} for feature in feature_list],
                value='danceability',
                multi=False,
                style={'width': '200px', 'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
            )
        ], width=6),
        dbc.Col([
            html.Label("Feature 2"),
            dcc.Dropdown(
                id='feature2-dropdown',
                options=[{'label': feature, 'value': feature} for feature in feature_list],
                value='liveness',
                multi=False,
                style={'width': '200px', 'backgroundColor': 'black', 'color': 'rgb(4, 184, 4)'}
            )
        ], width=6)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Button('Apply', id='apply-button-2a', n_clicks=0,
                    style={'border': '2px solid #1f1e1e',
                           'background-color': '#1f1e1e',
                           'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                           'border-radius': '10px', 'margin-right': '5px'}),
                width=3
        ),
        dbc.Col(html.Button('Reset', id='reset-button-2a', n_clicks=0,
                    style={'border': '2px solid #1f1e1e',
                           'background-color': '#1f1e1e',
                           'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                           'border-radius': '10px'}),
                width=3
        )
    ])
])

tab2_content_popularity_distribution = html.Div(id='tab-2-popularity-distribution-content',children=[
        dbc.Row([
            dbc.Col(
                [html.H5("Popularity Distribution"),
                    html.Div(id='feature-charts-container')
                ],width="1000"
            )
        ])])
     
popularity_distribution_filters = html.Div(children=[
    dbc.Row([
        dbc.Col(
            html.Div([
                #feature_description_card(),
                html.P("Release Date"),
                dcc.DatePickerRange(
                    id='date-picker-range-2',
                    start_date=datetime.date(2020, 1, 1),
                    start_date_placeholder_text='01/01/2020',
                    end_date_placeholder_text='End Date',
                ),
                html.Br(),
                html.P("Genre"),
                dcc.Dropdown(
                    id='genre-dropdown-2',
                    options=[{'label': 'Select All', 'value': 'all'}] + [{'label': genre, 'value': genre} for genre in genre_list],
                    value=['all'],  # Default value
                    multi=True,
                    style={'backgroundColor': 'black', 'color': '#16E536', 'width': '100%'},
                    className='custom-dropdown'
                ),
                html.Br(),
                html.P("SubGenre"),
                dcc.Dropdown(
                    id='subgenre-dropdown-2',
                    options=[{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list],
                    value=['all'],  # Default value
                    multi=True,
                    style={'backgroundColor': 'black', 'color': '#16E536', 'width': '100%'}
                ),
                html.Br(),
                html.P("Artist"),
                dcc.Dropdown(
                    id='artist-dropdown-2',
                    options=[{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list],
                    value=['all'],  # Default value
                    multi=True,
                    style={'backgroundColor': 'black', 'color': '#16E536', 'width': '100%'}
                ),
                html.Br(),
                html.P("Feature"),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': 'Select All', 'value': 'all'}] + [{'label': feature, 'value': feature} for feature in feature_list],
                    value=['all'],  # Default value
                    multi=True,
                    style={'backgroundColor': 'black', 'color': '#16E536', 'width': '100%'}
                ),
                html.Br(),
                dbc.Row([
                    dbc.Col(html.Button('Apply', id='apply-button-2', n_clicks=0,
                                        style={'border': '2px solid #1f1e1e',
                                                'background-color': '#1f1e1e',
                                                'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                                                'border-radius': '10px','margin-right': '5px'}),
                            width={'size': 5, 'offset': 1}), # Adjust the offset and size to match your desired layout
                    dbc.Col(html.Button('Reset', id='reset-button-2', n_clicks=0,
                                        style={'border': '2px solid #1f1e1e',
                                                'background-color': '#1f1e1e',
                                                'color': 'rgb(4, 184, 4)', 'fontweight': 'bold',
                                                'border-radius': '10px'}),
                            width=5)
                ])
            ]),
            width=12, # Set to 12 to take full width or adjust as needed
        )
    ])
])


tab2 = html.Div(id='tab-2-content', children=[
    dbc.Row([
        dbc.Col([
            html.H3("Filter Menu"),
            html.Br(),
            # This is where the filter components will be dynamically inserted
            html.Div(id='tab-2-filter-placeholder'),
        ], width=4),
        # Sub-tabs definition
        dbc.Col([
            dbc.Row([
                dbc.Col([
                dcc.Tabs(id='tab-2-sub-tabs', value='tab-2-bivariate-scatter', children=[
                    dcc.Tab(label='Bivariate', style={'backgroundColor': 'rgba(0,0,0,0)', 'border': '2px solid rgba(0,0,0,0)', 'border-radius': '10px', 'padding': '0px', 'fontSize': 20}, selected_style={'backgroundColor': '#116307', 'border': '2px solid #116307', 'border-radius': '10px', 'padding': '0px', 'color': '#F9FCF9', 'fontSize': 20},value='tab-2-bivariate-scatter'),
                    dcc.Tab(label='Multi-Feature', style={'backgroundColor': 'rgba(0,0,0,0)', 'border': '2px solid rgba(0,0,0,0)', 'border-radius': '10px', 'padding': '0px', 'fontSize': 20, 'margin-left': '5%'}, selected_style={'backgroundColor': '#116307', 'border': '2px solid #116307', 'border-radius': '10px', 'padding': '0px', 'color': '#F9FCF9', 'fontSize': 20, 'margin-left': '5%'}, value='tab-2-popularity-distribution')
                ], style={'width': '300px', 'height':'40px', 'margin-right': '100px'}),
                ], width='8')
            ], style={'margin-top': '2%'}),
            html.Br(),    
            dbc.Row([
                html.Div(id='tab-2-content-placeholder'),
                ])
            ], width=8)               
        ]),
])