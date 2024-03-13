import altair as alt
import pandas as pd
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
                        html.H4("Top 10 Popular Songs"),
                        dash_table.DataTable(
                            id = 'top-10-songs',
                            columns=[
                                {"name": "Rank", "id": "rank"},
                                {"name": "Song", "id": "track_name"},
                                {"name": "Artist", "id": "track_artist"},
                                {"name": "Popularity", "id": "track_popularity"},
                                {'name': 'Track ID', 'id': 'track_id'}
                            ],
                            hidden_columns=['track_id'],
                            style_data_conditional=[
                                # Odd rows
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#464545'
                                },
                                # Even rows
                                {
                                    'if': {'row_index': 'even'},
                                    'backgroundColor': '#111111'
                                },
                                {
                                    'if': {'state': 'selected'}, 
                                    'backgroundColor': '#096C01', 
                                    'border': '1px solid green'}
                            ],
                            style_table={
                                'border': '0px'
                            },
                            export_format='none',
                            style_cell={
                                'textAlign': 'center',
                                'fontFamily': "Nunito",
                                'border': '0px',
                                'color': 'white'
                            },
                            style_header={
                                'fontWeight': 'bold',
                                'fontFamily': "Nunito",
                                'border': '0px',
                                'background-color': '#888A87',
                                'color': 'white'
                            }),
                        html.Div(id='redirect-instructions', children=
                            [html.A(
                                [html.Img(src='assets/play.png', style={'height': '25px', 'width': '25px', 'margin-top': '-1%'}),
                                "Listen to the Music on Spotify"]
                                , id='song-link', style={'color': 'rgb(4, 184, 4)'}, target="_blank")])
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
                html.Div(children=[
                    html.H4("Top 10 Popular Artists"),
                        dash_table.DataTable(
                            id = 'top-10-artists',
                            columns=[
                                {"name": "Rank", "id": "rank"},
                                {"name": "Artist", "id": "track_artist"},
                                {"name": "Popularity", "id": "track_popularity"}
                            ],
                            style_data_conditional=[
                                # Odd rows
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#464545'
                                },
                                # Even rows
                                {
                                    'if': {'row_index': 'even'},
                                    'backgroundColor': '#111111'
                                },
                                {
                                    'if': {'state': 'selected'}, 
                                    'backgroundColor': '#096C01', 
                                    'border': '1px solid green'}
                            ],
                            style_table={
                                'border': '0px'
                            },
                            export_format='none',
                            style_cell={
                                'textAlign': 'center',
                                'fontFamily': "Nunito",
                                'border': '0px',
                                'color': 'white'
                            },
                            style_header={
                                'fontWeight': 'bold',
                                'fontFamily': "Nunito",
                                'border': '0px',
                                'background-color': '#888A87',
                                'color': 'white'
                            },
                            row_selectable=None)
                    
            ]),
                ],width="4")]
            )])