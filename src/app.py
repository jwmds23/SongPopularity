# python
import json
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import altair as alt
import dash
from dash import dcc, html, Input, Output, ClientsideFunction
alt.data_transformers.disable_max_rows()
# alt.data_transformers.enable("vegafusion")

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
)
app.title = "Spotify Song Popularity"

server = app.server
app.config.suppress_callback_exceptions = True


# Read data
df = pd.read_csv('../data/processed/spotify_songs_processed.csv', index_col=0)
object_columns = df.select_dtypes(include=['object']).columns
for column in object_columns:
    df[column] = df[column].astype('string')
features=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'duration_ms']
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

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H3("Spotify Song Popularity Summary"),
            html.Div(
                id="intro",
                children="In this section, users can find visual representations of the average popularity trending, the general distribution of all songs' popularity, as well as breakdowns by different genres and rankings of songs and artists. Users are able to filter the data by release date, genre, and artist to tailor the analysis to specific interests, which is particularly useful if users look to focus on a certain timeframe or music style.",
            ),
        html.Br(),    
        ],
    )

def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Release Date"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=dt(1957, 1, 1),
                end_date=dt(2020, 1, 29),
                min_date_allowed=dt(1957, 1, 1),
                max_date_allowed=dt(2020, 1, 29),
                initial_visible_month=dt(1957, 1, 1),
            ),
            html.Br(),
            html.Br(),
            html.P("Genre"),
            dcc.Dropdown(
                id="genre-select",
                options=[{"label": i, "value": i} for i in genre_list],
                value=genre_list[:],
                multi=True,
            ),
            html.Br(),
            html.P("SubGenre"),
            dcc.Dropdown(
                id="subgenre-select",
                options=[{"label": i, "value": i} for i in subgenre_list],
                value=subgenre_list[:],
                multi=True,
            ),
            html.Br(),
            html.P("Artist"),
            dcc.Input(
                id="artist-input",
                type="text",
                placeholder="Enter artist names",
            ),
            html.Br(),
            html.P("Feature"),
            dcc.Dropdown(
                id="feature-select",
                options=[{"label": i, "value": i} for i in features],
                value=features[:],
                multi=True,
            ),
        ],
    )


# Define the layout of the Dash app
app.layout = html.Div(
    id="app-container",
    children=[
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[
                description_card(),
                generate_control_card(),
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # First row containing the first three plots
                html.Div(
                    id="first-row",
                    className="row",
                    children=[
                        html.Div(
                            id="decade-trend-line-chart",
                            className="four columns",
                            children=[
                                html.H6("Decade Trend Line Chart"),
                                html.Iframe(
                                    id="decade-trend-line-chart-iframe",
                                    style={'border-width': '0', 'width': '100%', 'height': '400px'},
                                ),
                            ],
                        ),
                        html.Div(
                            id="popularity-level-distribution-chart",
                            className="three columns",
                            children=[
                                html.H6("Popularity Distribution"),
                                html.Iframe(
                                    id="popularity-level-distribution-chart-iframe",
                                    style={'border-width': '0', 'width': '100%', 'height': '400px'},
                                ),
                            ],
                        ),
                        html.Div(
                            id="genre-distribution-pie-chart",
                            className="five columns",
                            children=[
                                html.H6("Genre Distribution"),
                                html.Iframe(
                                    id="genre-distribution-pie-chart-iframe",
                                    style={'border-width': '0', 'width': '100%', 'height': '400px'},
                                ),
                            ],
                        ),
                    ],
                ),
                # Second row containing the last two plots
                html.Div(
                    id="second-row",
                    className="row",
                    children=[
                        html.Div(
                            id="top-10-popularity-songs-chart",
                            className="six columns",
                            children=[
                                html.H6("Top 10 Popularity Songs"),
                                html.Iframe(
                                    id="top-10-popularity-songs-chart-iframe",
                                    style={'border-width': '0', 'width': '100%', 'height': '400px'},
                                ),
                            ],
                        ),
                        html.Div(
                            id="top-10-average-popularity-artists-chart",
                            className="six columns",
                            children=[
                                html.H6("Top 10 Average Popularity Artists"),
                                html.Iframe(
                                    id="top-10-average-popularity-artists-chart-iframe",
                                    style={'border-width': '0', 'width': '100%', 'height': '400px'},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)



@app.callback(
    Output('decade-trend-line-chart-iframe', 'srcDoc'),
    [Input('date-picker-select', 'start_date'),
     Input('date-picker-select', 'end_date'),
     Input('genre-select', 'value'),
     Input('subgenre-select', 'value'),
     Input('artist-input', 'value')]
)
def update_decade_trend_line(start_date, end_date, genre, subgenre, artist_input):
    if artist_input:
        artists = [artist.strip() for artist in artist_input.split(',')]
    else:
        artists = artist_list
    filtered_df = df[(df['track_album_release_date'] >= start_date) & 
                     (df['track_album_release_date'] <= end_date) &
                     (df['playlist_genre'].isin(genre)) &
                     (df['playlist_subgenre'].isin(subgenre)) &
                     (df['track_artist'].isin(artists))]
    chart=alt.Chart(filtered_df).mark_line(color='red',opacity=0.4).encode(
        x=alt.X('decade',type='ordinal',title=None),
        y=alt.Y('mean(track_popularity)',scale=alt.Scale(zero=False),title='average popularity'),
    ).interactive()
    return chart.to_html()


@app.callback(
    Output('popularity-level-distribution-chart-iframe', 'srcDoc'),
    [Input('date-picker-select', 'start_date'),
     Input('date-picker-select', 'end_date'),
     Input('genre-select', 'value'),
     Input('subgenre-select', 'value'),
     Input('artist-input', 'value')]
)
def update_popularity_level_distribution(start_date, end_date, genre, subgenre, artist_input):
    if artist_input:
        artists = [artist.strip() for artist in artist_input.split(',')]
    else:
        artists = artist_list
    filtered_df = df[(df['track_album_release_date'] >= start_date) & 
                     (df['track_album_release_date'] <= end_date) &
                     (df['playlist_genre'].isin(genre)) &
                     (df['playlist_subgenre'].isin(subgenre)) &
                     (df['track_artist'].isin(artists))]
    chart=alt.Chart(filtered_df).mark_bar(color='orange',opacity=0.7).encode(
            x=alt.X('nominal_popularity',type='ordinal',title=None,sort=['low','medium','high']),
            y=alt.Y('count()',title='Count of Records'),
        ).properties(width=100)
    return chart.to_html()

@app.callback(
    Output('genre-distribution-pie-chart-iframe', 'srcDoc'),
    [Input('date-picker-select', 'start_date'),
     Input('date-picker-select', 'end_date'),
     Input('genre-select', 'value'),
     Input('subgenre-select', 'value'),
     Input('artist-input', 'value')]
)
def update_genre_distribution(start_date, end_date, genre, subgenre, artist_input):
    if artist_input:
        artists = [artist.strip() for artist in artist_input.split(',')]
    else:
        artists = artist_list
    filtered_df = df[(df['track_album_release_date'] >= start_date) & 
                     (df['track_album_release_date'] <= end_date) &
                     (df['playlist_genre'].isin(genre)) &
                     (df['playlist_subgenre'].isin(subgenre)) &
                     (df['track_artist'].isin(artists))]
    chart=alt.Chart(filtered_df).mark_arc().encode(
            color=alt.Color('playlist_genre',legend=alt.Legend(title=None)),
            theta='count()',
        ).properties(height=300,width=200)
    return chart.to_html()

@app.callback(
    Output('top-10-popularity-songs-chart-iframe', 'srcDoc'),
    [Input('date-picker-select', 'start_date'),
     Input('date-picker-select', 'end_date'),
     Input('genre-select', 'value'),
     Input('subgenre-select', 'value'),
     Input('artist-input', 'value')]
)
def update_top_10_popularity_songs(start_date, end_date, genre, subgenre, artist_input):
    if artist_input:
        artists = [artist.strip() for artist in artist_input.split(',')]
    else:
        artists = artist_list
    filtered_df = df[(df['track_album_release_date'] >= start_date) & 
                     (df['track_album_release_date'] <= end_date) &
                     (df['playlist_genre'].isin(genre)) &
                     (df['playlist_subgenre'].isin(subgenre)) &
                     (df['track_artist'].isin(artists))]
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
    [Input('date-picker-select', 'start_date'),
     Input('date-picker-select', 'end_date'),
     Input('genre-select', 'value'),
     Input('subgenre-select', 'value'),
     Input('artist-input', 'value')]
)
def update_top_10_popularity_artists(start_date, end_date, genre, subgenre, artist_input):
    if artist_input:
        artists = [artist.strip() for artist in artist_input.split(',')]
    else:
        artists = artist_list
    filtered_df = df[(df['track_album_release_date'] >= start_date) & 
                     (df['track_album_release_date'] <= end_date) &
                     (df['playlist_genre'].isin(genre)) &
                     (df['playlist_subgenre'].isin(subgenre)) &
                     (df['track_artist'].isin(artists))]
    popularity_by_artists = filtered_df[['track_artist','track_popularity']].groupby('track_artist').mean('track_popularity').reset_index()
    top10_artists=popularity_by_artists.nlargest(10,"track_popularity")
    popularity_min=top10_artists['track_popularity'].min()-5
    chart = alt.Chart(top10_artists).mark_bar(clip=True).encode(
        x=alt.X("track_popularity",scale=alt.Scale(domain=[popularity_min,100]),title='Average Popularity'),
        y=alt.Y("track_artist", sort='-x') # sort the x value in descent order
    )
    return chart.to_html()

@app.callback(
    Output('subgenre-select', 'options'),
    [Input('genre-select', 'value')]
)
def update_subgenre_options(selected_genres):
    if not selected_genres:
        # If no genres are selected, return an empty list of options
        return []
    else:
        # Filter subgenres based on the selected genres
        available_subgenres = df[df['playlist_genre'].isin(selected_genres)]['playlist_subgenre'].unique()
        options = [{'label': subgenre, 'value': subgenre} for subgenre in available_subgenres]
        return options

if __name__ == "__main__":
    app.run_server(debug=True)