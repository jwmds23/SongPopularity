# python
import json
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import altair as alt
import dash
from dash import dcc, html, Input, Output, ClientsideFunction
from dash.dependencies import Input, Output, State
from vega_datasets import data
alt.data_transformers.disable_max_rows()
import dash_bootstrap_components as dbc
#alt.data_transformers.enable("vegafusion")


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "Spotify Song Popularity by Feature"

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


app.layout = dbc.Container(
    dcc.Tabs([
        dcc.Tab(label='Summary', children=[
            # Add components for Summary tab
        ]),
        dcc.Tab(label='By Feature', children=[  
            dbc.Row([
                dbc.Col(
                    html.Div([
                        dcc.DatePickerRange(
                            id='date-picker-range',
                            start_date_placeholder_text='Start Date',
                            end_date_placeholder_text='End Date',
                    # Set default dates if needed
                        ),
                        dcc.Dropdown(
                            id='genre-dropdown',
                            options=[{'label': 'Select All', 'value': 'all'}] + [{'label': genre, 'value': genre} for genre in genre_list],
                            value=['all'],  # Default value
                            multi=True
                        ),
                        dcc.Dropdown(
                            id='subgenre-dropdown',
                            options=[{'label': 'Select All', 'value': 'all'}] + [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list],
                            value=['all'],  # Default value
                            multi=True
                        ),
                        dcc.Dropdown(
                            id='artist-dropdown',
                            options=[{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list],
                            value=['all'],  # Default value
                            multi=True
                        ),
                        dcc.Dropdown(
                            id='feature-dropdown',
                            options=[{'label': 'Select All', 'value': 'all'}] + [{'label': feature, 'value': feature} for feature in feature_list],
                            value=['all'],  # Default value
                            multi=True
                        ),
                        html.Button('Apply', id='apply-button', n_clicks=0),
                    ]),
                ),
                dbc.Col(
                    html.Div(id='feature-charts-container'),
                ),            
            ]),
        ]),    
        dcc.Tab(label='Prediction', children=[
            # Add components for Prediction tab
        ]),
    ]),
    fluid=True
)

# Define callback to update charts
def handle_select_all(selected_values, options_list):
    if 'all' in selected_values:
        # Exclude 'Select All' option itself
        return [option['value'] for option in options_list if option['value'] != 'all']
    return selected_values

@app.callback(
    Output('feature-charts-container', 'children'),
    [Input('apply-button', 'n_clicks')],
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('genre-dropdown', 'value'),
     State('subgenre-dropdown', 'value'),
     State('artist-dropdown', 'value'),
     State('feature-dropdown', 'value')]
)

def update_output(n_clicks, start_date, end_date, selected_genres, selected_subgenres, selected_artists, selected_features):
    genres_list = [{'label': genre, 'value': genre} for genre in genre_list]
    subgenres_list = [{'label': subgenre, 'value': subgenre} for subgenre in subgenre_list]
    artists_list = [{'label': artist, 'value': artist} for artist in artist_list]
    features_list = [{'label': feature, 'value': feature} for feature in feature_list]

    # Handle 'Select All' selections for each dropdown
    selected_genres = handle_select_all(selected_genres, genres_list)
    selected_subgenres = handle_select_all(selected_subgenres, subgenres_list)
    selected_artists = handle_select_all(selected_artists, artists_list)
    selected_features = handle_select_all(selected_features, features_list)

    selected_genres = [] if 'all' in selected_genres else selected_genres
    selected_subgenres = [] if 'all' in selected_subgenres else selected_subgenres
    selected_artists = [] if 'all' in selected_artists else selected_artists

    selected_features = feature_list if 'all' in selected_features else selected_features

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

    if start_date and end_date:
        filtered_df= filtered_df[(filtered_df['track_album_release_date'] >= start_date) & (filtered_df['track_album_release_date'] <= end_date)]
        
        # Create Altair charts based on the selected features
    chart_html = create_feature_distribution_charts(filtered_df, selected_features)
    return html.Iframe(srcDoc=chart_html, style={"width": "100%", "height": "400px"})


if __name__ == "__main__":
    app.run_server(debug=True)
