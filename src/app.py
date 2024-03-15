# python
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
import numpy as np
import datetime
import requests
from datetime import datetime as dt
import altair as alt
import dash
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helper import handle_select_all, create_feature_distribution_charts, pred_chart, track_radar, pop_predict, music_play, transparent_bg, grey_bg, parse_date, calculate_decade
from tab1 import tab1
from tab2 import tab2, tab2_content_popularity_distribution, popularity_distribution_filters, tab2_content_bivariate_scatter, bivariate_scatter_filters
from tab3 import tab3
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

df["track_album_release_date"] = df["track_album_release_date"].apply(parse_date)


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
        dbc.Col([html.H2('Spotify Song Popularity', style={'fontSize': 36, 'textAlign': 'left', 'margin-top': '10px'})], width=True),
        dbc.Col(tabs, width='auto')]),
    html.Div([
        dbc.Row(html.Div(id='tabs-content'))],
        style={'background-color':'#191A19'})
])



# ---------------------------------------------------------Callback Starts Here-----------------------------------------------------------------------------
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
    start_date = '2020/01/01'
    end_date = None
    genre_value = ['all']
    subgenre_value = ['all']
    artist_value = ['all']
    feature_value = ['all']
    return start_date, end_date, genre_value,subgenre_value,artist_value,feature_value

@app.callback(
    [Output('year-slider', 'value'),
     Output('feature1-dropdown', 'value'),
     Output('feature2-dropdown', 'value'),
     Output('genre-dropdown-2a', 'value'),
     Output('subgenre-dropdown-2a', 'value'),
     Output('artist-dropdown-2a', 'value'),],
    [Input('reset-button-2a', 'n_clicks')]
)
def reset_all_filters(n_clicks):
    year_slider = 2000
    feature1='danceability'
    feature2='liveness'
    genre_value = ['all']
    subgenre_value = ['all']
    artist_value = ['all']
    return year_slider,feature1,feature2,genre_value,subgenre_value,artist_value

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

# Define callback to update filters
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
    Output('subgenre-dropdown-2a', 'options'),
    [Input('genre-dropdown-2a', 'value')]
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
    Output('artist-dropdown-1', 'options'),
    [Input('genre-dropdown-1', 'value'),
     Input('subgenre-dropdown-1', 'value'),]
)
def update_artist_options(selected_genres,selected_subgenres):
    if 'all' not in selected_genres:
        if 'all' not in selected_subgenres:
            available_artists = df[df['playlist_genre'].isin(selected_genres) & df['playlist_subgenre'].isin(selected_subgenres)]['track_artist'].unique()
            options = [{'label': artist, 'value': artist} for artist in available_artists]
            return options
        else:
            available_artists = df[df['playlist_genre'].isin(selected_genres)]['track_artist'].unique()
            options = [{'label': artist, 'value': artist} for artist in available_artists]
            return options
    else: 
        options = [{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list]
        return options
    

@app.callback(
    Output('artist-dropdown-2', 'options'),
    [Input('genre-dropdown-2', 'value'),
     Input('subgenre-dropdown-2', 'value'),]
)
def update_artist_options(selected_genres,selected_subgenres):
    if 'all' not in selected_genres:
        if 'all' not in selected_subgenres:
            available_artists = df[df['playlist_genre'].isin(selected_genres) & df['playlist_subgenre'].isin(selected_subgenres)]['track_artist'].unique()
            options = [{'label': artist, 'value': artist} for artist in available_artists]
            return options
        else:
            available_artists = df[df['playlist_genre'].isin(selected_genres)]['track_artist'].unique()
            options = [{'label': artist, 'value': artist} for artist in available_artists]
            return options
    else: 
        options = [{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list]
        return options
    

@app.callback(
    Output('artist-dropdown-2a', 'options'),
    [Input('genre-dropdown-2a', 'value'),
     Input('subgenre-dropdown-2a', 'value'),]
)
def update_artist_options(selected_genres,selected_subgenres):
    if 'all' not in selected_genres:
        if 'all' not in selected_subgenres:
            available_artists = df[df['playlist_genre'].isin(selected_genres) & df['playlist_subgenre'].isin(selected_subgenres)]['track_artist'].unique()
            options = [{'label': artist, 'value': artist} for artist in available_artists]
            return options
        else:
            available_artists = df[df['playlist_genre'].isin(selected_genres)]['track_artist'].unique()
            options = [{'label': artist, 'value': artist} for artist in available_artists]
            return options
    else: 
        options = [{'label': 'Select All', 'value': 'all'}] + [{'label': artist, 'value': artist} for artist in artist_list]
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
    ).properties(height=300, width=350)
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
    popularity_colors = {
        'high': '#38AD48',  
        'medium':  '#E8CC52',  
        'low': '#5777A5', 
    }
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    sel1 = alt.selection_point(fields = ['playlist_genre'])
    sel2 = alt.selection_point(fields = ['nominal_popularity'])
    filtered_df_popularity_level = filtered_df[['playlist_genre','nominal_popularity','track_id']].groupby(['playlist_genre','nominal_popularity']).count().reset_index()
    chart1=alt.Chart(filtered_df_popularity_level).mark_bar(stroke=None).encode(
        y=alt.Y('nominal_popularity',type='ordinal',title=None,
                sort=['low','medium','high'],
                axis=alt.Axis(labelColor='white', titleColor='white')),
        x=alt.X('sum(track_id)',title='',axis=alt.Axis(labels=False)),
        color= alt.Color('nominal_popularity', legend=None,scale=alt.Scale(domain=list(popularity_colors.keys()), range=list(popularity_colors.values()))),
        tooltip = [alt.Tooltip('nominal_popularity', title='Popularity'), alt.Tooltip('sum(track_id)', title='Count of Records')]
        ).properties(height=100,width=300).transform_filter(sel1).add_params(sel2)
    text = chart1.mark_text(align='left', dx=4, baseline='middle').encode(
    x=alt.X('sum(track_id):Q', stack='zero'),
    text=alt.Text('sum(track_id):Q', format=','),
    color=alt.value('white'))
    chart1=chart1+text
    chart2=alt.Chart(filtered_df_popularity_level).mark_arc().encode(
            color=alt.Color('playlist_genre',legend=alt.Legend(title=None,labelColor='white')),
            theta='sum(track_id)',
            tooltip = [alt.Tooltip('playlist_genre', title='Genre'), alt.Tooltip('sum(track_id)', title='Count of Records')]
        ).properties(height=200,width=300).add_params(sel1).transform_filter(sel2)
    return (alt.vconcat(chart1, chart2,config={'axis': {'grid': False,'domain':False}}).resolve_scale(color='independent').configure_view(strokeWidth=0)).to_html()


@app.callback(
    Output('top-10-songs', 'data'),
    [Input('apply-button-1', 'n_clicks')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]
)
def update_top_10_popularity_songs_artists(n_clicks, start_date, end_date,selected_genres, selected_subgenres, selected_artists):
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    df_new=filtered_df[['track_id','track_name','track_artist','track_popularity']].drop_duplicates()
    popularity_by_songs = df_new.groupby('track_name').agg({
        'track_popularity': 'mean',
        'track_artist': 'max',
        'track_id': 'max'
    }).reset_index()
    top10_songs = popularity_by_songs.nlargest(10,"track_popularity")
    top10_songs['rank'] = [i for i in range(1, 11)]
    return top10_songs.to_dict('records')

@app.callback(
    Output('top-10-artists', 'data'),
    [Input('apply-button-1', 'n_clicks')],
    [State('date-picker-range-1', 'start_date'),
     State('date-picker-range-1', 'end_date'),
     State('genre-dropdown-1', 'value'),
     State('subgenre-dropdown-1', 'value'),
     State('artist-dropdown-1', 'value')]
)
def update_top_10_popularity_songs_artists(n_clicks, start_date, end_date,selected_genres, selected_subgenres, selected_artists):
    filtered_df = update_df(df, start_date, end_date, selected_genres, selected_subgenres, selected_artists)
    df_new=filtered_df[['track_artist','track_popularity']].drop_duplicates()
    popularity_by_artists = df_new[['track_artist','track_popularity']].groupby('track_artist').mean('track_popularity').reset_index()
    top10_artists=popularity_by_artists.nlargest(10,"track_popularity")
    top10_artists['rank'] = [i for i in range(1, len(top10_artists)+1)]
    top10_artists['track_popularity'] = round(top10_artists.track_popularity, 0)
    return top10_artists.to_dict('records')

@app.callback(
    Output('feature_scatter-chart-iframe', 'srcDoc'),
    [Input('apply-button-2a', 'n_clicks')],
    [State('year-slider', 'value'),
     State('feature1-dropdown', 'value'),
     State('feature2-dropdown', 'value'),
     State('genre-dropdown-2a', 'value'),
     State('subgenre-dropdown-2a', 'value'),
     State('artist-dropdown-2a', 'value')]

)

def update_feature_scatter(n_clicks, select_year, f1, f2, selected_genres, selected_subgenres, selected_artists):
    alt.themes.enable('grey_bg')
    popularity_colors = {
        'high': '#13EC6F',  
        'medium':  '#E8CC52',  
        'low': '#138DEC', 
    }
    filtered_df = update_df_2a(df, selected_genres, selected_subgenres, selected_artists)
    df_new = filtered_df[filtered_df['track_album_release_date'].dt.year == select_year][['track_name', 'track_artist', 'track_album_name', 'track_popularity', 'playlist_genre', 'nominal_popularity', f1, f2]]
    
    brush = alt.selection_interval()
    sel = alt.selection_point(fields=['playlist_genre'])
    
    # Scatter plot with updated title and size
    scatter_plot = alt.Chart(df_new).mark_point().encode(
        x=alt.X(f1, axis=alt.Axis(labelColor='white', titleColor='white',titleFontSize=12)),
        y=alt.Y(f2, axis=alt.Axis(labelColor='white', titleColor='white',titleFontSize=12)),
        # size='track_popularity',
        color=alt.condition(
            brush,
            'nominal_popularity',
            alt.value('grey'),
            title='Popularity',
            legend=alt.Legend(title=None, labelColor='white',labelFontSize=15),
            scale=alt.Scale(domain=list(popularity_colors.keys()), range=list(popularity_colors.values()))
        ),
        tooltip=[alt.Tooltip('track_name', title='Track Name'), alt.Tooltip('track_artist', title='Artist'), alt.Tooltip('track_album_name', title='Album Name'), alt.Tooltip('track_popularity', title='Popularity')]
    ).properties(
        title="",
        width=380,  # Adjust width as needed
        height=400   # Adjust height as needed
    ).add_params(brush).transform_filter(sel)

    # Bar chart with a title and size adjusted
    bar_chart = alt.Chart(df_new).mark_bar().encode(
        x=alt.X('count()', axis=alt.Axis(labelColor='white', titleColor='white')),
        y=alt.Y('playlist_genre', title=None, axis=alt.Axis(labelFontSize=12,labelColor='white', titleColor='white')),
        color=alt.Color('playlist_genre', legend=None),
        tooltip=[alt.Tooltip('playlist_genre', title='Genre'), alt.Tooltip('count()', title='Count of Records')]
    ).properties(
        title="",
        width=380,  # Adjust width as needed
        height=400   # Make the height the same as the scatter plot
    ).add_params(sel).transform_filter(brush)

    # Combine the plots horizontally
    combined_plots = alt.hconcat(scatter_plot, bar_chart).resolve_scale(color='independent')

    return combined_plots.to_html()

def update_df_2a(df, selected_genres, selected_subgenres, selected_artists):
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
    
    return filtered_df


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
    
@app.callback(
    Output('song-link', 'href'),
    Input('top-10-songs', 'active_cell'),
    State('top-10-songs', 'data')
)
def update_audio_link(active_cell, rows):
    if active_cell:
        try:
            track_id = rows[active_cell['row']]['track_id']
            preview_url = music_play(track_id)
        except Exception as e:
            print(f"Error updating audio link: {e}")
            return ""
        return preview_url
    return ""

@app.callback(
    [Output('tab-2-filter-placeholder', 'children'),
     Output('tab-2-content-placeholder', 'children')],
    [Input('tab-2-sub-tabs', 'value')]
)
def render_subtab_content(selected_sub_tab):
    if selected_sub_tab == 'tab-2-bivariate-scatter':
        return bivariate_scatter_filters, tab2_content_bivariate_scatter
    elif selected_sub_tab == 'tab-2-popularity-distribution':
        return popularity_distribution_filters, tab2_content_popularity_distribution
    else:
        # You can provide a default or handle other cases here
        return html.Div(), html.Div()
#------------------------------------------------------Callback  Ends Here--------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)