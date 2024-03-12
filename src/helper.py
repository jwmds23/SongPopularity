import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import altair as alt
import requests
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import load

# load the prediction model and feature scalar
model = load('src/support_model/spotify_model.joblib')
key_scalar = load('src/support_model/key_scalar.joblib')
loud_scalar = load('src/support_model/loud_scalar.joblib')
dur_scalar = load('src/support_model/duration_scalar.joblib')
tempo_scalar = load('src/support_model/tempo_scalar.joblib')
key_scalar_mm = load('src/support_model/key_scalar_mm.joblib')
loud_scalar_mm = load('src/support_model/loud_scalar_mm.joblib')
dur_scalar_mm = load('src/support_model/duration_scalar_mm.joblib')
tempo_scalar_mm = load('src/support_model/tempo_scalar_mm.joblib')

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
    alt.themes.enable('transparent_bg')
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
    alt.themes.enable('grey_bg')
    charts = []
    
    # Define a single selection that binds to the legend and allows toggling
    selection = alt.selection_point(fields=['nominal_popularity'], bind='legend')

    popularity_colors = {
        'high': '#38AD48',  
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
                alt.Color('nominal_popularity:N', legend=alt.Legend(title="Select Popularity Level",  titleColor='white', symbolSize= 500, labelColor='white'), scale=alt.Scale(domain=list(popularity_colors.keys()), range=list(popularity_colors.values()))),
                tooltip=[alt.Tooltip(f"{feature}:N"), alt.Tooltip('count()', title='Count')]
            ).add_params(
                selection
            ).transform_filter(
                selection
            ).properties(
            width = 220,
            height = 220
            )
        else:  # Numerical features
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(f"{feature}:Q", bin=True, title=feature.capitalize().replace(' (binned)', ''),axis=alt.Axis(labelColor='white', titleColor='white')),
                alt.Y('count()', title=None,axis=alt.Axis(labelColor='white', titleColor='white')),
                alt.Color('nominal_popularity:N', legend=alt.Legend(title="Select Popularity Level", titleColor='white', symbolSize= 500,labelColor='white'), scale=alt.Scale(domain=list(popularity_colors.keys()), range=list(popularity_colors.values()))),
                tooltip=[alt.Tooltip(f"{feature}:Q", bin=True), alt.Tooltip('count()', title='Count')]
            ).add_params(
                selection
            ).transform_filter(
                selection
            ).properties(
            width = 220,
            height = 220
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
    ).configure_legend(
    direction='horizontal',
    orient='none',  # This takes the legend out of the normal flow
    titleAlign='right',
    titleAnchor='end',
    titleFontSize=15,
    legendX=650,  # Manually adjust the X-coordinate position
    legendY= -70,  # Manually adjust the Y-coordinate position to move it closer to the title
    columns=3  # Assuming you want 3 items in one row, change as needed
    )
    
    return combined_chart.to_html()

def music_play(track_id):
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

    auth_response = requests.post('https://accounts.spotify.com/api/token', {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })

    # Convert the response to JSON
    auth_response_data = auth_response.json()

    # Save the access token
    access_token = auth_response_data['access_token']

    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    response = requests.get(f'https://api.spotify.com/v1/tracks/{track_id}', headers=headers)

    track_details = response.json()

    return track_details['external_urls']['spotify']