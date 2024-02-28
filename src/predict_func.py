import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import load

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
            name='Track Features'
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
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