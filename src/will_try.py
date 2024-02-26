import altair as alt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import load

# load the data set
df = pd.read_csv('../data/processed/spotify_songs_processed.csv', parse_dates = ['track_album_release_date'], index_col=0)

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

key_options = [{'label': str(i), 'value': i} for i in range(12)]

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

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Label('Genre'),
        dcc.Dropdown(
            id = 'genre',
            options = [
                {'label': 'EDM', 'value': 'edm'},
                {'label': 'Latin', 'value': 'latin'},
                {'label': 'Pop', 'value': 'pop'},
                {'label': 'R&B', 'value': 'r&b'},
                {'label': 'Rap', 'value': 'rap'},
                {'label': 'Rock', 'value': 'rock'}
            ],
            multi = False,
            clearable = True,
            searchable = True,
            placeholder = 'Please select the genre of the song...'
        ),
        html.Label('Danceability'),
        dcc.Slider(
            id = 'danceability',
            min = 0, max = 1, value = 0, marks = {0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'},
            updatemode='drag'
        ),
        html.Label('Energy'),
        dcc.Slider(
            id = 'energy',
            min = 0, max = 1, value = 0, marks = {0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
        ),
        html.Label('Key'),
        dcc.Dropdown(
            id = 'key',
            options = key_options,
            multi = False,
            clearable = True,
            placeholder = 'Please select the key...'
        ),
        html.Label('Loudness'),
        dcc.Slider(
            id = 'loudness',
            min = -47, max = 1.5, value = -47, marks = {-47: '-47', -40: '-40', -30: '-30', -20: '-20', -10: '-10', 1.5: '1.5'}
        ),
        html.Label('Mode'),
        dcc.Dropdown(
            id = 'mode',
            options = [{'label': '0', 'value': 0}, {'label': '1', 'value': 1}],
            multi = False,
            clearable = True,
            placeholder = 'Please select the mode...'
        ),
        html.Label('Speechiness'),
        dcc.Slider(
            id = 'speechiness',
            min = 0, max = 1, value = 0, marks = {0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
        ),
        html.Label('Acousticness'),
        dcc.Slider(
            id = 'acousticness',
            min = 0, max = 1, value = 0, marks = {0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
        ),
        html.Label('Instrumentalness'),
        dcc.Slider(
            id = 'instrumentalness',
            min = 0, max = 1, value = 0, marks = {0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
        ),
        html.Label('Liveness'),
        dcc.Slider(
            id = 'liveness',
            min = 0, max = 1, value = 0, marks = {0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
        ),
        html.Label('Valence'),
        dcc.Slider(
            id = 'valence',
            min = 0, max = 1, value = 0, marks = {0: '0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1'}
        ),
        html.Label('Tempo'),
        dcc.Slider(
            id = 'tempo',
            min = 0, max = 240, value = 0, marks = {0: '0', 30: '30', 60: '60', 90: '90', 120: '120', 150: '150', 180: '180', 210: '210', 240: '240'}
        ),
        html.Label('Minutes:'),
        dcc.Input(
            id='minutes',
            type='number',
            placeholder='Enter minutes...',
            min=0
        ),
        html.Label('Seconds:'),
        dcc.Input(
            id='seconds',
            type='number',
            placeholder='Enter seconds...',
            min=0,
            max=59
        ),
        html.Button('Apply', id = 'apply', n_clicks = 0),
        html.Div(id='test1'),
        dcc.Graph(id='test2')
    ])
])

@app.callback(
    [Output('test1', 'children'),
     Output('test2', 'figure')],
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
        duration_ms = minutes * 6e+7 + seconds * 1e+6   # transfer the time to microseconds
        test1_output = round(pop_predict(genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms), 2)
        test2_output = track_radar(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, livenss, valence, tempo, duration_ms)
        return test1_output, test2_output
    else:
        return '', track_radar(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)