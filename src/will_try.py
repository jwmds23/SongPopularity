import altair as alt
import pandas as pd
import matplotlib as plt
import numpy as np
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import load

# load the data set
df = pd.read_csv('../data/processed/spotify_songs_processed.csv', parse_dates = ['track_album_release_date'], index_col=0)

# load the prediction model and feature scalar
model = load('support_model/spotify_model.joblib')
loud_scalar = load('support_model/loud_scalar.joblib')
dur_scalar = load('support_model/duration_scalar.joblib')
tempo_scalar = load('support_model/tempo_scalar.joblib')
loud_scalar_mm = load('support_model/loud_scalar_mm.joblib')
dur_scalar_mm = load('support_model/duration_scalar_mm.joblib')
tempo_scalar_mm = load('support_model/tempo_scalar_mm.joblib')

def track_radar(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, loud_scalar_mm, dur_scalar_mm, tempo_scalar_mm):
    # initialize the value
    categories = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration']
    N = len(categories)
    danceability = danceability
    energy = energy
    key = key
    loudness = loud_scalar_mm.transform(pd.DataFrame({'loudness': [loudness]}))[0][0]
    mode = mode
    speechiness = speechiness
    acousticness = acousticness
    instrumentalness = instrumentalness
    liveness = liveness
    valence = valence
    tempo = tempo_scalar_mm.transform(pd.DataFrame({'tempo': [tempo]}))[0][0]
    duration = dur_scalar_mm.transform(pd.DataFrame({'duration_ms': [duration]}))[0][0]
    values = [danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration]

    # repeat the first value in values to close the circle
    values += values[:1]

    # calculate the angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # plot the radar graph
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Plot each entity
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    return fig

def pop_predict(type, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, loud_scalar, tempo_scalar, dur_scalar, model):
    # standardize loud, tempo and duartion
    loudness = loud_scalar.transform(pd.DataFrame({'loudness': [loudness]}))[0][0]
    tempo = tempo_scalar.transform(pd.DataFrame({'tempo': [tempo]}))[0][0]
    duration = dur_scalar.transform(pd.DataFrame({'duration_ms': [duration]}))[0][0]

    # initialize the prediction df
    pred_df = pd.DataFrame(
        np.array([danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration]).reshape(1, 12),
        columns = ['danceability', 'energy','key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
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
# the func to generate 'Average Popularity Trend' Graph
# def pop_trend(df, start, end):
#     new_df = df[df['track_album_release_date'].between(start, end)].loc[:,['track_album_release_date', 'track_popularity']]
#     new_df = new_df.groupby(new_df.track_album_release_date.dt.year).mean(numeric_only=True)
#     new_df = new_df.reset_index()
#     chart = alt.Chart(
#             new_df, title=alt.TitleParams('Average Popularity Trend', fontSize=22)
#             ).mark_line().encode(
#             x = alt.X('track_album_release_date', axis=alt.Axis(format='Y', grid=False, tickCount=5, titleFontSize=16, labelFontSize=12), title='Release Year'),
#             y = alt.Y('track_popularity', axis=alt.Axis(titleFontSize=16, labelFontSize=12), scale=alt.Scale(zero=False), title='Popularity'),
#             color = alt.value('orange')
#         ).configure_view(
#             stroke=None
#         ).properties(
#             width=600,
#             height=400
#         )
#     return chart.to_html()

app = Dash(__name__)

# app.layout = html.Div([
#     html.Label('Release Date'),
#     html.Div([dcc.DatePickerRange(
#         id = 'date-select',
#         display_format = 'YYYY-MM-DD',
#         start_date = df.track_album_release_date.min().strftime('%Y-%m-%d'),
#         end_date = df.track_album_release_date.max().strftime('%Y-%m-%d'),
#         start_date_placeholder_text='Start date',
#         end_date_placeholder_text='End date',
#         minimum_nights=1800,
#         stay_open_on_select=True,
#         clearable=True
#     )]),
#     html.Iframe(
#         id='average-pop-trend',
#         srcDoc=pop_trend(df, start=df.track_album_release_date.min(), end=df.track_album_release_date.max()),
#         style={'width':'100%', 'height':'800px'}
#     )
# ])

# @app.callback(
#     Output('average-pop-trend', 'srcDoc'),
#     [Input('date-select', 'start_date'),
#      Input('date-select', 'end_date')]
# )
# def update_output(start, end):
#     return pop_trend(df, start, end)



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)