from dash import Dash, dcc, html, dash_table, Input, Output, State, dependencies
import dash_bootstrap_components as dbc
import requests
import pandas as pd

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

df = pd.read_csv('../data/processed/spotify_songs_processed.csv', index_col=0)
df_new=df[['track_id','track_name','track_artist','track_popularity']].drop_duplicates()
popularity_by_songs = df_new.groupby('track_name').agg({
    'track_popularity': 'mean',
    'track_artist': 'max',
    'track_id': 'max'
}).reset_index()
top10_songs = popularity_by_songs.nlargest(10,"track_popularity")

def music_play(track_id):
    client_id = 'e07829826a52472bb6cd2fe4ef80515e'
    client_secret = '34a0c73b34864d398ac393c73ac03a9e'

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

app.layout = html.Div([
    dash_table.DataTable(
        id = 'top-10-songs',
        columns=[
            {"name": "Song", "id": "track_name"},
            {"name": "Artist", "id": "track_artist"},
            {"name": "Popularity", "id": "track_popularity"},
            {'name': 'Track ID', 'id': 'track_id'}
        ],
        hidden_columns=['track_id'],
        data=top10_songs.to_dict('records'),
        style_cell_conditional=[
            {'if': {'column_id': 'track_name'}, 'textAlign': 'center', 'cursor': 'pointer'}
        ]),
        html.Div(id='redirect-instructions', children=
                 [html.A("Listen On Spotify", id='link')])
])

@app.callback(
    Output('link', 'href'),
    Input('top-10-songs', 'active_cell'),
    State('top-10-songs', 'data')
)
def update_audio_src(active_cell, rows):
    if active_cell:
        try:
            track_id = rows[active_cell['row']]['track_id']
            preview_url = music_play(track_id)
        except:
            return ""
        return preview_url
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)