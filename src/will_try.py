import altair as alt
import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

# load the data set
df = pd.read_csv('../data/processed/spotify_songs_processed.csv', parse_dates = ['track_album_release_date'], index_col=0)

# the func to generate 'Average Popularity Trend' Graph
def pop_trend(df, start, end):
    new_df = df[df['track_album_release_date'].between(start, end)].loc[:,['track_album_release_date', 'track_popularity']]
    new_df = new_df.groupby(new_df.track_album_release_date.dt.year).mean(numeric_only=True)
    new_df = new_df.reset_index()
    chart = alt.Chart(
            new_df, title=alt.TitleParams('Average Popularity Trend', fontSize=22)
            ).mark_line().encode(
            x = alt.X('track_album_release_date', axis=alt.Axis(format='Y', grid=False, tickCount=5, titleFontSize=16, labelFontSize=12), title='Release Year'),
            y = alt.Y('track_popularity', axis=alt.Axis(titleFontSize=16, labelFontSize=12), scale=alt.Scale(zero=False), title='Popularity'),
            color = alt.value('orange')
        ).configure_view(
            stroke=None
        ).properties(
            width=600,
            height=400
        )
    return chart.to_html()

app = Dash(__name__)

app.layout = html.Div([
    html.Label('Release Date'),
    html.Div([dcc.DatePickerRange(
        id = 'date-select',
        display_format = 'YYYY-MM-DD',
        start_date = df.track_album_release_date.min().strftime('%Y-%m-%d'),
        end_date = df.track_album_release_date.max().strftime('%Y-%m-%d'),
        start_date_placeholder_text='Start date',
        end_date_placeholder_text='End date',
        minimum_nights=1800,
        stay_open_on_select=True,
        clearable=True
    )]),
    html.Iframe(
        id='average-pop-trend',
        srcDoc=pop_trend(df, start=df.track_album_release_date.min(), end=df.track_album_release_date.max()),
        style={'width':'100%', 'height':'800px'}
    )
])

@app.callback(
    Output('average-pop-trend', 'srcDoc'),
    [Input('date-select', 'start_date'),
     Input('date-select', 'end_date')]
)
def update_output(start, end):
    return pop_trend(df, start, end)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)