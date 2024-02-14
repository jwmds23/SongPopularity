## Motivation and Purpose

**Our role**: Data Analytics Team in Spotify

**Target audience**: Executive in the Market Department

**Purpose**: To occupy the musical market as a large online media service provider is all about capturing the popularity trend. 
It is difficult to know which aspects of a song dominate its popularity when there are thousands of options for song purchase. 
As a role of Spotify’s data analytics team, we proposed to create a tool for assisting our marketing and management teams 
with accessible insights into the music trends, named as Spotify Songs Popularity Breadth Dashboard. 
This application is designed to be user-friendly and allows for an in-depth analysis of song features and their influence on song popularity. 
By understanding these dynamics, our marketing executives and decision-makers can clearly see what drives a song's success. 
This insight is vital for shaping marketing strategies and making informed choices about song selection, 
ensuring our efforts align with listener preferences and market demands.

## Description of the data
We will visualize a dataset of about 30,000 songs from Spotify. 
The songs were released between 1957 and 2020, and each song has 23 features. 
Those features describe:
- what the song is (nominal features: `track_id`, `track_name`, `track_artist`); 
- the song's popularity (`track_popularity`, ranging from 0 to 100 with the center around 45); 
- which album and playlist the song belongs to (nominal features: `track_album_id`, `track_album_name`, 
`track_album_release_date`, `playlist_name`, `playlist_id`, `playlist_genre`, `playlist_subgenre`); 
- and 12 key features of each song (numeric features with different scales: `danceability`, `energy`, `key`, 
`loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`).
The relationship between popularity and 12 key features will be the main visualization 
and other features will serve as the filter, allowing the user to explore more details.

## Research questions and usage scenario
**Research questions**
- what kinds of factors are highly correlated to song popularity or affects it?
- how to predict the potential popularity of a new song based on its features?

**usage scenario**

Will is a member of the marketing team at Spotify who wants to [investigate] the current characteristics of the stream music market on Spotify 
and [detect] music popularity trends to adjust marketing strategies for future music and artist selection. 
When he logs in to this app, he will see an overview of the distributions of the popularity, 
the distributions of all genres, the top 10 songs/artists, and popularity of all 12 available features, 
including Danceability, Energy, Speechness, Loudness, Key and other features in this dataset. 
By [multi-selecting] different features, he can [explore] the relationship between the selected features and the song's popularity. 
To inspect a more detailed and specific area, Will can filter out the data by album release date, 
music genre/subgenre or artist name to narrow down the search scope. 
Also he can [predict] a song’s popularity based on current data by inputting/selecting some feature values 
to decide whether it is worthwhile to purchase this song for Spotify.  
