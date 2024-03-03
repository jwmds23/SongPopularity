## Reflection for Milestone2

### Summary Tab

**Expected Functionalities**

In the `summary` tab, we allowed users to have an overall understanding of the spotify popularity data and do some exploratory data analysis.

**What have been done**

The filter menu on the left side can be used to set the data range of all charts on the right side. By assigning specific date range, artist(s), genre(s) and subgenre(s) it can be seen that a bar plot with one linked pie chart showing the distribution of popularity and genres, the line graph showing the trend of average popularity, two bar charts showing the top 10 artists and songs, and one scatter plot with one linked bar chart showing the relationship of any two features and the respective popularity level, along with the distribution of genres.

**What should still be modified or improved**

Currently, we are still working on the interactive performance of charts so some have interactive functionalities but others don’t have yet.

### Feature Tab

**Expected Functionalities**

In the `Feature` tab, users can explore the popularity of Spotify songs by analyzing various technical music aspects. 

**What have been done**

Results can be filtered by release date, genre, subgenre, artist, and specific song features to see the popularity distribution in bar charts. 
Users can also plot the distribution for different popularity levels by interacting with the legends. Tooltips on each color-coded bar provide the count of song records within a specific popularity range for the selected feature. 

**What should still be modified or improved**

We plan to enhance the page's appearance by refining the format, color styles, and layout.


### Prediction Tab
**Expected Functionalities**

In the `prediction` tab, we allowed users to use the slider, dropdown menu, or input box to input the value for various features of the song. The popularity of the song will be predicted using the models developed with the data set. 

**What have been done**

The predicted popularity result will be displayed in a circular graph with the numeric result in the center and the feature of the song will be displayed in a radar graph which can give the user a graphical view of how the song looks like.

**What should still be modified or improved**

Currently, the prediction tab is well-functional if users input values for all features. The dashboard may crash if users leave out certain features and click the ‘Apply’ button. We need to improve by handling the possible errors in a more user-friendly way.

