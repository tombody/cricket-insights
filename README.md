## Match level insights for ODI cricket matches
##### Written: 19-November-2018 by *Thomas Body*

In this post I will briefly go over the process of how I used Python to scrape and analyze match level data for one day international (ODI) cricket matches. To accomplish this I used a combination of the BeautifulSoup and the requests library to scrape the data, and then the pandas, sklearn and the fastai library to analyze it. 

An outline of the code for the webscraper can be found [here](https://github.com/tombody/cricket-insights/blob/master/scraper.py), and the code for the analysis can be found [here](https://github.com/tombody/cricket-insights/blob/master/Player%20embeddings.ipynb). The code was written entirely in python 3 using both visual code and jupyter notebooks. The original data was obtained from http://www.howstat.com.

The entire repository can be found on my [tombody](https://github.com/tombody/cricket-insights) github profile page.

### Scraping the data using BeautifulSoup

First of all, let's look at how the data appears on the [website](http://www.howstat.com).

![image](https://github.com/tombody/cricket-insights/blob/master/images/raw_stats.png?raw=true)

The data is in the form of an ODI scorecard, which are the aggregate statistics of each players batting and bowling contributions to the match. In addition to this, there is also information regarding the date, location, and result of the match. 

In order to analyze the data, I needed to get it into a more readily accessible form. To do this, all of the data from the raw html was scraped using BeautifulSoup and then put into a CSV file so that it could be easily imported into a Pandas dataframe for analysis.

Using the requests library, the raw html was pulled from the url using the following code.

``` python
url = 'http://www.howstat.com/cricket/Statistics/Matches/MatchScorecard_ODI.asp?MatchCode='
page = '1619'
source = requests.get(f'{url}{page}').text
soup = BeautifulSoup(source, 'lxml')
```
This returned the raw html, which appeared like this.

```html
<html>
<head><meta content="IE=edge" http-equiv="X-UA-Compatible"/>
<title>Scorecard - 1999-2000 New Zealand v West Indies - 02/01/2000</title>
<meta content="One Day Internationals - Scorecard: 1999-2000 New Zealand v West Indies - 1st ODI - 2nd January, 2000 - Eden Park" name="description"/>
<link href="../../styles/howstat.css" rel="stylesheet"/>
<script type="text/javascript">
  ```
By manually parsing the data, I could find all of the useful information that I was interested in. 

```html
<td class="TextBlackBold8" valign="top" width="160">
                  Match Date:
                </td>
<td class="TextBlack8" valign="top" width="485">
                  2nd January, 2000
                </td>
```
For instance, located in under the `TextBlack8` html class tag was all of the match information. By using the `find_all` function from beautifulsoup and basic python string manipulation methods, I could pull out all of this information and place it into python lists or dictionaries, which I would later import into a pandas dataframe.

```python
repeatables = np.array([item.text.strip() for item in soup.find_all(class_="TextBlack8")])
date, location, result, rr_inn_1, rr_inn_2 = repeatables[[0,1,4,6,8]]

repeatables
>>>     array(['2nd January, 2000', 'Eden Park, Auckland', '50 Overs, Day Match',
       'West Indies', 'New Zealand won by 3 wickets [Duckworth-Lewis]',
       'N J Astle', '&nbsp(50.0 overs @ 5.36 rpo)', '(target 250)',
       '&nbsp(45.1 overs @ 5.54 rpo)',
       'Rain interrupted play. New Zealand target altered to 250 from 46 overs.'],
        dtype='<U71')

date, location, result, rr_inn_1, rr_inn_2
>>>   ('2nd January, 2000',
      'Eden Park, Auckland',
      'New Zealand won by 3 wickets [Duckworth-Lewis]',
      5.36,
      5.54)
```
After a bunch of [trial and error](https://github.com/tombody/cricket-insights/blob/master/scraper_outline.ipynb), the raw data was put togehter into a pandas dataframe. The end result looked something like this.

<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Dismissal</th>
      <th>Runs</th>
      <th>BallsFaced</th>
      <th>Fours</th>
      <th>Sixes</th>
      <th>StrikeRate</th>
      <th>Date</th>
      <th>Location</th>
      <th>Result</th>
      <th>Innings</th>
      <th>Team</th>
      <th>BatSideRR</th>
      <th>BatSideWicksLost</th>
      <th>BatSideScore</th>
      <th>Win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S L Campbell</td>
      <td>st Parore b  Vettori</td>
      <td>51</td>
      <td>67</td>
      <td>6</td>
      <td>0</td>
      <td>76.12</td>
      <td>2nd January, 2000</td>
      <td>Eden Park, Auckland</td>
      <td>New Zealand won by 3 wickets [Duckworth-Lewis]</td>
      <td>1</td>
      <td>West Indies</td>
      <td>5.36</td>
      <td>7</td>
      <td>268</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R D Jacobs</td>
      <td>lbw b  Harris</td>
      <td>65</td>
      <td>61</td>
      <td>7</td>
      <td>2</td>
      <td>106.56</td>
      <td>2nd January, 2000</td>
      <td>Eden Park, Auckland</td>
      <td>New Zealand won by 3 wickets [Duckworth-Lewis]</td>
      <td>1</td>
      <td>West Indies</td>
      <td>5.36</td>
      <td>7</td>
      <td>268</td>
      <td>0</td>
     </tr>
  </tbody>
</table>

The data was split between the bowlers and the batters. All of the scorecards for all ODIs played between 1971 and 2018 were collected. There was over 70000 rows of data for individual batters innings, and over 40000 rows of bowling statistics. 

## Analyzing the data

The analysis of the data is still an [ongoing process](https://github.com/tombody/cricket-insights/blob/master/Player%20embeddings.ipynb), however some preliminary investigation has already been undertaken. Using the k-means clustering algorithm, I wanted to investigate if there were similarities between bowlers using their match level statistics. 

The [k-means algorithm](https://en.wikipedia.org/wiki/K-means_clustering) is an unsupervised machine learning algorithm that clusters data points based on their euclidean distance to a centroid point. Basically, a pre-determined number of clusters *n* is chosen, and *n* number of centroid points are distributed within the space of the data. The data points are then assigned to a cluster based on how close they are to nearest centroid point. Once every data point is assigned, the centroid points are recalculated based on the average distances of every assigned data point that the centroid has. Once that is complete, every data point is then reassigned to its new closest centroid. This process is repeated until no more data points have been reassigned, at which point the algorithm is considered to have convereged on a result. 

The kmeans algorithm can be easily implemented in python using the sklearn library.

```python

kmeans = KMeans(n_clusters=15, n_init=100).fit(X)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = X.index.values
cluster_map['cluster'] = kmeans.labels_
```
The purpose of doing this was to see if there are some similarities between the bowlers based on their match level statistics (i.e. how many maidens did they bowl, how many wickets did they get, how many runs scored against them, bowling average etc). An arbitrary number of 15 clusters were chosen, which would split all of the bowlers into 15 separate groups. As the dataset contained 1600 bowlers (turns out a lot of players have tried their hand at bowling in ODIs over the years), a select group of players were investigated.

The algorithm produced the following distribution of players across the 15 groups.

![image](https://github.com/tombody/cricket-insights/blob/master/images/graph.png?raw=true)

As we can see, most of the bowlers are contained within groups 2, 5, and 11. 

```python

players_of_interest = ['G D McGrath', 'A A Donald', 'Waqar Younis', 'Wasim Akram', 'S K Warne',
                      'M Muralitharan', 'R J Hadlee', 'D L Vettori', 'A Kumble', 'S C G MacGill',
                       'D W Steyn', 'M J Clarke', 'M A Starc', 'D K Lillee']
                       
cluster_map[cluster_map.data_index.isin(players_of_interest)].sort_values(by='cluster')
```          

Investigating a small number of players:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Player</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A Kumble</td>
      <td>2</td>
    </tr>
    <tr>
      <td>D K Lillee</td>
      <td>2</td>
    </tr>
    <tr>
      <td>D L Vettori</td>
      <td>2</td>
    </tr>
    <tr>
      <td>M Muralitharan</td>
      <td>2</td>
    </tr>
    <tr>
      <td>S C G MacGill</td>
      <td>2</td>
    </tr>
    <tr>
      <td>S K Warne</td>
      <td>2</td>
    </tr>
    <tr>
      <td>D W Steyn</td>
      <td>5</td>
    </tr>
    <tr>
      <td>M A Starc</td>
      <td>5</td>
    </tr>
    <tr>
      <td>M J Clarke</td>
      <td>8</td>
    </tr>
    <tr>
      <td>A A Donald</td>
      <td>11</td>
    </tr>
    <tr>
      <td>G D McGrath</td>
      <td>11</td>
    </tr>
    <tr>
      <td>R J Hadlee</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Waqar Younis</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Wasim Akram</td>
      <td>11</td>
    </tr>
  </tbody>
</table>

So what can we infer from these results? There are some definite trends that can be seen here, with a number of spin bowlers being mapped to cluster 2, and a number of pace bowlers being mapped to cluster 11. Without knowing anything about cricket at all, the kmeans algorithm was able to determine that a bowler like Daniel Vettori is more similar to Shane Warne than he is to Wasim Akram, and that Glen McGrath is more similar to Richard Hadlee than he is to Mitchell Starc. There are some oddities in this data, such as Dennis Lillee being in the same group as Muralitharan, but an investigation into the match statistics may show that there are similarites between the two bowlers (both players are for instance known as being effective wicket takers with very good bowling averages and economy rates).

As this is only the start of the investigation into this data, there are a number of questions that I plan to investigate, and I'll be updating this post as I get further along.

- Which batters similar based on match level statistics?
- Can outcomes of matches be predicted based on the the players present?
