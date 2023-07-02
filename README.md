EDSA_Movie 
First of I tried a Content based filtering as there was a lot of data to support my idea.
First I grouped the tag data together by the movieId to get all the tags together. Then I droped the 'userId', 'timestamp'
from the tags data.
I Then I replaced the movies gernes | with a space that I get all the keywords and droped all the duplicats.
I then merged the training data with the movie data and droped the 'timestamp'
Then in the imdb_data I replaced the | with a space in the following coloumns title_cast,director and plot_keywords
After that I merged it with the movie data that haaf been merged with th train data already and extracted the year from the title 
into another field. Then I droped the 'timestamp'
Then I displayed all the rating to make have a look how the users are rating the movies.
After trying to process the data, My laptop allways ran out of Memory and I had to find another solution. 
After building a Server with 64GB Ram it finaly executed and after 50hours of processing it gave me a score of 1.1 on Kaggle.

After that I decided to use Colleborative filtering.
In the training data there is more than enough data to use  Colleborative filtering.
First I checked how many numique users and movies I have.
Then I removed the timestamp from the training data.
After that I checked for outliers and saw there is an outlier. The first user with over 12000 ratings. that is almost impossible so I
removed him. 
Then I did a train test split

BaselineOnly
The BaselineOnly did not prefore that well and was alot higher than my there algorithems

NMF
The NMF was also much higher than the SVDpp
SVDpp
Iused the SVDpp to do the split as I got the best results. I tryed to do a GridSearchCV, but it took too long so T tried to manually set 
the hyperparametes. These where the best values I got SVDpp(n_factor=100, n_epochs=20, lr_all=0.005, reg_all=0.03)
Then I ran it on my test data and exported it to an CSV.


