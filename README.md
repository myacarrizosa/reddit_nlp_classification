# Executive Summary 
## Problem Statement
Reddit is a social media platform that consists of pages dedicated to specific topics, called subreddits, where users can post and comment within the guidelines of that page and topic. Two of these subreddits are the r/Religion page, and the r/AskPhilosophy page. We are a sociology research group who has theorized that the thematic elements of religious discussions and philosophical discussions are very similar; we anecdotally see overlap in areas like ethics, morality, meaning, human nature, and more. Because of these thematic similarities, we are interested in seeing if a machine learning model could predict whether a submission was taken from the Religion or AskPhilosophy subreddit. 


## Description of Data
The data used in this project is text data consisting of Reddit submissions (title and text) posted on the r/AskPhilosophy and r/Religion subreddits. The AskPhilosophy subreddit was chosen instead of the Philosophy subreddit because the Philosophy subreddit mainly consisted of links to external pages and videos, which wouldn't provide the text data needed to compare and classify the pages. After data cleaning, there were 4945 data records.

## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|subreddit|integer|all_posts| Identifies the subreddit that submission was posted on. AskPhilosophy is coded as 0 and Religion is coded as 1.| 
|title|object|all_posts| The title text from a submission.|
|selftext|object|all_posts| The body text from a submission.| 
|all_text|object|all_posts| The title and body text from a submission.| 
|text_no_rp|object|all_posts| The title and body text from a submission with the words religion and philosophy removed.|
|status_length|integer|all_posts| The number of characters in a submission's body text.| 
|status_word_count|integer|all_posts| The number of words in a submission's body text.| 
|title_len|integer|all_posts| The number of characters in a submission's title.| 
|title_word_count|integer|all_posts| The number of words in a submission's title.|  

## Data Collection
Data was pulled from the r/AskPhilosophy and r/Religion subreddits using Pushshift's API. 

## Analysis
#### Exploratory Data Analysis
A few operations were performed as an intial exploration of the subreddit data. 

First, we created variables for status length, status word count, title length, and title word count. We then created distibutions of the status word count for posts from the philosophy and religion subreddits respectively to visualize and compare the spread of status word count for each subreddit. They both were right-skewed and had a clear mode of zero. We also created distibutions of the title word count for posts from the philosophy and religion subreddits respectively to compare the spread of title word count for each subreddit. They both were right-skewed, but the philosophy subreddit had a larger range of title word counts. 

Next, we ran our text data through a CountVectorizer transformer so that the frequency of words could be examined. After transforming the data, we found the 10 most common words for each subreddit to compare to one another. There was some overlap in these words, with words like 'just', 'people', 'know', and 'think' being found in the top ten words in both subreddits. 

## Classification Modeling 
We fit four different types of models to find which model best performed at classifying which subreddit our reddit posts came from. 

Prior to fitting our models, we split our data into a training set and a test set so that we would be able to see how well our model performs on new data after fitting on the training data. 

Next, we created a null model by identifying the majority class in our data. For any given submission, the null model would classify the submission as being from the AskPhilosophy subreddit, since it was the majority class. This model has an accuracy score of 0.57.

Although we already transformed our text data using CountVectorizer for our EDA, we wanted to be able to gridsearch over the best hyperparameters for CountVectorizer, so for each model we re-fed the data through a pipeline including CountVectorizer. 

#### Logistic Regression 
For our logistic regression, we built a pipeline with CountVectorizer and Logistic Regression so we could gridsearch for the best hyperparameters. We used GridSeach to check whether CountVectorizer stop words should be none or english, whether CountVectorizer ngram_range should be only single words or single words and word pairs, and whether CountVectorizer max_df should be 0.9 or 1.0. We also GridSearched over Logistic Regression to check whether logreg_solver should be lbfgs or liblinear. The GridSearch returns the combination hyperparameters that built the best performing model. The Logistic Regression best parameters were: 'cvec__max_df': 0.9,
 'cvec__ngram_range': (1, 1),'cvec__stop_words': 'english',
 'logreg__solver': 'liblinear'. 
 
We then fit this model on our X_train data. This model had an accuracy score of 0.98 on the train data and 0.89 on the test data. On test data, the model sensitivity was 0.91, the model specificity was 0.90, and the model precision was 0.87. This was a pretty well performing model, with slight evidence of overfitting, seen by the decrease in performance from train data to test data. 


#### K Nearest Neighbors Classifier
For our K Nearest Neighbors Classifier, we built a pipeline with CountVectorizer, StandardScaler and KNeighborsClassifier so we could gridsearch for the best hyperparameters. StandardScaler was included in this pipeline because it is necessary to scale data before using KNN models. We used GridSeach to check whether CountVectorizer stop words should be none or english, whether CountVectorizer ngram_range should be only single words or single words and word pairs, and whether CountVectorizer max_df should be 0.9 or 1.0. We did not test any hyperparameters for StandardScaler in GridSearch, but we set with_mean = False. We also GridSearched over KNeighborsClassifier to check whether n_neighbors should be 3, 5, or 7, whether weights should be uniform or distance, and whether metric should be minkowski, euclidian, or manhattan. The GridSearch returns the combination hyperparameters that built the best performing model. The KNN best parameters were: 'cvec__max_df': 0.9, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': 'english', 'knn__metric': 'minkowski', 'knn__n_neighbors': 3, 'knn__weights': 'distance'.
 
We then fit this model on our X_train data. This model had an accuracy score of 0.9997 on the train data and 0.56 on the test data. This model was severely overfit. On test data, the model sensitivity was 0.93, the model specificity was 0.29, and the model precision was 0.50. This model had difficulty classifying AskPhilosophy posts, labeling them as Religion, as shown by the very low specificity score. 

 

#### Random Forests Decision Trees
For our Random Forests Decision Trees Classifier, we built a pipeline with CountVectorizer and Random Forest Classifier so we could gridsearch for the best hyperparameters. We used GridSeach to check whether CountVectorizer stop words should be none or english, whether CountVectorizer ngram_range should be only single words or single words and word pairs, and whether CountVectorizer max_df should be 0.9 or 1.0. We also GridSearched over Random Forest Classifier to check whether n_estimators should be 100, 125, or 150, whether max depth of branches should be 3, 5, or 8, and whether the minimum samples for a leaf should be 1, 2, or 5. The GridSearch returns the combination hyperparameters that built the best performing model. The Random Forest best parameters were: 'cvec__max_df': 0.9, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': None, 'rf__max_depth': 8, 'rf__min_samples_leaf': 1, 'rf__n_estimators': 150. 
 
We then fit this model on our X_train data. This model had an accuracy score of 0.75 on the train data and 0.73 on the test data. On test data, the model sensitivity was 0.40, the model specificity was 0.99, and the model precision was 0.96. This model had difficulty classifying Religion posts, frequently labeling them as AskPhilosophy, as shown by the very low sensitivity score. 

#### AdaBoosted Decision Trees
For our AdaBoost Classifier, we built a pipeline with CountVectorizer and AdaBoost Classifier so we could gridsearch for the best hyperparameters. We used GridSeach to check whether CountVectorizer stop words should be none or english, whether CountVectorizer ngram_range should be only single words or single words and word pairs, and whether CountVectorizer max_df should be 0.9 or 1.0. We also GridSearched over AdaBoost Classifier to check whether n_estimators should be 100, 125, or 150, and whether the learning rate that weights incorrect classifications each iteration of the model should be 1.0 or 1.5. The GridSearch returns the combination hyperparameters that built the best performing model. The AdaBoost best parameters were: 'cvec__max_df': 0.9, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': None, 'ada__learning_rate': 1.0, 'ada__n_estimators': 150. 

    
We then fit this model on our X_train data. This model had an accuracy score of 0.91 on the train data and 0.87 on the test data. On test data, the model sensitivity was 0.89, the model specificity was 0.89, and the model precision was 0.86. This model performed almost as well as the Logistic Regression model and had less variance. 

#### Logistic Regression on Data With No Subreddit Names 

We decided to see if our model could still perform well at predicting subreddit classification if the submission did not contain the words 'religion' or 'philosophy', as these words likely were pretty strong indicators of what subreddit a submission came from. We created a new column of the combined title and body text of a submission with the words 'religion' and 'philosophy' removed. We then created a pipeline of CountVectorizer and Logistic Regression and set the hyperparameters equal to the best parameters that GridSearch identified in our previous Logistic Regression model ('cvec__max_df': 0.9, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': 'english', 'logreg__solver': 'liblinear'). 

We then performed a train-test-split on the text with words removed and fit this model on our X_train data. This model had an accuracy score of 0.99 on the train data and 0.90 on the test data. On test data, the model sensitivity was 0.90, the model specificity was 0.89, and the model precision was 0.85. This model performed as well as the Logistic Regression model with 'religion' and 'philosophy' left in, showing that our model was still able to classify submissions based on other features besides direct references in the text to the subreddits. 



#### Data Visualizations 
To visualize the performance of our models, we created confusion matrix plots that displayed the counts of true positives, true negatives, false positives, and false negatives. These visualizations helped us see which models were peforming best, as well as see what types of posts some models were struggling at classifying (for example, the random forests model did well classifying AskPhilosophy posts correctly but struggled at classifying Religion posts, which can be clearly seen in the confusion matrix). 


![CM]('./visualizations/rfcm.jpg')

We also looked at instances where the model misclassified submissions to hypothesize why these posts were difficult for our model to classify.  

## Conclusions and Recommendations 

By analyzing text data from the Religion and AskPhilosophy subreddits using natural language processing and classification models, a machine learning model can predict which subreddit the text data came from with high accuracy. Our logistic regression model was able to classify submisssions with 89% accuracy, a sensitivity of 0.91, a specificity of 0.90, and a precision of 0.87. 

Even without key words like 'religion' and 'philosophy', the model is able to pick up on differences in the way that 