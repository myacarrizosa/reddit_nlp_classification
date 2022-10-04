# Executive Summary 
## Problem Statement
Reddit is social media platform that   . There is 


## Description of Data
The data used in this project is text data consisting of Reddit submissions (title and text) posted on the r/AskPhilosophy and r/Religion subreddits. 

## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|MS Zoning|object|selected_data| Identifies the general zoning classification of the sale. Dummy variables were created for each possible value of MS Zoning and coded 1 in the column that corresponded to the value of the data record| 
|Utilities|object|selected_data| The type of utilities available. Dummy variables were created for each possible value of Utilities and coded 1 in the column that corresponded to the value of the data record| | 
|Neighborhood|object|selected_data|Physical location within the Ames city limits. Dummy variables were created for each possible value of Neighborhood and coded 1 in the column that corresponded to the value of the data record| | 
|House Style|object|selected_data| Style of dwelling. Dummy variables were created for each possible value of House Style and coded 1 in the column that corresponded to the value of the data record|
|Overall Qual|integer|selected_data|Rating of the overall material and finish of the house from 1 to 10|
|Overall Cond|integer|selected_data|Rating of the overall condition of the house from 1 to 10|
|Year Built|integer|selected_data|The original construction date of the house|
|Heating QC|object|selected_data|Heating quality and condition, rated either Excellent, Good, Average, Fair, or Poor. Dummy variables were created for each possible value of Heating QC and coded 1 in the column that corresponded to the value of the data record|
|Central Air|object|selected_data| Whether the house has central air conditioning, either yes or no. Dummy variables were created for each possible value of Central Air and coded 1 in the column that corresponded to the value of the data record|
|TotRms AbvGrd|integer|selected_data|Total rooms above grade|
|Functional|object|selected_data|Home functionality, rated either Typical, Minor Deductions 1, Minor Deductions 2, Moderate Deductions, Major Deductions 1, Major Deductions 2, Severely Damaged, or Salvage Only. Dummy variables were created for each possible value of Functional and coded 1 in the column that corresponded to the value of the data record|
|Garage Cars|integer|selected_data|Size of garage in car capacity|
|Fireplaces|integer|selected_data|Number of fireplaces|
|1st Flr SF|integer|selected_data|First floor square feet|
|2nd Flr SF|integer|selected_data|Second floor square feet|
|Full Bath|integer|selected_data|Number of full bathrooms above grade|
|Half Bath|integer|selected_data|Number of half bathrooms above grade|

Feature information sourced from Ames, Iowa Assessor's Office
## Analysis
#### Exploratory Data Analysis
A few operations were performed as an intial exploration of the data to look for trends and potential areas of interest. 

First, we selected a narrowed down list of variables that we thought might have an influence on sale price. This list was comprised of With this smaller set of variables, we checked for null values, which we then filled with the most frequent value in the column. We then created histograms to look at the distribution of values for all the categorical variables (MS Zoning, Utilites, Neighborhoopd, House Style, Heating QC, Central Air, and Functional). For the numerical data, we created box plots to look at the spread and identify outliers. Every numerical variable except for Half Baths had some outliers, but when examining the outliers further and contextualizing them with the information on the variables, none of them seemed to be due to error. Because of this, we left them in for most of our models.

We then created dummy variables for each of our categorical variables so that they could be included in linear regression. For each dummy variable, one of the categories was set as the baseline.

MS Zoning Baseline: Agriculture  
Utilities Baseline: All Pub  
Neighborhood Baseline: Blmngtn  
House Style Baseline: 1.5 Fin  
Heating QC Baseline: Ex  
Central Air Baseline: No  
Functional Baseline: Maj1  

Next, we checked for multicollinerity by finding correlations between variables we thought might be related. We found a correlation of 0.81 between the variable for 2nd Floor House Style and 2nd floor square feet, indicating multicollinearity. To address this, we made an interaction term of the two by adding them together to be used in place of the two variables in our model. We also made an interaction term between Overall Quality and Overall Condition by multiplying those two variables together, to see if the combination of the two influences sale price. Finally, we made an interaction term by adding first and second floor square feet to see if the total square feet above ground has an influence on sale price. 

We then created scatterplots of the correlations between some of our X variables and sale price to get an initial idea of which variables might be more highly influencing sale price. We found positive relations between sale price and variables like 1st and 2nd floor square feet, suggesting that these variables would be instrumental in the models we fit to predict sale price. 

### Linear Modeling 
We fit several different models using different combinations of selected variables to find which combination of variables explained the highest percent of the variability in sale price. 

Prior to fitting our models, we split our data into a training set and a verification set so that we would be able to see how well our model performs on new data after fitting using the training data. We then standardized the X-variables to ensure that the large ranges in scale (for example, square feet measured in the thousands and number of rooms measured in ones) weren't creating issues when fitting our models. 

Next, we created a null model by taking the mean of sale price. For any given inputs, the null model will return the average price. This allows us to compare our models to a baseline to see if they are performing better than if we just guessed using the average price. 

For our first linear regression, we fit a model using all of the variables we selected for our narrowed down Selected Variables dataset from the larger housing dataset. It was likely that this model would be overfit and not be the best choice to predict sale price. The model had a coefficient of determination of 0.842 on the validation data, meaning that 84.2% of the variation in sale price could be explained by our X variables.

Next we fit a model (referred to as Model 2) that dropped the variables that were possibly multicollinear, 2nd floor square feet and 2nd floor house style. This model had a coefficient of determination of 0.842 on the validation data, meaning that 84.2% of the variation in sale price can be explained by our X variables. This was the same as the all variable model, but had the advantage of not including multicollinear variables. 

We fit four more variations of our models, dropping variables like heating quality and condition, MS zoning, fireplaces, and functionality ratings. None of these model performed as well as Model 2.

We noticed in all of the scatterplots of the relation between our actual sale prices and predicted sale prices that the plots seemed to fan out. We then decided to remove outliers from the variables Total Rooms Above Grade and 1st Floor Square Feet to see if this resulted in a more linear relationship. We chose these variables because in our box plots constructed in our early data analysis, these two variables had the most amount of outliers. We then fit a model with the same variables as Model 2, using the data with outliers removed. This model saw an increase in the coefficient of determination for the data it was trained on, but did not see improvement when tested on validation data, stay around 0.84. Removing the outliers did not seem to help fit a model with stronger predictive ability. 

We noticed that there was a slight curve in the data in all of our scatterplots of the relation between actual and predicted sale price. We thought this could be addressed by constructing a model for the log of sale price, instead of sale price. This model had a coefficient of determination of 0.876 on the validation data, meaning that 87.6% of the variation in sale price could be explained by our X variables. This was the biggest improvement yet in our models. The scatterplot of the relation between predicted log sale price and actual log sale price was more linear than the scatterplots of the other models. A drawback of using this model is that other metrics, like the root mean square error, cannot be easily compared to previous models due to their difference in scale of the residuals. Additionally, when using this model, the output will need to be transformed back into sale price. This model remained our best performing model. 

Because several of our earlier models were all performing similarly, we wanted to see if we could find a simplified model that still maintained roughly the same performance as our earlier models. We decided to shoot for a coefficient of determination of around 0.84, while parsing down our model as much as possible. The simplest model we came up with had a coefficient of determination of 0.839 while only needing information on overall quality and condition of the house, total rooms above grade, first and second floor square feet, number of half and full baths, and neighborhood. 

#### Regularizations

Because our best model, the log of sale price, consisted of so many x variables, we wanted to see if regularizing the model had any impact on its performance. We used both ridge regularization and lasso regularization on the model. We tested a variety of values for our hyperparameter alpha, and found that for ridge regularization the best value for alpha was 70.55, and for lasso regularization the best value for alpha was 0.0023. Both methods of regularization resulted in virtually identicial train and validation coeffients of determinations, meaning there was very little if any error due to variation. However, there was not improvement in R^2 compared to the non-regularized log model; the performance of the two models were both an R^2 value of 0.87. Because error due to variation was not really a concern with the non-regularized model, we decided it wasn't necessary to use the regularized model.

We then used ridge regularization on our Model 2 with the outliers removed. This model had the greatest decrease in coefficient of determination from train data to validation data, indicating potential error due to variance. We wanted to see if regularizing would improve model fit. We tested a variety of values for our hyperparameter alpha and found that 10.72 was the best value for alpha. After regularizing, the difference in train and validation R^2 only decreased by 0.003, so the regularization did not have much of an impact. 

After testing many models, our best performing model was our model predicting the log of sale price, with an R^2 of 0.876. Our simiplest model that still performed comparably to the majority of other models was our model that took the following X variables: overall quality and condition of the house, total rooms above grade, first and second floor square feet, number of half and full baths, and neighborhood. This model had an R^2 of 0.839. 



#### Data Visualizations 
To visualize the fit of our models, we created scatterplots of the relation between the sale prices our model predicted and the actual sale price of the property using the validation data we set aside to test our trained models on. These scatterplots allowed us to detect the curved pattern in these residuals in our original models, which indicated that we might need to take the log of sale price. The scatterplot of the model with the log of sale price shows a more linear relation between predicted and actual price, which is a reflection of the higher R^2 and better predictions of the model. 



## Conclusions and Reccomendations 

Our models to predict sale price were able to explain a range of 82-87% of the variation in sale price given the x variables. Depending on the goals and resources of the user, we recommend one of two models. 

If you have the ability to collect extensive information on a property, we recommend using the model that predicts the log of the sale price. The variables in that model explain 87% of the variation in the log in sale price. The prediction of the log of sale price can be easily transformed back to sale price by raising e to the power of the predicted value. This model is most likely to generate predictions close to the true sale price of a property. 

If you are trying to predict sale price but don't have the resources or access to collect information on an exhaustive list of features, we recommend using the simple model we fit. With just information on the overall quality and condition, total rooms above grade, first and second floor square feet, number of half and full baths, and neighborhood, this model can explain 84% of the variance in sale price. This model is a good choice if you want to save time and money collecting data on the house while still getting a comparable sale price prediction. 

These models were designed for prediction, not necessarily inference. Although multicollinearity was checked for in variables that seemed related and addressed when found, a thorough check of every variable correlated with the others was not conducted, and some instances of multicollinearity may have been missed.