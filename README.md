# house-price-predictions
developing a product that will quickly price a consumer's house with an interpretable confidence range

*Pegah Mirghafari*

<br/> 

## Table of Contents
<br/> 

- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Data Dictionary](#Data-Dictionary)
- [Data Cleaning](#Data-Cleaning)  
    - [Data Types and Data Dictionary Interpretation](#Data-Types-and-Data-Dictionary-Interpretation)
    - [NA Values](#NA-Values)
    - [Numeric-Variables](#Numeric-Variables)
    - [Categorical Variables](#Categorical-Variables)
    - [Low-occurrence column values identified in EDA](#Low-occurrence-column-values-identified-in-EDA)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Feature Engineering](#Feature-Engineering)
- [Model Preparation](#Model-Preparation)
- [Modeling](#Modeling)
- [Model Selection](#Model-Selection)
- [Model Evaluation](#Model-Evaluation)
- [Conclusion & Recommendations](#Conclusion-&-Recommendations)
- [APP](#APP)
<br/> 


## Problem Statement
<br/> 
We'd like to create a proof-of-concept model that demonstrates we can achieve a low error metric despite incomplete or missing data. The only currently available dataset is the Ames Housing Price dataset, so we must use the resources at hand to construct an accurate model to show to non-technical stakeholders within the company.
<br/> 


## Executive Summary
<br/> 
Our dataset is composed of ~3000 observations of 80 variables, plus our target variable: the final sale price of the house. Our final evaluation for the model is based on a test set which has been stripped of final sale prices, so in order to evaluate our model we will need to conduct performance analysis against a holdout data set.

We will first reference the provided data dictionary to examine missing values in the data set and subjectively determine their most likely true value.

Next, we will tidy up variables which have small numbers of extreme observations (outliers) by binning the outliers with nearby values. We will also drop some features which we believe are too sparse (e.g. 2997 identical observations and 3 distinct ones) for models to interpret sensibly.

Third, we will conduct feature engineering to create combinations of our different variables based on subject-matter knowledge (For example, total interior square footage of the house may be more meaningful than the square footage of the basement and the square footage of non-basement interior areas on their own), then isolate variables which have high independent predictive power and emphasize them to our models by adding them raised to the 2nd, 3rd, and 1/2 powers.

Finally, we will select appropriate models to test plus a null model, and compare them individually against a weighted ensemble model, which predicts the average of all the underlying model's predictions for each data point.

Our model evaluation metric has been chosen as Root Mean Squared Error. In context, the RMSE of our models reflects the idea that "Our model has an average error of $______ across all predictions." This is a concise and useful metric which is easily representable to stakeholders, or our fictional company's customers. 
<br/> 


## Data Dictionary
<br/> 

[Plaintext Data Dictionary Here](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)

[Formatted Data Dictionary Here](https://www.kaggle.com/c/dsi-us-12-project-2-regression-challenge/data)
<br/> 


## Data Cleaning
<br/> 

### Data Types and Data Dictionary Interpretation
<br/> 
In order to clean and prepare this data correctly, we need to bring the training and testing data together, otherwise we may miss categorical values present in one set but not another, and have other difficulties with replicating our data cleaning on both sets.
to not loose track of which is which i will add a dummy column.
<br/> 

### NA Values
<br/> 

Reviewing our Data Dictionary, we see that 

- an "NA" in "Pool QC" means "No Pool." 

- An "NA" in "Misc Feature" means "No miscellaneous feature."

- An "NA" in "Alley" means "No alley access."

- An "NA" in "Fence" means "No Fence"  

- An "NA" in "FireplaceQu" means "No Fireplace" 

- "NA" in Garage Finish, Condition, Quality, Yr Built, and Type means "No Garage" 

- "NA" in the Basement columns means "No Basement"  

- "NA" in Masonry Veneer means no masonry  

- We'd like to look closer at missing entries in "Garage Cars" and "Garage Area"

- Missing entries in Basement Square Footage most likely represent no basement.

- An "NA" in "Lot Frontage" seems to mean that the property is not connected to streets.

- "NA" in Electrical should be assigned the minimum value of the ordinal system

- Missing values in "Sale Price" are simply the testing data, which have the column removed.
<br/> 

### Numeric Variables
<br/>
We note that Month Sold is given as 1-12, we should convert this to a categorical, as there is no numeric significance to the number of a month.  

We should likewise convert Year Sold into a categorical variable.

We note that there is an entry error in Garage Year Built, we should adjust this number or drop the row after investigating.

We note that MS SubClass is a numeric column which is actually meant to be interpreted as a categorical, we should convert the values to strings.
<br/>

### Categorical Variables
<br/>
As we noted after reviewing the Data Dictionary, a number of ordinal variables share a consistent scale. Let's adjust these to an integer scale.

We will raise all ordinals to the second power to increase the model's sense of scale between a very bad and very good quality.
<br/>

### Low-occurrence column values identified in EDA
<br/>

***for the compelete data cleaning process please visit [this notebook](https://github.com/pegahmirghafari/house-price-predictions/blob/main/house%20price%20preds.ipynb).***

<br/> 


## Exploratory Data Analysis
<br/> 

<img src="./assets/heatmap.png" width="50%" height="50%">

**Analysis:**

- Unsurprisingly, some of the generic "Quality" features are our best standalone predictors. 

- Some of the "Area" features are also important.

- Year Built/Remodeled are important.

- Number of bathrooms, rooms, and fireplaces also seem important.


<img src="./assets/Distribution and RelationshipDistribution and Relationship to Sale Price of Lot Area.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipDistribution and Relationship to Sale Price of Lot Frontage.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipDistribution and Relationship to Sale Price of Lot Shape.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipDistribution and Relationship to Sale Price of Land Slope.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipDistribution and Relationship to Sale Price of Overall Cond.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipDistribution and Relationship to Sale Price of Overall Qual.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipDistribution and Relationship to Sale Price of Year Built.png" width="30%" height="30%">

**Analysis:**

- In our first runthrough, these visualizations allowed us to identify numerous "garbage" columns which we could throw out, or needed to be rebinned due to low frequency of observations


<br/> 


## Feature Engineering
<br/>

<img src="./assets/pre-log price.png" width="50%" height="50%"><img src="./assets/post-log price.png" width="50%" height="50%">
<br/>

***refer to [this notebook](https://github.com/pegahmirghafari/house-price-predictions/blob/main/house%20price%20preds.ipynb) for an indepth look at the feature engineering process***
<br/> 


## Model Preparation
<br/> 
We need to create dummy columns for all nominal and ordinal variables. Next we establish our final sets for modeling, and conduct a train/test split for model evaluation.  We still need a holdout out of our training data, and Set up scaled data for models requiring it.

<br/> 


## Model Selection
<br/> 
We were originally including an Ordinary Least Squares model, but the extent of our feature engineering damages it too much to be useful. Instead, we're running with Lasso, Ridge, ElasticNet, and Stochastic Gradient Descent regression models, along with the XGBoost gradient descent regressor.

<br/> 


## Model Evaluation
<br/> 
our best model has a train R2 Score: 0.9986 and a test R2 Score: 0.94845 and RMSE 18052.98588

<img src="./assets/Distribution and RelationshipLasso Model Residual Plots.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipRidge Model Residual Plots.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipElasticNet Model Residual Plots.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipStochastic Gradient Descent Residual Plots.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipXGBoost Residual Plots.png" width="30%" height="30%"><img src="./assets/Distribution and RelationshipEnsemble Model Residual Plots.png" width="30%" height="30%">

**Analysis:**

- Examining our residual plots against holdout data, we see solid normality of residuals, tolerable but noteworthy heteroscedasticity, and strong linearity in the relationship between predicted and true values. While our model has some issues with predicting very high values, we're confident in the overall fit.

<br/> 


## Conclusion & Recommendations
<br/> 
Our final model has an MSRE of 17894, which can be taken to mean "The average error in predicted sale price for the model is $17,894."

We believe this model is fundamentally sound, as it violates only one LINE assumption (homoscedasticity), and does not show signs of overfitting based on the difference in R2 scores for training and holdout data. 

While the low-level projection of the model (Sale Price plus or minus MSRE) is easily interpretable, it is important to note that we have not selected models whose coefficients can be interpreted straightforwardly, particularly due to transformations of our independent variables. This is still addressable in a production context with some work, but is an evaluation factor depending on the context of application. Since the uniform-weight ensemble model is simply the mean of all 5 input model's predictions, we can distinguish how much weight each individual X-variable in a given prediction, and what that impact had on the overall valuation.

We believe this is a strong proof-of-concept for an on-demand housing valuation model.

<br/> 


## APP
<br/> 

**plrease refere to this link for the [app](http://house-price-estimator.pegahmirghafari.com/)**

<br/> 
