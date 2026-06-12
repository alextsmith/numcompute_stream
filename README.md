# numcompute_stream

A machine library for classification based on decision-trees that supports streaming data. Includes methods for calculating streaming statistics, performing preprocessing procedures such as scaling, imputing missing values and encoding categorical variables, as well as fitting decision tree and ensemble models. It provides a pipeline for preprocessing, model fitting and prediction. Visualisation tools for classification metrics are included. 

## Core functionality

### stats.py
+ Allows online calculation of statistics for streaming data. Designed for calculating statistics for features with data inputted as a matrix with rows representing observations and columns representing features.
  + mean, variance, standard deviation

### preprocessing.py
+ Provides functions for preprocessing
    + StandardScaler
    + Inputer (inputes missing values with feature mean)
    + OneHotEncoder
 
### metrics.py
+ Allows online calculation of classification metrics for streaming data.
    + Accuracy, precision, recall, confusion matrix
 
### tree.py
+ Fits a decision tree for classification. Allows online updates of tree for streaming data.

### ensemble.py
+ Fits ensemble of decision trees using bootstrap aggregating (bagging) for classification. Allows online updates of ensemble for streaming data.

### pipeline.py
+ Provides a pipeline for preprocessing, model fitting and prediction. Allows online updates of preprocessing parameters and fitted models for streaming data.

### stream.py
+ Applies pipeline to streaming data and updates classification metrics.

### visualisation.py
+ Allows visualisation of model performance as measured by classification metrics.
    + Plot metric over time
    + Compare models
    + Compare predictions vs ground truth

### Demo

Refer to demo.ipynb for an example usage.
