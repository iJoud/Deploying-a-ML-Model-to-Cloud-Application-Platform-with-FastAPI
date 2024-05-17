# Model Card

## Model Details
Joud Alghamdi created the model as a prerequisite to pass the MLOps Nanodegree project. It is a scikit-learn ensemble model called Extra Trees Classifier.

## Intended Use
The model is used to predict individual salary, whether it's above 50K or less than 50K, based on given data.

## Training Data
Training data can be accessed [here](https://archive.ics.uci.edu/dataset/20/census+income).

## Evaluation Data
We used 20% of the data as the testing set.

## Metrics
Model performance has been measured using F-beta score, precision score, and recall score metrics from scikit-learn.

#### Model Performance: 

**precision:**  0.6739766081871345 

**recall:**  0.5967637540453075 

**fbeta:**  0.6330243734981119


## Ethical Considerations
Obtained data could include sensitive and personal information.

## Caveats and Recommendations
For further improvements, hyperparameter tuning and k-fold cross-validation techniques can be used.
