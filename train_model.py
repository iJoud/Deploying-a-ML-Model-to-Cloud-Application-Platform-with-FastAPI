# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, performance_on_data_slices
import pandas as pd
import pickle


data_path = './data/census.csv'
model_folder_path = './model/'

# Add code to load in the data.
data = pd.read_csv(data_path)

# remove all spaces in col names
data.columns = data.columns.str.strip()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train model
model = train_model(X_train, y_train)
predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print("model metrics: ")
print("precision: ", precision, "\nrecall: ", recall, "\nfbeta: ", fbeta)

# save model and encoders
pickle.dump(model, open(model_folder_path + 'model.pkl', 'wb'))
pickle.dump(encoder, open(model_folder_path + 'encoder.pkl', 'wb'))
pickle.dump(lb, open(model_folder_path + 'lb.pkl', 'wb'))

# get the model performance on data slices of categorical features
output_file_path = performance_on_data_slices(model, data, cat_features, encoder, lb)
print('results saved on path: ', output_file_path)