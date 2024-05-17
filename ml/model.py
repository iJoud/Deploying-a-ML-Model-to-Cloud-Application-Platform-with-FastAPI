from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from .data import process_data
import os 

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # initialize model
    model = ExtraTreesClassifier(n_estimators=50)
    # train the model
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble._forest.ExtraTreesClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def performance_on_data_slices(model, data, cat_features, encoder, lb):
    """
    outputs the performance of the model on slices of the data.

    Inputs
    ------
    model : sklearn.ensemble._forest.ExtraTreesClassifier
        Trained machine learning model.
    data : pd.DataFrame
        test data.
    cat_features : list
        list of categorical features in the data.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        trained encoder for the data.
    lb : sklearn.preprocessing._label.LabelBinarizer
        trained lb for the data.

    Returns
    -------
    output_file_path : str
    """

    output_file_path = os.path.join(os.getcwd(), "slice_output.txt")

    with open(output_file_path, "w") as f:
        for feature in cat_features:
            unique_vals = data[feature].unique()

            for val in unique_vals:
                # filter data based on current data slice
                current_data = data[data[feature] == val]

                # process data
                X_test, y_test, _, _ = process_data(
                    current_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
                )

                # get predictions and compute performance metrics
                predictions = inference(model, X_test)
                precision, recall, fbeta = compute_model_metrics(y_test, predictions)

                # printout results
                print(f"Model performance for category {val} in column {feature}:")
                print("Precision:", precision)
                print("Recall:", recall)
                print("F-beta:", fbeta)
                print()

                # write results to the file 
                f.write(f"Model performance for category {val} in column {feature}:\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"F-beta: {fbeta}\n\n")
    
    return output_file_path

