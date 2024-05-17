import pytest
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, performance_on_data_slices


@pytest.fixture(scope="session")
def data():
    data_path = './data/census.csv'

    # load data
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()

    return data


@pytest.fixture(scope="session")
def train_test_data(data):

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

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    return X_train, y_train, X_test, y_test


def test_input_data(data):
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_train_model(train_test_data):

    X_train, y_train, _, _ = train_test_data

    model = train_model(X_train, y_train)

    assert isinstance(model, ExtraTreesClassifier)


# def test_inference(train_test_data):

#     model_path = 'model/model.pkl'

#     _, _, X_test, y_test = train_test_data

#     model = pickle.load(open(model_path, 'rb'))
#     preds = inference(model, X_test)
    
#     assert isinstance(preds, np.ndarray)
#     assert len(preds) == len(y_test)

def test_performance_on_data_slices(data):
    model_path = 'model'

    model_folder_path = './model/'

    with open(os.path.join(model_folder_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(model_folder_path, 'lb.pkl'), 'rb') as f:
        lb = pickle.load(f)

    with open(os.path.join(model_folder_path, 'encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)

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

    output_file_path = performance_on_data_slices(model, data, cat_features, encoder, lb)

    assert isinstance(output_file_path, str)


# def test_compute_model_metrics(train_test_data):

#     model_path = 'model/model.pkl'

#     _, _, X_test, y_test = train_test_data

#     model = pickle.load(open(model_path, 'rb'))

#     preds = inference(model, X_test)

#     precision, recall, fbeta = compute_model_metrics(y_test, preds)

#     assert isinstance(precision, float)
#     assert isinstance(recall, float)
#     assert isinstance(fbeta, float)
