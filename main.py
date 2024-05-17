# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import os
import pandas as pd
from ml.data import process_data
from ml.model import inference

app = FastAPI()


cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

class data(BaseModel):
    age: int = Field(examples=[37])
    workclass: str = Field(examples=["Private"])
    fnlgt: int = Field(examples=[284582])
    education: str = Field(examples=['Masters'])
    education_num: int = Field(examples=[14])
    marital_status: str = Field(examples=['Married-civ-spouse'])
    occupation: str = Field(examples=['Exec-managerial'])
    relationship: str = Field(examples=['Wife'])
    race: str = Field(examples=['White'])
    sex: str = Field(examples=['Female'])
    capital_gain: int = Field(examples=[0])
    capital_loss: int = Field(examples=[0])
    hours_per_week: int = Field(examples=[40])
    native_country: str = Field(examples=['United-States'])


model_folder_path = './model/'

with open(os.path.join(model_folder_path, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(model_folder_path, 'lb.pkl'), 'rb') as f:
    lb = pickle.load(f)

with open(os.path.join(model_folder_path, 'encoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)


@app.get("/")
async def greetings():
    return "Welcome to my FastAPI app!"


@app.post("/predictions")
async def infrence(data: data):

    data = pd.DataFrame([dict(data)])
    data['salary'] = ''

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

    pred = inference(model, X)

    pred = list(lb.inverse_transform(pred))

    return {"predictions": pred}
